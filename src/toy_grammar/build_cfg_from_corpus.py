from toy_grammar.gen_nary_grammar import TreeNode
from actor_critic import ActorCritic
from grammar_env.corpus.sentence import Sentence
import torch
from collections import defaultdict
from typing import DefaultDict
from grammar_env.corpus.corpus import Corpus
import argparse
import os
from pathlib import Path

def preprocess_rule_count(rule_count: DefaultDict[tuple[str], int]):
    nary_rule_count = defaultdict(int)
    nb_nary_rule_by_lhs = defaultdict(int)
    for rule, count in rule_count.items():
        if len(rule) >= 3:
            nary_rule_count[rule] = count
            nary_rule_count[(rule[0],)] += count
            nb_nary_rule_by_lhs[rule[0]] += count
    binary_frequencies = {rule: count / nary_rule_count[(rule[0],)] for rule, count in nary_rule_count.items()}

    print(f"Binary rule frequencies: {binary_frequencies}")

def rule_count_to_cfg(rule_count: DefaultDict[tuple[str], int]) -> str:
    """
    Converts a rule count dictionary to a string representation of a nltk.CFG.
    """
    #preprocess_rule_count(rule_count)
    def wrap_terminals(symbol: str) -> str:
        if symbol[:2] == 'TT':
            return f"'{symbol}'"
        return symbol
    cfg_lines = []
    for rule, count in rule_count.items():
        cfg_line = f"{rule[0]} -> {' '.join([wrap_terminals(sym) for sym in rule[1:]])}"
        if rule[0] == 'NT0':
            cfg_lines.insert(0, cfg_line)
        else:
            cfg_lines.append(f"{cfg_line}")
    return "\n".join(cfg_lines)

def fuse_and_shift(arr: torch.Tensor, idx: torch.Tensor, value: torch.Tensor, padding_value):
    """
    For each row i in 'arr', fuses the values at position 'idx[i]' and 'idx[i]+1' into 'value[i]' and shifts the rest of the elements to the left.
    The last element of each row is set to 'padding_value'.
    """
    assert torch.all((idx >= 0) & (idx < arr.shape[1] - 1)), "Index out of bounds"
    
    arr_copy = arr.clone()
    value = value.to(arr_copy.dtype)
    
    for i in range(arr.shape[0]):
        fusion_index = idx[i].item()
        # Insert fusion value at position idx[i]
        arr_copy[i, fusion_index] = value[i]
        # Shift left the elements after idx[i]+1
        arr_copy[i, fusion_index+1:-1] = arr[i, fusion_index+2:]
        # Pad last element
        arr_copy[i, -1] = padding_value
    
    return arr_copy

def parse_sentences(actor_critic: ActorCritic, batch_sentences: list[Sentence], device: torch.device, rule_count: DefaultDict[tuple[str], int]) -> list[TreeNode]:

    actor_critic.eval()

    batch_s_embeddings = actor_critic.encode_sentence(batch_sentences)
    current_states = batch_s_embeddings
    current_lengths = torch.tensor([len(sen.symbols) for sen in batch_sentences], device=device, dtype=torch.int32)
    tree_nodes : list[list[TreeNode]] = [[TreeNode(type='PT', index=pos_idx, children=[TreeNode(type='TT', index=word_idx)]) for pos_idx, word_idx in zip(sen.pos_tags, sen.symbols_idx)] for sen in batch_sentences]
    for sentence_tree_nodes in tree_nodes:
        for tree_node in sentence_tree_nodes:
            rule_count[(tree_node.symbol(), tree_node.children[0].symbol())] += 1
    sum_log_probs = torch.zeros(len(batch_sentences), device=device, dtype=torch.float32)

    while not torch.all(current_lengths == 1):
        not_done = current_lengths > 1
        not_done_range = torch.arange(not_done.sum().item(), device=device)
        not_done_idx = torch.arange(not_done.shape[0], device=device)[not_done]
        curr_mask = torch.arange(current_states.shape[1], device=device)[None, :] < current_lengths[:, None]
        curr_nd_mask = curr_mask[not_done]

        current_nd_states = current_states[not_done]
        position, position_log_prob, symbol, symbol_log_prob, state_value = actor_critic.act(current_nd_states, curr_nd_mask, max_prob=True)
        sum_log_probs[not_done] += position_log_prob + symbol_log_prob

        update_position, update_symbol = position, symbol

        # Compute new constituents embeddings
        left = current_states[not_done_range, update_position, :]
        right = current_states[not_done_range, update_position + 1, :]
        emb_sym = actor_critic.tag_embedder(update_symbol)
        new_constituent = actor_critic.fuse_constituents(emb_sym, left, right)
        
        new_states = fuse_and_shift(current_nd_states, update_position, new_constituent, 0.)
        current_states[not_done] = new_states

        # Update tree nodes and rule counts
        for i in range(not_done.sum().item()):
            left_child : TreeNode = tree_nodes[not_done_idx[i]][update_position[i]]
            right_child : TreeNode = tree_nodes[not_done_idx[i]][update_position[i] + 1]
            '''
            if right_child.type == 'NT' and right_child.index % 2 == 1: # If the right child is an artificial NT (binarization)
                parent : TreeNode = TreeNode(type='NT', index=update_symbol[i], children=[left_child]+right_child.children)
            else:
                parent : TreeNode = TreeNode(type='NT', index=update_symbol[i], children=[left_child, right_child])
            '''
            parent : TreeNode = TreeNode(type='NT', index=update_symbol[i], children=[left_child, right_child])
            tree_nodes[not_done_idx[i]][update_position[i]] = parent
            del tree_nodes[not_done_idx[i]][update_position[i] + 1]
            '''
            if parent.index % 2 == 0: # If the parent is a real NT (not an artificial one)
                rule_count[(sym.type+str(sym.index//2) if sym.type=='NT' else sym.type+str(sym.index) for sym in [parent]+parent.children)] += 1
            '''
            rule_count[(parent.symbol(), left_child.symbol(), right_child.symbol())] += 1



        current_lengths[not_done] -= 1
    
    tree_nodes = [tree[0] for tree in tree_nodes]
    return tree_nodes, sum_log_probs

def build_cfg_from_corpus(actor_critic: ActorCritic, device: torch.device, corpus: Corpus) -> str:
    rule_count: DefaultDict[tuple[str], int] = defaultdict(int)
    dataloader = corpus.get_dataloader(batch_size=128)
    for batch_sentences in dataloader:
        _, _ = parse_sentences(actor_critic, batch_sentences, device, rule_count)
    cfg_str = rule_count_to_cfg(rule_count)
    return cfg_str

def save_model_cfg(actor_critic: ActorCritic, device: torch.device, corpus: Corpus, filename:str):
    """
    Converts a corpus of sentences into a CFG string representation using the provided actor-critic model.
    
    :param actor_critic: The actor-critic model used for parsing sentences.
    :param device: The device on which the model is running (CPU or GPU).
    :param corpus: The corpus containing sentences to be parsed.
    :return: A string representation of the CFG derived from the corpus.
    """
    cfg_str = build_cfg_from_corpus(actor_critic, device, corpus)
    with open(filename, 'w') as f:
        f.write(cfg_str)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build CFG from corpus using an actor-critic model.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the saved actor-critic model.")
    parser.add_argument("--corpus_path", type=str, required=True, help="Path to the corpus file.")
    parser.add_argument("--num_nt", type=int, required=True, help="Number of non-terminals to use in the CFG.")
    parser.add_argument("--num_pt", type=int, required=True, help="Number of pre-terminals to use in the CFG.")
    parser.add_argument("--ac_embedding_dim", type=int, default=64, help="Embedding dimension for the actor-critic model.")
    parser.add_argument("--ac_n_layer", type=int, default=2, help="Number of layers in the actor-critic model.")
    parser.add_argument("--vocab_size", type=int, required=True, help="Vocabulary size for the actor-critic model.")
    args = parser.parse_args()

    # Load the corpus
    corpus = Corpus(args.corpus_path, max_vocab_size=args.vocab_size)

    # Load the actor-critic model
    actor_critic = ActorCritic(state_dim=args.num_nt + args.num_pt,
                              embedding_dim=args.ac_embedding_dim,
                              action_dim=args.num_nt,
                                n_layer=args.ac_n_layer,
                                num_heads=2,
                                vocab_size=corpus.vocab_size)
    actor_critic.load_state_dict(torch.load(args.model_path))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    actor_critic.to(device)

    current_dir = Path(__file__).resolve().parent
    output_dir = current_dir / "model_grammar"
    output_path = output_dir / "model_cfg.txt"

    # Build and save the CFG
    save_model_cfg(actor_critic, device, corpus, args.output_cfg)

    
    
    