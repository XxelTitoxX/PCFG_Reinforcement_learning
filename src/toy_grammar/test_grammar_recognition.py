from pathlib import Path
import sys

CURRENT_DIR = Path(__file__).resolve().parent
src_dir = CURRENT_DIR.parent
if str(src_dir) not in sys.path:
    sys.path.append(str(src_dir))

from gen_nary_grammar import CFG, to_nltk_cfg, nltk_cfg_to_string
from grammar_env.corpus.corpus import Corpus
from ppo import PPOConfig, PPO
import argparse
from pathlib import Path
from writer import Writer
from device import get_device
from build_cfg_from_corpus import save_model_cfg
from nltk import CFG as NLTK_CFG
from nltk.parse import EarleyChartParser
import signal

class TimeoutException(Exception):
    pass

def handler(signum, frame):
    raise TimeoutException("Operation timed out!")

# Set the signal handler
signal.signal(signal.SIGALRM, handler)

def save_to_file(text, file_path:Path):
    if isinstance(text, list):
        text = '\n'.join(text)
    with open(file_path, 'w') as f:
        f.write(text)

def coverage_f1(n_test_grammars:int):
    tp:int = 0
    fp:int = 0
    fn:int = 0
    parsing_errors:int = 0
    interrupt_errors:int = 0
    timeout_errors:int = 0
    for i in range(n_test_grammars):
        target_grammar_path = CURRENT_DIR / 'target_grammar' / f'target_cfg_{i}.txt'
        model_grammar_path = CURRENT_DIR / 'model_grammar' / f'model_cfg_{i}.txt'
        
        with open(target_grammar_path, 'r') as f:
            target_grammar = NLTK_CFG.fromstring(f.read())
        terminal_productions = [str(prod) for prod in target_grammar.productions()
                        if all(isinstance(sym, str) for sym in prod.rhs())]
        with open(model_grammar_path, 'r') as f:
            model_productions = f.readlines()
            total_productions = list(set(model_productions + terminal_productions))
            model_grammar_string = '\n'.join(total_productions)
            model_grammar = NLTK_CFG.fromstring(model_grammar_string)


        target_parser = EarleyChartParser(target_grammar)
        model_parser = EarleyChartParser(model_grammar)
        
        for j in range(n_test_grammars):
            print(f"Parsing corpus {j} with grammars {i}")
            test_corpus_path = CURRENT_DIR / 'target_grammar' / f'test_corpus_{j}.txt'
            with open(test_corpus_path, 'r') as f:
                sentence_idx = 0
                for test_sentence in f:
                    print(f"Parsing sentence {sentence_idx} of corpus {j} with grammars {i}")
                    test_sentence = test_sentence.strip().split()
                    if len(test_sentence)==0:
                        continue
                    err_target = False
                    err_model = False
                    # Set a timeout for parsing
                    signal.alarm(10)  # Set a timeout of 5 seconds
                    try:
                        target_parse_trees = list(target_parser.parse(test_sentence))
                        signal.alarm(0)  # Disable the alarm
                    except ValueError as e:
                        print(f"ERROR PARSING SENTENCE {test_sentence} WITH TARGET GRAMMAR {i}: {e}")
                        print(f"THIS SHOULD NOT HAPPEN")
                        input("Press Enter to continue...")
                        parsing_errors += 1
                        target_parse_trees = []
                        err_target = True
                    except KeyboardInterrupt as e:
                        print(f"Interrupted while parsing sentence {test_sentence} with target grammar {i}: {e}")
                        interrupt_errors += 1
                        target_parse_trees = []
                        err_target = True
                    except TimeoutException as e:
                        print(f"Timeout while parsing sentence {test_sentence} with target grammar {i}: {e}")
                        timeout_errors += 1
                        target_parse_trees = []
                        err_target = True
                    if err_target:
                        print(f"Skipping model parsing due to target grammar error")
                        continue
                    parsed_by_target = len(target_parse_trees) > 0
                    signal.alarm(10)
                    try:
                        model_parse_trees = list(model_parser.parse(test_sentence))
                        signal.alarm(0)  # Disable the alarm
                    except ValueError as e:
                        #print(f"Error parsing sentence {test_sentence} with model grammar {i}: {e}")
                        parsing_errors += 1
                        model_parse_trees = []
                        err_model = True
                    except KeyboardInterrupt as e:
                        print(f"Interrupted while parsing sentence {test_sentence} with model grammar {i}: {e}")
                        interrupt_errors += 1
                        model_parse_trees = []
                        err_model = True
                    except TimeoutException as e:
                        print(f"Timeout while parsing sentence {test_sentence} with model grammar {i}: {e}")
                        timeout_errors += 1
                        model_parse_trees = []
                        err_model = True
                    if err_model:
                        print(f"Skipping precision/recall update due to model grammar error")
                        continue
                    parsed_by_model = len(model_parse_trees) > 0
                    
                    if parsed_by_target and parsed_by_model:
                        tp += 1
                    elif parsed_by_target and not parsed_by_model:
                        fn += 1
                    elif not parsed_by_target and parsed_by_model:
                        fp += 1
                    sentence_idx += 1
        print("STATISTICS SO FAR:")
        print(f"TP: {tp} | FP: {fp} | FN: {fn}")
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1_score:.4f}")
    print(f"Parsing errors: {parsing_errors} | Interrupt errors: {interrupt_errors} | Timeout errors: {timeout_errors}")
    print(f"Total errors: {parsing_errors + interrupt_errors + timeout_errors}")
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f1_score
    



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a CFG from a file containing grammar rules.")
    parser.add_argument('--n_non_terminals', type=int, required=True, help="Number of non-terminals in the grammar.")
    parser.add_argument('--n_pre_terminals', type=int, required=True, help="Number of pre-terminals in the grammar.")
    parser.add_argument('--n_terminals', type=int, required=True, help="Number of terminals in the grammar.")
    parser.add_argument('--n_nary_rules', type=int, required=True, help="Number of n-ary rules in the grammar.")
    parser.add_argument('--max_pt_per_terminal', type=int, default=2, help="Maximum number of pre-terminals per terminal.")
    parser.add_argument('--ambiguity_prob', type=float, default=0.1, help="Probability of ambiguity in the grammar.")
    parser.add_argument('--max_unary_rules', type=int, default=None, help="Maximum number of unary rules in the grammar.")
    parser.add_argument('--value_of_nt', type=float, default=2.0, help="Value of non-terminals in the grammar.")
    parser.add_argument('--max_value_per_rhs', type=int, default=7, help="Maximum value per right-hand side in the grammar.")
    parser.add_argument('--n_test_sentences', type=int, default=2000, help="Number of test sentences to generate.")
    parser.add_argument('--n_train_sentences', type=int, default=10000, help="Number of sentences from target grammar for the model training.")
    parser.add_argument('--max_length', type=int, default=70, help="Maximum length of the sentences to generate.")
    parser.add_argument('--n_grammars', type=int, default=8, help="Number of target grammars to generate.")
    parser.add_argument('--gen_grammars', action='store_true', help="Generate target grammars and save them to files.")

    args = parser.parse_args()

    cfg_params = {
        'n_non_terminals': args.n_non_terminals,
        'n_pre_terminals': args.n_pre_terminals,
        'n_terminals': args.n_terminals,
        'n_nary_rules': args.n_nary_rules,
        'max_pt_per_terminal': args.max_pt_per_terminal,
        'ambiguity_prob': args.ambiguity_prob,
        'max_unary_rules': args.max_unary_rules,
        'value_of_nt': args.value_of_nt,
        'max_value_per_rhs': args.max_value_per_rhs,
    }

    for i in range(args.n_grammars):
        target_grammar_path = CURRENT_DIR / 'target_grammar' / f'target_cfg_{i}.txt'
        training_corpus_path = CURRENT_DIR / 'target_grammar' / f'training_corpus_{i}.txt'
        val_corpus_path = CURRENT_DIR / 'target_grammar' / f'val_corpus_{i}.txt'
        test_corpus_path = CURRENT_DIR / 'target_grammar' / f'test_corpus_{i}.txt'
        if args.gen_grammars:
            target_grammar = CFG(**cfg_params)
            target_grammar.random_initialize()
            target_grammar.save_grammar_as_nltk(target_grammar_path)
            sentences = target_grammar.generate_unique_strings(args.n_train_sentences, args.max_length, format='ptb')
            print(f"Generated {len(sentences)} sentences for target grammar {i} with max length {args.max_length}")
            val_split_idx = int(0.8 * len(sentences))
            save_to_file(sentences[:val_split_idx], training_corpus_path)
            save_to_file(sentences[val_split_idx:], val_corpus_path)
            test_sentences = target_grammar.generate_unique_strings(args.n_test_sentences, args.max_length, format='raw')
            save_to_file(test_sentences, test_corpus_path)
            print(f"Generated target grammar {i} with {len(sentences)} training sentences and {len(test_sentences)} test sentences, saved to {str(CURRENT_DIR / 'target_grammar')}")

        # Load the corpus
        train_corpus = Corpus(training_corpus_path)
        val_corpus = Corpus(val_corpus_path)
        train_corpus._initialize_symbol_idx()
        # train and valid corpus must use the same symbol_to_idx and idx_to_symbol for the correct results
        val_corpus.symbol_to_idx = train_corpus.symbol_to_idx
        val_corpus.idx_to_symbol = train_corpus.idx_to_symbol
        val_corpus.vocab_size = train_corpus.vocab_size
        train_corpus._apply_symbol_idx()
        val_corpus._apply_symbol_idx()

        ppo_config = PPOConfig(num_non_terminals= args.n_non_terminals, num_epochs=2, num_sentences_per_score=128, gamma=0.0)
        writer = Writer(f"grammar_induction_{i}", vars(ppo_config), use_wandb=False)
        device = get_device("cuda:0")
        ppo: PPO = PPO(
            train_corpus, val_corpus, CURRENT_DIR / "model_grammar",
            writer, device,
            ppo_config,
            n_gram=None,
        )
        ppo.learn(10000000)

        actor_critic = ppo.actor_critic

        output_dir = CURRENT_DIR / "model_grammar"
        output_path = output_dir / f"model_cfg_{i}.txt"

        # Build and save the CFG
        save_model_cfg(actor_critic, device, train_corpus, output_path)
        print(f"Model grammar {i} saved to {output_path}")

    precision, recall, f1_score = coverage_f1(args.n_grammars)
    print(f"Final coverage statistics after testing {args.n_grammars} grammars:")
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1_score:.4f}")