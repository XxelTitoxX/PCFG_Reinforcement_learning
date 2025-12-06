import torch
import torch.nn.functional as F

@torch.no_grad()
def viterbi_parse_sentence(actor_critic, sentence, device=None):
    """
    Viterbi-style bottom-up binary parse for a single sentence using the trained ActorCritic.
    - Uses the Conv/Transformer layer only once at the start.
    - Uses SymbolActor.actor_network as local label scorer.
    - Uses fusion (induction_embedder) to build constituent embeddings.

    Args:
        actor_critic: ActorCritic
        sentence: grammar_env.corpus.sentence.Sentence
        device: torch.device or None

    Returns:
        root_label: int
        backpointers: nested structure you can use to reconstruct the full tree
        score_chart: (L, n, n) tensor of best log-scores
    """
    actor_critic.eval()
    if device is None:
        device = next(actor_critic.parameters()).device

    # ----- 1) Encode sentence once -----
    # state: (1, T, E)
    state = actor_critic.encode_sentence([sentence]).to(device)
    T = state.shape[1]
    mask = torch.ones(1, T, dtype=torch.bool, device=device)

    # Conv/Transformer only here
    _, seq_emb = actor_critic.encode_state(state, mask)  # (1, T, E)
    seq_emb = seq_emb[0]  # (T, E)

    # actual length (if padding is possible)
    # here assume no padding; otherwise derive from sentence or mask
    n = T

    # ----- 2) Prepare chart structures -----
    L = actor_critic.action_dim
    E = actor_critic.embedding_dim

    NEG_INF = -1e9

    # score[l, i, j]
    score = seq_emb.new_full((L, n, n), NEG_INF)
    # emb[l, i, j, :]
    emb = seq_emb.new_zeros((L, n, n, E))
    # back[l][i][j] = (k, l_left, l_right) or None
    back = [[[None for _ in range(n)] for _ in range(n)] for _ in range(L)]

    # Precompute tag embeddings for all labels
    tag_ids = torch.arange(L, device=device)
    tag_emb_all = actor_critic.tag_embedder(tag_ids)  # (L, E)

    # Leaves: store word embeddings for spans of length 1, unlabeled
    leaf_emb = seq_emb  # (n, E)

    # ----- 3) Dynamic program over span length -----
    # length = span width (number of tokens)
    for length in range(2, n + 1):       # spans of length >= 2
        for i in range(0, n - length + 1):
            j = i + length - 1

            # Try all splits i..k | k+1..j
            for k in range(i, j):
                # Left/right embeddings for best subtree of those spans.
                # For length 1 we just use leaf_emb; for longer spans we
                # take the best label at that span.
                if k == i:
                    left_best_score = seq_emb.new_tensor(0.0)   # score 0 for leaf
                    left_best_emb = leaf_emb[i]                 # (E,)
                    left_best_label = None                      # no label at leaf
                else:
                    # best label at span (i, k)
                    span_scores = score[:, i, k]                # (L,)
                    l_left = torch.argmax(span_scores)          # scalar
                    left_best_score = span_scores[l_left]
                    left_best_emb = emb[l_left, i, k]           # (E,)
                    left_best_label = int(l_left.item())

                if k + 1 == j:
                    right_best_score = seq_emb.new_tensor(0.0)
                    right_best_emb = leaf_emb[j]
                    right_best_label = None
                else:
                    span_scores = score[:, k+1, j]
                    l_right = torch.argmax(span_scores)
                    right_best_score = span_scores[l_right]
                    right_best_emb = emb[l_right, k+1, j]
                    right_best_label = int(l_right.item())

                # ----- 3a) Score all parent labels for this split -----
                # SymbolActor expects pair embeddings; we can directly reuse its MLP
                pair_emb = torch.cat([left_best_emb, right_best_emb], dim=-1)  # (2E,)
                pair_emb = pair_emb.unsqueeze(0)  # (1, 2E)

                # local logits for all labels: (1, L)
                local_logits = actor_critic.symbol_actor.actor_network(pair_emb)
                local_logprobs = F.log_softmax(local_logits, dim=-1).squeeze(0)  # (L,)

                # ----- 3b) Compose embeddings for all parent labels -----
                # We want one candidate embedding per parent label.
                # batch_size = L
                tag_batch = tag_emb_all                              # (L, E)
                left_batch = left_best_emb.unsqueeze(0).expand(L, -1)   # (L, E)
                right_batch = right_best_emb.unsqueeze(0).expand(L, -1) # (L, E)
                new_emb_all = actor_critic.fuse_constituents(tag_batch,
                                                             left_batch,
                                                             right_batch,
                                                             dropout=False)     # (L, E)

                # ----- 3c) Total scores per parent label for this split -----
                total_scores = left_best_score + right_best_score + local_logprobs  # (L,)

                # ----- 3d) Update chart whenever we improve a label -----
                better = total_scores > score[:, i, j]  # (L,)
                if better.any():
                    improved_idx = torch.nonzero(better, as_tuple=True)[0]
                    score[improved_idx, i, j] = total_scores[improved_idx]
                    emb[improved_idx, i, j] = new_emb_all[improved_idx]

                    # store backpointers for those labels
                    for l_parent in improved_idx.tolist():
                        back[l_parent][i][j] = (k, left_best_label, right_best_label)

    # ----- 4) Choose best root label at span (0, n-1) -----
    root_scores = score[:, 0, n-1]  # (L,)
    root_label = int(torch.argmax(root_scores).item())
    root_score = root_scores[root_label].item()

    return root_label, root_score, back, score
