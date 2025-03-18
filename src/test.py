import argparse
import logging
from pathlib import Path

import numpy as np
import torch

from device import get_device
from grammar_env.corpus.corpus import Corpus
from grammar_env.criterion import CoverageCriterion, Criterion, F1Criterion, ProbabilityCriterion
from grammar_env.grammar.binary_grammar import BinaryGrammar
from grammar_env.grammar.unary_grammar import SupervisedUnaryGrammar, UnaryGrammar
from result_saver import Result

logger = logging.getLogger(__name__)


def test(args):
    device: torch.device = get_device(args.device)

    train_corpus: Corpus = Corpus(
        str(args.directory / "ptb-train.txt"), max_vocab_size=args.max_vocab_size, max_len=args.max_len
    )
    valid_corpus: Corpus = Corpus(str(args.directory / "ptb-valid.txt"))
    # train and valid corpus must use the same symbol_to_idx and idx_to_symbol for the correct results
    valid_corpus.symbol_to_idx = train_corpus.symbol_to_idx
    valid_corpus.idx_to_symbol = train_corpus.idx_to_symbol

    # Load the result from the result_path
    result: Result = Result.schema().loads(
        args.result_path.read_text()
    )
    # TODO: Once we also learn the unary grammar, it should be loaded from the result as well
    unary_grammar: UnaryGrammar = SupervisedUnaryGrammar(train_corpus).to(device)
    binary_grammar: BinaryGrammar = result.binary_grammar

    coverage_criterion: Criterion = CoverageCriterion(
        valid_corpus, device,
        len(valid_corpus), 256
    )
    f1_criterion: Criterion = F1Criterion(
        valid_corpus, device,
        len(valid_corpus), 256
    )
    probability_criterion: Criterion = ProbabilityCriterion(
        valid_corpus, device, -200.,
        len(valid_corpus), 256
    )

    coverage_criterion.score(binary_grammar, unary_grammar)
    f1_criterion.score(binary_grammar, unary_grammar)
    probability_criterion.score(binary_grammar, unary_grammar)

    coverage_scores: np.ndarray = coverage_criterion._scores.numpy(force=True)
    f1_scores: np.ndarray = f1_criterion._scores.numpy(force=True)
    probability_scores: np.ndarray = probability_criterion._scores.numpy(force=True)

    for i in range(len(valid_corpus)):
        logger.info(
            f"coverage: {coverage_scores[i]:.5f}, f1: {f1_scores[i]:.5f}, probability: {probability_scores[i]:.5f}"
        )

    covered: np.ndarray = coverage_scores > 0.
    assert np.all(f1_scores[~covered] == 0.)

    logger.info(f"coverage: {coverage_scores.mean():.5f}")
    logger.info(f"f1: {f1_scores.mean():.5f}")
    logger.info(f"probability: {probability_scores.mean():.5f}")

    logger.info(f"coverage_greedy: {coverage_scores[covered].mean():.5f}")
    logger.info(f"f1_greedy: {f1_scores[covered].mean():.5f}")
    logger.info(f"probability_greedy: {probability_scores[covered].mean():.5f}")


if __name__ == '__main__':
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s -- %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--max_vocab_size", type=int, default=10000)
    parser.add_argument("--max_len", type=int)

    parser.add_argument("--result_path", type=Path, required=True)

    parser.add_argument("--device", type=str, default="cuda:0")

    test(parser.parse_args())
