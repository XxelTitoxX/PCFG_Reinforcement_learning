from dataclasses import dataclass
from logging import getLogger
from pathlib import Path

import jsbeautifier
import torch
from dataclasses_json import dataclass_json

from grammar_env.corpus.corpus import Corpus
from grammar_env.criterion import CoverageCriterion, Criterion, F1Criterion, ProbabilityCriterion
from grammar_env.grammar.binary_grammar import BinaryGrammar
from grammar_env.grammar.unary_grammar import UnaryGrammar
from actor_critic import ActorCritic
from env import Environment
from writer import Writer

logger = getLogger(__name__)


@dataclass_json
@dataclass(frozen=True)
class Result:
    train_f1: float
    train_prob: float
    train_cov: float
    valid_f1: float
    valid_prob: float
    valid_cov: float
    #len: int
    #binary_grammar: BinaryGrammar


class ResultSaver:
    def __init__(
            self, persistent_dir: Path, writer: Writer,
            train_corpus: Corpus, valid_corpus: Corpus, device: torch.device,
            num_sentences_per_score: int, max_num_steps: int, binary_grammar: BinaryGrammar
    ):
        self.persistent_dir: Path = persistent_dir
        self.writer: Writer = writer
        self.train_corpus: Corpus = train_corpus
        self.valid_corpus: Corpus = valid_corpus
        self.device: torch.device = device
        self.num_sentences_per_score: int = num_sentences_per_score
        self.max_num_steps: int = max_num_steps

        self.train_f1_criterion: Criterion = F1Criterion(
            train_corpus, device
        )
        self.train_probability_criterion: Criterion = ProbabilityCriterion(
            train_corpus, device
        )
        self.train_coverage_criterion: Criterion = CoverageCriterion(
            train_corpus, device
        )
        self.valid_f1_criterion: Criterion = F1Criterion(
            valid_corpus, device
        )
        self.valid_probability_criterion: Criterion = ProbabilityCriterion(
            valid_corpus, device
        )
        self.valid_coverage_criterion: Criterion = CoverageCriterion(
            valid_corpus, device
        )

        self.train_env: Environment = Environment(num_sentences_per_score, max_num_steps, binary_grammar, self.train_f1_criterion, 0.0, device)
        self.valid_env: Environment = Environment(num_sentences_per_score, max_num_steps, binary_grammar, self.valid_f1_criterion, 0.0, device)

    def save(
            self, name: str, i_so_far: int, actor_critic: ActorCritic,
            commit: bool
    ):
        with torch.no_grad():
            self.train_env.rollout(self.train_corpus, actor_critic, self.num_sentences_per_score, self.max_num_steps)
            self.valid_env.rollout(self.valid_corpus, actor_critic, self.num_sentences_per_score, self.max_num_steps)
            self.train_env.criterion = self.train_f1_criterion
            self.valid_env.criterion = self.valid_f1_criterion
            train_f1: float = torch.mean(self.train_env.success_reward()).item()
            valid_f1: float = torch.mean(self.valid_env.success_reward()).item()

            self.train_env.criterion = self.train_probability_criterion
            self.valid_env.criterion = self.valid_probability_criterion
            train_prob: float = torch.mean(self.train_env.success_reward()).item()
            valid_prob: float = torch.mean(self.valid_env.success_reward()).item()

            self.train_env.criterion = self.train_coverage_criterion
            self.valid_env.criterion = self.valid_coverage_criterion
            train_cov: float = torch.mean(self.train_env.success_reward()).item()
            valid_cov: float = torch.mean(self.valid_env.success_reward()).item()

            result: Result = Result(
                train_f1=train_f1, train_prob=train_prob, train_cov=train_cov,
                valid_f1=valid_f1, valid_prob=valid_prob, valid_cov=valid_cov,
                #len=len(binary_grammar), binary_grammar=binary_grammar
            )

            logger.info(f"saving name: {name}, i_so_far: {i_so_far} result: {result}")

            path: Path = self.persistent_dir / "result" / f"{i_so_far}_{name}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            result_str: str = jsbeautifier.beautify(result.to_json())
            path.write_text(result_str)

            self.writer.log(
                {
                    f"train/{name}_f1": train_f1,
                    f"train/{name}_prob": train_prob,
                    f"train/{name}_cov": train_cov,
                    f"valid/{name}_f1": valid_f1,
                    f"valid/{name}_prob": valid_prob,
                    f"valid/{name}_cov": valid_cov,
                }, commit=commit
            )
