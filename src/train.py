import argparse
import datetime
import logging
from pathlib import Path
from typing import Any

from device import get_device
from grammar_env.corpus.corpus import Corpus
from n_gram import NGram
from ppo import PPO, PPOConfig
from writer import Writer

logger = logging.getLogger(__name__)

TRAIN_DATASET_FILENAME: str = "ptb-train.txt"
#TRAIN_DATASET_FILENAME: str = "train_sentences.txt"
VALID_DATASET_FILENAME: str = "ptb-valid.txt"
#VALID_DATASET_FILENAME: str = "val_sentences.txt"


def datetime_tag() -> str:
    return datetime.datetime.now().strftime("%m%d_%H-%M-%S")


def train(name: str, persistent_dir: Path, args: argparse.Namespace, ppo_config: PPOConfig) -> None:
    logger.info(f"Training name: {name}, persistent_dir: {persistent_dir} with args: {args}")

    train_corpus: Corpus = Corpus(
        str(args.directory / TRAIN_DATASET_FILENAME), max_vocab_size=args.max_vocab_size, max_len=args.max_len, max_sentence_length=60, multiprocessing=not args.no_multiprocess
    )
    valid_corpus: Corpus = Corpus(str(args.directory / VALID_DATASET_FILENAME), multiprocessing=not args.no_multiprocess)
    train_corpus._initialize_symbol_idx()
    # train and valid corpus must use the same symbol_to_idx and idx_to_symbol for the correct results
    valid_corpus.symbol_to_idx = train_corpus.symbol_to_idx
    valid_corpus.idx_to_symbol = train_corpus.idx_to_symbol
    valid_corpus.vocab_size = train_corpus.vocab_size
    train_corpus._apply_symbol_idx()
    valid_corpus._apply_symbol_idx()

    n_gram: NGram = NGram(
        n=4, padding_idx=-1
    )
    #n_gram.import_from_pkl(str(args.directory / "four_gram.pkl"))

    writer: Writer = Writer(name, vars(ppo_config), args.use_wandb)
    ppo: PPO = PPO(
        train_corpus, valid_corpus, persistent_dir,
        writer, get_device(args.device),
        ppo_config,
        n_gram=n_gram,
    )
    ppo.learn(args.timesteps)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--max_vocab_size", type=int, default=10000)
    parser.add_argument("--max_len", type=int)
    parser.add_argument("--no_multiprocess", action='store_true',
                        help="Disable multiprocessing for corpus initialization")
    parser.add_argument("--timesteps", type=int, default=int(1e7))
    parser.add_argument("--device", type=str, default="cuda:0")

    # use probability criterion by default
    parser.add_argument("--criterion", type=str)
    parser.add_argument("--use_wandb", action='store_true')
    parser.add_argument("--num_non_terminals", type=int)

    parser.add_argument("--n_updates_per_iteration", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--gamma", type=float)
    parser.add_argument("--clip", type=float)
    parser.add_argument("--actor_weight", type=float)
    parser.add_argument("--critic_weight", type=float)
    parser.add_argument("--entropy_weight", type=float)

    parser.add_argument("--save_freq", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--min_ep_rews_threshold", type=float)
    parser.add_argument("--num_sentences_per_score", type=int)
    parser.add_argument("--num_sentences_per_batch", type=int)
    parser.add_argument("--max_num_steps", type=int)
    
    parser.add_argument("--embedding_dim", type=int)
    parser.add_argument("--n_layer", type=int)
    parser.add_argument("--n_head", type=int)
    parser.add_argument("--gradient_clip", type=float)
    parser.add_argument("--pure_reinforce", action='store_true')
    parser.add_argument("--name", type=str, default="")
    parser.add_argument("--nt_freq_offset", action='store_true')

    args: argparse.Namespace = parser.parse_args()

    ppo_config: PPOConfig = PPOConfig()
    hyperparameters: dict[str, Any] = {
        param: val for param, val in vars(args).items() if val is not None and hasattr(ppo_config, param)
    }
    ppo_config: PPOConfig = PPOConfig(**hyperparameters)

    name: str = f"{args.name}_{ppo_config.num_non_terminals}_{ppo_config.num_sentences_per_batch}_{datetime_tag()}_{'REINFORCE' if ppo_config.pure_reinforce else 'PPO'}"
    persistent_dir: Path = Path("log") / name
    persistent_dir.mkdir(parents=True, exist_ok=False)

    fh: logging.FileHandler = logging.FileHandler(
        persistent_dir / "training.log", mode="w"
    )
    sh: logging.StreamHandler = logging.StreamHandler()
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s -- %(message)s",
        datefmt="%m/%d/%Y %H-%M-%S",
        level=logging.INFO,
        handlers=[fh, sh]
    )

    train(name, persistent_dir, args, ppo_config)
