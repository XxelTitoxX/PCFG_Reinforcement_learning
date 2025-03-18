from logging import getLogger
from pathlib import Path

import jsbeautifier

from grammar_env.corpus.corpus import Corpus

__all__ = [
    'write_corpus_to_json', 'write_corpus_to_file'
]

logger = getLogger(__name__)


def write_corpus_to_json(corpus: Corpus) -> str:
    return corpus.to_json()


def write_corpus_to_file(corpus: Corpus, path: Path):
    with path.open('w') as file:
        json_str: str = write_corpus_to_json(corpus)
        json_str = jsbeautifier.beautify(json_str)
        file.write(json_str)
    logger.info(f"Wrote corpus of length {len(corpus)} to {path}")
