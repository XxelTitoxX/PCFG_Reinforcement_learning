from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from logging import getLogger
from typing import Optional
from torch.utils.data import DataLoader, Dataset
import numpy as np
import torch

from dataclasses_json import dataclass_json

from grammar_env.corpus.sentence import Sentence

__all__ = ['Corpus']

logger = getLogger(__name__)


class POSDataset(Dataset):
    def __init__(self, pos_tags: list[list[int]]):
        self.sentences = pos_tags
    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return self.sentences[idx]
    
def collate_fn(batch):
    list_max_len = max(len(lst) for lst in batch)  # Find max length in batch
    batch_size = len(batch)

    # Create a (batch_size, list_max_len) array filled with zeros
    padded_batch = np.zeros((batch_size, list_max_len), dtype=int)
    padded_batch.fill(-1)  # Fill with padding index (indexes are shifted by 1 before passed to the embedding layers)

    # Copy each list into the padded array
    for i, lst in enumerate(batch):
        padded_batch[i, :len(lst)] = np.array(lst)  # Fill up to the list's length, 0 is reserved for padding

    return padded_batch



@dataclass_json
@dataclass
class Corpus:
    ptb_path: str
    min_sentence_length: Optional[int] = 1
    max_sentence_length: Optional[int] = 60
    max_vocab_size: Optional[int] = None
    max_len: Optional[int] = 40000

    sentences: list[Sentence] = field(init=False)
    symbol_count: list[tuple[str, int]] = field(init=False)
    symbol_to_idx: dict[str, int] = field(init=False)
    idx_to_symbol: dict[int, str] = field(init=False)
    pos_tags: list[list[int]] = field(init=False)

    def __post_init__(self):
        self._initialize_sentences()
        self._initialize_symbol_idx()
        self.__get_pos_tag()
        logger.info(
            f"Read corpus of length {len(self)} from {self.ptb_path}, "
            f"min_sentence_length={self.min_sentence_length}, "
            f"max_sentence_length={self.max_sentence_length}, "
            f"max_vocab_size={self.max_vocab_size}, "
            f"max_len={self.max_len}"
        )

    def _initialize_sentences(self) -> None:
        with open(self.ptb_path, 'r') as file:
            lines = file.readlines()
        with ProcessPoolExecutor(max_workers=4) as executor:
            all_sentences = list(executor.map(Sentence, lines))

        # Filter sentences based on length criteria.
        self.sentences = []
        for sentence in all_sentences:
            if self.min_sentence_length is not None and len(sentence) < self.min_sentence_length:
                continue
            if self.max_sentence_length is not None and len(sentence) > self.max_sentence_length:
                continue
            self.sentences.append(sentence)
        if self.max_len is not None:
            self.sentences = self.sentences[:self.max_len]

    def _initialize_symbol_idx(self) -> None:
        symbols_count: defaultdict[str, int] = defaultdict(int)
        for sentence in self.sentences:
            for symbol in sentence.symbols:
                symbols_count[symbol] += 1

        self.symbol_count: list[tuple[str, int]] = sorted(
            symbols_count.items(), key=lambda x: x[1], reverse=True
        )

        self.symbol_to_idx = {'<pad>': 0, '<unk>': 1, '<num>': 2}
        """
        Dictionary of symbols to idx. 
        When indexing the dictionary, one must not index it directly. 
        Rather, use get(key, 1) to map unknown symbols to the <unk> index.
        """
        self.idx_to_symbol = {0: '<pad>', 1: '<unk>', 2: '<num>'}
        """
        Dictionary of idx to symbols.
        You can index this dictionary directly.
        """

        if self.max_vocab_size is not None:
            symbol_count = self.symbol_count[:self.max_vocab_size]
        else:
            symbol_count = self.symbol_count

        symbol_count = [(symbol, count) for symbol, count in symbol_count if symbol not in self.symbol_to_idx]
        for idx, (symbol, _) in enumerate(symbol_count, 3):
            self.symbol_to_idx[symbol] = idx
            self.idx_to_symbol[idx] = symbol

    def __len__(self):
        return len(self.sentences)
    
    def __get_pos_tag(self) :
        pos_tags : list[int] = []


        pos_cluster: dict[str, str] = {
            "CC": "CC",

            "CD": "CD",

            "DT": "DT",
            "PDT": "DT",

            "IN": "IN",

            "JJ": "JJ",
            "JJR": "JJ",
            "JJS": "JJ",

            "NN": "NN",
            "NNS": "NN",
            "NNP": "NN",
            "NNPS": "NN",

            "PRP": "PRP",
            "PRP$": "PRP",

            "RB": "RB",
            "RBR": "RB",
            "RBS": "RB",

            "TO": "TO",

            "VB": "VB",
            "VBD": "VB",
            "VBG": "VB",
            "VBN": "VB",
            "VBP": "VB",
            "VBZ": "VB",

            "WDT": "WH-",
            "WP": "WH-",
            "WP$": "WH-",
            "WRB": "WH-",

            "MD": "MD",

            "POS": "POS",

            "EX": "ETC",
            "FW": "ETC",
            "LS": "ETC",
            "RP": "ETC",
            "SYM": "ETC",
            "UH": "ETC",
        }

        pos_cluster_to_idx: dict[str, int] = {
            "CC": 0,
            "CD": 1,
            "DT": 2,
            "IN": 3,
            "JJ": 4,
            "NN": 5,
            "PRP": 6,
            "RB": 7,
            "TO": 8,
            "VB": 9,
            "WH-": 10,
            "MD": 11,
            "POS": 12,
            "ETC": 13
        }

        for sentence in self.sentences:
            sentence_pos_tags = []
            for action in sentence.actions_sanitized:
                match action:
                    case Shift(pos_tag, symbol):
                        sentence_pos_tags.append(pos_cluster_to_idx[pos_cluster[pos_tag]])
                    case _:
                        pass
            pos_tags.append(sentence_pos_tags)
        self.pos_tags = pos_tags

    def get_dataloader(self, batch_size: int) -> DataLoader:
        """
        Get a DataLoader for the corpus.
        :param batch_size: Batch size for the DataLoader.
        :return: DataLoader for the corpus.
        """
        dataset = POSDataset(self.pos_tags)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
