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


class SentenceDataset(Dataset):
    def __init__(self, sentences: list[Sentence]):
        self.raw_sentences : list[Sentence] = sentences
        self.pos_tags : list[list[int]] = [sentence.pos_tags for sentence in sentences]
        self.gold_spans : list[list[tuple[int, int]]] = [sentence.start_end_spans for sentence in sentences]
    def __len__(self):
        return len(self.raw_sentences)

    def __getitem__(self, idx):
        return self.pos_tags[idx], self.gold_spans[idx]
    
def collate_fn(batch):
    data, spans = zip(*batch)  # Unzip into data and indices

    list_max_len = max(len(lst) for lst in data)  # Find max length in batch
    batch_size = len(data)

    padded_batch = np.full((batch_size, list_max_len), fill_value=-1, dtype=int)

    for i, lst in enumerate(data):
        padded_batch[i, :len(lst)] = np.array(lst)

    return torch.tensor(padded_batch), spans




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

    def __post_init__(self):
        self._initialize_sentences()
        self._initialize_symbol_idx()
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
    

    def get_dataloader(self, batch_size: int) -> DataLoader:
        """
        Get a DataLoader for the corpus.
        :param batch_size: Batch size for the DataLoader.
        :return: DataLoader for the corpus.
        """
        dataset = SentenceDataset(self.sentences)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
