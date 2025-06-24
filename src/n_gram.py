from collections import defaultdict
from grammar_env.corpus.sentence import Sentence
from grammar_env.corpus.corpus import Corpus
import pickle
import math
import logging

logger = logging.getLogger(__name__)

class NGram:
    def __init__(self, n : int, padding_idx : int = -1):
        assert n > 1, "N-gram model only works for n > 1"
        self.n = n
        self.padding_idx = padding_idx
        self.ngrams_count = defaultdict(int)

    def count_sentence(self, sen : Sentence):
        list_pos = sen.pos_tags
        # We count subsequences of lengths l <= n
        # We count subsequences of length l < n two different ways : 1) as seq+[padding] if seq is not the end of sentence 2) as seq if seq is the end of sentence
        paddings = {k:(self.padding_idx,)*k for k in range(1, self.n+1)}
        for i in range(len(list_pos)):
            for j in range(1,self.n+1): # length of n-gram
                if i+1 < j:
                    break
                self.ngrams_count[paddings[j]] += 1 # count all subsequences of length j
                jgram = tuple(list_pos[i+1-j:i+1])
                self.ngrams_count[jgram] += 1
                if j< self.n:  # if j < n, we can add padding
                    if i < len(list_pos) - 1:  # if not the end of sentence, we can add padding
                        self.ngrams_count[jgram+paddings[1]] += 1 # count j-gram that have a next word
                    if i > 0:
                        self.ngrams_count[paddings[1]+jgram] += 1 # count j-gram that have a previous word

    def count_corpus(self, corpus : Corpus):
        for sen in corpus.sentences:
            self.count_sentence(sen)

    def compute_word_after(self, prefix : tuple, word : int):
        """Compute the probability of word following the prefix."""
        prefix_count = self.ngrams_count[prefix+(self.padding_idx,)]
        ngram_count = self.ngrams_count[prefix+(word,)]
        if prefix_count == 0:
            if len(prefix) == 0:
                raise ValueError("Count of words is zero")
            else:
                return self.compute_word_after(prefix[1:], word)  # if prefix is empty, we can use the padding as prefix
        if ngram_count == 0:
            return 1/ (prefix_count + 1)
        return ngram_count / prefix_count
    
    def compute_word_before(self, word : int, suffix : tuple):
        """Compute the probability of word preceding the suffix."""
        suffix_count = self.ngrams_count[(self.padding_idx,)+suffix]
        ngram_count = self.ngrams_count[(word,)+suffix]
        if suffix_count == 0:
            if len(suffix) == 0:
                raise ValueError("Count of words is zero")
            else:
                return self.compute_word_before(word, suffix[:-1])
        if ngram_count == 0:
            return 1/ (suffix_count + 1)
        return ngram_count / suffix_count

    def compute_prob(self, ngramA : tuple, ngramB : tuple):

        """Compute the average of the probabilities that ngramA is followed by ngramB in the corpus and ngramB is preceded by ngramA in the corpus."""
        assert len(ngramA) > 0 and len(ngramB) > 0, "N-grams must be non-empty"
        end_of_A = ngramA[len(ngramA)-self.n-1:]
        start_of_B = ngramB[:self.n-1]

        p_B_after_A = 1.
        prefix = end_of_A
        for i in range(len(ngramB)): # chain rule
            p_B_after_A *= self.compute_word_after(prefix, ngramB[i])

            if len(prefix) >= self.n-1:
                prefix = prefix[1:] + (ngramB[i],)
            else:
                prefix = prefix + (ngramB[i],)
        p_B_after_A = math.pow(p_B_after_A, 1/len(ngramB)) # Normalize by the length of ngramB
        
        p_A_before_B = 1.
        suffix = start_of_B
        for i in reversed(range(len(ngramA))): # chain rule
            p_A_before_B *= self.compute_word_before(ngramA[i], suffix)
            if len(suffix) >= self.n-1:
                suffix = (ngramA[i],) + suffix[:-1]
            else:
                suffix = (ngramA[i],) + suffix
        p_A_before_B = math.pow(p_A_before_B, 1/len(ngramA)) # Normalize by the length of ngramA

        average_prob = math.sqrt(p_B_after_A * p_A_before_B)
        #average_prob = (p_B_after_A + p_A_before_B) / 2
        #logger.info(f"Computed probability for A={ngramA} and B={ngramB}: {average_prob}")
        return average_prob

        


    def export_to_pkl(self, filename : str):
        with open(filename, "wb") as f:
            pickle.dump(self.ngrams_count, f)
    
    def import_from_pkl(self, filename : str):
        with open(filename, "rb") as f:
            self.ngrams_count = pickle.load(f)
        logger.info(f"Loaded {self.n}-gram model from {filename}")
        