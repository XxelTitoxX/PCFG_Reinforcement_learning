from collections import defaultdict
from grammar_env.corpus.sentence import Sentence
from grammar_env.corpus.corpus import Corpus
import pickle
import math
import logging

logger = logging.getLogger(__name__)

def sigmoid(x: float) -> float:
    """Sigmoid function."""
    return 1 / (1 + math.exp(-x))

def prob_norm(p: float, reference_prob: float) -> float:
    if p < reference_prob:
        return p
    extended_p = reference_prob + (p - reference_prob) * 3
    return min(extended_p, 1.0)

class NGram:
    def __init__(self, n : int, padding_idx : int = -1):
        assert n > 1, "N-gram model only works for n > 1"
        self.n = n
        self.padding_idx = padding_idx
        self.ngrams_count = defaultdict(int)
        self.reference_prob : float = 1/14 # Reference probability for the n-gram model, used for normalization

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
    
    def compute_B_after_A(self, ngramA : tuple, ngramB : tuple):
        """Compute the probability that ngramB follows ngramA in the corpus."""
        assert len(ngramA) > 0 and len(ngramB) > 0, "N-grams must be non-empty"
        end_of_A = ngramA[len(ngramA)-self.n-1:]
        p_B_after_A = 1.
        prefix = end_of_A
        for i in range(len(ngramB)):
            p_B_after_A *= self.compute_word_after(prefix, ngramB[i])
            if len(prefix) >= self.n-1:
                prefix = prefix[1:] + (ngramB[i],)
            else:
                prefix = prefix + (ngramB[i],)
        p_B_after_A = math.pow(p_B_after_A, 1/len(ngramB))
        return p_B_after_A
    
    def compute_A_before_B(self, ngramA : tuple, ngramB : tuple):
        """Compute the probability that ngramA precedes ngramB in the corpus."""
        assert len(ngramA) > 0 and len(ngramB) > 0, "N-grams must be non-empty"
        start_of_B = ngramB[:self.n-1]
        p_A_before_B = 1.
        suffix = start_of_B
        for i in reversed(range(len(ngramA))):
            p_A_before_B *= self.compute_word_before(ngramA[i], suffix)
            if len(suffix) >= self.n-1:
                suffix = (ngramA[i],) + suffix[:-1]
            else:
                suffix = (ngramA[i],) + suffix
        p_A_before_B = math.pow(p_A_before_B, 1/len(ngramA))
        return p_A_before_B

    def compute_prob(self, ngramA : tuple, ngramB : tuple):
        """Compute the probability of associating ngramA and ngramB as one grammatical constituent."""
        assert len(ngramA) > 0 and len(ngramB) > 0, "N-grams must be non-empty"
        
        p_A_before_B = self.compute_A_before_B(ngramA, ngramB)
        p_B_after_A = self.compute_B_after_A(ngramA, ngramB)
        average_prob = math.sqrt(p_A_before_B * p_B_after_A)
        return average_prob
    
    def compute_attraction(self, left : tuple, center : tuple, right : tuple):
        """
        Compute the signed attraction of each constituent in the sentence.
        A positive attraction means that the center constituent is more likely to be associated with the left constituent than with the right one.
        The reason for this scoring method is that the two best constituents to merge are the ones situated at the most extreme right of their parent node.
        If the center constituent is the right edge of the parent node, it is more likely to be associated with the left constituent. 
        """
        assert len(center) > 0, "Center constituent must be non-empty"
        if  len(right) == 0:
            right_attraction = self.reference_prob
        else:
            #right_attraction = self.compute_B_after_A(center, right)
            right_attraction = self.compute_prob(center, right)
        if len(left) == 0:
            left_attraction = self.reference_prob
        else:
            #left_attraction = self.compute_A_before_B(left, center)
            left_attraction = self.compute_prob(left, center)
        attraction = sigmoid((left_attraction - right_attraction)/self.reference_prob)
        #attraction = max(left_attraction - right_attraction, 0) / self.reference_prob
        #attraction = prob_norm(attraction, self.reference_prob)
        #attraction = prob_norm(left_attraction, self.reference_prob)
        return attraction
        #return math.sqrt(left_attraction * self.compute_B_after_A(left, center))
        


    def export_to_pkl(self, filename : str):
        with open(filename, "wb") as f:
            pickle.dump(self.ngrams_count, f)
    
    def import_from_pkl(self, filename : str):
        with open(filename, "rb") as f:
            self.ngrams_count = pickle.load(f)
        logger.info(f"Loaded {self.n}-gram model from {filename}")
        