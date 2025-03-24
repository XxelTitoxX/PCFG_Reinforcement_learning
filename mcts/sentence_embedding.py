import numpy as np
import pandas as pd
import spacy
from sklearn.decomposition import PCA
from collections import Counter
from sklearn.preprocessing import normalize

# Load Spacy model for tokenization
nlp = spacy.load("en_core_web_sm")

# Sample corpus
corpus = [
    "The quick brown fox jumps over the lazy dog",
    "The dog barks at the fox",
    "Foxes are wild animals",
    "Dogs are loyal animals"
]

# Tokenize and preprocess
sentences = [[token.lemma_.lower() for token in nlp(sentence) if token.is_alpha] for sentence in corpus]

# Build vocabulary
vocab = sorted(set(word for sentence in sentences for word in sentence))
word2idx = {word: i for i, word in enumerate(vocab)}

# Compute word-word co-occurrence matrix (PPMI weighting optional)
window_size = 2
co_matrix = np.zeros((len(vocab), len(vocab)))
for sentence in sentences:
    for i, word in enumerate(sentence):
        word_idx = word2idx[word]
        for j in range(max(0, i - window_size), min(len(sentence), i + window_size + 1)):
            if i != j:  # Avoid self-co-occurrence
                co_matrix[word_idx, word2idx[sentence[j]]] += 1

# Apply PCA for dimensionality reduction
pca = PCA(n_components=50)  # Reduce to 50D (adjust as needed)
word_embeddings = pca.fit_transform(co_matrix)

# Compute word frequency (for weighted sentence embeddings)
word_freq = Counter(word for sentence in sentences for word in sentence)
word_weights = {word: 1 / (word_freq[word] + 1e-5) for word in vocab}  # Smooth inverse frequency

# Compute sentence embeddings with frequency-based weighting
def get_sentence_embedding(sentence):
    words = [word2idx[word] for word in sentence if word in word2idx]
    if not words:
        return np.zeros(pca.n_components)  # Return zero vector if sentence has no known words
    weighted_vectors = np.array([word_embeddings[idx] * word_weights[vocab[idx]] for idx in words])
    return np.mean(weighted_vectors, axis=0)

sentence_embeddings = np.array([get_sentence_embedding(sentence) for sentence in sentences])

# Remove common component (first principal component)
def remove_common_component(embeddings):
    pca = PCA(n_components=1)
    common_component = pca.fit_transform(embeddings)
    return embeddings - pca.inverse_transform(common_component)

sentence_embeddings = remove_common_component(sentence_embeddings)

# Output embeddings
print("Final Sentence Embeddings:")
print(sentence_embeddings)
