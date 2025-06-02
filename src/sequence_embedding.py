
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
from grammar_env.corpus.sentence import Sentence




class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, max_seq_len=1000):
        super().__init__()
        
        # Create position encodings once and re-use
        pe = torch.zeros(max_seq_len, embedding_dim)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-np.log(10000.0) / embedding_dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        # Register buffer (won't be updated during backprop)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class EnhancedTransformerEmbedding(nn.Module):
    def __init__(self, state_dim, embedding_dim, num_layers, max_seq_len=60, num_heads=2, dropout=0.1):
        super().__init__()

        self.action_embedding = nn.Embedding(state_dim + 1, embedding_dim, padding_idx=0)  # Adjusted for padding
        self.embedding_norm = nn.LayerNorm(embedding_dim)
        self.positional_encoding = PositionalEncoding(embedding_dim, max_seq_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim*2,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))  # CLS token for global representation

    def forward(self, action_sequence):
        """
        action_sequence: Tensor of shape (batch, max_seq_len) with -1 as null
        """
        batch_size = action_sequence.size(0)

        action_sequence = action_sequence + 1  # Shift -1 to 0 for padding
        mask = action_sequence != 0  # (batch, seq_len)

        # Embed actions
        x = self.action_embedding(action_sequence)  # (batch, seq_len, embedding_dim)
        x = self.positional_encoding(x)
        #x = self.embedding_norm(x)

        # Add CLS token at position 0
        cls_token = self.cls_token.expand(batch_size, -1, -1)  # (batch, 1, embedding_dim)
        x = torch.cat([cls_token, x], dim=1)  # (batch, seq_len + 1, embedding_dim)

        # Update the mask (CLS token is not masked)
        cls_mask = torch.ones(batch_size, 1, dtype=torch.bool, device=mask.device)
        mask = torch.cat([cls_mask, mask], dim=1)  # (batch, seq_len + 1)

        attn_mask = ~mask  # Transformer expects True for padding
        x = self.transformer_encoder(x, src_key_padding_mask=attn_mask)

        # Extract the CLS token embedding (global representation)
        cls_representation = x[:, 0]  # (batch, embedding_dim)

        encoded_sequence = x[:, 1:]

        return cls_representation, encoded_sequence
    

class MLPEncoder(nn.Module):
    def __init__(self, state_dim : int, embedding_dim : int):
        """
        Initialize a sequence encoder that converts integer-indexed symbols to embeddings.
        
        Args:
            state_dim (int): The number of possible symbol values (excluding padding)
            embedding_dim (int): The dimension of the output embeddings
            hidden_dims (list): Dimensions of hidden layers in the MLP
        """
        super(MLPEncoder, self).__init__()
        
        # Combine all layers into a sequential model
        self.mlp = nn.Sequential(nn.Linear(state_dim, embedding_dim),
                                 nn.ReLU(),
                                 nn.Linear(embedding_dim, embedding_dim),
                                 )
        self.state_dim = state_dim
        self.embedding_dim = embedding_dim
        
    def forward(self, x):
        """
        Forward pass through the encoder.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, max_seq_len)
                             with integer indices and -1 as padding
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, max_seq_len, embedding_dim)
        """
        batch_size, max_seq_len = x.shape
        
        # Replace padding indices with 0 to avoid index errors when one-hot encoding
        x = torch.clamp(x, min=0)
        
        # Convert to one-hot encoding
        x_one_hot = F.one_hot(x, num_classes=self.state_dim).float()  # Shape: (batch_size, max_seq_len, state_dim)
        
        # Process each sequence element through the MLP
        # Reshape for easier processing
        x_flat = x_one_hot.view(-1, self.state_dim)  # Shape: (batch_size * max_seq_len, state_dim)
        
        # Forward through MLP
        embeddings_flat = self.mlp(x_flat)  # Shape: (batch_size * max_seq_len, embedding_dim)
        
        # Reshape back to sequence form
        embeddings = embeddings_flat.view(batch_size, max_seq_len, -1)  # Shape: (batch_size, max_seq_len, embedding_dim)
        
        # The embeddings for padding positions don't matter according to the requirements
        # But we could zero them out if needed with: embeddings = embeddings * mask
        
        return embeddings
    

class TransformerEmbedding(nn.Module):
    def __init__(self, state_dim, embedding_dim, num_layers, max_seq_len=60, num_heads=2, dropout=0.1):
        super().__init__()

        self.action_embedding = MLPEncoder(state_dim, embedding_dim)  # Adjusted for padding
        self.embedding_norm = nn.LayerNorm(embedding_dim)
        self.positional_encoding = PositionalEncoding(embedding_dim, max_seq_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim*2,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))  # CLS token for global representation

    def forward(self, action_sequence):
        """
        action_sequence: Tensor of shape (batch, max_seq_len) with -1 as null
        """
        batch_size = action_sequence.size(0)

        mask = action_sequence != -1  # (batch, seq_len)

        # Embed actions
        x = self.action_embedding(action_sequence.long())  # (batch, seq_len, embedding_dim)
        x = self.positional_encoding(x)
        #x = self.embedding_norm(x)

        # Add CLS token at position 0
        cls_token = self.cls_token.expand(batch_size, -1, -1)  # (batch, 1, embedding_dim)
        x = torch.cat([cls_token, x], dim=1)  # (batch, seq_len + 1, embedding_dim)

        # Update the mask (CLS token is not masked)
        cls_mask = torch.ones(batch_size, 1, dtype=torch.bool, device=mask.device)
        mask = torch.cat([cls_mask, mask], dim=1)  # (batch, seq_len + 1)

        attn_mask = ~mask  # Transformer expects True for padding
        x = self.transformer_encoder(x, src_key_padding_mask=attn_mask)

        # Extract the CLS token embedding (global representation)
        cls_representation = x[:, 0]  # (batch, embedding_dim)

        encoded_sequence = x[:, 1:]

        return cls_representation, encoded_sequence
    
from transformers import XLNetTokenizer, XLNetModel

class XLNetWordEmbedder(torch.nn.Module):
    def __init__(self, embedding_dim, model_name='xlnet-base-cased'):
        super().__init__()
        self.tokenizer = XLNetTokenizer.from_pretrained(model_name)
        self.model = XLNetModel.from_pretrained(model_name)
        self.model.eval()  # no dropout in eval mode
        self.embedding_dim = embedding_dim
        self.projection = nn.Linear(self.model.config.hidden_size, embedding_dim)
    
    def forward(self, batch_sentences : list[Sentence]):
        """
        Args:
            batch_sentences (list[Sentence]): A batch of sentences
        Returns:
            torch.Tensor: A tensor of shape (batch_size, sequence_length, embedding_dim) cotaining word-level embeddings
        """
        device = next(self.parameters()).device
        # Join word tokens into sentences (XLNet tokenizer expects string input)
        batch_word_sentences = [sentence.symbols for sentence in batch_sentences]
        sentences = [" ".join(words) for words in batch_word_sentences]
        
        # Tokenize sentences as a batch
        encoded_inputs = self.tokenizer(
            sentences,
            return_tensors='pt',
            padding=True,
            truncation=True,
            is_split_into_words=False  # we joined them above
        )
        
        # Move tensors to the correct device
        encoded_inputs = {k: v.to(device) for k, v in encoded_inputs.items()}
        
        # Forward pass through XLNet
        with torch.no_grad():
            outputs = self.model(**encoded_inputs)
        
        # Last hidden states: (batch_size, sequence_length, hidden_size)
        last_hidden_states = outputs.last_hidden_state

        projected_states = self.projection(last_hidden_states)  # Project to embedding_dim
        # The output shape is (batch_size, sequence_length, embedding_dim)
        
        return projected_states
    
class TagEmbedder(nn.Module):
    def __init__(self, tag_dim, embedding_dim):
        super(TagEmbedder, self).__init__()
        self.embedding = nn.Embedding(tag_dim+1, embedding_dim, padding_idx=0) # +1 for padding index
        self.embedding_dim = embedding_dim
        self.tag_dim = tag_dim
        
    def forward(self, tags):
        """
        Args:
            tags (torch.Tensor): Tensor of shape (batch_size, max_seq_len) with integer indices for tags (-1 for padding)
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, max_seq_len, embedding_dim)
        """
        shifted_tags = tags + 1
        return self.embedding(shifted_tags.long())  # Ensure tags are long type for embedding lookup
    
class IndexWordEmbedder(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(IndexWordEmbedder, self).__init__()
        self.embedding = nn.Embedding(vocab_size + 1, embedding_dim, padding_idx=0)  # +1 for padding index
        self.embedding_dim = embedding_dim
        self.state_dim = vocab_size
        
    def forward(self, batch_sentences: list[Sentence]):
        """
        Args:
            indices (torch.Tensor): Tensor of shape (batch_size, max_seq_len) with integer indices (-1 for padding)
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, max_seq_len, embedding_dim)
        """
        device = next(self.parameters()).device
        indices : list[torch.Tensor] = [torch.tensor(sentence.symbols_idx, device=device) for sentence in batch_sentences]
        indices = torch.nn.utils.rnn.pad_sequence(indices, batch_first=True, padding_value=-1)

        shifted_indices = indices + 1
        return self.embedding(shifted_indices.long())  # Ensure indices are long type for embedding lookup
    
class WordTagEmbedder(nn.Module):
    def __init__(self, tag_embedder : nn.Module, word_embedder : nn.Module, embedding_dim):
        super(WordTagEmbedder, self).__init__()
        self.word_embedder = word_embedder
        self.tag_embedder = tag_embedder
        self.embedding_dim = embedding_dim
        self.projection = nn.Sequential(
            nn.Linear(word_embedder.embedding_dim + tag_embedder.embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
    def forward(self, batch_sentences: list[Sentence]):
        """
        Args:
            batch_sentences (list[Sentence]): A batch of sentences
        Returns:
            torch.Tensor: A tensor of shape (batch_size, max_seq_len, embedding_dim) containing word-tag embeddings
            It concatenates word and tag embeddings and projects them onto embedding_dim.
        """
        word_embeddings = self.word_embedder(batch_sentences)
        device = word_embeddings.device
        pos_tags = [torch.tensor(sentence.pos_tags, device=device) for sentence in batch_sentences]
        pos_tags = torch.nn.utils.rnn.pad_sequence(pos_tags, batch_first=True, padding_value=-1)
        tag_embeddings = self.tag_embedder(pos_tags)
        
        # Concatenate word and tag embeddings
        combined_embeddings = torch.cat((word_embeddings, tag_embeddings), dim=-1)
        # Project to the desired embedding dimension
        projected_embeddings = self.projection(combined_embeddings)
        return projected_embeddings


class InductionEmbedder(nn.Module):
    def __init__(self, embedding_dim):
        super(InductionEmbedder, self).__init__()
        self.embedding_dim = embedding_dim
        self.mlp = nn.Sequential(
            nn.Linear(3*embedding_dim, 3*embedding_dim),
            nn.ReLU(),
            nn.Linear(3*embedding_dim, embedding_dim)
        )
        
    def forward(self, tag_embeddings: torch.Tensor, left_embeddings: torch.Tensor, right_embeddings: torch.Tensor):
        """
        Args:
            tag_embeddings (torch.Tensor): Tensor of shape (batch_size, embedding_dim) containing tag embeddings for new abstract constituent
            left_embeddings (torch.Tensor): Tensor of shape (batch_size, embedding_dim) containing embedding for left child
            right_embeddings (torch.Tensor): Tensor of shape (batch_size, embedding_dim) containing embedding for right child
        Returns:
            torch.Tensor: A tensor of shape (batch_size, embedding_dim) containing new abstract constituent embeddings
        """
        assert tag_embeddings.shape[1] == left_embeddings.shape[1] == right_embeddings.shape[1] == self.embedding_dim, f"Tag ({tag_embeddings.shape[1]}), left child({left_embeddings.shape[1]}) and right child({right_embeddings.shape[1]}) embeddings must match embedding dimension ({self.embedding_dim})"
        assert left_embeddings.shape[1] == self.embedding_dim, "Left embeddings must match embedding dimension"
        assert right_embeddings.shape[1] == self.embedding_dim, "Right embeddings must match embedding dimension"
        # Concatenate tag, left, and right embeddings
        combined_embeddings = torch.cat((tag_embeddings, left_embeddings, right_embeddings), dim=-1)
        # Pass through MLP
        new_embeddings = self.mlp(combined_embeddings)
        return new_embeddings
    
class TransformerLayer(nn.Module):
    def __init__(self, embedding_dim, num_layers, max_seq_len=60, num_heads=2, dropout=0.1):
        super().__init__()

        self.embedding_norm = nn.LayerNorm(embedding_dim)
        self.positional_encoding = PositionalEncoding(embedding_dim, max_seq_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim*2,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))  # CLS token for global representation

    def forward(self, emb_sequence, mask=None):
        """
        emb_sequence: Tensor of shape (batch, max_seq_len, embedding_dim)
        mask: Optional mask of shape (batch, max_seq_len) where False indicates padding
        """
        batch_size = emb_sequence.size(0)

        if mask is None:
            mask = torch.ones(batch_size, emb_sequence.size(1), dtype=torch.bool, device=emb_sequence.device)

        # Embed actions
        x = self.positional_encoding(emb_sequence)  # (batch, seq_len, embedding_dim)
        x = self.embedding_norm(x)

        # Add CLS token at position 0
        cls_token = self.cls_token.expand(batch_size, -1, -1)  # (batch, 1, embedding_dim)
        x = torch.cat([cls_token, x], dim=1)  # (batch, seq_len + 1, embedding_dim)

        # Update the mask (CLS token is not masked)
        cls_mask = torch.ones(batch_size, 1, dtype=torch.bool, device=mask.device)
        mask = torch.cat([cls_mask, mask], dim=1)  # (batch, seq_len + 1)

        attn_mask = ~mask  # Transformer expects True for padding
        x = self.transformer_encoder(x, src_key_padding_mask=attn_mask)

        # Extract the CLS token embedding (global representation)
        cls_representation = x[:, 0]  # (batch, embedding_dim)

        encoded_sequence = x[:, 1:]

        return cls_representation, encoded_sequence