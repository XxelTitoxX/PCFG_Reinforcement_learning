
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils

class LSTMEmbedding(nn.Module):
    def __init__(self, state_dim, embedding_dim, hidden_dim, num_layers, bidirectional=False, dropout=0.1):
        super().__init__()
        self.action_embedding = nn.Embedding(state_dim + 1, embedding_dim, padding_idx=0)  # Shifted indices
        self.norm_emb = nn.LayerNorm(embedding_dim)  # Normalize embeddings

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0  # Dropout only applies if num_layers > 1
        )

        if bidirectional:
            self.projection = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, action_sequence):
        """
        action_sequence: Tensor of shape (batch, max_seq_len) with -1 as null
        seq_lengths: Tensor of actual sequence lengths before padding
        """
        seq_lengths = (action_sequence != -1).sum(dim=1)  # Calculate sequence lengths
        action_sequence = action_sequence + 1  # Shift -1 to 0 for padding
        embedded_actions = self.action_embedding(action_sequence)  # (batch, max_seq_len, embedding_dim)
        embedded_actions = self.norm_emb(embedded_actions)  # Normalize embeddings

        # Pack the sequence
        packed_embedded = rnn_utils.pack_padded_sequence(embedded_actions, seq_lengths.cpu(), batch_first=True, enforce_sorted=False)

        _, (hidden_state, _) = self.lstm(packed_embedded)  # Process with LSTM

        # Extract the last hidden state
        if self.lstm.bidirectional:
            lstm_output = torch.cat((hidden_state[-2], hidden_state[-1]), dim=-1)  # Concatenate forward & backward
            lstm_output = self.projection(lstm_output)  # Reduce dimensionality
        else:
            lstm_output = hidden_state[-1]

        return lstm_output




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

class AttentionPooling(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.attention = nn.Linear(embedding_dim, 1)
        
    def forward(self, x, mask=None):
        # x shape: (batch_size, seq_len, embedding_dim)
        attn_scores = self.attention(x).squeeze(-1)  # (batch_size, seq_len)
        
        # Apply mask if provided
        if mask is not None:
            assert mask.size(1) == x.size(1), f"Mask size mismatch: {mask.size(1)} vs {x.size(1)}"
            attn_scores = attn_scores.masked_fill(~mask, -1e9)
        
        # Normalize attention scores
        attn_weights = F.softmax(attn_scores, dim=1).unsqueeze(1)  # (batch_size, 1, seq_len)
        
        # Apply attention weights
        pooled = torch.bmm(attn_weights, x).squeeze(1)  # (batch_size, embedding_dim)
        return pooled

class EnhancedTransformerEmbedding(nn.Module):
    def __init__(self, state_dim, embedding_dim, hidden_dim, num_layers, max_seq_len=60, num_heads=2, pooling_type='attention'):
        super().__init__()

        self.action_embedding = nn.Embedding(state_dim + 1, embedding_dim, padding_idx=0)  # Adjusted for padding
        self.embedding_norm = nn.LayerNorm(embedding_dim)
        self.positional_encoding = PositionalEncoding(embedding_dim, max_seq_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim*2,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.pooling_type = pooling_type
        if pooling_type == 'attention':
            self.attention_pooling = AttentionPooling(embedding_dim)

        self.output_projection = nn.Linear(embedding_dim, hidden_dim)

    def forward(self, action_sequence):
        """
        action_sequence: Tensor of shape (batch, max_seq_len) with -1 as null
        seq_lengths: Tensor of actual sequence lengths before padding
        """
        action_sequence = action_sequence + 1  # Shift -1 to 0 for padding
        mask = action_sequence != 0  # Create mask where 0 (padded) is False

        x = self.action_embedding(action_sequence)  # (batch, seq_len, embedding_dim)
        x = self.embedding_norm(x)
        x = self.positional_encoding(x)

        attn_mask = ~mask  # Transformer uses True for padding positions
        x = self.transformer_encoder(x, src_key_padding_mask=attn_mask)

        # Here the sequence length may not correspond to the mask anymore (some position on dimension 1 may have been discarded), leading to errors
        mask = mask[:, :x.size(1)]
        
        if self.pooling_type == 'mean':
            pooled = (x * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True)
        elif self.pooling_type == 'max':
            x = x.masked_fill(~mask.unsqueeze(-1), -1e9)
            pooled = x.max(dim=1)[0]
        elif self.pooling_type == 'attention':
            pooled = self.attention_pooling(x, mask)

        pooled = self.output_projection(pooled)

        return pooled
