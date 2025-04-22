
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils




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

