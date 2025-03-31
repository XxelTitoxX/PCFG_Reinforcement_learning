
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class LSTMEmbedding(nn.Module):
    def __init__(self, num_actions, embedding_dim, hidden_dim, num_layers, bidirectional=False, dropout=0.1):
        super().__init__()
        self.action_embedding = nn.Embedding(num_actions, embedding_dim)  # Learnable embeddings
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
            lstm_out_dim = hidden_dim * 2
            # Projection layer to reduce LSTM output to a fixed size
            self.projection = nn.Linear(lstm_out_dim, hidden_dim)
        else:
            lstm_out_dim = hidden_dim

    def forward(self, action_sequence):
        embedded_actions = self.action_embedding(action_sequence)  # Shape: (batch, seq_len, embedding_dim)
        embedded_actions = self.norm_emb(embedded_actions)  # Normalize embeddings

        _, (hidden_state, _) = self.lstm(embedded_actions)  

        if self.lstm.bidirectional:
            lstm_output = torch.cat((hidden_state[-2], hidden_state[-1]), dim=-1)  # Concatenate last hidden states
        else:
            lstm_output = hidden_state[-1]

        if (self.lstm.bidirectional):
            lstm_output = self.projection(lstm_output)
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
            attn_scores = attn_scores.masked_fill(~mask, -1e9)
        
        # Normalize attention scores
        attn_weights = F.softmax(attn_scores, dim=1).unsqueeze(1)  # (batch_size, 1, seq_len)
        
        # Apply attention weights
        pooled = torch.bmm(attn_weights, x).squeeze(1)  # (batch_size, embedding_dim)
        return pooled

class EnhancedTransformerEmbedding(nn.Module):
    def __init__(self, num_actions, embedding_dim, num_heads, hidden_dim, num_layers, max_seq_len,
                 pooling_type='attention'):
        super().__init__()
        
        # Action embedding layer
        self.action_embedding = nn.Embedding(num_actions, embedding_dim)
        self.embedding_norm = nn.LayerNorm(embedding_dim)
        self.positional_encoding = PositionalEncoding(embedding_dim, max_seq_len)

        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Pooling strategy
        self.pooling_type = pooling_type
        if pooling_type == 'attention':
            self.attention_pooling = AttentionPooling(embedding_dim)
        
        
    def forward(self, action_sequence, state=None, mask=None):
        """
        action_sequence: Tensor of shape (batch_size, seq_len) containing action indices
        state: Optional tensor of shape (batch_size, state_dim) for conditioning
        mask: Optional boolean mask of shape (batch_size, seq_len) indicating valid positions
        """
        batch_size, seq_len = action_sequence.shape
        
        # Create embeddings
        x = self.action_embedding(action_sequence)  # (batch, seq_len, embedding_dim)
        x = self.embedding_norm(x)
        
        # Add positional encoding
        x = self.positional_encoding(x)
        
        
        # Create attention mask for transformer (converting from boolean to float)
        attn_mask = None
        if mask is not None:
            attn_mask = ~mask  # Invert because transformer uses 1 for masked positions
        
        # Apply transformer encoder
        x = self.transformer_encoder(x, src_key_padding_mask=attn_mask)
        

        if self.pooling_type == 'mean':
            # Mean pooling (with mask)
            if mask is not None:
                pooled = (x * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True)
            else:
                pooled = x.mean(dim=1)
        elif self.pooling_type == 'max':
            # Max pooling (with mask)
            if mask is not None:
                x = x.masked_fill(~mask.unsqueeze(-1), -1e9)
            pooled = x.max(dim=1)[0]
        elif self.pooling_type == 'attention':
            # Attention pooling
            pooled = self.attention_pooling(x, mask)
        
        return pooled
