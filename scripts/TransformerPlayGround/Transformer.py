"""
    1. PositionalEncoding:
Purpose: Since transformers don't inherently know the order of tokens, we add positional encodings to the input embeddings to give the model information about token positions in the sequence.

"""

class positionEncoding(nn.module):
    def __init__(self, d_model, max_len=5000):
        # d_model: Embedding size (dimensionality of the model)
        # max_len: Maximum sequence length (default 5000)
        
        # Create a matrix 'pe' where each row corresponds to the positional encoding for a specific position.
        pe = torch.zeros(max_len, d_model)  # Shape: [max_len, d_model]

        # Position is an array [0, 1, 2, ..., max_len-1] reshaped to a column vector.
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # Generate angles for sine/cosine functions based on the position and the dimensions.
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # Apply sine on even indices and cosine on odd indices (this alternating pattern helps learn positions).
        pe[:, 0::2] = torch.sin(position * div_term)  # Sin for even dimensions
        pe[:, 1::2] = torch.cos(position * div_term)  # Cos for odd dimensions

        pe = pe.unsqueeze(0).transpose(0, 1)  # Reshape for broadcasting: [max_len, 1, d_model]
        self.register_buffer('pe', pe)  # Register as buffer (non-trainable parameter)

    def forward(self, x):
        # 'x' is the input embeddings of shape [sequence_len, batch_size, d_model]
        # We add positional encoding to input embeddings.
        x = x + self.pe[:x.size(0), :]  # Ensure shape compatibility
        return x
