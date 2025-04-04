import torch
import torch.nn as nn

from .attention import MultiHeadAttention, FeedForwardNN
from .encoder import PositionalEncoding

class DecoderLayer(nn.Module):
    
    def __init__(self, 
                 num_heads: int, 
                 embedding_dim: int,
                 ffn_hidden_dim: int,
                 qk_length: int, 
                 value_length: int,
                 dropout: float = 0.1):
        """
        Each decoder layer will take in two embeddings of
        shape (B, T, C):

        1. The `target` embedding, which comes from the decoder
        2. The `source` embedding, which comes from the encoder

        and will output a representation
        of the same shape.

        The decoder layer will have three main components:
            1. A Masked Multi-Head Attention layer (you'll need to
               modify the MultiHeadAttention layer to handle this!)
            2. A Multi-Head Attention layer for cross-attention
               between the target and source embeddings.
            3. A Feed-Forward Neural Network layer.

        Remember that for each Multi-Head Attention layer, we
        need create Q, K, and V matrices from the input embedding(s)!
        """
        super().__init__()

        self.num_heads = num_heads
        self.embedding_dim = embedding_dim
        self.ffn_hidden_dim = ffn_hidden_dim
        self.qk_length = qk_length
        self.value_length = value_length

        # Define any layers you'll need in the forward pass
        # raise NotImplementedError("Implement the DecoderLayer layer definitions!")
        self.attention1 = MultiHeadAttention(num_heads, embedding_dim, qk_length, value_length)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.attention2 = MultiHeadAttention(num_heads, embedding_dim, qk_length, value_length)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.ffn = FeedForwardNN(embedding_dim, ffn_hidden_dim)
        self.drop = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(embedding_dim)

    
    def forward(self, x: torch.Tensor, enc_x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        The forward pass of the DecoderLayer.
        """
        # raise NotImplementedError("Implement the DecoderLayer forward method!")
        self.out1 = self.attention1(x, x, x, mask)
        self.out2 = x + self.norm1(self.out1)
        self.out3 = self.attention2(self.out2, enc_x, enc_x)
        self.out4 = self.out2 + self.norm2(self.out3)
        self.out5 = self.ffn(self.out4)
        self.out6 = self.drop(self.out5)
        self.out7 = self.out4 + self.norm3(self.out6)
        return self.out7


class Decoder(nn.Module):

    def __init__(self, 
                 vocab_size: int, 
                 num_layers: int, 
                 num_heads: int,
                 embedding_dim: int,
                 ffn_hidden_dim: int,
                 qk_length: int,
                 value_length: int,
                 max_length: int,
                 dropout: float = 0.1):
        """
        Remember that the decoder will take in a sequence
        of tokens AND a source embedding
        and will output an encoded representation
        of shape (B, T, C).

        First, we need to create an embedding from the sequence
        of tokens. For this, we need the vocab size.

        Next, we want to create a series of Decoder layers.
        For this, we need to specify the number of layers 
        and the number of heads.

        Additionally, for every Multi-Head Attention layer, we
        need to know how long each query/key is, and how long
        each value is.
        """
        super().__init__()

        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.embedding_dim = embedding_dim
        self.ffn_hidden_dim = ffn_hidden_dim

        self.qk_length = qk_length
        self.value_length = value_length

        # Define any layers you'll need in the forward pass
        # Hint: You may find `ModuleList`s useful for creating
        # multiple layers in some kind of list comprehension.
        # 
        # Recall that the input is just a sequence of tokens,
        # so we'll have to first create some kind of embedding
        # and then use the other layers we've implemented to
        # build out the Transformer decoder.
        # raise NotImplementedError("Implement the Decoder layer definitions!")
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.positional_encoding = PositionalEncoding(embedding_dim, dropout=dropout, max_len=max_length)
        self.layers = nn.ModuleList([DecoderLayer(num_heads, embedding_dim, ffn_hidden_dim, qk_length, value_length, dropout) for _ in range(num_layers)])
        self.linear = nn.Linear(embedding_dim, vocab_size)

    def make_mask(self, x: torch.Tensor) -> torch.Tensor:
        """
        Create a mask to prevent attention to future tokens.
        """
        # raise NotImplementedError("Implement the make_mask method!")

        return torch.triu(torch.ones(x.size(0), self.num_heads, x.size(1), x.size(1)), diagonal=0)

    def forward(self, x: torch.Tensor, enc_x: torch.Tensor) -> torch.Tensor:
        """
        The forward pass of the Decoder.
        """
        # raise NotImplementedError("Implement the Decoder forward method!")
        mask = self.make_mask(x)
        self.out1 = self.embedding(x)
        self.out2 = self.positional_encoding(self.out1)
        mask = mask.to(x.device)
        for decoder in self.layers:
            self.out2 = decoder(self.out2, enc_x, mask)

        self.out3 = self.linear(self.out2)
        return self.out3