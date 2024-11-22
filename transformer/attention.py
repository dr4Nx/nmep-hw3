from typing import Optional

import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):

    def __init__(self, 
                 num_heads: int,
                 embedding_dim: int,
                 qk_length: int,
                 value_length: int
                 ):
        """
        The Multi-Head Attention layer will take in Q, K, and V
        matrices and will output an attention matrix of shape <TODO>.

        First, Q, K, and V should be projected to have
        a shape of (B, T, C) where C = num_heads * qk_length 
        (OR value_length). You are then expected to split 
        the C dimension into num_heads different heads, each 
        with shape (B, T, vec_length).

        Next, you will compute the scaled dot-product attention
        between Q, K, and V.

        Finally, you will concatenate the heads and project the
        output to have a shape of (B, T, C).

        Check out the `masked_fill` method in PyTorch to help
        you implement the masking step!
        """
        super().__init__()

        self.num_heads = num_heads
        self.qk_length = qk_length
        self.value_length = value_length

        # Define any layers you'll need in the forward pass
        # (hint: number of Linear layers needed != 3)
        raise NotImplementedError("Implement the Multi-Head Attention layer definitions!")

    def split_heads(self, x: torch.Tensor, vec_length: int) -> torch.Tensor:
        """
        Split the C dimension of the input tensor into num_heads
        different heads, each with shape (B, T, vec_length).

        Args:
            x: torch.Tensor of shape (B, T, C), where C = num_heads * vec_length
            vec_length: int, the length of the query/key/value vectors

        Returns:
            torch.Tensor of shape (B, num_heads, T, vec_length)
        """
        raise NotImplementedError("Implement the split_heads method!")
        

    def combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Combine the num_heads different heads into a single tensor.
        Hint: check out the `contiguous` method in PyTorch to help
        you reshape the tensor.

        Args:
            x: torch.Tensor of shape (B, num_heads, T, vec_length)

        Returns:
            torch.Tensor of shape (B, T, num_heads * vec_length)
        """
        raise NotImplementedError("Implement the combine_heads method!")

    def scaled_dot_product_attention(self, 
                                     Q: torch.Tensor, 
                                     K: torch.Tensor, 
                                     V: torch.Tensor, 
                                     mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute the scaled dot-product attention given Q, K, and V.

        Args:
            Q: torch.Tensor of shape (B, num_heads, T, qk_length)
            K: torch.Tensor of shape (B, num_heads, T, qk_length)
            V: torch.Tensor of shape (B, num_heads, T, value_length)
            mask: Optional torch.Tensor of shape (B, T, T) or None
        """
        raise NotImplementedError("Implement the scaled_dot_product_attention method!")


    def forward(self,
                Q: torch.Tensor, 
                K: torch.Tensor, 
                V: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        The forward pass of the Multi-Head Attention layer.

        Args:
            Q: torch.Tensor of shape (B, T, C)
            K: torch.Tensor of shape (B, T, C)
            V: torch.Tensor of shape (B, T, C)
            mask: Optional torch.Tensor of shape (B, T, T) or None

        Returns:
            torch.Tensor of shape (B, T, C)
        """
        raise NotImplementedError("Implement the forward method!")

class FeedForwardNN(nn.Module):

    def __init__(self, 
                 embedding_dim: int,
                 hidden_dim: int):
        """
        The Feed-Forward Neural Network layer will take in
        an input tensor of shape (B, T, C) and will output
        a tensor of the same shape.

        The FFNN will have two linear layers, with a ReLU
        activation function in between.

        Args:
            hidden_dim: int, the size of the hidden layer
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim

        # Define any layers you'll need in the forward pass
        self.relu = nn.ReLU()
        raise NotImplementedError("Implement the FeedForwardNN layer definitions!")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        The forward pass of the FeedForwardNN.
        """
        raise NotImplementedError("Implement the FFN forward method!")
