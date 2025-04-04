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
        # raise NotImplementedError("Implement the Multi-Head Attention layer definitions!")

        self.linearQ = nn.Linear(embedding_dim, num_heads * qk_length)
        self.linearK = nn.Linear(embedding_dim, num_heads * qk_length)
        self.linearV = nn.Linear(embedding_dim, num_heads * qk_length)

        self.linearConcat = nn.Linear(num_heads * value_length, embedding_dim)

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
        # raise NotImplementedError("Implement the split_heads method!")

        if x.shape[2] == self.num_heads * vec_length:
            # Reshape the tensor to split the last dimension into num_heads and vec_length
            return x.view(x.shape[0], x.shape[1], self.num_heads, vec_length).transpose(1, 2)
        else:
            raise AssertionError 
        

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
        # raise NotImplementedError("Implement the combine_heads method!")

        return x.view(x.shape[0], x.shape[2], x.shape[1] * x.shape[3])

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
        # raise NotImplementedError("Implement the scaled_dot_product_attention method!")

        attentionscores = torch.matmul(Q, K.transpose(-2, -1)) / (self.qk_length ** 0.5)
        attentionweights = torch.nn.functional.softmax(attentionscores, dim=-1)
        if mask is not None:
            attentionscores = attentionscores.masked_fill(mask == 0, 0)
        attentionoutput = torch.matmul(attentionweights, V)
        return attentionoutput


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
        # raise NotImplementedError("Implement the forward method!")
        qout = self.split_heads(self.linearQ(Q), self.qk_length)
        kout = self.split_heads(self.linearK(K), self.qk_length)
        vout = self.split_heads(self.linearV(V), self.value_length)
        attentionoutput = self.scaled_dot_product_attention(qout, kout, vout, mask)
        attentionoutput = self.combine_heads(attentionoutput)
        attentionoutput = self.linearConcat(attentionoutput)
        return attentionoutput

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
        # raise NotImplementedError("Implement the FeedForwardNN layer definitions!")

        self.linear1 = nn.Linear(embedding_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, embedding_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        The forward pass of the FeedForwardNN.
        """
        # raise NotImplementedError("Implement the FFN forward method!")

        self.out1 = self.linear1(x)
        self.out2 = self.relu(self.out1)
        self.out3 = self.linear2(self.out2)
        return self.out3
