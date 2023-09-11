import torch
import torch.nn as nn
import math
from labml import tracker
from typing import Optional, List
from labml_helpers.module import Module
from .util import shift_right

class SquaredReLU(nn.Module):
    """
    A PyTorch module that applies the ReLU (Rectified Linear Unit) activation function
    followed by squaring the output element-wise.

    Attributes:
        relu (nn.ReLU): The ReLU activation function module.

    Args:
        None

    Example:
        >>> squared_relu = SquaredReLU()
        >>> input_tensor = torch.randn(3, 3)
        >>> output_tensor = squared_relu(input_tensor)
    """

    def __init__(self):
        """
        Initializes a new instance of the SquaredReLU module.
        """
        super().__init__()
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the SquaredReLU module.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor with each element squared.

        Example:
            >>> squared_relu = SquaredReLU()
            >>> input_tensor = torch.randn(3, 3)
            >>> output_tensor = squared_relu(input_tensor)
        """
        x = self.relu(x)
        return x * x

class SpatialDepthWiseConvolution(nn.Module):
    """
    A PyTorch module for spatial depth-wise convolution on a sequence of feature maps.

    This module applies 1D depth-wise convolution to each position in a sequence of feature maps.

    Args:
        d_k (int): The number of input channels and output channels.
        kernel_size (int, optional): The size of the convolutional kernel. Default is 3.

    Example:
        >>> spatial_conv = SpatialDepthWiseConvolution(d_k=64, kernel_size=3)
        >>> input_tensor = torch.randn(seq_len, batch_size, heads, d_k)
        >>> output_tensor = spatial_conv(input_tensor)
    """

    def __init__(self, d_k: int, kernel_size: int = 3):
        """
        Initializes a new instance of the SpatialDepthWiseConvolution module.

        Args:
            d_k (int): The number of input channels and output channels.
            kernel_size (int, optional): The size of the convolutional kernel. Default is 3.
        """
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(in_channels=d_k,
                            out_channels=d_k,
                            kernel_size=(kernel_size,),
                            padding=(kernel_size - 1,),
                            groups=d_k)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the SpatialDepthWiseConvolution module.

        Args:
            x (torch.Tensor): The input tensor with shape (seq_len, batch_size, heads, d_k).

        Returns:
            torch.Tensor: The output tensor after applying spatial depth-wise convolution.

        Example:
            >>> spatial_conv = SpatialDepthWiseConvolution(d_k=64, kernel_size=3)
            >>> input_tensor = torch.randn(seq_len, batch_size, heads, d_k)
            >>> output_tensor = spatial_conv(input_tensor)
        """
        seq_len, batch_size, heads, d_k = x.shape
        x = x.permute(1, 2, 3, 0)
        x = x.view(batch_size * heads, d_k, seq_len)
        x = self.conv(x)
        x = x[:, :, :-(self.kernel_size - 1)]
        x = x.view(batch_size, heads, d_k, seq_len)
        x = x.permute(3, 0, 1, 2)
        return x

class PrepareForMultiHeadAttention(nn.Module):
    """
    A PyTorch module to prepare input data for multi-head attention.

    This module linearly transforms input features into a format suitable for multi-head attention
    by splitting the input into multiple heads.

    Args:
        d_model (int): The dimension of the input features.
        heads (int): The number of attention heads.
        d_k (int): The dimension of each head.
        bias (bool): Whether to include bias in the linear transformation.

    Example:
        >>> prep_for_attention = PrepareForMultiHeadAttention(d_model=512, heads=8, d_k=64, bias=True)
        >>> input_tensor = torch.randn(seq_len, batch_size, d_model)
        >>> output_tensor = prep_for_attention(input_tensor)
    """

    def __init__(self, d_model: int, heads: int, d_k: int, bias: bool):
        """
        Initializes a new instance of the PrepareForMultiHeadAttention module.

        Args:
            d_model (int): The dimension of the input features.
            heads (int): The number of attention heads.
            d_k (int): The dimension of each head.
            bias (bool): Whether to include bias in the linear transformation.
        """
        super().__init__()
        self.linear = nn.Linear(d_model, heads * d_k, bias=bias)
        self.heads = heads
        self.d_k = d_k

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the PrepareForMultiHeadAttention module.

        Args:
            x (torch.Tensor): The input tensor with shape (seq_len, batch_size, d_model).

        Returns:
            torch.Tensor: The output tensor suitable for multi-head attention
                        with shape (seq_len, batch_size, heads, d_k).

        Example:
            >>> prep_for_attention = PrepareForMultiHeadAttention(d_model=512, heads=8, d_k=64, bias=True)
            >>> input_tensor = torch.randn(seq_len, batch_size, d_model)
            >>> output_tensor = prep_for_attention(input_tensor)
        """
        head_shape = x.shape[:-1]
        x = self.linear(x)
        x = x.view(*head_shape, self.heads, self.d_k)
        return x

class MultiHeadAttention(nn.Module):
    """
    A PyTorch module for multi-head attention.

    This module performs multi-head attention on input sequences, allowing the model to focus on
    different parts of the input for different tasks.

    Args:
        heads (int): The number of attention heads.
        d_model (int): The dimension of the input features.
        dropout_prob (float, optional): Dropout probability. Default is 0.1.
        bias (bool, optional): Whether to include bias in linear transformations. Default is True.

    Example:
        >>> multihead_attention = MultiHeadAttention(heads=8, d_model=512, dropout_prob=0.1, bias=True)
        >>> query_tensor = torch.randn(seq_len, batch_size, d_model)
        >>> key_tensor = torch.randn(seq_len, batch_size, d_model)
        >>> value_tensor = torch.randn(seq_len, batch_size, d_model)
        >>> output_tensor = multihead_attention(query=query_tensor, key=key_tensor, value=value_tensor)
    """

    def __init__(self, heads: int, d_model: int, dropout_prob: float = 0.1, bias: bool = True):
        """
        Initializes a new instance of the MultiHeadAttention module.

        Args:
            heads (int): The number of attention heads.
            d_model (int): The dimension of the input features.
            dropout_prob (float, optional): Dropout probability. Default is 0.1.
            bias (bool, optional): Whether to include bias in linear transformations. Default is True.
        """
        super().__init__()
        self.d_k = d_model // heads
        self.heads = heads
        self.query = PrepareForMultiHeadAttention(d_model, heads, self.d_k, bias=bias)
        self.key = PrepareForMultiHeadAttention(d_model, heads, self.d_k, bias=bias)
        self.value = PrepareForMultiHeadAttention(d_model, heads, self.d_k, bias=True)
        self.softmax = nn.Softmax(dim=1)
        self.output = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout_prob)
        self.scale = 1 / math.sqrt(self.d_k)
        self.attn = None

    def get_scores(self, query: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
        """
        Compute attention scores between query and key.

        Args:
            query (torch.Tensor): The query tensor.
            key (torch.Tensor): The key tensor.

        Returns:
            torch.Tensor: The attention scores.

        Example:
            >>> scores = self.get_scores(query, key)
        """
        return torch.einsum('ibhd,jbhd->ijbh', query, key)

    def prepare_mask(self, mask: torch.Tensor, query_shape: List[int], key_shape: List[int]) -> torch.Tensor:
        """
        Prepare a mask for attention.

        Args:
            mask (torch.Tensor): The mask tensor.
            query_shape (List[int]): The shape of the query tensor.
            key_shape (List[int]): The shape of the key tensor.

        Returns:
            torch.Tensor: The prepared mask.

        Example:
            >>> mask = self.prepare_mask(mask, query.shape, key.shape)
        """
        assert mask.shape[0] == 1 or mask.shape[0] == query_shape[0]
        assert mask.shape[1] == key_shape[0]
        assert mask.shape[2] == 1 or mask.shape[2] == query_shape[1]
        mask = mask.unsqueeze(-1)
        return mask

    def forward(self, *, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of the MultiHeadAttention module.

        Args:
            query (torch.Tensor): The query tensor.
            key (torch.Tensor): The key tensor.
            value (torch.Tensor): The value tensor.
            mask (Optional[torch.Tensor]): The attention mask tensor. Default is None.

        Returns:
            torch.Tensor: The output tensor after multi-head attention.

        Example:
            >>> output_tensor = self.forward(query=query_tensor, key=key_tensor, value=value_tensor, mask=mask_tensor)
        """
        seq_len, batch_size, _ = query.shape

        if mask is not None:
            mask = self.prepare_mask(mask, query.shape, key.shape)

        query = self.query(query)
        key = self.key(key)
        value = self.value(value)

        scores = self.get_scores(query, key)
        scores *= self.scale

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn = self.softmax(scores)

        tracker.debug('attn', attn)  # Replace with your debugging mechanism

        attn = self.dropout(attn)
        x = torch.einsum("ijbh,jbhd->ibhd", attn, value)

        self.attn = attn.detach()

        x = x.reshape(seq_len, batch_size, -1)

        return self.output(x)

class RelativeMultiHeadAttention(MultiHeadAttention):
    """
    A PyTorch module for relative multi-head attention.

    This module extends the MultiHeadAttention class to incorporate relative positional embeddings.

    Args:
        heads (int): The number of attention heads.
        d_model (int): The dimension of the input features.
        dropout_prob (float, optional): Dropout probability. Default is 0.1.

    Example:
        >>> relative_multihead_attention = RelativeMultiHeadAttention(heads=8, d_model=512, dropout_prob=0.1)
        >>> query_tensor = torch.randn(seq_len, batch_size, d_model)
        >>> key_tensor = torch.randn(seq_len, batch_size, d_model)
        >>> value_tensor = torch.randn(seq_len, batch_size, d_model)
        >>> output_tensor = relative_multihead_attention(query=query_tensor, key=key_tensor, value=value_tensor)
    """

    def __init__(self, heads: int, d_model: int, dropout_prob: float = 0.1):
        """
        Initializes a new instance of the RelativeMultiHeadAttention module.

        Args:
            heads (int): The number of attention heads.
            d_model (int): The dimension of the input features.
            dropout_prob (float, optional): Dropout probability. Default is 0.1.
        """
        super().__init__(heads, d_model, dropout_prob, bias=False)

        self.P = 2 ** 12

        self.query = nn.Sequential(self.query, SpatialDepthWiseConvolution(self.d_k))
        self.key = nn.Sequential(self.key, SpatialDepthWiseConvolution(self.d_k))
        self.value = nn.Sequential(self.value, SpatialDepthWiseConvolution(self.d_k))

        self.key_pos_embeddings = nn.Parameter(torch.zeros((self.P * 2, heads, self.d_k)), requires_grad=True)
        self.key_pos_bias = nn.Parameter(torch.zeros((self.P * 2, heads)), requires_grad=True)
        self.query_pos_bias = nn.Parameter(torch.zeros((heads, self.d_k)), requires_grad=True)

    def get_scores(self, query: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
        """
        Compute attention scores between query and key with relative positional embeddings.

        Args:
            query (torch.Tensor): The query tensor.
            key (torch.Tensor): The key tensor.

        Returns:
            torch.Tensor: The attention scores with relative positional embeddings.

        Example:
            >>> scores = self.get_scores(query, key)
        """
        key_pos_emb = self.key_pos_embeddings[self.P - key.shape[0]:self.P + query.shape[0]]
        key_pos_bias = self.key_pos_bias[self.P - key.shape[0]:self.P + query.shape[0]]
        query_pos_bias = self.query_pos_bias[None, None, :, :]

        ac = torch.einsum('ibhd,jbhd->ijbh', query + query_pos_bias, key)
        b = torch.einsum('ibhd,jhd->ijbh', query, key_pos_emb)
        d = key_pos_bias[None, :, None, :]
        bd = shift_right(b + d)
        bd = bd[:, -key.shape[0]:]

        return ac + bd
