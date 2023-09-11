import torch
import torch.nn as nn
from labml_helpers.module import Module
from typing import Optional, List
from labml_nn.transformers.utils import subsequent_mask

# transformer
from labml_nn.transformers.feed_forward import FeedForward
from labml_nn.utils import clone_module_list
from .attention import RelativeMultiHeadAttention

class TransformerXLLayer(nn.Module):
    """
    A PyTorch module for a single layer of the Transformer-XL model.

    Args:
        d_model (int): The dimension of the input features.
        self_attn (RelativeMultiHeadAttention): The relative multi-head self-attention module.
        feed_forward (FeedForward): The feed-forward neural network module.
        dropout_prob (float): Dropout probability.

    Example:
        >>> self_attn_layer = TransformerXLLayer(d_model=512, self_attn=relative_multihead_attention,
        >>>                                      feed_forward=feed_forward_network, dropout_prob=0.1)
        >>> input_tensor = torch.randn(seq_len, batch_size, d_model)
        >>> mask_tensor = torch.ones(seq_len, seq_len)
        >>> output_tensor = self_attn_layer(x=input_tensor, mem=None, mask=mask_tensor)
    """

    def __init__(self, *, 
                d_model: int, 
                self_attn: RelativeMultiHeadAttention,
                feed_forward: FeedForward, 
                dropout_prob: float):
        """
        Initializes a new instance of the TransformerXLLayer module.

        Args:
            d_model (int): The dimension of the input features.
            self_attn (RelativeMultiHeadAttention): The relative multi-head self-attention module.
            feed_forward (FeedForward): The feed-forward neural network module.
            dropout_prob (float): Dropout probability.
        """
        super().__init__()
        self.size = d_model
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.dropout = nn.Dropout(dropout_prob)
        self.norm_self_attn = nn.LayerNorm([d_model])
        self.norm_ff = nn.LayerNorm([d_model])

    def forward(self, *,
                x: torch.Tensor,
                mem: Optional[torch.Tensor],
                mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the TransformerXLLayer module.

        Args:
            x (torch.Tensor): The input tensor.
            mem (Optional[torch.Tensor]): The memory tensor. Can be None.
            mask (torch.Tensor): The attention mask tensor.

        Returns:
            torch.Tensor: The output tensor after processing through the layer.

        Example:
            >>> input_tensor = torch.randn(seq_len, batch_size, d_model)
            >>> mask_tensor = torch.ones(seq_len, seq_len)
            >>> output_tensor = self_attn_layer(x=input_tensor, mem=None, mask=mask_tensor)
        """
        z = self.norm_self_attn(x)
        if mem is not None:
            mem = self.norm_self_attn(mem)
            m_z = torch.cat((mem, z), dim=0)
        else:
            m_z = z
        self_attn = self.self_attn(query=z, key=m_z, value=m_z, mask=mask)
        x = x + self.dropout(self_attn)

        z = self.norm_ff(x)
        ff = self.feed_forward(z)
        x = x + self.dropout(ff)

        return x

class TransformerXL(nn.Module):
    """
    A PyTorch module for the Transformer-XL model.

    Args:
        layer (TransformerXLLayer): The Transformer-XL layer to be repeated.
        n_layers (int): The number of layers.

    Example:
        >>> transformer_xl = TransformerXL(layer=transformer_xl_layer, n_layers=6)
        >>> input_tensor = torch.randn(seq_len, batch_size, d_model)
        >>> memory_list = [torch.randn(seq_len, batch_size, d_model) for _ in range(n_layers)]
        >>> mask_tensor = torch.ones(seq_len, seq_len)
        >>> output_tensor, new_memory = transformer_xl(x=input_tensor, mem=memory_list, mask=mask_tensor)
    """

    def __init__(self, layer: TransformerXLLayer, n_layers: int):
        """
        Initializes a new instance of the TransformerXL module.

        Args:
            layer (TransformerXLLayer): The Transformer-XL layer to be repeated.
            n_layers (int): The number of layers.
        """
        super().__init__()
        self.layers = clone_module_list(layer, n_layers)
        self.norm = nn.LayerNorm([layer.size])

    def forward(self, x: torch.Tensor,
                mem: List[torch.Tensor],
                mask: torch.Tensor):
        """
        Forward pass of the TransformerXL module.

        Args:
            x (torch.Tensor): The input tensor.
            mem (List[torch.Tensor]): List of memory tensors.
            mask (torch.Tensor): The attention mask tensor.

        Returns:
            torch.Tensor: The output tensor after processing through the layers.
            List[torch.Tensor]: List of updated memory tensors.

        Example:
            >>> input_tensor = torch.randn(seq_len, batch_size, d_model)
            >>> memory_list = [torch.randn(seq_len, batch_size, d_model) for _ in range(n_layers)]
            >>> mask_tensor = torch.ones(seq_len, seq_len)
            >>> output_tensor, new_memory = transformer_xl(x=input_tensor, mem=memory_list, mask=mask_tensor)
        """
        new_mem = []
        for i, layer in enumerate(self.layers):
            new_mem.append(x.detach())
            m = mem[i] if mem else None
            x = layer(x=x, mem=m, mask=mask)
        return self.norm(x), new_mem

class AutoregressiveModel(nn.Module):
    """
    A PyTorch module for an autoregressive model.

    Args:
        n_vocab (int): The size of the vocabulary.
        d_model (int): The dimension of the input features.
        transformer (TransformerXL): The Transformer-XL model.

    Example:
        >>> autoregressive_model = AutoregressiveModel(n_vocab=10000, d_model=512, transformer=transformer_xl)
        >>> input_tensor = torch.tensor([1, 2, 3, 4])
        >>> memory_list = [torch.randn(seq_len, batch_size, d_model) for _ in range(n_layers)]
        >>> output_tensor, new_memory = autoregressive_model(x=input_tensor, mem=memory_list)
    """

    def __init__(self, n_vocab: int, d_model: int, transformer: TransformerXL):
        """
        Initializes a new instance of the AutoregressiveModel module.

        Args:
            n_vocab (int): The size of the vocabulary.
            d_model (int): The dimension of the input features.
            transformer (TransformerXL): The Transformer-XL model.
        """
        super().__init__()
        self.src_embed = nn.Embedding(n_vocab, d_model)
        self.transformer = transformer
        self.generator = nn.Linear(d_model, n_vocab)
        self.mask_x = None
        self.mask_mem = None

    def forward(self, x: torch.Tensor, mem: List[torch.Tensor]):
        """
        Forward pass of the AutoregressiveModel module.

        Args:
            x (torch.Tensor): The input tensor.
            mem (List[torch.Tensor]): List of memory tensors.

        Returns:
            torch.Tensor: The output tensor after processing through the model.
            List[torch.Tensor]: List of updated memory tensors.

        Example:
            >>> input_tensor = torch.tensor([1, 2, 3, 4])
            >>> memory_list = [torch.randn(seq_len, batch_size, d_model) for _ in range(n_layers)]
            >>> output_tensor, new_memory = autoregressive_model(x=input_tensor, mem=memory_list)
        """
        m_len = len(mem[0]) if mem else 0
        if self.mask_x is None or self.mask_x.shape[0] < len(x):
            self.mask_x = subsequent_mask(len(x)).to(x.device)
        if self.mask_mem is None or self.mask_mem.shape[1] < m_len or self.mask_mem.shape[0] < len(x):
            self.mask_mem = self.mask_x.new_ones(len(x), m_len, 1)

        if m_len:
            mask = torch.cat((self.mask_mem[:len(x), :m_len], self.mask_x[:len(x), :len(x)]), dim=1)
        else:
            mask = self.mask_x[:len(x), :len(x)]

        x = self.src_embed(x)
        res, mem = self.transformer(x, mem, mask)
        res = self.generator(res)
        return res, mem