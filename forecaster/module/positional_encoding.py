"""
Code for positional encoding. From:
https://github.com/pytorch/examples/blob/main/word_language_model/model.py
"""
import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    # 通过正弦和余弦函数为输入序列引入位置信息。
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)  # 存储位置编码
        # 生成一个从 0 到 max_len 的位置索引向量，并通过 unsqueeze(1) 将其转换为列向量，形状为 [max_len, 1]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # 计算位置编码的分母部分。div_term 和 div_term_single 是指数函数结果的向量，用于缩放位置索引
        # div_term 计算的是偶数位置的缩放因子，div_term_single 计算的是奇数位置的缩放因子
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        div_term_single = torch.exp(torch.arange(0, d_model - 1, 2).float() * (-math.log(10000.0) / d_model))
        # 对 pe 矩阵的偶数列位置填入 sin 函数值，对奇数列位置填入 cos 函数值
        # 这里用到了广播机制，position 和 div_term（或 div_term_single）相乘，生成正弦和余弦的输入。
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term_single)
        # 使用 unsqueeze(0) 将 pe 的维度扩展为 [1, max_len, d_model]，
        # 再使用 transpose(0, 1) 交换第 0 维和第 1 维，最终形状为 [max_len, 1, d_model]
        # register_buffer 方法将 pe 注册为模型的 buffer，这样 pe 就不会作为模型参数进行优化
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """

        # 将 pe 的前 x.size(0) 行（对应输入序列的长度）加到输入张量 x 上
        # 位置编码 pe 的形状为 [max_len, 1, d_model]，通过广播机制与 x 相加
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class PositionEmbedding(nn.Module):
    def __init__(self, emb_type: str, input_size: int, max_len: int = 5000, dropout=0.1):
        """Addictive position embedding on time-series data.

        Args:
            emb_type: The type of embedding. Can be learn or static. If learn, the embedding is learned as the model parameter. If static, apply the sinusoidal encoding.
            input_size: Dimension of the input.
            max_len: Maximum length of the input.

        Raises:
            ValueError: If emb_type is not learn or static.
        """
        super(PositionEmbedding, self).__init__()
        self.emb_type = emb_type
        if emb_type == "learn":
            # 创建一个可学习的嵌入层，输入大小为 [max_len, input_size]
            self.emb = nn.Embedding(max_len, input_size)
        elif emb_type == "static":
            # 创建一个基于正弦余弦函数的静态位置编码层
            self.emb = PositionalEncoding(input_size, max_len=max_len, dropout=dropout)
        else:
            raise ValueError("Unknown embedding type: {}".format(emb_type))

    def forward(self, x):
        if self.emb_type == "" or self.emb_type == "learn":
            # 生成一个从 0 到 x.size()[1]-1 的整数序列，表示序列长度
            embedding = self.emb(torch.arange(end=x.size()[1], device=x.device))
            # 重复嵌入以匹配输入的批次大小和序列长度
            embedding = embedding.repeat([x.size()[0], 1, 1])
            x = x + embedding
        elif self.emb_type == "static":
            x = self.emb(x.transpose(0, 1)).transpose(0, 1)

        return x
