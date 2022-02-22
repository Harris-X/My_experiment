# ViT Online Class
# Author: Dr. Zhu
# Project: PaddleViT (https://github.com/BR-IDL/PaddleViT)
# 2021.11
import math
from typing import Union, Tuple, List

import paddle
import paddle.nn as nn
from paddle import Tensor

from paddle import ParamAttr

# paddle.set_device('cpu')
paddle.set_device('gpu')  # 有免费的GPU不用，你是不是山炮啊！


class Attention(nn.Layer):
    """
    第一种实现方式（默认实现）

    """
    def __init__(self,
                 embed_dim: int,
                 num_attn_heads: int,
                 qkv_bias: Union[ParamAttr, bool] = False,
                 dropout: float = 0.,
                 attention_dropout: float = 0.):
        super().__init__()

        self.num_attn_heads = num_attn_heads
        self.attn_head_size = int(embed_dim / self.num_attn_heads)
        self.all_heads_size = self.num_attn_heads * self.attn_head_size

        self.qkv_layer = nn.Linear(in_features=embed_dim,
                                   out_features=self.all_heads_size * 3,
                                   bias_attr=qkv_bias)

        self.output_proj = nn.Linear(in_features=embed_dim, out_features=embed_dim)

        self.scales = 1.0 / math.sqrt(float(self.attn_head_size))

        self.attn_dropout = nn.Dropout(attention_dropout)
        self.proj_dropout = nn.Dropout(dropout)

        self.softmax = nn.Softmax(axis=-1)

    def _transpose_for_scores(self,
                              input_tensor: Tensor,
                              batch_size: int,
                              num_attn_heads: int,
                              num_patches: int,
                              attn_head_size: int) -> Tensor:
        """
        [batch_size, num_patches, embed_dim] -> [batch_size, num_patches, num_attn_heads, attn_head_size]
        """
        output_tensor = paddle.reshape(
            input_tensor, [batch_size, num_patches, num_attn_heads, attn_head_size])
        output_tensor = paddle.transpose(output_tensor, [0, 2, 1, 3])
        return output_tensor

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        # x.shape: [batch_size, num_patches, embed_dim]
        batch_size, num_patches, _ = x.shape

        # 1. Linear Projection
        # [batch_size, num_patches, num_attn_heads * attn_head_size * 3]
        x = self.qkv_layer(x)
        # [batch_size, num_patches, num_attn_heads * attn_head_size]
        query, key, value = paddle.chunk(x, chunks=3, axis=-1)

        # 2. Multi-heads
        # [batch_size, num_attn_heads, num_patches, attn_head_size]
        query = self._transpose_for_scores(
            query, batch_size, self.num_attn_heads, num_patches, self.attn_head_size)
        key = self._transpose_for_scores(
            key, batch_size, self.num_attn_heads, num_patches, self.attn_head_size)
        value = self._transpose_for_scores(
            value, batch_size, self.num_attn_heads, num_patches, self.attn_head_size)

        # 3. Scaled Dot Production
        # [batch_size, num_attn_heads, num_patches, num_patches]
        attention_scores = paddle.matmul(query, key, transpose_y=True)
        attention_scores = self.scales * attention_scores

        # 4. Normalize the attention scores to probabilities.
        # [batch_size, num_attn_heads, num_patches, num_patches]
        attnetion_probs_ = self.softmax(attention_scores)
        attnetion_probs = attnetion_probs_
        attnetion_probs_ = self.attn_dropout(attnetion_probs_)

        # 5. weighted attention
        # [batch_size, num_attn_heads, num_patches, attn_head_size]
        context = paddle.matmul(attnetion_probs_, value)

        # 6. reshape to original shape
        # [batch_size, num_patches, num_attn_heads, attn_head_size]
        context = paddle.transpose(context, perm=[0, 2, 1, 3])
        # [batch_size, num_patches, num_attn_heads * attn_head_size]
        # context = paddle.flatten(context, start_axis=2)
        context = paddle.reshape(context, shape=context.shape[: -2] + [self.all_heads_size])

        # 7. Output Projection
        # [batch_size, num_patches, embed_dim]
        output = self.output_proj(context)
        output = self.proj_dropout(output)

        return output, attnetion_probs

    def num_parameters(self):
        # total = 0
        # for param in self.parameters():
        #     total += param.numel().item()

        total = sum([param.numel().item() for param in self.parameters()])

        return total


class AttentionSecond(nn.Layer):
    """
    第二种实现方式
    参考：https://github.com/google-research/bert/blob/master/modeling.py#L558

    """
    def __init__(self,
                 embed_dim: int,
                 num_attn_heads: int,
                 query_bias: Union[bool, ParamAttr] = False,
                 key_bias: Union[bool, ParamAttr] = False,
                 value_bias: Union[bool, ParamAttr] = False,
                 dropout: float = 0.,
                 attention_dropout: float = 0.):
        super().__init__()

        self.num_attn_heads = num_attn_heads
        self.attn_head_size = int(embed_dim / self.num_attn_heads)
        self.all_heads_size = self.num_attn_heads * self.attn_head_size

        self.query_layer = nn.Linear(
            in_features=embed_dim,
            out_features=self.all_heads_size,
            bias_attr=query_bias
        )
        self.key_layer = nn.Linear(
            in_features=embed_dim,
            out_features=self.all_heads_size,
            bias_attr=key_bias
        )
        self.value_layer = nn.Linear(
            in_features=embed_dim,
            out_features=self.all_heads_size,
            bias_attr=value_bias
        )

        self.output_proj = nn.Linear(in_features=embed_dim, out_features=embed_dim)

        self.scales = 1.0 / math.sqrt(float(self.attn_head_size))

        self.attn_dropout = nn.Dropout(attention_dropout)
        self.proj_dropout = nn.Dropout(dropout)

        self.softmax = nn.Softmax(axis=-1)

    def _transpose_for_scores(self,
                              input_tensor: Tensor,
                              batch_size: int,
                              num_attn_heads: int,
                              num_patches: int,
                              attn_head_size: int) -> Tensor:
        output_tensor = paddle.reshape(
            input_tensor, [batch_size, num_patches, num_attn_heads, attn_head_size])
        output_tensor = paddle.transpose(output_tensor, [0, 2, 1, 3])
        return output_tensor

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        # x.shape: [batch_size, num_patches, embed_dim]
        batch_size, num_patches, _ = x.shape

        # 1. Linear Projection
        # [batch_size, num_patches, num_attn_heads * attn_head_size]
        query = self.query_layer(x)
        key = self.key_layer(x)
        value = self.value_layer(x)

        # 2. Multi-heads
        # [batch_size, num_attn_heads, num_patches, attn_head_size]
        query = self._transpose_for_scores(
            query, batch_size, self.num_attn_heads, num_patches, self.attn_head_size)
        key = self._transpose_for_scores(
            key, batch_size, self.num_attn_heads, num_patches, self.attn_head_size)
        value = self._transpose_for_scores(
            value, batch_size, self.num_attn_heads, num_patches, self.attn_head_size)

        # 3. Scaled Dot Production
        # [batch_size, num_attn_heads, num_patches, num_patches]
        attention_scores = paddle.matmul(query, key, transpose_y=True)
        attention_scores = self.scales * attention_scores

        # 4. Normalize the attention scores to probabilities.
        # [batch_size, num_attn_heads, num_patches, num_patches]
        attnetion_probs_ = self.softmax(attention_scores)
        attnetion_probs = attnetion_probs_
        attnetion_probs_ = self.attn_dropout(attnetion_probs_)

        # 5. weighted attention
        # [batch_size, num_attn_heads, num_patches, attn_head_size]
        context = paddle.matmul(attnetion_probs_, value)

        # 6. reshape to original shape
        # [batch_size, num_patches, num_attn_heads, attn_head_size]
        context = paddle.transpose(context, perm=[0, 2, 1, 3])
        # [batch_size, num_patches, num_attn_heads * attn_head_size]
        context = paddle.flatten(context, start_axis=2)

        # 7. Output Projection
        # [batch_size, num_patches, embed_dim]
        output = self.output_proj(context)
        output = self.proj_dropout(output)

        return output, attnetion_probs

    def num_parameters(self):
        total = 0
        for param in self.parameters():
            total += param.numel().item()

        # total = sum([param.numel().item() for param in self.parameters()])

        return total


class AttentionTHETHIRD(nn.Layer):
    """
    第三种实现方式 (^_^ THE THIRD ^_^)
    """
    def __init__(self,
                 embed_dim: int,
                 num_attn_heads: int,
                 query_bias: Union[bool, ParamAttr] = False,
                 key_bias: Union[bool, ParamAttr] = False,
                 value_bias: Union[bool, ParamAttr] = False,
                 dropout: float = 0.,
                 attention_dropout: float = 0.):
        super().__init__()

        self.num_attn_heads = num_attn_heads
        self.attn_head_size = int(embed_dim / self.num_attn_heads)
        self.all_heads_size = self.num_attn_heads * self.attn_head_size

        self.query_layer_list = nn.LayerList()
        self.key_layer_list = nn.LayerList()
        self.value_layer_list = nn.LayerList()
        for head_idx in range(self.num_attn_heads):
            layer_list = nn.LayerList()
            query_layer = nn.Linear(
                in_features=self.attn_head_size,
                out_features=self.attn_head_size,
                bias_attr=query_bias,
                name=f'query_layer_{head_idx}'
            )
            key_layer = nn.Linear(
                in_features=self.attn_head_size,
                out_features=self.attn_head_size,
                bias_attr=key_bias,
                name=f'key_layer_{head_idx}'
            )
            value_layer = nn.Linear(
                in_features=self.attn_head_size,
                out_features=self.attn_head_size,
                bias_attr=value_bias,
                name=f'value_layer_{head_idx}'
            )
            self.query_layer_list.append(query_layer)
            self.key_layer_list.append(key_layer)
            self.value_layer_list.append(value_layer)

        self.output_proj = nn.Linear(in_features=embed_dim, out_features=embed_dim)

        self.scales = 1.0 / math.sqrt(float(self.attn_head_size))

        self.attn_dropout = nn.Dropout(attention_dropout)
        self.proj_dropout = nn.Dropout(dropout)

        self.softmax = nn.Softmax(axis=-1)

    def _transpose_for_scores(self,
                              input_tensor: Tensor,
                              batch_size: int,
                              num_attn_heads: int,
                              num_patches: int,
                              attn_head_size: int) -> Tensor:
        output_tensor = paddle.reshape(
            input_tensor, [batch_size, num_patches, num_attn_heads, attn_head_size])
        output_tensor = paddle.transpose(output_tensor, [0, 2, 1, 3])
        return output_tensor

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        # x.shape: [batch_size, num_patches, embed_dim]
        batch_size, num_patches, _ = x.shape
        # [batch_size, num_patches, attn_head_size]
        x_chunks = paddle.chunk(x, chunks=self.num_attn_heads, axis=-1)
        query_list, key_list, value_list = [], [], []
        for head_idx in range(self.num_attn_heads):
            query_layer = self.query_layer_list[head_idx]
            key_layer = self.key_layer_list[head_idx]
            value_layer = self.value_layer_list[head_idx]
            x = x_chunks[head_idx]
            query = query_layer(x)
            key = key_layer(x)
            value = value_layer(x)
            query_list.append(query)
            key_list.append(key)
            value_list.append(value)

        # [batch_size, num_patches, num_attn_heads * attn_head_size]
        query = paddle.concat(query_list, axis=-1)
        key = paddle.concat(key_list, axis=-1)
        value = paddle.concat(value_list, axis=-1)

        # 2. Multi-heads
        # [batch_size, num_attn_heads, num_patches, attn_head_size]
        query = self._transpose_for_scores(
            query, batch_size, self.num_attn_heads, num_patches, self.attn_head_size)
        key = self._transpose_for_scores(
            key, batch_size, self.num_attn_heads, num_patches, self.attn_head_size)
        value = self._transpose_for_scores(
            value, batch_size, self.num_attn_heads, num_patches, self.attn_head_size)

        # 3. Scaled Dot Production
        # [batch_size, num_attn_heads, num_patches, num_patches]
        attention_scores = paddle.matmul(query, key, transpose_y=True)
        attention_scores = self.scales * attention_scores

        # 4. Normalize the attention scores to probabilities.
        # [batch_size, num_attn_heads, num_patches, num_patches]
        attnetion_probs_ = self.softmax(attention_scores)
        attnetion_probs = attnetion_probs_
        attnetion_probs_ = self.attn_dropout(attnetion_probs_)

        # 5. weighted attention
        # [batch_size, num_attn_heads, num_patches, attn_head_size]
        context = paddle.matmul(attnetion_probs_, value)

        # 6. reshape to original shape
        # [batch_size, num_patches, num_attn_heads, attn_head_size]
        context = paddle.transpose(context, perm=[0, 2, 1, 3])
        # [batch_size, num_patches, num_attn_heads * attn_head_size]
        context = paddle.flatten(context, start_axis=2)

        # 7. Output Projection
        # [batch_size, num_patches, embed_dim]
        output = self.output_proj(context)
        output = self.proj_dropout(output)

        return output, attnetion_probs

    def num_parameters(self):
        total = 0
        for param in self.parameters():
            total += param.numel().item()

        # total = sum([param.numel().item() for param in self.parameters()])

        return total


def main():
    t = paddle.randn([4, 16, 96])     # [batch_size: int = 4, num_patches: int = 16, embed_dim: int = 96]
    print('input shape = ', t.shape)

    print("----------------------------第一种实现方式----------------------------------")
    model = Attention(embed_dim=96,
                      num_attn_heads=16,
                      qkv_bias=False,
                      dropout=0.,
                      attention_dropout=0.)

    print(model)

    out, attn_weights = model(t)
    print(f'out.shape: {out.shape}')
    print(f'attn_weights.shape: {attn_weights.shape}')
    print()
    for name, param in model.named_parameters():
        print(f'param name: {name},\tparam shape: {param.shape} ')

    print(f'Total parameters: {model.num_parameters()}')


    print("----------------------------第二种实现方式----------------------------------")
    
    model = AttentionSecond(embed_dim=96,
                            num_attn_heads=8,
                            query_bias=False,
                            key_bias=False,
                            value_bias=False,
                            dropout=0.,
                            attention_dropout=0.)

    print(model)

    out, attn_weights = model(t)
    print(f'out.shape: {out.shape}')
    print(f'attn_weights.shape: {attn_weights.shape}')
    print()
    for name, param in model.named_parameters():
        print(f'param name: {name},\tparam shape: {param.shape} ')

    print(f'Total parameters: {model.num_parameters()}')


    print("----------------------------第三种实现方式----------------------------------")
    
    model = AttentionTHETHIRD(embed_dim=96,
                              num_attn_heads=8,
                              query_bias=False,
                              key_bias=False,
                              value_bias=False,
                              dropout=0.,
                              attention_dropout=0.)

    print(model)

    out, attn_weights = model(t)
    print(f'out.shape: {out.shape}')
    print(f'attn_weights.shape: {attn_weights.shape}')
    print()

    for name, param in model.named_parameters():
        print(f'param name: {name},\tparam shape: {param.shape} ')

    print(f'total parameters: {model.num_parameters()}')


if __name__ == "__main__":
    main()
