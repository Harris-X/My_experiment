from typing import Union

import paddle
import paddle.nn as nn
from paddle import ParamAttr

class Attention(nn.Layer):
    """ Attention module

    Attention module for ViT, here q, k, v are assumed the same.
    The qkv mappings are stored as one single param.

    Attributes:
        num_heads: number of heads
        attn_head_size: feature dim of single head
        all_head_size: feature dim of all heads
        qkv: a nn.Linear for q, k, v mapping
        scales: 1 / sqrt(single_head_feature_dim)
        out: projection of multi-head attention
        attn_dropout: dropout for attention
        proj_dropout: final dropout before output
        softmax: softmax op for attention
    """
    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 attn_head_size: int,
                 qkv_bias: Union[bool, ParamAttr],
                 dropout: float = 0.,
                 attention_dropout: float = 0.):
        super().__init__()
        """
        增加了一个attn_head_size的参数，attn_head_size和num_heads的大小不受embed_dim的限制。
        """
        self.num_heads = num_heads
        self.attn_head_size = attn_head_size
        self.all_head_size = self.attn_head_size * self.num_heads  # Attention Layer's hidden_size

        w_attr_1, b_attr_1 = self._init_weights()
        self.qkv = nn.Linear(embed_dim,
                             self.all_head_size*3,  # weights for q, k, and v
                             weight_attr=w_attr_1,
                             bias_attr=b_attr_1 if qkv_bias else False)

        self.scales = self.attn_head_size ** -0.5

        w_attr_2, b_attr_2 = self._init_weights()
        # self.out = nn.Linear(embed_dim,
        #                      embed_dim,
        #                      weight_attr=w_attr_2,
        #                      bias_attr=b_attr_2)
        # 用于将维度映射回 embed_dim，方便残差连接
        self.out = nn.Linear(self.all_head_size,
                             embed_dim,
                             weight_attr=w_attr_2,
                             bias_attr=b_attr_2)

        self.attn_dropout = nn.Dropout(attention_dropout)
        self.proj_dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(axis=-1)

    def _init_weights(self):
        weight_attr = paddle.ParamAttr(initializer=nn.initializer.KaimingUniform())
        bias_attr = paddle.ParamAttr(initializer=nn.initializer.KaimingUniform())
        return weight_attr, bias_attr

    def transpose_multihead(self, x):
        new_shape = x.shape[:-1] + [self.num_heads, self.attn_head_size]
        x = x.reshape(new_shape)
        x = x.transpose([0, 2, 1, 3])
        return x

    def forward(self, x):
        qkv = self.qkv(x).chunk(3, axis=-1)
        q, k, v = map(self.transpose_multihead, qkv)

        attn = paddle.matmul(q, k, transpose_y=True)
        attn = attn * self.scales
        attn = self.softmax(attn)
        attn_weights = attn
        attn = self.attn_dropout(attn)

        z = paddle.matmul(attn, v)
        z = z.transpose([0, 2, 1, 3])
        new_shape = z.shape[:-2] + [self.all_head_size]
        z = z.reshape(new_shape)
        # reshape，将维度映射回输入维度embed_dim，方便残差连接
        z = self.out(z)
        z = self.proj_dropout(z)
        return z, attn_weights

def main():
    t = paddle.randn([4, 16, 96])     # [batch_size, num_patches, embed_dim]
    print('input shape = ', t.shape)

    model = Attention(embed_dim=96,
                      num_heads=8,
                      attn_head_size=128,
                      qkv_bias=False,
                      dropout=0.,
                      attention_dropout=0.)

    print(model)

    out, attn_weights = model(t)
    print(out.shape)
    print(attn_weights.shape)

    for name, param in model.named_parameters():
        print(f'param name: {name},\tparam shape: {param.shape} ')


if __name__ == "__main__":
    main()


"""Output:
input shape =  [4, 16, 96]
Attention(
  (qkv): Linear(in_features=96, out_features=3072, dtype=float32)
  (out): Linear(in_features=1024, out_features=96, dtype=float32)
  (attn_dropout): Dropout(p=0.0, axis=None, mode=upscale_in_train)
  (proj_dropout): Dropout(p=0.0, axis=None, mode=upscale_in_train)
  (softmax): Softmax(axis=-1)
)
[4, 16, 96]
[4, 8, 16, 16]
param name: qkv.weight,	param shape: [96, 3072] 
param name: out.weight,	param shape: [1024, 96] 
param name: out.bias,	param shape: [96] 

"""

