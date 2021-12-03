"""
 @Time    : 2021/5/7 19:53
 @Author  : WuW15
"""
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def attention(query, key, value, dropout=None):
    """
    This function used to calculate the dot product attention

        Attention(Q,K,V) = softmax(QK^T/sqrt(dk))V

    :param dropout: drop out rate
    :param query: query array
    :param key: key array
    :param value: value array
    :return: scaled value, attention
    """
    dk = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(dk)
    att = F.softmax(scores, dim=-1)
    if dropout:
        att = dropout(att)
    return torch.matmul(att, value), att


class ObjectBehaviorModel(nn.Module):
    """
    Main detector model for behavior detection
    returns the motion class [left/follow lane/right] and time to perform regression [s]

    Main idea and attention block implantation took form :
    Transformer: Attention Is All You Need
    https://arxiv.org/abs/1706.03762
    ViT: An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale
    https://arxiv.org/abs/2010.11929

    Stacks the Encoder like block as the feature extractor
    """

    def __init__(self, dim, num_obj, num_head, num_block, num_cls, num_reg, hist_len):
        super(ObjectBehaviorModel, self).__init__()
        self.extractor_layer = clones(ExtractorBlock(num_head, dim * num_obj, dim * num_obj * 4), num_block)
        self.classifier_layer = BehaviorClassifier(hist_len * dim * num_obj, hist_len * dim * num_obj * 4, num_cls,
                                                   num_reg)
        self.norm = LayerNorm(dim * num_obj * hist_len)

    def forward(self, x):
        for layer in self.extractor_layer:
            x = layer(x)
        x = x.contiguous().view(x.size(0), 1, -1)
        x = self.norm(x)
        return self.classifier_layer(x)


class ExtractorBlock(nn.Module):
    def __init__(self, num_header, dim_model, dim_feed_forward, dropout=0.2):
        super(ExtractorBlock, self).__init__()
        self.multi_att = MultiHeadAttention(num_header, dim_model, dropout)
        self.norm = LayerNorm(dim_model)
        self.feed_forward = PositionwiseFeedForward(dim_model, dim_feed_forward)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        y = self.norm(x)
        x = x + self.dropout(self.multi_att(y, y, y))
        y = self.norm(x)
        x = x + self.dropout(self.feed_forward(y))
        return x


class BehaviorClassifier(nn.Module):
    def __init__(self, dim_model, dim_feed_forward, dim_cls, dim_reg, dropout=0.2):
        super(BehaviorClassifier, self).__init__()
        self.feed_forward = PositionwiseFeedForward(dim_model, dim_feed_forward)
        self.liner_cls0 = nn.Linear(dim_model, dim_cls)
        self.liner_cls1 = nn.Linear(dim_model, dim_cls)
        self.liner_cls2 = nn.Linear(dim_model, dim_cls)
        self.liner_reg = nn.Linear(dim_model, dim_reg)
        self.dropout = nn.Dropout(dropout)
        self.dim_cls = dim_cls

    def forward(self, x):
        x = self.dropout(self.feed_forward(x))
        return F.softmax(self.liner_cls0(x), dim=-1).squeeze(1), F.softmax(self.liner_cls1(x), dim=-1).squeeze(1), \
               F.softmax(self.liner_cls2(x), dim=-1).squeeze(1), \
               self.liner_reg(x)


class PositionwiseFeedForward(nn.Module):
    """
    feed forward
    """

    def __init__(self, dim_model, dim_feed_forward, dropout=0.2):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(dim_model, dim_feed_forward)
        self.w_2 = nn.Linear(dim_feed_forward, dim_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class MultiHeadAttention(nn.Module):
    """
    Multi head attention block, used to extract different key, query, value set

                      ------ Header1 ---> K Q V set1 ->attention

    input array ----> ------ Header2 ---> K Q V set2 ->attention  -----> concat --> n* scaled attention feature

                      ------ HeaderN ---> K Q V setN ->attention
    """

    def __init__(self, num_header, dim_model, dropout=0.2):
        super(MultiHeadAttention, self).__init__()
        self.dim_k = dim_model // num_header
        self.num_header = num_header
        self.linears = clones(nn.Linear(dim_model, dim_model), 3)
        self.linear = nn.Linear(dim_model, dim_model)
        self.attention = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value):
        num_batches = query.size(0)
        query, key, value = [l(x).view(num_batches, -1, self.num_header, self.dim_k).transpose(1, 2)
                             for l, x in zip(self.linears, (query, key, value))]
        x, self.attention = attention(query, key, value, self.dropout)
        x = x.transpose(1, 2).contiguous().view(num_batches, -1, self.num_header * self.dim_k)
        return self.linear(x)


class LayerNorm(nn.Module):
    """
    Normalization layer

    LN(xi) = alpha*(xi-mean)/sqrt(var^2+e) + beta
    """

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
