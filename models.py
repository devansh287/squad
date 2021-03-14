"""Top-level model classes.

Author:
    Chris Chute (chute@stanford.edu)
"""

import layers
import torch
import torch.nn as nn


class BiDAF(nn.Module):
    """Baseline BiDAF model for SQuAD.

    Based on the paper:
    "Bidirectional Attention Flow for Machine Comprehension"
    by Minjoon Seo, Aniruddha Kembhavi, Ali Farhadi, Hannaneh Hajishirzi
    (https://arxiv.org/abs/1611.01603).

    Follows a high-level structure commonly found in SQuAD models:
        - Embedding layer: Embed word indices to get word vectors.
        - Encoder layer: Encode the embedded sequence.
        - Attention layer: Apply an attention mechanism to the encoded sequence.
        - Model encoder layer: Encode the sequence again.
        - Output layer: Simple layer (e.g., fc + softmax) to get final outputs.

    Args:
        word_vectors (torch.Tensor): Pre-trained word vectors.
        hidden_size (int): Number of features in the hidden state at each layer.
        drop_prob (float): Dropout probability.
    """
    def __init__(self, word_vectors, hidden_size, drop_prob=0.):
        super(BiDAF, self).__init__()
        self.emb = layers.Embedding(word_vectors=word_vectors,
                                    hidden_size=hidden_size,
                                    drop_prob=drop_prob)

        self.enc = layers.RNNEncoder(input_size=hidden_size,
                                     hidden_size=hidden_size,
                                     num_layers=1,
                                     drop_prob=drop_prob)

        self.att = layers.BiDAFAttention(hidden_size=2 * hidden_size,
                                         drop_prob=drop_prob)

        self.mod = layers.RNNEncoder(input_size=8 * hidden_size,
                                     hidden_size=hidden_size,
                                     num_layers=2,
                                     drop_prob=drop_prob)

        self.out = layers.BiDAFOutput(hidden_size=hidden_size,
                                      drop_prob=drop_prob)

    def forward(self, cw_idxs, cc_idxs, qw_idxs, qc_idxs):
        c_mask = torch.zeros_like(cw_idxs) != cw_idxs
        q_mask = torch.zeros_like(qw_idxs) != qw_idxs
        c_len, q_len = c_mask.sum(-1), q_mask.sum(-1)

        c_emb = self.emb(cw_idxs)         # (batch_size, c_len, hidden_size)
        q_emb = self.emb(qw_idxs)         # (batch_size, q_len, hidden_size)

        c_enc = self.enc(c_emb, c_len)    # (batch_size, c_len, 2 * hidden_size)
        q_enc = self.enc(q_emb, q_len)    # (batch_size, q_len, 2 * hidden_size)

        att = self.att(c_enc, q_enc,
                       c_mask, q_mask)    # (batch_size, c_len, 8 * hidden_size)

        mod = self.mod(att, c_len)        # (batch_size, c_len, 2 * hidden_size)

        out = self.out(att, mod, c_mask)  # 2 tensors, each (batch_size, c_len)

        return out

class charBiDAF(nn.Module):
    """Modified Character Embedding BiDAF
    """
    def __init__(self, word_vectors, char_vectors, emb_size, hidden_size, drop_prob=0.):
        super(charBiDAF, self).__init__()
        self.emb = layers.charEmbedding(word_vectors=word_vectors,
                                        char_vectors=char_vectors,
                                        emb_size=emb_size,
                                        hidden_size=hidden_size,
                                        drop_prob=drop_prob)

        self.enc = layers.RNNEncoder(input_size=hidden_size,
                                     hidden_size=hidden_size,
                                     num_layers=1,
                                     drop_prob=drop_prob)

        self.att = layers.BiDAFAttention(hidden_size=2 * hidden_size,
                                         drop_prob=drop_prob)

        self.mod = layers.RNNEncoder(input_size=8 * hidden_size,
                                     hidden_size=hidden_size,
                                     num_layers=2,
                                     drop_prob=drop_prob)

        self.out = layers.BiDAFOutput(hidden_size=hidden_size,
                                      drop_prob=drop_prob)

    def forward(self, cw_idxs, cc_idxs, qw_idxs, qc_idxs):
        c_mask = torch.zeros_like(cw_idxs) != cw_idxs
        q_mask = torch.zeros_like(qw_idxs) != qw_idxs
        c_len, q_len = c_mask.sum(-1), q_mask.sum(-1)

        c_emb = self.emb(cw_idxs, cc_idxs)         # (batch_size, c_len, hidden_size)
        q_emb = self.emb(qw_idxs, qc_idxs)         # (batch_size, q_len, hidden_size)

        c_enc = self.enc(c_emb, c_len)    # (batch_size, c_len, 2 * hidden_size)
        q_enc = self.enc(q_emb, q_len)    # (batch_size, q_len, 2 * hidden_size)

        att = self.att(c_enc, q_enc,
                       c_mask, q_mask)    # (batch_size, c_len, 8 * hidden_size)

        mod = self.mod(att, c_len)        # (batch_size, c_len, 2 * hidden_size)

        out = self.out(att, mod, c_mask)  # 2 tensors, each (batch_size, c_len)

        return out


"""
---------------------------- YOU ARE ENTERING TRANSFORMER ZONE ---------------------------
"""

class QANet(nn.Module):
    """
    Inspired by: https://arxiv.org/pdf/1804.09541.pdf
    """
    def __init__(self, word_vectors, char_vectors, emb_size, hidden_size, drop_prob=0.):
        super(QANet, self).__init__()
        self.num_model_blocks = 4       # as suggested by QANet paper
        self.encoder_conv_layers = 4    # as suggested by QANet paper
        self.model_conv_layers = 2      # as suggested by QANet paper

        self.emb = layers.charEmbedding(word_vectors=word_vectors,
                                        char_vectors=char_vectors,
                                        emb_size=emb_size,
                                        hidden_size=hidden_size,
                                        drop_prob=drop_prob)

        self.enc = layers.QAEncoder(input_size=hidden_size,
                                    hidden_size=hidden_size,
                                    num_layers=self.encoder_conv_layers,
                                    drop_prob=drop_prob)

        self.att = layers.ContextQueryAttention(hidden_size=hidden_size,
                                         drop_prob=drop_prob)

        self.model_blocks = []
        for i in range(self.num_model_blocks):
            self.model_blocks.append(layers.QAEncoder(input_size=4*hidden_size,
                                                      hidden_size=4*hidden_size,
                                                      num_layers=self.model_conv_layers,
                                                      drop_prob=drop_prob))

        # Caution: may have to write new output block in layers.py
        self.out = layers.QAOutput(hidden_size=8*hidden_size)

    def forward(self, cw_idxs, cc_idxs, qw_idxs, qc_idxs):

        c_mask = torch.zeros_like(cw_idxs) != cw_idxs
        q_mask = torch.zeros_like(qw_idxs) != qw_idxs

        # Embedding
        c_emb = self.emb(cw_idxs, cc_idxs)         # (batch_size, c_len, hidden_size)
        q_emb = self.emb(qw_idxs, qc_idxs)         # (batch_size, q_len, hidden_size)

        # Encoding
        c_enc = self.enc(c_emb)     # (batch_size, c_len, hidden_size)
        q_enc = self.enc(q_emb)     # (batch_size, q_len, hidden_size)

        # Context-query attention
        att = self.att(c_enc.cuda(), q_enc.cuda(), c_mask.cuda(), q_mask.cuda())   # (batch_size, c_len, hidden_size)
        del c_enc
        del q_enc

        # Model blocks - repeat 3 times with same parameters
        block1 = att
        for model_block in self.model_blocks:
            block1 = model_block(block1)
        del att

        block2 = block1
        for model_block in self.model_blocks:
            block2 = model_block(block2)

        block3 = block2
        for model_block in self.model_blocks:
            block3 = model_block(block3)

        # Concatenate model blocks to obtain start and end representations
        start = torch.cat((block1, block2), 2)
        end = torch.cat((block2, block3), 2)

        out = self.out(start, end, c_mask)

        return out
