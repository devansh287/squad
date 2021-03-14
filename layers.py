"""Assortment of layers for use in models.py.

Author:
    Chris Chute (chute@stanford.edu)
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from util import masked_softmax

device = torch.device("cuda:0")
back = torch.device("cpu:0")
class Embedding(nn.Module):
    """Embedding layer used by BiDAF, without the character-level component.

    Word-level embeddings are further refined using a 2-layer Highway Encoder
    (see `HighwayEncoder` class for details).

    Args:
        word_vectors (torch.Tensor): Pre-trained word vectors.
        hidden_size (int): Size of hidden activations.
        drop_prob (float): Probability of zero-ing out activations
    """
    def __init__(self, word_vectors, hidden_size, drop_prob):
        super(Embedding, self).__init__()
        self.drop_prob = drop_prob
        self.embed = nn.Embedding.from_pretrained(word_vectors)
        self.proj = nn.Linear(word_vectors.size(1), hidden_size, bias=False)
        self.hwy = HighwayEncoder(2, hidden_size)

    def forward(self, x):
        emb = self.embed(x)   # (batch_size, seq_len, embed_size)
        emb = F.dropout(emb, self.drop_prob, self.training)
        emb = self.proj(emb)  # (batch_size, seq_len, hidden_size)
        emb = self.hwy(emb)   # (batch_size, seq_len, hidden_size)

        return emb

class charEmbedding(nn.Module):
    def __init__(self, word_vectors, char_vectors, emb_size, hidden_size, drop_prob):
        super(charEmbedding, self).__init__()
        self.drop_prob = drop_prob
        self.wordEmbed = nn.Embedding.from_pretrained(word_vectors)
        self.charVectors = nn.Embedding.from_pretrained(char_vectors)
        self.convs = []
        num_kernels = 0
        for i in range(2, 5):
            self.convs.append(nn.Conv1d(in_channels=emb_size, out_channels=1, kernel_size=i))
            num_kernels += 1
        self.ReLU = nn.ReLU()
        self.pooling = nn.AdaptiveMaxPool1d(1)
        #self.charEmbed = nn.Conv1d(in_channels = 1, out_channels = 1)
        #we might have to set the channels to zero
        self.proj = nn.Linear(word_vectors.size(1) + num_kernels, hidden_size, bias=False)
        self.hwy = HighwayEncoder(2, hidden_size)

    def forward(self, word_idx, char_idx):
        word_emb = self.wordEmbed(word_idx)     # (batch_size, seq_len, word_embed_size)
        char_vec = self.charVectors(char_idx)   # (batch_sze, seq_len, max_word_len, char_embed_size)

        char_vec = torch.transpose(char_vec, 2, 3)  # (batch_size, seq_len, char_embed_size, max_word_len)
        (batch_size, seq_len, embed_size, max_word_len) = char_vec.size()
        char_vec = char_vec.view(batch_size * seq_len, embed_size, max_word_len)
        char_pooled = []
        char_vec = char_vec.cpu()

        for conv in self.convs:
            char_val = self.pooling(self.ReLU(conv(char_vec)))
            char_pooled.append(char_val)
        char_emb = torch.cat(char_pooled, 1)  # (batch_size*seq_len, num_kernels, 1)
        char_emb = char_emb.squeeze(2)  # (batch_size*seq_len, num_kernels)

        output_dim = char_emb.size(1)
        char_emb = char_emb.view(batch_size, seq_len, output_dim)  # (batch_size, seq_len, num_kernels)

        char_emb = char_emb.cuda()
        emb = torch.cat((word_emb, char_emb), 2) # (batch_size, seq_len, word_embed_size + num_kernels)
        emb = F.dropout(emb, self.drop_prob, self.training)
        emb = self.proj(emb)  # (batch_size, seq_len, hidden_size)
        emb = self.hwy(emb)   # (batch_size, seq_len, hidden_size)
        return emb


class HighwayEncoder(nn.Module):
    """Encode an input sequence using a highway network.

    Based on the paper:
    "Highway Networks"
    by Rupesh Kumar Srivastava, Klaus Greff, Jürgen Schmidhuber
    (https://arxiv.org/abs/1505.00387).

    Args:
        num_layers (int): Number of layers in the highway encoder.
        hidden_size (int): Size of hidden activations.
    """
    def __init__(self, num_layers, hidden_size):
        super(HighwayEncoder, self).__init__()
        self.transforms = nn.ModuleList([nn.Linear(hidden_size, hidden_size)
                                         for _ in range(num_layers)])
        self.gates = nn.ModuleList([nn.Linear(hidden_size, hidden_size)
                                    for _ in range(num_layers)])

    def forward(self, x):
        for gate, transform in zip(self.gates, self.transforms):
            # Shapes of g, t, and x are all (batch_size, seq_len, hidden_size)
            g = torch.sigmoid(gate(x))
            t = F.relu(transform(x))
            x = g * t + (1 - g) * x

        return x


class RNNEncoder(nn.Module):
    """General-purpose layer for encoding a sequence using a bidirectional RNN.

    Encoded output is the RNN's hidden state at each position, which
    has shape `(batch_size, seq_len, hidden_size * 2)`.

    Args:
        input_size (int): Size of a single timestep in the input.
        hidden_size (int): Size of the RNN hidden state.
        num_layers (int): Number of layers of RNN cells to use.
        drop_prob (float): Probability of zero-ing out activations.
    """
    def __init__(self,
                 input_size,
                 hidden_size,
                 num_layers,
                 drop_prob=0.):
        super(RNNEncoder, self).__init__()
        self.drop_prob = drop_prob
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers,
                           batch_first=True,
                           bidirectional=True,
                           dropout=drop_prob if num_layers > 1 else 0.)

    def forward(self, x, lengths):
        # Save original padded length for use by pad_packed_sequence
        orig_len = x.size(1)

        # Sort by length and pack sequence for RNN
        lengths, sort_idx = lengths.sort(0, descending=True)
        x = x[sort_idx]     # (batch_size, seq_len, input_size)
        x = pack_padded_sequence(x, lengths, batch_first=True)

        # Apply RNN
        x, _ = self.rnn(x)  # (batch_size, seq_len, 2 * hidden_size)

        # Unpack and reverse sort
        x, _ = pad_packed_sequence(x, batch_first=True, total_length=orig_len)
        _, unsort_idx = sort_idx.sort(0)
        x = x[unsort_idx]   # (batch_size, seq_len, 2 * hidden_size)

        # Apply dropout (RNN applies dropout after all but the last layer)
        x = F.dropout(x, self.drop_prob, self.training)

        return x


class BiDAFAttention(nn.Module):
    """Bidirectional attention originally used by BiDAF.

    Bidirectional attention computes attention in two directions:
    The context attends to the query and the query attends to the context.
    The output of this layer is the concatenation of [context, c2q_attention,
    context * c2q_attention, context * q2c_attention]. This concatenation allows
    the attention vector at each timestep, along with the embeddings from
    previous layers, to flow through the attention layer to the modeling layer.
    The output has shape (batch_size, context_len, 8 * hidden_size).

    Args:
        hidden_size (int): Size of hidden activations.
        drop_prob (float): Probability of zero-ing out activations.
    """
    def __init__(self, hidden_size, drop_prob=0.1):
        super(BiDAFAttention, self).__init__()
        self.drop_prob = drop_prob
        self.c_weight = nn.Parameter(torch.zeros(hidden_size, 1))
        self.q_weight = nn.Parameter(torch.zeros(hidden_size, 1))
        self.cq_weight = nn.Parameter(torch.zeros(1, 1, hidden_size))
        for weight in (self.c_weight, self.q_weight, self.cq_weight):
            nn.init.xavier_uniform_(weight)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, c, q, c_mask, q_mask):
        batch_size, c_len, _ = c.size()
        q_len = q.size(1)
        s = self.get_similarity_matrix(c, q)        # (batch_size, c_len, q_len)
        c_mask = c_mask.view(batch_size, c_len, 1)  # (batch_size, c_len, 1)
        q_mask = q_mask.view(batch_size, 1, q_len)  # (batch_size, 1, q_len)
        s1 = masked_softmax(s, q_mask, dim=2)       # (batch_size, c_len, q_len)
        s2 = masked_softmax(s, c_mask, dim=1)       # (batch_size, c_len, q_len)

        # (bs, c_len, q_len) x (bs, q_len, hid_size) => (bs, c_len, hid_size)
        a = torch.bmm(s1, q)
        # (bs, c_len, c_len) x (bs, c_len, hid_size) => (bs, c_len, hid_size)
        b = torch.bmm(torch.bmm(s1, s2.transpose(1, 2)), c)

        x = torch.cat([c, a, c * a, c * b], dim=2)  # (bs, c_len, 4 * hid_size)

        return x

    def get_similarity_matrix(self, c, q):
        """Get the "similarity matrix" between context and query (using the
        terminology of the BiDAF paper).

        A naive implementation as described in BiDAF would concatenate the
        three vectors then project the result with a single weight matrix. This
        method is a more memory-efficient implementation of the same operation.

        See Also:
            Equation 1 in https://arxiv.org/abs/1611.01603
        """
        c_len, q_len = c.size(1), q.size(1)
        c = F.dropout(c, self.drop_prob, self.training)  # (bs, c_len, hid_size)
        q = F.dropout(q, self.drop_prob, self.training)  # (bs, q_len, hid_size)

        # Shapes: (batch_size, c_len, q_len)
        s0 = torch.matmul(c, self.c_weight).expand([-1, -1, q_len])
        s1 = torch.matmul(q, self.q_weight).transpose(1, 2)\
                                           .expand([-1, c_len, -1])
        s2 = torch.matmul(c * self.cq_weight, q.transpose(1, 2))
        s = s0 + s1 + s2 + self.bias

        return s


class CoAttention(nn.Module):
    """
    Co-Attention
    """
    def __init__(self, hidden_size, drop_prob=0.1):
        super(CoAttention, self).__init__()
        self.hidden_proj = nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.tanh = nn.Tanh()
        self.c_sentinel = nn.Parameter(torch.rand(hidden_size))
        self.q_sentinel = nn.Parameter(torch.rand(hidden_size))
        self.a_softmax = nn.Softmax(2)
        self.b_softmax = nn.Softmax(1)
        self.biLSTM = nn.LSTM(input_size=2*hidden_size, hidden_size=2*hidden_size, num_layers=1, dropout=drop_prob, bidirectional=True)

    def forward(self, c, q, c_mask, q_mask):
        (batch_size, q_len, h_size) = q.size()
        c_len = c.size(1)

        q_prime = self.tanh(self.hidden_proj(q))  # (batch_size, q_len, h_size)
        q_sentinel = self.q_sentinel.expand(batch_size, 1, h_size)  # (batch_size, 1, h_size)
        q_prime = torch.cat((q_prime, q_sentinel), 1)  # (batch_size, q_len + 1, h_size)

        c_sentinel = self.c_sentinel.expand(batch_size, 1, h_size)  # (batch_size, 1, h_size)
        c = torch.cat((c, c_sentinel), 1)  # (batch_size, c_len + 1, h_size)

        # Affinity Matrix
        affinity = torch.matmul(c, q_prime.transpose(1, 2))  # (batch_size, c_len + 1, q_len + 1)

        # C2Q attention
        alpha = self.a_softmax(affinity)  # (batch_size, c_len + 1, q_len + 1)
        a = torch.matmul(alpha, q_prime)  # (batch_size, c_len + 1, h_size)

        # Q2C attention
        beta = self.b_softmax(affinity)  # (batch_size, c_len + 1, q_len + 1)
        b = torch.matmul(beta.transpose(1, 2), c)  # (batch_size, q_len + 1, h_size)

        s = torch.matmul(alpha[:, :-1, :], b)  # (batch_size, c_len, h_size)

        # BiLISTM layer
        lstm_input = torch.cat((s, a[:, :-1, :]), 2)  # (batch_size, c_len, 2 * h_size)
        h0 = torch.randn(2, c_len, 2 * h_size).cuda()
        c0 = torch.randn(2, c_len, 2 * h_size).cuda()
        u, (hn, cn) = self.biLSTM(lstm_input, (h0, c0))  # (batch_size, c_len, 4 * h_size)
        return u


class BiDAFOutput(nn.Module):
    """Output layer used by BiDAF for question answering.

    Computes a linear transformation of the attention and modeling
    outputs, then takes the softmax of the result to get the start pointer.
    A bidirectional LSTM is then applied the modeling output to produce `mod_2`.
    A second linear+softmax of the attention output and `mod_2` is used
    to get the end pointer.

    Args:
        hidden_size (int): Hidden size used in the BiDAF model.
        drop_prob (float): Probability of zero-ing out activations.
    """
    def __init__(self, hidden_size, drop_prob):
        super(BiDAFOutput, self).__init__()
        self.att_linear_1 = nn.Linear(8 * hidden_size, 1)
        self.mod_linear_1 = nn.Linear(2 * hidden_size, 1)

        self.rnn = RNNEncoder(input_size=2 * hidden_size,
                              hidden_size=hidden_size,
                              num_layers=1,
                              drop_prob=drop_prob)

        self.att_linear_2 = nn.Linear(8 * hidden_size, 1)
        self.mod_linear_2 = nn.Linear(2 * hidden_size, 1)

    def forward(self, att, mod, mask):
        # Shapes: (batch_size, seq_len, 1)
        logits_1 = self.att_linear_1(att) + self.mod_linear_1(mod)
        mod_2 = self.rnn(mod, mask.sum(-1))
        logits_2 = self.att_linear_2(att) + self.mod_linear_2(mod_2)

        # Shapes: (batch_size, seq_len)
        log_p1 = masked_softmax(logits_1.squeeze(), mask, log_softmax=True)
        log_p2 = masked_softmax(logits_2.squeeze(), mask, log_softmax=True)

        return log_p1, log_p2


"""
---------------------------- YOU ARE ENTERING TRANSFORMER ZONE ---------------------------
"""

class QAEncoder(nn.Module):
    """
    General QANet Encoder Block
    Instantiated many times in QAnet:
        One encoder each for encoding query / context embeddings.
        A batch of encoders in the model. This batch is reused 3 times.
    Intended to handle input size different to hidden size (this may not be necessary for QAnet).
    """
    def __init__(self,
                 input_size,
                 hidden_size,
                 num_layers,
                 drop_prob=0.):
        super(QAEncoder, self).__init__()
        # Hyperparameters
        self.kernel_size = 7 # paper suggest constant kernel size of 7 for both embeding encoder & model encoder
        self.num_heads = 10 # Likewise, this was suggested by the QANet paper
        self.drop_prob = drop_prob
        # Layer Norms - N.B. designed to handle input size different to hidden size
        self.init_layer_norm = nn.LayerNorm(input_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        # Convolutions - N.B. designed to handle input size different to hidden size
        self.init_conv = nn.Conv1d(in_channels=input_size,
                                   out_channels=hidden_size,
                                   kernel_size=self.kernel_size,
                                   padding=3,
                                   groups=hidden_size)
        self.convs = []
        for i in range(num_layers-1):
            self.convs.append(nn.Conv1d(in_channels=hidden_size,
                                        out_channels=hidden_size,
                                        kernel_size=self.kernel_size,
                                        padding=3,
                                        groups=hidden_size))
        # Multi-Head Self Attention
        self.att = MultiHeadSelfAttention(hidden_size, self.num_heads, drop_prob=drop_prob)
        self.pos_encoder = PositionalEncoding(input_size, dropout=drop_prob)
        #Feedforward Network
        self.feedforward = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()

    def forward(self, x):

        #convert to cuda
        self.init_layer_norm = self.init_layer_norm.to(device)
        self.layer_norm = self.layer_norm.to(device)
        self.init_conv = self.init_conv.to(device)
        for i in range(len(self.convs)):
            self.convs[i] = self.convs[i].to(device)
        self.att = self.att.to(device)
        self.feedforward = self.feedforward.to(device)

        # Convolution layers
        x = self.pos_encoder(x)         # (batch_size, seq_len, input_size)
        x = x.to(device)
        x = self.init_layer_norm(x)     # (batch_size, seq_len, input_size)
        x = torch.transpose(x, 1, 2)    # (batch_size, input_size, seq_len)
        x = self.init_conv(x)           # (batch_size, hidden_size, seq_len)
        x = torch.transpose(x, 1, 2)
        for conv in self.convs:
            start_state = x
            x = self.layer_norm(x)
            x = torch.transpose(x, 1, 2)
            x = conv(x)
            x = torch.transpose(x, 1, 2)
            x = x + start_state

        # Self-attention layer
        start_state = x
        x = self.layer_norm(x)
        x = self.att(x)
        x = x + start_state
        del start_state

        # Feedforward layer (preliminarily a single-layer perceptron)
        start_state = x
        x = self.layer_norm(x)
        x = self.feedforward(x)
        x = self.relu(x)
        x = x + start_state

        self.init_layer_norm = self.init_layer_norm.cpu()
        self.layer_norm = self.layer_norm.cpu()
        self.init_conv = self.init_conv.cpu()
        for i in range(len(self.convs)):
            self.convs[i] = self.convs[i].cpu()
        self.att = self.att.cpu()
        self.feedforward = self.feedforward.cpu()
        x = x.cpu()

        return x
   

class QAOutput(nn.Module):
    """
    QANet output layer - adapted for compatibility with BiDAF output
    """
    def __init__(self, hidden_size):
        super(QAOutput, self).__init__()
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, start, end, mask):
        # Shapes: (batch_size, seq_len, 1)
        start_logits = self.linear(start)
        end_logits = self.linear(end)

        # Shapes: (batch_size, seq_len)
        log_p1 = masked_softmax(start_logits.squeeze(), mask, log_softmax=True)
        log_p2 = masked_softmax(end_logits.squeeze(), mask, log_softmax=True)
        return log_p1, log_p2


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x.to(device)
        self.pe = self.pe.to(device)
        x = x + self.pe[:x.size(0), :]
        x = x.cpu()
        self.pe = self.pe.cpu()
        return self.dropout(x)


class ContextQueryAttention(nn.Module):
    """
    Context-Query Attention for QANet
    Currently identical to BiDAF attention
    Potentially should be simplified to DCN attention

    Args:
        hidden_size (int): Size of hidden activations.
        drop_prob (float): Probability of zero-ing out activations.
    """
    def __init__(self, hidden_size, drop_prob=0.1):
        super(ContextQueryAttention, self).__init__()
        self.drop_prob = drop_prob
        self.c_weight = nn.Parameter(torch.zeros(hidden_size, 1))
        self.q_weight = nn.Parameter(torch.zeros(hidden_size, 1))
        self.cq_weight = nn.Parameter(torch.zeros(1, 1, hidden_size))
        for weight in (self.c_weight, self.q_weight, self.cq_weight):
            nn.init.xavier_uniform_(weight)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, c, q, c_mask, q_mask):
        batch_size, c_len, _ = c.size()
        q_len = q.size(1)
        s = self.get_similarity_matrix(c, q)        # (batch_size, c_len, q_len)
        c_mask = c_mask.view(batch_size, c_len, 1)  # (batch_size, c_len, 1)
        q_mask = q_mask.view(batch_size, 1, q_len)  # (batch_size, 1, q_len)
        s1 = masked_softmax(s, q_mask, dim=2)       # (batch_size, c_len, q_len)
        s2 = masked_softmax(s, c_mask, dim=1)       # (batch_size, c_len, q_len)

        # (bs, c_len, q_len) x (bs, q_len, hid_size) => (bs, c_len, hid_size)
        a = torch.bmm(s1, q)
        # (bs, c_len, c_len) x (bs, c_len, hid_size) => (bs, c_len, hid_size)
        b = torch.bmm(torch.bmm(s1, s2.transpose(1, 2)), c)

        x = torch.cat([c, a, c * a, c * b], dim=2)  # (bs, c_len, 4 * hid_size)

        return x

    def get_similarity_matrix(self, c, q):
        """Get the "similarity matrix" between context and query (using the
        terminology of the BiDAF paper).

        A naive implementation as described in BiDAF would concatenate the
        three vectors then project the result with a single weight matrix. This
        method is a more memory-efficient implementation of the same operation.

        See Also:
            Equation 1 in https://arxiv.org/abs/1611.01603
        """
        c_len, q_len = c.size(1), q.size(1)
        c = F.dropout(c, self.drop_prob, self.training)  # (bs, c_len, hid_size)
        q = F.dropout(q, self.drop_prob, self.training)  # (bs, q_len, hid_size)

        # Shapes: (batch_size, c_len, q_len)
        s0 = torch.matmul(c, self.c_weight).expand([-1, -1, q_len])
        s1 = torch.matmul(q, self.q_weight).transpose(1, 2)\
                                           .expand([-1, c_len, -1])
        s2 = torch.matmul(c * self.cq_weight, q.transpose(1, 2))
        s = s0 + s1 + s2 + self.bias

        return s


class MultiHeadSelfAttention(nn.Module):
    """
    Adapted from https://towardsdatascience.com/how-to-code-the-transformer-in-pytorch-24db27c8f9ec
    """
    def __init__(self, hidden_size, num_heads, drop_prob = 0.1):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.d_k = self.hidden_size // self.num_heads

        self.key_lin = nn.Linear(hidden_size, hidden_size)
        self.query_lin = nn.Linear(hidden_size, hidden_size)
        self.val_lin = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(drop_prob)
        self.out = nn.Linear(hidden_size, hidden_size)

    def forward(self, x, mask=None):
        batch_size = x.size(0)
        seq_len = x.size(1)

        #converting to cuda
        """
        self.key_lin=self.key_lin.to(device)
        self.query_lin=self.query_lin.to(device)
        self.val_lin=self.val_lin.to(device)
        self.out=self.out.to(device)
        """

        key = self.key_lin(x)
        key = key.view(batch_size, seq_len, self.num_heads, self.d_k)
        query = self.query_lin(x).view(batch_size, seq_len, self.num_heads, self.d_k)
        value = self.val_lin(x).view(batch_size, seq_len, self.num_heads, self.d_k)

        key = key.transpose(1,2)
        query = query.transpose(1,2)
        value = value.transpose(1,2)\
        # .contiguous().view(batch_size * self.num_heads, seq_len, self.d_k)


        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)
        scores = F.softmax(scores, dim=-1)
        scores = self.dropout(scores)
        scores = torch.matmul(scores, value)

        #scores = attention(query, key, value, self.d_k, mask, self.dropout)

        del key
        del query
        del value

        result = scores.transpose(1,2).contiguous()
        result = result.view(batch_size, seq_len, self.hidden_size)
        output = self.out(result)
        del scores
        del result

        #converting back
        """
        self.key_lin=self.key_lin.cpu()
        self.query_lin=self.query_lin.cpu()
        self.val_lin=self.val_lin.cpu()
        self.out=self.out.cpu()
        """

        return output


def attention(q, k, v, d_k, mask=None, dropout=None):
    """
    Taken from https://towardsdatascience.com/how-to-code-the-transformer-in-pytorch-24db27c8f9ec
    """
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 0, -1e9)
    scores = F.softmax(scores, dim=-1)

    if dropout is not None:
        scores = dropout(scores)

    output = torch.matmul(scores, v)
    del scores
    return output