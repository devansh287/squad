'''
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
        self.key_lin=self.key_lin.to(device)
        self.query_lin=self.query_lin.to(device)
        self.val_lin=self.val_lin.to(device)
        self.out=self.out.to(device)

        key = self.key_lin(x)
        key = key.view(batch_size, seq_len, self.num_heads, self.d_k)
        query = self.query_lin(x).view(batch_size, seq_len, self.num_heads, self.d_k)
        value = self.val_lin(x).view(batch_size, seq_len, self.num_heads, self.d_k)

        key = key.transpose(1,2)
        query = query.transpose(1,2)
        value = value.transpose(1,2)
        # .contiguous().view(batch_size * self.num_heads, seq_len, self.d_k)


        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)
        scores = F.softmax(scores, dim=-1)
        scores = self.dropout(scores)
        scores = torch.matmul(scores, value)

        #scores = attention(query, key, value, self.d_k, mask, self.dropout)

        #del key
        #del query
        #del value

        result = scores.transpose(1,2).contiguous()
        result = result.view(batch_size, seq_len, self.hidden_size)
        output = self.out(result)
        #del scores
        #del result

        #converting back
        '''
        '''
        self.key_lin=self.key_lin.cpu()
        self.query_lin=self.query_lin.cpu()
        self.val_lin=self.val_lin.cpu()
        self.out=self.out.cpu()
        ''
        '''
        #return output
    '''

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
