import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np

class WORDEBD(nn.Module):
    '''
        An embedding layer that maps the token id into its corresponding word
        embeddings. The word embeddings are kept as fixed once initialized.
    '''
    def __init__(self, vocab, finetune_ebd):
        super(WORDEBD, self).__init__()

        self.vocab_size, self.embedding_dim = vocab.vectors.size()
        self.embedding_layer = nn.Embedding(
                self.vocab_size, self.embedding_dim)
        self.embedding_layer.weight.data = vocab.vectors

        self.finetune_ebd = finetune_ebd

        self.drop_out = nn.Dropout(0.2)

        if self.finetune_ebd:
            self.embedding_layer.weight.requires_grad = True
        else:
            self.embedding_layer.weight.requires_grad = False

        

    def forward(self, data, weights=None):
        '''
            @param text: batch_size * max_text_len
            @return output: batch_size * max_text_len * embedding_dim
        '''
        if (weights is None) or (self.finetune_ebd == False):
            emb = self.embedding_layer(data['text'])
            
            return emb

        else:
            return F.embedding(data['text'],
                               weights['ebd.embedding_layer.weight'])
