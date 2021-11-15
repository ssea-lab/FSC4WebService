import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from embedding.wordebd import WORDEBD
from embedding.auxiliary.factory import get_embedding

class RNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, bidirectional,
            dropout):
        super(RNN, self).__init__()

        self.rnn = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True,
                bidirectional=bidirectional, dropout=dropout)

    def _sort_tensor(self, input, lengths):
        '''
        pack_padded_sequence  requires the length of seq be in descending order
        to work.
        Returns the sorted tensor, the sorted seq length, and the
        indices for inverting the order.

        Input:
                input: batch_size, seq_len, *
                lengths: batch_size
        Output:
                sorted_tensor: batch_size-num_zero, seq_len, *
                sorted_len:    batch_size-num_zero
                sorted_order:  batch_size
                num_zero
        '''
        sorted_lengths, sorted_order = lengths.sort(0, descending=True)
        sorted_input = input[sorted_order]
        _, invert_order  = sorted_order.sort(0, descending=False)

        # Calculate the num. of sequences that have len 0
        nonzero_idx = sorted_lengths.nonzero(as_tuple=False)
        num_nonzero = nonzero_idx.size()[0]
        num_zero = sorted_lengths.size()[0] - num_nonzero

        # temporarily remove seq with len zero
        sorted_input = sorted_input[:num_nonzero]
        sorted_lengths = sorted_lengths[:num_nonzero]

        return sorted_input, sorted_lengths, invert_order, num_zero

    def _unsort_tensor(self, input, invert_order, num_zero):
        '''
        Recover the origin order

        Input:
                input:        batch_size-num_zero, seq_len, hidden_dim
                invert_order: batch_size
                num_zero
        Output:
                out:   batch_size, seq_len, *
        '''
        if num_zero == 0:
            input = input[invert_order]

        else:
            dim0, dim1, dim2 = input.size()
            zero = torch.zeros((num_zero, dim1, dim2), device=input.device,
                    dtype=input.dtype)
            input = torch.cat((input, zero), dim=0)
            input = input[invert_order]

        return input

    def forward(self, text, text_len):
        '''
        Input: text, text_len
            text       Variable  batch_size * max_text_len * input_dim
            text_len   Tensor    batch_size

        Output: text
            text       Variable  batch_size * max_text_len * output_dim
        '''
        # Go through the rnn
        # Sort the word tensor according to the sentence length, and pack them together
        sort_text, sort_len, invert_order, num_zero = self._sort_tensor(input=text, lengths=text_len)
        text = pack_padded_sequence(sort_text, lengths=sort_len.cpu().numpy(), batch_first=True)


        # Run through the word level RNN
        text, _ = self.rnn(text)         # batch_size, max_doc_len, args.word_hidden_size

        # Unpack the output, and invert the sorting
        text = pad_packed_sequence(text, batch_first=True)[0] # batch_size, max_doc_len, rnn_size
        text = self._unsort_tensor(text, invert_order, num_zero) # batch_size, max_doc_len, rnn_size
        return text


class META(nn.Module):
    def __init__(self, ebd, args):
        super(META, self).__init__()
        self.args = args

        self.ebd = ebd
        self.aux = get_embedding(args)

        self.ebd_dim = self.ebd.embedding_dim

        self.add_ebds = nn.ModuleList([nn.Embedding(d,8) for d in add_key_counts])
        a = 1e-3
        for add_ebd,d in zip(self.add_ebds,add_key_counts):
            add_ebd.weight = nn.Parameter(torch.FloatTensor(d, 8).uniform_(-a, a))
        # self.auth_ebd = nn.Embedding(129, 16)
        # a = 1e-2
        # self.auth_ebd.weight = nn.Parameter(torch.FloatTensor(129, 10).uniform_(-a, a))

        input_dim = int(args.meta_idf) + self.aux.embedding_dim + \
            int(args.meta_w_target) + int(args.meta_iwf) + int(args.meta_task_idf) + int(args.meta_label_sim)
 
        if args.meta_ebd:
            # abalation use distributional signatures with word ebd may fail
            input_dim += self.ebd_dim
        if args.embedding == 'meta':
            self.rnn = RNN(input_dim, 25, 1, True, 0)

            self.seq = nn.Sequential(
                    nn.Dropout(self.args.dropout),
                    nn.Linear(50, 1),
                    )
        else:
            # use a mlp to predict the weight individually
            self.seq = nn.Sequential(
                nn.Linear(input_dim, 50),
                nn.ReLU(),
                nn.Dropout(self.args.dropout),
                nn.Linear(50, 1))
            
            # self.seq = nn.Sequential(
            #     nn.Linear(input_dim, 1))
            
        
        da = 300
        self.head = nn.Parameter(torch.Tensor(da, 1).uniform_(-0.1, 0.1))
        
        self.liner1 = nn.Linear(300, 1)
        self.liner2 = nn.Linear(300, 1)
        
        self.cnn_filter_sizes = [1,2,3]
        self.cnn_num_filters = 100
        
        self.convs = nn.ModuleList([nn.Conv1d(
                    in_channels=300,
                    out_channels=self.cnn_num_filters,
                    kernel_size=K) for K in self.cnn_filter_sizes])

    def change_ebd_frozen(self, state):
        for name,value in self.ebd.named_parameters():
            value.requires_grad = state
        # self.ebd.embedding_layer.weight.requires_grad = state

    def change_rnn_frozen(self, state):
        for name,value in self.named_parameters():
            if 'ebd' not in name:
                value.requires_grad = state

    def _conv_max_pool(self, x, conv_filter=None, weights=None):
        '''
        Compute sentence level convolution
        Input:
            x:      batch_size, max_doc_len, embedding_dim
        Output:     batch_size, num_filters_total
        '''
        assert(len(x.size()) == 3)

        x = x.permute(0, 2, 1)  # batch_size, embedding_dim, doc_len
        x = x.contiguous()

        # Apply the 1d conv. Resulting dimension is
        # [batch_size, num_filters, doc_len-filter_size+1] * len(filter_size)
        assert(not ((conv_filter is None) and (weights is None)))
        if conv_filter is not None:
            x = [conv(x) for conv in conv_filter]

        elif weights is not None:
            x = [F.conv1d(x, weight=weights['convs.{}.weight'.format(i)],
                          bias=weights['convs.{}.bias'.format(i)])
                 for i in range(len(self.cnn_filter_sizes))]

        # max pool over time. Resulting dimension is
        # [batch_size, num_filters] * len(filter_size)
        x = [F.max_pool1d(sub_x, sub_x.size(2)).squeeze(2) for sub_x in x]

        # concatenate along all filters. Resulting dimension is
        # [batch_size, num_filters_total]
        x = torch.cat(x, 1)
        x = F.relu(x)

        return x
    
    def forward(self, data, return_score=False):
        '''
            @param data dictionary
                @key text: batch_size * max_text_len
                @key text_len: batch_size
            @param return_score bool
                set to true for visualization purpose

            @return output: batch_size * embedding_dim
        '''
        ebd = self.ebd(data)
        
        if return_score:
            scale,iwf,task_idf  = self.compute_score(data, ebd, return_stats=True)
        else:
            scale  = self.compute_score(data, ebd)
        
        if self.args.feature_convergence == 'sum':
            ebd = torch.sum(ebd * scale, dim=1)
        elif self.args.feature_convergence == 'mean':
            ebd = torch.mean(ebd * scale, dim=1)
        elif self.args.feature_convergence == 'max':
            ebd = torch.max(ebd * scale, dim=1)
        elif self.args.feature_convergence == 'cnn':
            ebd = self._conv_max_pool(ebd * scale, conv_filter=self.convs)
        else:
            raise ValueError('Invalid feature_convergence.'
                         'feature convergence can only be: sum, max, mean, cnn.')
        ebds= [ebd]
        ebd = torch.cat(ebds,-1)
        
        if self.args.aug_support and data['is_support']:
            factor1 = torch.sigmoid(self.liner1(ebd))
            factor2 = torch.sigmoid(self.liner2(data['label_vectors'].float()))
            factor1 = factor1/(factor1+factor2)
            factor2 = 1 - factor1
            ebd = ebd * factor1 + data['label_vectors'].float() * factor2
        
        if return_score:
            return ebd, scale,iwf,task_idf 
        else:
            return ebd

    def _varlen_softmax(self, logit, text_len):
        '''
            Compute softmax for sentences with variable length
            @param: logit: batch_size * max_text_len
            @param: text_len: batch_size

            @return: score: batch_size * max_text_len
        '''
        logit = torch.exp(logit)
        mask = torch.arange(
                logit.size()[-1], device=logit.device,
                dtype=text_len.dtype).expand(*logit.size()
                        ) < text_len.unsqueeze(-1)

        logit = mask.float() * logit
        score = logit / torch.sum(logit, dim=1, keepdim=True)

        return score

    def _attention(self, x, text_len):
        '''
            text:     batch, max_text_len, input_dim
            text_len: batch, max_text_len
        '''
        batch_size, max_text_len, _ = x.size()

        proj_x = torch.tanh(x.view(batch_size * max_text_len, -1))
        att = torch.mm(proj_x, self.head)
        att = att.view(batch_size, max_text_len, 1)  # unnormalized

        # create mask
        idxes = torch.arange(max_text_len, out=torch.cuda.LongTensor(max_text_len,
                                 device=text_len.device)).unsqueeze(0)
        mask = (idxes < text_len.unsqueeze(1))
        att[~mask] = float('-inf')

        # apply softmax
        att = F.softmax(att, dim=1)  # batch, max_text_len

        return att

    
    def compute_score(self, data, ebd, return_stats=False):
        '''
            Compute the weight for each word

            @param data dictionary
            @param return_stats bool
                return statistics (input and output) for visualization purpose
            @return scale: batch_size * max_text_len * 1
        '''

        # preparing the input for the meta model
        x = self.aux(data)
        if self.args.meta_idf:
            idf = F.embedding(data['text'], data['idf']).detach()
            x = torch.cat([x, idf], dim=-1)

        if self.args.meta_task_idf:
            task_idf = F.embedding(data['text'], data['task_idf']).detach()
            x = torch.cat([x, task_idf], dim=-1)
            # x = torch.cat([x, task_idf ** 2], dim=-1)
            # x = torch.cat([x, task_idf ** 0.5], dim=-1)

        if self.args.meta_iwf:
            iwf = F.embedding(data['text'], data['iwf']).detach()
            x = torch.cat([x, iwf], dim=-1)

            # x = torch.cat([x, iwf ** 2], dim=-1)
            # x = torch.cat([x, iwf ** 0.5], dim=-1)
            
        # x = torch.cat([x, iwf * task_idf], dim=-1)
        
        if self.args.meta_ebd:
            x = torch.cat([x, ebd], dim=-1)

        if self.args.meta_w_target:
            if self.args.meta_target_entropy:
                w_target = ebd @ data['w_target']
                w_target = F.softmax(w_target, dim=2) * F.log_softmax(w_target,
                        dim=2)
                w_target = -torch.sum(w_target, dim=2, keepdim=True)
                w_target = 1.0 / w_target
                x = torch.cat([x, w_target.detach()], dim=-1)
            else:
                # for rr approxmiation, use the max weight to approximate
                # task-specific importance
                w_target = torch.abs(ebd @ data['w_target'])
                w_target = w_target.max(dim=2, keepdim=True)[0]
                x = torch.cat([x, w_target.detach()], dim=-1)

        if self.args.meta_label_sim:
            beta = ebd @ data['unique_label_vectors'].transpose(0,1).float()
            beta = beta.max(-1)[0].unsqueeze(-1) / 10
            # beta = beta / beta.max()
            x = torch.cat([x, beta], dim=-1)
        
        
        if self.args.embedding == 'meta':
            # run the LSTM
            hidden = self.rnn(x, data['text_len'])
        else:
            hidden = x

        # predict the logit
        logit = self.seq(hidden).squeeze(-1)  # batch_size * max_text_len
        # logit = hidden.squeeze(-1)

        score = self._varlen_softmax(logit, data['text_len']).unsqueeze(-1)

        if return_stats:
            return score, self._varlen_softmax(iwf.squeeze(-1), data['text_len']).unsqueeze(-1),\
                                self._varlen_softmax(task_idf.squeeze(-1), data['text_len']).unsqueeze(-1) 
        else:
            return score

def top_k_logits(logits, k):
    """Mask logits so that only top-k logits remain
    """
    values, _ = torch.topk(logits, k,1)
    min_values = values[:, -1, :].unsqueeze(1).repeat(1, logits.shape[1], 1)
    res = torch.where(logits < min_values, torch.zeros_like(logits, dtype=logits.dtype), logits)
    return res
