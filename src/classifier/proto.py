import torch
import torch.nn as nn
import torch.nn.functional as F
from classifier.base import BASE


class PROTO(BASE):
    '''
        PROTOTIPICAL NETWORK FOR FEW SHOT LEARNING
    '''
    def __init__(self, ebd_dim, args):
        super(PROTO, self).__init__(args)
        self.ebd_dim = ebd_dim

        if args.embedding == 'meta':
            self.mlp = None
            print('No MLP')
        else:
            self.mlp = self._init_mlp(
                    self.ebd_dim, self.args.proto_hidden, self.args.dropout)

        # for instance-level attention
        self.fc = nn.Linear(self.ebd_dim, self.ebd_dim, bias=True)

    def _compute_prototype(self, XS, YS, XQ):
        '''
            Compute the prototype for each class by averaging over the ebd.

            @param XS (support x): support_size x ebd_dim
            @param YS (support y): support_size
            @param XQ (query x): query_size x ebd_dim

            @return prototype: way x ebd_dim
        '''
        # sort YS to make sure classes of the same labels are clustered together
        sorted_YS, indices = torch.sort(YS)
        sorted_XS = XS[indices] # support_size x ebd_dim

        if self.args.iatt:
            NQ = XQ.size(0)
            support = sorted_XS.contiguous().view((self.args.way, self.args.shot, -1)) # N,K,D
            # instance-level attention 
            support = support.unsqueeze(0).expand(NQ, -1, -1, -1) # (NQ, N, K, D)
            support_for_att = self.fc(support)  # (NQ, N, K, D)
            query_for_att = self.fc(XQ.unsqueeze(1).unsqueeze(2).expand(-1, self.args.way, self.args.shot, -1)) # (NQ, N, K, D)
            ins_att_score = F.softmax(torch.tanh(support_for_att * query_for_att).sum(-1), dim=-1) # (NQ, N, K)
            prototype = (support * ins_att_score.unsqueeze(3).expand(-1, -1, -1, self.ebd_dim)).sum(-1) # (NQ, N, D)
        else:
            prototype = []
            for i in range(self.args.way):
                prototype.append(torch.mean(
                    sorted_XS[i*self.args.shot:(i+1)*self.args.shot], dim=0,
                    keepdim=True))

            prototype = torch.cat(prototype, dim=0)

        return prototype

    def forward(self, XS, YS, XQ, YQ):
        '''
            @param XS (support x): support_size x ebd_dim
            @param YS (support y): support_size
            @param XQ (support x): query_size x ebd_dim
            @param YQ (support y): query_size

            @return acc
            @return loss
        '''
        if self.mlp is not None:
            XS = self.mlp(XS)
            XQ = self.mlp(XQ)

        YS, YQ = self.reidx_y(YS, YQ)

        prototype = self._compute_prototype(XS, YS, XQ)

        pred = -self._compute_l2(prototype, XQ)

        loss = F.cross_entropy(pred, YQ)

        acc = BASE.compute_acc(pred, YQ)

        return acc, loss
