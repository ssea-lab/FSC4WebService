import torch
from torch.autograd import Variable
from torch.nn import functional as F
from classifier.gnn_iclr import *
from classifier.base import BASE
import logging
logging.basicConfig(level=logging.INFO)

class GNN(BASE):
    
    def __init__(self, ebd_dim, args):
        '''
        N: Num of classes
        '''
        super(GNN, self).__init__(args)
        self._cuda = args.cuda
        self.ebd_dim = ebd_dim
        N = self.args.way
        self.node_dim = ebd_dim + N
        self.gnn_obj = GNN_nl(args, N, self.node_dim, nf=96, J=1)

    def forward(self, XS, YS, XQ, YQ):
        '''
                @param XS (support x): support_size x ebd_dim
                @param YS (support y): support_size
                @param XQ (support x): query_size x ebd_dim
                @param YQ (support y): query_size

                @return acc
                @return loss
        '''
        YS, YQ = self.reidx_y(YS, YQ)
        support = XS.view(self.args.way, self.args.shot, self.ebd_dim)
        query = XQ.view(-1, self.ebd_dim)
        N = self.args.way
        K = self.args.shot

        NQ = query.size(0)
        D = self.ebd_dim

        support = support.unsqueeze(0).expand(NQ, -1, -1, -1).contiguous().view(-1, N * K, D) # (NQ, N * K, D)
        query = query.view(-1, 1, D) # (NQ, 1, D)
        labels = Variable(torch.zeros((NQ, 1 + N * K, N), dtype=torch.float)).cuda(self._cuda)
        for b in range(NQ):
            for i in range(N):
                for k in range(K):
                    labels[b][1 + i * K + k][YS[i*K+k]] = 1
        nodes = torch.cat([torch.cat([query, support], 1), labels], -1) # (NQ, 1 + N * K, D + N)
        logits = self.gnn_obj(nodes) # (NQ, N)
        # _, pred = torch.max(logits, 1)

        loss = F.cross_entropy(logits, YQ)

        acc = BASE.compute_acc(logits, YQ)

        return acc, loss
