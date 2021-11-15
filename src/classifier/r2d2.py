import torch
import torch.nn as nn
import torch.nn.functional as F
from classifier.base import BASE

class R2D2(BASE):
    '''
        META-LEARNING WITH DIFFERENTIABLE CLOSED-FORM SOLVERS
    '''
    def __init__(self, ebd_dim, args):
        super(R2D2, self).__init__(args)
        self.ebd_dim = ebd_dim * 1

        # meta parameters to learn
        self.lam = nn.Parameter(torch.tensor(-1, dtype=torch.float))
        self.alpha = nn.Parameter(torch.tensor(0, dtype=torch.float))
        self.beta = nn.Parameter(torch.tensor(1, dtype=torch.float))
        # lambda and alpha is learned in the log space

        # cached tensor for speed
        self.I_support = nn.Parameter(
            torch.eye(self.args.shot * self.args.way, dtype=torch.float),
            requires_grad=False)
        self.I_way = nn.Parameter(torch.eye(self.args.way, dtype=torch.float),
                                  requires_grad=False)

        self.ls = LabelSmoothing(smoothing=0.1)

    def _compute_w(self, XS, YS_onehot):
        '''
            Compute the W matrix of ridge regression
            @param XS: support_size x ebd_dim
            @param YS_onehot: support_size x way

            @return W: ebd_dim * way
        '''
        # I_support = nn.Parameter(
        #     torch.eye(XS.size(0), dtype=torch.float),
        #     requires_grad=False).to('cuda:0')
        W = XS.t() @ torch.inverse(
                XS @ XS.t() + (10. ** self.lam) * self.I_support) @ YS_onehot

        return W

    def _label2onehot(self, Y):
        '''
            Map the labels into 0,..., way
            @param Y: batch_size

            @return Y_onehot: batch_size * ways
        '''
        Y_onehot = F.embedding(Y, self.I_way)

        return Y_onehot


    def forward(self, XS, YS, XQ, YQ, testing=False):
        '''
            @param XS (support x): support_size x ebd_dim
            @param YS (support y): support_size
            @param XQ (support x): query_size x ebd_dim
            @param YQ (support y): query_size

            @return acc
            @return loss
        '''

        
        YS, YQ = self.reidx_y(YS, YQ)

        YS_onehot = self._label2onehot(YS)

        W = self._compute_w(XS, YS_onehot)
    
        pred = (10.0 ** self.alpha) * XQ @ W + self.beta

        acc1 = BASE.compute_acc(pred, YQ)
        loss = F.cross_entropy(pred, YQ)
        # loss = self.ls(pred, YQ)

        # pred = nn.Softmax(dim=1)(pred)
        # row_mx_value = torch.max(pred,dim=1)[0]
        # # indices = torch.argmax(row_mx_value)
        # indices = row_mx_value>0.5
        # if indices.sum() > 0:
        #     pseud_y = torch.argmax(pred[indices,:],-1)
        #     all_x = torch.cat([XQ[indices,:],XS],0)
        #     all_y = torch.cat([pseud_y,YS])
        # else:
        #     all_x = XS
        #     all_y = YS
        # YS_onehot = self._label2onehot(all_y)
        # W = self._compute_w(all_x, YS_onehot)
        # pred = (10.0 ** self.alpha) * XQ @ W + self.beta
        # loss = F.cross_entropy(pred, YQ)

        # if testing:
        #     acc2 = BASE.compute_acc(pred, YQ)
        #     return acc1,acc2,loss

        return acc1, loss


class LabelSmoothing(nn.Module):
    """
    NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.0):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)

        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()