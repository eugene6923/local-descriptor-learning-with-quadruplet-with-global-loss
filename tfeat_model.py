import torch
from torch import nn
import torch.nn.functional as F

class TNet(nn.Module):
    """TFeat model definition
    """
    def __init__(self):
        super(TNet, self).__init__()
        self.features = nn.Sequential(
            nn.InstanceNorm2d(1, affine=False),
            nn.Conv2d(1, 32, kernel_size=7),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=6),
            nn.ReLU()
        )
        self.descr = nn.Sequential(
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU()
        )
#        self.metriclearn = nn.Sequential(
#            nn.Dense(),
#            nn.Dense(),
#            nn.Dense()
#
#                )
#
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.descr(x)
        return x

#    def forward(self,anchor,pos,neg1,neg2) :
#
#        anchor=self.network1(anchor)
#        pos=self.network1(pos)
#        neg1=self.network1(neg1)
#        neg2=self.network2(neg2)
#
#        anchor_p=torch.cat([anchor,pos],dim=1)
#        anchor_n1=torch.cat([anchor,neg1],dim=1)
#        n1_n2=torch.cat([neg1,neg2],dim1)
#
#        pos_d=self.metriclearn(anchor_p)
#        neg1_d=self.metriclearn(anchor_n1)
#        neg2_d=self.metriclearn(anchor_n2)
#
#        return pos_d,neg1_d,neg2_d
#

class newQuadrupletLoss(nn.Module):
    """
    Quadruplet loss function.
    Builds on the Triplet Loss and takes 4 data input: one anchor, one positive and two negative examples.
    The negative examples needs not to be matching the anchor, the positive and each other.
    """
    def __init__(self, margin1=2.0, margin2=1.0,gamma=0.8,ramda=0.8,t1=0.1,t2=0.01):
        super(newQuadrupletLoss, self).__init__()
        self.margin1 = margin1
        self.margin2 = margin2
        self.gamma=gamma
        self.ramda=ramda
        self.t1=t1
        self.t2=t2

    def loss(self,anchor, positive, negative1, negative2):

        squarred_distance_pos = (anchor - positive).pow(2).sum(1)
        squarred_distance_neg = (anchor - negative1).pow(2).sum(1)
        squarred_distance_neg_b = (negative1 - negative2).pow(2).sum(1)

        pos_d=(((anchor-positive).pow(2))/4).mean(1)
        neg1_d=(((anchor-negative1).pow(2))/4).mean(1)
        neg2_d=(((negative1-negative2).pow(2))/4).mean(1)

        pos_var=(((anchor-positive).pow(2))/4).var(1)
        neg1_var=(((anchor-positive).pow(2))/4).var(1)
        neg2_var=(((anchor-positive).pow(2))/4).var(1)


        quadruplet_loss = \
            self.gamma*(F.relu(self.margin1 + squarred_distance_pos - squarred_distance_neg)
            + F.relu(self.margin2 + squarred_distance_pos - squarred_distance_neg_b))\
            + self.ramda*(F.relu(pos_d-neg1_d+self.t1)+F.relu(pos_d-neg2_d+self.t2))\
            + pos_var+neg1_var+neg2_var

        return quadruplet_loss.mean()

