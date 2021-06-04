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
#            nn.Conv2d(1, 32, kernel_size=3,padding=1),
#            nn.BatchNorm2d(32),
#            nn.ReLU(),
#
#            nn.Conv2d(32,32,kernel_size=3,padding=1),
#            nn.BatchNorm2d(32),
#            nn.ReLU(),
#
#            nn.Conv2d(32,64,kernel_size=3,stride=2,padding=1),
#            nn.BatchNorm2d(64),
#            nn.ReLU(),
#
#            nn.Conv2d(64, 64, kernel_size=3,padding=1),
#            nn.BatchNorm2d(64),
#            nn.ReLU(),
#
#            nn.Conv2d(64,128,kernel_size=3, stride=2,padding=1),
#            nn.BatchNorm2d(128),
#            nn.ReLU(),
#
#            nn.Conv2d(128,128,kernel_size=3,padding=1),
#            nn.BatchNorm2d(128),
#            nn.ReLU(),
#            nn.Dropout2d(p=0.3),
#
#            nn.Conv2d(128,128,kernel_size=8),
#            nn.BatchNorm2d(128)
#
 
            nn.Conv2d(1,32,kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.Conv2d(32,64,kernel_size=3),
            nn.ReLU(),
            #nn.BatchNorm2d(64),
            #nn.MaxPool2d(kernel_size=2,stride=2),

            nn.Conv2d(64,128,kernel_size=3),
            nn.ReLU(),
            #nn.MaxPool2d(kernel_size=2,stride=2),
            nn.Conv2d(128,256,kernel_size=5),
            nn.ReLU()
            #nn.MaxPool2d(kernel_size=2,stride=2)

)
        self.descr = nn.Sequential(
            nn.Linear(256*7*7,128),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.descr(x)
        return x
    
class Quadruplet(nn.Module):
    def __init__(self,margin1=2.0,margin2=1.0) :
        super(Quadruplet,self).__init__()
        self.margin1=margin1
        self.margin2=margin2

    def loss(self,anchor,positive,negative1,negative2) :
        distance_pos=(anchor-positive).pow(2)
        distance_neg=(anchor-negative1).pow(2)
        distance_neg_b=(negative1-negative2).pow(2)
        #print(distance_pos.size())
        #self.margin1=
        #self.margin2=

        squarred_distance_pos = distance_pos.sum(1)
        squarred_distance_neg = distance_neg.sum(1)
        squarred_distance_neg_b = distance_neg_b.sum(1)
        #print(squarred_distance_pos.size())
        
        #self.margin1=min(-squarred_distance_pos+squarred_distance_neg)
        #self.margin2=min(-squarred_distance_pos+squarred_distance_neg_b)
        quadruplet_loss = \
            F.relu(1-squarred_distance_neg/(squarred_distance_pos+self.margin1))\
            + F.relu(1- squarred_distance_neg_b/(squarred_distance_pos+self.margin2))

        return quadruplet_loss.mean()

class newQuadruplet(nn.Module):
    """
    Quadruplet loss function.
    Builds on the Triplet Loss and takes 4 data input: one anchor, one positive and two negative examples.
    The negative examples needs not to be matching the anchor, the positive and each other.
    """
    def __init__(self, margin1=2.0, margin2=1.0,gamma=0.8,ramda=0.8,t1=0.1):
        super(newQuadruplet, self).__init__()
        self.margin1 = margin1
        self.margin2 = margin2
        self.gamma=gamma
        self.ramda=ramda
        self.t1=t1
        #self.t2=t2

    def loss(self,anchor, positive, negative1, negative2,swap=True):

#        distance_pos=F.pairwise_distance(anchor,positive).pow(2)
#        distance_neg=F.pairwise_distance(anchor,negative1).pow(2)
#        distance_neg_b=F.pairwise_distance(negative1,negative2).pow(2)


        distance_pos=(anchor-positive).pow(2).sum(1)
        distance_neg=(anchor-negative1).pow(2).sum(1)
        distance_neg_b=(negative1-negative2).pow(2).sum(1)
        
        pos_neg2=(anchor-negative2).pow(2).sum(1)

##        for i,a in enumerate(distance_neg):
##            if a>pos_neg2[i] :
##                temporal=a
##                a=pos_neg2[i]
##                pos_neg2[i]=temporal
##
        #import copy
        if swap==True :

            a=distance_neg-pos_neg2>0
            copy=negative1.clone().detach()
            negative1[a==1]=negative2[a==1]
            negative2[a==1]=copy[a==1]

            distance_neg=(anchor-negative2).pow(2).sum(1)
            pos_neg2=(anchor-negative1).pow(2).sum(1)

#        if distance_neg.mean()>pos_neg2.mean() :
#        
#            distance_neg=(anchor-negative2).pow(2).sum(1)
#            pos_neg2=(anchor-negative1).pow(2).sum(1)
#
#
        pos_d=(distance_pos/4).mean()
        neg1_d=(distance_neg/4).mean()
        neg2_d=(pos_neg2/4).mean()
        
        pos_var=(distance_pos/4).var()
        neg1_var=(distance_neg/4).var()
        neg2_var=(pos_neg2/4).var()
        
        self.margin1=distance_neg.mean()-distance_pos.mean()
        self.margin2=0.5*self.margin1
        #self.t1=0.25*(distance_neg-distance_pos).min() # maybe I can try moving average here

        quadruplet_loss = F.relu(1-distance_neg/(distance_pos+self.margin1))\
            + F.relu(1- distance_neg_b/(distance_pos+self.margin2))
        
        global_loss = self.ramda*(F.relu(2*pos_d-neg1_d-neg2_d+self.t1))\
            + pos_var+neg1_var+neg2_var

        loss=self.gamma*quadruplet_loss.sum()+global_loss
        
        return loss
