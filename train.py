import torchvision as tv
import phototour
import torch
from tqdm import tqdm
import numpy as np
import torch.nn as nn
import math

lib_train = phototour.PhotoTour('.','liberty', download=True, train=True, mode = 'Quadruwithglobal', augment = True, nsamples=409600)
yos_train = phototour.PhotoTour('.','yosemite', download=True, train=True, mode = 'Quadruwithglobal', augment = True)
nd_train = phototour.PhotoTour('.','notredame', download=True, train=True, mode = 'Quadruwithglobal', augment = True)

eval_db = phototour.PhotoTour('.','yosemite', download=True, train=False)
# train_db = torch.utils.data.ConcatDataset((lib_train, yos_train))
train_db = nd_train
train_name = 'notredame'

import tfeat_model
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

tfeat = tfeat_model.TNet()
tfeat = tfeat.cuda()

# this kind of works
optimizer = optim.Adam(tfeat.parameters(),betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False) # change this to Adam
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)

seed=42
torch.manual_seed(seed)
np.random.seed(seed)
# cv2.setRNGSeed(seed)

train_loader = torch.utils.data.DataLoader(train_db,
                                             batch_size=300, shuffle=False,
                                             num_workers=30)

eval_loader = torch.utils.data.DataLoader(eval_db,
                                             batch_size=1024, shuffle=False,
                                             num_workers=32)

class newQuadrupletLoss(torch.nn.Module):
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

    def forward(self, anchor, positive, negative1, negative2):

        squarred_distance_pos = (anchor - positive).pow(2).sum(1)
        squarred_distance_neg = (anchor - negative1).pow(2).sum(1)
        squarred_distance_neg_b = (negative1 - negative2).pow(2).sum(1)

        pos_d=squarred_distance_pos/4
        neg1_d=squarred_distance_neg/4
        neg2_d=squarred_distance_neg_b/4

        quadruplet_loss = \
            self.gamma*(F.relu(self.margin1 + squarred_distance_pos - squarred_distance_neg)
            + F.relu(self.margin2 + squarred_distance_pos - squarred_distance_neg_b))\
            + self.ramda*((F.relu(pos_d)-neg1_d+self.t1)+(F.relu(pos_d)-neg2_d+t2))\
            + F.var(pos_d)+F.var(neg1_d)+F.var(neg2_d)

        return quadruplet_loss.mean()

fpr_per_epoch = []

for e in range(300):
    tfeat.train()
    for batch_idx, (data_a, data_p, data_n1, data_n2) in tqdm(enumerate(train_loader)):
        data_a = data_a.unsqueeze(1).float().cuda()
        data_p = data_p.unsqueeze(1).float().cuda()
        data_n1 = data_n1.unsqueeze(1).float().cuda()
        data_n2 = data_n2.unsqueeze(1).float().cuda()
        out_a, out_p, out_n1,out_n2 = tfeat(data_a), tfeat(data_p), tfeat(data_n1), tfeat(data_n2)
        loss = newQuadrupletLoss(out_a, out_p, out_n1,out_n2,margin1=2.0, margin2=1.0,gamma=0.8,ramda=0.8,t1=0.1,t2=0.01)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    tfeat.eval()
    l = np.empty((0,))
    d = np.empty((0,))
    # evaluate the network after each epoch
    for batch_idx, (data_l, data_r, lbls) in enumerate(eval_loader):
        data_l = data_l.unsqueeze(1).float().cuda()
        data_r = data_r.unsqueeze(1).float().cuda()
        out_l, out_r = tfeat(data_l), tfeat(data_r)
        dists = torch.norm(out_l - out_r, 2, 1).detach().cpu().numpy()
        l = np.hstack((l, lbls.numpy()))
        d = np.hstack((d, dists))

    # FPR95 code from Yurun Tian
    d = torch.from_numpy(d)
    l = torch.from_numpy(l)
    dist_pos = d[l == 1]
    dist_neg = d[l != 1]
    dist_pos, indice = torch.sort(dist_pos)
    loc_thr = int(np.ceil(dist_pos.numel() * 0.95))
    thr = dist_pos[loc_thr]
    fpr95 = float(dist_neg.le(thr).sum()) / dist_neg.numel()
    print(e, fpr95)
    fpr_per_epoch.append([e, fpr95])
    scheduler.step()
    np.savetxt('fpr.txt', np.array(fpr_per_epoch), delimiter=',')
    torch.save(tfeat.state_dict(), train_name + '-tfeat.params')