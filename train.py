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
optimizer = optim.Adam(tfeat.parameters(),betas=(0.9, 0.999), lr=1e-3, betas=(0., 0.999))
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
fpr_per_epoch = []

for e in range(300):
    tfeat.train()
    for batch_idx, (data_a, data_p, data_n1, data_n2) in tqdm(enumerate(train_loader)):
        data_a = data_a.unsqueeze(1).float().cuda()
        data_p = data_p.unsqueeze(1).float().cuda()
        data_n1 = data_n1.unsqueeze(1).float().cuda()
        data_n2 = data_n2.unsqueeze(1).float().cuda()
        out_a, out_p, out_n1,out_n2 = tfeat(data_a), tfeat(data_p), tfeat(data_n1), tfeat(data_n2)
        #pos_d,neg1_d,neg2_d=network(out_a,out_p,out_n1,out_n2)
        hyperparameter = tfeat_model.newQuadrupletLoss(margin1=2.0, margin2=1.0,gamma=0.8,ramda=0.8,t1=0.1,t2=0.01)
        loss=hyperparameter.loss(out_a,out_p,out_n1,out_n2)
        #loss=hyperparameter.loss(pos_d,neg1_d,neg2_d)

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
