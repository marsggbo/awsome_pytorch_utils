'''
refer to: https://github.com/hongyi-zhang/mixup/blob/master/cifar/utils.py

suit for both 2d and 3d data
'''

import numpy as np
import torch

def mixup_data(x, y, alpha=1.0):

    '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''
    if alpha > 0.:
        lam = [np.random.beta(alpha, alpha) for _ in range(len(x))]
        if len(x.shape)==5:
            lam = torch.tensor(lam).to(x.device).view(-1, 1, 1, 1, 1)
        elif len(x.shape)==4:
            lam = torch.tensor(lam).to(x.device).view(-1, 1, 1, 1)
    else:
        lam = torch.ones(x.shape[0], 1, 1, 1, 1).to(x.device)
    batch_size = x.shape[0]
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index,:]
    y_a, y_b = y, y[index]
    mixed_y = torch.cat((y_a.view(-1, 1), y_b.view(-1, 1), lam.view(-1, 1)), dim=1)
    return mixed_x, mixed_y
  
  
  if __name__ == '__main__':
    # 3d data
    x = torch.rand(2,3,8,8,8)
    y = torch.randint(0,10,(2,))
    mx, my = mixup_data(x, y)
  
