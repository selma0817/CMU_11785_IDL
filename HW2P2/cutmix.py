import torch
import numpy as np

def cutmix_data(images, labels, alpha=1.0):
    ''' Returns mixed inputs, pairs of targets, and lambda '''
    lam = np.random.beta(alpha, alpha)
    rand_index = torch.randperm(images.size()[0]).to(images.device)
    target_a = labels
    target_b = labels[rand_index]

    bbx1, bby1, bbx2, bby2 = rand_bbox(images.size(), lam)
    images[:, :, bbx1:bbx2, bby1:bby2] = images[rand_index, :, bbx1:bbx2, bby1:bby2]

    return images, target_a, target_b, lam

def rand_bbox(size, lam):
    ''' Generates random bounding box '''
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # Uniformly sample the bounding box center
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2
