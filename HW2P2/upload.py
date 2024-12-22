import torch
from torchsummary import summary
import torchvision
from torchvision import transforms
import torch.nn.functional as F
import os
import gc
from tqdm import tqdm
from PIL import Image
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn import metrics as mt
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import glob
import wandb
import matplotlib.pyplot as plt
import csv
from dataset import TestImagePairDataset, ImagePairDataset

def test_epoch_ver(model, pair_data_loader, config):

    model.eval()
    scores = []
    batch_bar = tqdm(total=len(pair_data_loader), dynamic_ncols=True, position=0, leave=False, desc='Val Veri.')
    for i, (images1, images2) in enumerate(pair_data_loader):

        images = torch.cat([images1, images2], dim=0).to(DEVICE)
        # Get model outputs
        with torch.inference_mode():
            outputs = model(images)

        feats = F.normalize(outputs['feats'], dim=1)
        feats1, feats2 = feats.chunk(2)
        similarity = F.cosine_similarity(feats1, feats2)
        scores.extend(similarity.cpu().numpy().tolist())
        batch_bar.update()

    return scores

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
config = {
    'batch_size'    : 512, # Increase this if your GPU can handle it
    'init_lr'       : 0.1, # 1e-2 for pretrain
    'epochs'        : 200, # 20 epochs is recommended ONLY for the early submission - you will have to train for much longer typically.
    'data_dir'      : "/ihome/hkarim/yip33/HW2P2/data/11-785-f24-hw2p2-verification/cls_data", #TODO
    'data_ver_dir'  : "/ihome/hkarim/yip33/HW2P2/data/11-785-f24-hw2p2-verification/ver_data", #TODO
    'checkpoint_dir': f"./checkpoints/",
    'optimizer'     : 'SGD',
    #'scheduler'     : "CosineAnnealingWarmRestarts",
    'scheduler'     : "ReducedLRonPlateau",
    'loss'          : 'CrossEntropy',
    'weight_decay'  : 1e-4,
    'batch_norm'    : True,
    'activation'    : 'ReLU',
    'T_0'           : 10,
    'T_mult'        : 2,
    'momentum'      : 0.9,
    # Include other parameters as needed.
}
from network import ResNet34
model = ResNet34(num_classes=8631).to(DEVICE) # convext small



def load_model(model, optimizer=None, scheduler=None, path='./checkpoint.pth'):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    else:
        optimizer = None
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    else:
        scheduler = None
    epoch = checkpoint['epoch']
    metrics = checkpoint['metric']
    return model, optimizer, scheduler, epoch, metrics


model, _, _, _, _ = load_model(model,
                                    None,
                                    None,
                                    path='./checkpoints/yiyan_run20241018_202943/best_ret.pth')
data_dir = config['data_ver_dir']
# TODO: Add your validation pair txt file
test_transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize(224),
    torchvision.transforms.ToTensor(),
    # use other distribution
    torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                    std=[0.5, 0.5, 0.5])])
test_pair_dataset = TestImagePairDataset(data_dir, csv_file='/ihome/hkarim/yip33/HW2P2/data/11-785-f24-hw2p2-verification/test_pairs.txt', transform=test_transforms)
test_pair_dataloader = torch.utils.data.DataLoader(test_pair_dataset,
                                              batch_size=config["batch_size"],
                                              shuffle=False,
                                              pin_memory=True,
                                              num_workers=4)




scores = test_epoch_ver(model, test_pair_dataloader, config)



with open(f"verification_resnet34.csv", "w+") as f:
    f.write("ID,Label\n")
    for i in range(len(scores)):
        f.write("{},{}\n".format(i, scores[i]))

