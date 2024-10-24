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
#from pytorch_metric_learning import samplers
import csv
# from torchvision.transforms import v2
# from pytorch_metric_learning import losses
# !mkdir '/content/data'

# !kaggle competitions download -c 11785-hw-2-p-2-face-verification-fall-2024
# !unzip -qo '11785-hw-2-p-2-face-verification-fall-2024.zip' -d '/content/data'


from metric import AverageMeter
from network import ConvNeXt
from dataset import TestImagePairDataset, ImagePairDataset
from arcface import ArcFaceLoss

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Device: ", DEVICE)

# print current partition
from datetime import datetime

#run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
run_name = f"yiyan_run{datetime.now().strftime('%Y%m%d_%H%M%S')}"



config = {
    'batch_size'    : 1024, # Increase this if your GPU can handle it
    'init_lr'       : 0.01, # 1e-2 for pretrain
    'epochs'        : 20, # 20 epochs is recommended ONLY for the early submission - you will have to train for much longer typically.
    'data_dir'      : "/ihome/hkarim/yip33/HW2P2/data/11-785-f24-hw2p2-verification/cls_data", #TODO
    'data_ver_dir'  : "/ihome/hkarim/yip33/HW2P2/data/11-785-f24-hw2p2-verification/ver_data", #TODO
    'checkpoint_dir': f"./checkpoints/{run_name}",
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
# Create the checkpoint directory if it doesn't exist
os.makedirs(config['checkpoint_dir'], exist_ok=True)


data_dir = config['data_dir']
# train_dir = os.path.join(data_dir)

train_dir = os.path.join(data_dir, 'train')
val_dir = os.path.join(data_dir, 'dev')

# train transforms
# for 112
# Mean: tensor([0.5309, 0.4139, 0.3587])
# Standard deviation: tensor([0.2463, 0.2132, 0.2008])

# for 224
# train Mean: tensor([0.5318, 0.4148, 0.3596])
# train Standard deviation: tensor([0.2441, 0.2109, 0.1984])



# Define CutMix
#from torchvision.transforms.v2 import CutMix
# cutmix_prob = 1.0  # Probability of applying CutMix (adjust this value as needed)
# cutmix_transform = v2.CutMix(num_classes=8631, alpha=1.0, p=1.0)



train_transforms = torchvision.transforms.Compose([

    torchvision.transforms.Resize(112), # Why are we resizing the Image?
    torchvision.transforms.TrivialAugmentWide(num_magnitude_bins=15),
    torchvision.transforms.ToTensor(),

    # should use other distribution value
    torchvision.transforms.Normalize(mean=[0.5309, 0.4139, 0.3587],
        std=[0.2463, 0.2132, 0.2008])
    ])

# val transforms
val_transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize(112),
    torchvision.transforms.ToTensor(),
    # use other distribution
    torchvision.transforms.Normalize(mean=[0.5315, 0.4144, 0.3591],
        std=[0.2463, 0.2134, 0.2009])])

test_transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize(224),
    torchvision.transforms.ToTensor(),
    # use other distribution
    torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                    std=[0.5, 0.5, 0.5])])


# get datasets
train_dataset = torchvision.datasets.ImageFolder(train_dir, transform=train_transforms)
val_dataset = torchvision.datasets.ImageFolder(val_dir, transform=val_transforms)

train_loader = torch.utils.data.DataLoader(train_dataset,
                                            batch_size=config["batch_size"],
                                            shuffle=True,
                                            pin_memory=True,
                                            num_workers=8,
                                            sampler=None
                                            )

val_loader = torch.utils.data.DataLoader(val_dataset,
                                          batch_size=config["batch_size"],
                                          shuffle=False,
                                          num_workers=4)



data_dir = config['data_ver_dir']


# get datasets

# TODO: Add your validation pair txt file
# test csv_file path
pair_dataset = ImagePairDataset(data_dir, csv_file='/ihome/hkarim/yip33/HW2P2/data/11-785-f24-hw2p2-verification/val_pairs.txt', transform=test_transforms)
pair_dataloader = torch.utils.data.DataLoader(pair_dataset,
                                              batch_size=config["batch_size"],
                                              shuffle=False,
                                              pin_memory=True,
                                              num_workers=4)

# TODO: Add your validation pair txt file
test_pair_dataset = TestImagePairDataset(data_dir, csv_file='/ihome/hkarim/yip33/HW2P2/data/11-785-f24-hw2p2-verification/test_pairs.txt', transform=test_transforms)
test_pair_dataloader = torch.utils.data.DataLoader(test_pair_dataset,
                                              batch_size=config["batch_size"],
                                              shuffle=False,
                                              pin_memory=True,
                                              num_workers=4)





# Double-check your dataset/dataloaders work as expected

print("Number of classes    : ", len(train_dataset.classes))
print("No. of train images  : ", train_dataset.__len__())
print("Shape of image       : ", train_dataset[0][0].shape)
print("Batch size           : ", config['batch_size'])
print("Train batches        : ", train_loader.__len__())
print("Val batches          : ", val_loader.__len__())

# Feel free to print more things if needed

# Visualize a few images in the dataset

"""
You can write your own code, and you don't need to understand the code
It is highly recommended that you visualize your data augmentation as sanity check
"""


#from network import ConvNeXtPreTrained
from network import ResNet34, ResNet50
#model = ConvNeXtPreTrained(num_classes=8631).to(DEVICE)
# model = ConvNeXt(num_classes=8631, drop_path_rate=0.1).to(DEVICE)
#model = ConvNeXt(num_classes = 8631).to(DEVICE)
#model = ResNet50(num_classes = 8631).to(DEVICE)
#summary(model, (3, 112, 112))
#
model = ResNet34(num_classes=8631).to(DEVICE) # convext small

# --------------------------------------------------- #
# embedding_size = 512
num_classes = 8631
# Defining Loss function
criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.15) #: # label_smoothing=0.2 What loss do you need for a multi class classification problem and would label smoothing be beneficial here?
# loss_func =
# criterion = losses.ArcFaceLoss(num_classes, embedding_size, margin=28.6, scale=64, **kwargs).to(DEVICE)
# criterion_optimizer = torch.optim.SGD(criterion.parameters(), lr=0.1)
# during trianing
# criterion_tune_optimizer.step()
#from arcface import ArcFace
#criterion = ArcFaceLoss(embedding_size, num_classes, s=64.0, m=0.5).to(DEVICE)
#arcface = ArcFace()
#criterion_optimizer = torch.optim.SGD(criterion.parameters(), lr=0.01)
#criterion = ArcFaceLoss(embedding_size, num_classes, s=64.0, m=0.5).to(DEVICE)
# --------------------------------------------------- #

# Defining Optimizer
#optimizer =  torch.optim.Adam(model.parameters(), lr=config['init_lr'], weight_decay=config['weight_decay'])# TODO: Feel free to pick a optimizer
optimizer = torch.optim.SGD(
    # add momentuum
    model.parameters(),
    momentum=config['momentum'],
    lr=config['init_lr'],
    weight_decay=config['weight_decay']
)
# --------------------------------------------------- #

# Defining Scheduler
#scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['T_max']) # TODO: Use a good scheduler such as ReduceLRonPlateau, StepLR, MultistepLR, CosineAnnealing, etc.
#scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=0.001, last_epoch=-1)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                        factor=0.9,
                                                        patience=3,
                                                        threshold=0.001)

# scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
#     optimizer, 
#     T_0=config['T_0'], 
#     T_mult=config['T_mult']
# )


# --------------------------------------------------- #

# Initialising mixed-precision training. # Good news. We've already implemented FP16 (Mixed precision training) for you
# It is useful only in the case of compatible GPUs such as T4/V100
scaler = torch.cuda.amp.GradScaler()



def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = min(max(topk), output.size()[1])
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    return [correct[:min(k, maxk)].reshape(-1).float().sum(0) * 100. / batch_size for k in topk]



def get_ver_metrics(labels, scores, FPRs):
    # eer and auc
    fpr, tpr, _ = mt.roc_curve(labels, scores, pos_label=1)
    roc_curve = interp1d(fpr, tpr)
    EER = 100. * brentq(lambda x : 1. - x - roc_curve(x), 0., 1.)
    AUC = 100. * mt.auc(fpr, tpr)

    # get acc
    tnr = 1. - fpr
    pos_num = labels.count(1)
    neg_num = labels.count(0)
    ACC = 100. * max(tpr * pos_num + tnr * neg_num) / len(labels)

    # TPR @ FPR
    if isinstance(FPRs, list):
        TPRs = [
            ('TPR@FPR={}'.format(FPR), 100. * roc_curve(float(FPR)))
            for FPR in FPRs
        ]
    else:
        TPRs = []

    return {
        'ACC': ACC,
        'EER': EER,
        'AUC': AUC,
        'TPRs': TPRs,
    }

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
    '''Generates random bounding box'''
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)  # Replaced np.int with int
    cut_h = int(H * cut_rat)  # Replaced np.int with int

    # Uniformly sample the bounding box center
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def mixup_data(x, y, alpha=1.0):
    '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


# class LogitLayer(torch.nn.Module):
#     def __init__(self, embedding_size, num_classes):
#         super(LogitLayer, self).__init__()
#         # A fully connected layer that maps embeddings to logits
#         self.fc = torch.nn.Linear(embedding_size, num_classes)

#     def forward(self, features):
#         # Features from the model's output (embedding)
#         features = F.normalize(features, p=2, dim=1)
#         logits = self.fc(features)  # This outputs logits of shape [batch_size, num_classes]
#         return logits

# logit_layer = LogitLayer(embedding_size=512, num_classes=8631).to(DEVICE)



def train_epoch(model, dataloader, optimizer, lr_scheduler, scaler, device, config):

    model.train()

    # metric meters
    loss_m = AverageMeter()
    acc_m = AverageMeter()

    # Progress Bar
    batch_bar = tqdm(total=len(dataloader), dynamic_ncols=True, leave=False, position=0, desc='Train', ncols=5)

    for i, (images, labels) in enumerate(dataloader):

        optimizer.zero_grad() # Zero gradients

        # send to cuda
        images = images.to(device, non_blocking=True)
        if isinstance(labels, (tuple, list)):
            targets1, targets2, lam = labels
            labels = (targets1.to(device), targets2.to(device), lam)
        else:
            labels = labels.to(device, non_blocking=True)

        # Apply CutMix with a probability of 50%
        if np.random.rand(1) < 0.5:
            images, targets_a, targets_b, lam = cutmix_data(images, labels, alpha=1.0)
        else:
            images, targets_a, targets_b, lam = mixup_data(images, labels, alpha=1.0)

            #images, targets_a, targets_b, lam = cutmix_data(images, labels, alpha=1.0)
        targets_a, targets_b = targets_a.to(device), targets_b.to(device)

            # Forward pass with mixed images
        with torch.cuda.amp.autocast():  # Mixed precision
            outputs = model(images)
            out = outputs['out']

                # Compute the loss as a combination of the two labels
            loss = lam * criterion(out, targets_a) + (1 - lam) * criterion(out, targets_b)


            #     outputs = model(images)

            #     # Use the type of output depending on the loss function you want to use
            
            #     #loss = criterion(outputs['out'], labels)
        scaler.scale(loss).backward() # This is a replacement for loss.backward()
        scaler.step(optimizer) # This is a replacement for optimizer.step()
        scaler.update()
        #loss.backward()
        #optimizer.step()
        #criterion_optimizer.step()

        # metrics
        loss_m.update(loss.item())
        if 'feats' in outputs:
            acc = accuracy(outputs['out'], labels)[0].item()
        else:
            acc = 0.0
        acc_m.update(acc)

        # tqdm lets you add some details so you can monitor training as you train.
        batch_bar.set_postfix(
            # acc         = "{:.04f}%".format(100*accuracy),
            acc="{:.04f}% ({:.04f})".format(acc, acc_m.avg),
            loss        = "{:.04f} ({:.04f})".format(loss.item(), loss_m.avg),
            lr          = "{:.04f}".format(float(optimizer.param_groups[0]['lr'])))

        batch_bar.update() # Update tqdm bar

    # You may want to call some schedulers inside the train function. What are these?
    # if lr_scheduler is not None:
    #     lr_scheduler.step(loss) # for reduceLRonPlaetau

    batch_bar.close()

    return acc_m.avg, loss_m.avg


@torch.no_grad()
def valid_epoch_cls(model, dataloader, device, config):

    model.eval()
    #arcface.eval()
    batch_bar = tqdm(total=len(dataloader), dynamic_ncols=True, position=0, leave=False, desc='Val Cls.', ncols=5)

    # metric meters
    loss_m = AverageMeter()
    acc_m = AverageMeter()

    for i, (images, labels) in enumerate(dataloader):

        # Move images to device
        images, labels = images.to(device), labels.to(device)

        # Get model outputs
        with torch.inference_mode():
            outputs = model(images)

            #features = outputs['feats']
            out = outputs['out']
            #logits = logit_layer(features)
            #logits = arcface(logits, labels)
            loss = criterion(out, labels)
            #loss = criterion(logits, labels)

        # metrics
        acc = accuracy(outputs['out'], labels)[0].item()
        loss_m.update(loss.item())
        acc_m.update(acc)

        batch_bar.set_postfix(
            acc         = "{:.04f}% ({:.04f})".format(acc, acc_m.avg),
            loss        = "{:.04f} ({:.04f})".format(loss.item(), loss_m.avg))

        batch_bar.update()

    batch_bar.close()
    return acc_m.avg, loss_m.avg

gc.collect() # These commands help you when you face CUDA OOM error
torch.cuda.empty_cache()



def valid_epoch_ver(model, pair_data_loader, device, config):

    model.eval()
    scores = []
    match_labels = []
    batch_bar = tqdm(total=len(pair_data_loader), dynamic_ncols=True, position=0, leave=False, desc='Val Veri.')
    for i, (images1, images2, labels) in enumerate(pair_data_loader):

        # match_labels = match_labels.to(device)
        images = torch.cat([images1, images2], dim=0).to(device)
        # Get model outputs
        with torch.inference_mode():
            outputs = model(images)

        feats = F.normalize(outputs['feats'], dim=1)
        feats1, feats2 = feats.chunk(2)
        similarity = F.cosine_similarity(feats1, feats2)
        scores.append(similarity.cpu().numpy())
        match_labels.append(labels.cpu().numpy())
        batch_bar.update()

    scores = np.concatenate(scores)
    match_labels = np.concatenate(match_labels)

    FPRs=['1e-4', '5e-4', '1e-3', '5e-3', '5e-2']
    metric_dict = get_ver_metrics(match_labels.tolist(), scores.tolist(), FPRs)
    print(metric_dict)
    # return metric_dict['ACC']

    return metric_dict



# weight and bias login

wandb.login(key="57c916d673703185e1b47000c74bd854db77bcf8") # API Key is in your wandb account, under settings (wandb.ai/settings)


# Create your wandb run
run = wandb.init(
    name = "ResNet34_test_modification", ## Wandb creates random run names if you skip this field
    reinit = True, ### Allows reinitalizing runs when you re-run this cell
    #id = "ahnsbg3k", # Insert specific run id here if you want to resume a previous run
    #resume = "must", ### You need this to resume previous runs, but comment out reinit = True when using this
    project = "hw2p2", ### Project should be created in your wandb account
    config = config ### Wandb Config for your run
)
# it seems that i am not starting my learning rate from 0.01. i should test again, only load model this time dont
# load anything else 

# Uncomment the line for saving the scheduler save dict if you are using a scheduler
def save_model(model, optimizer, scheduler, metrics, epoch, path):
    torch.save(
        {'model_state_dict'         : model.state_dict(),
         #'criterion_optimizer_dict' : criterion_optimizer.state_dict(),
         'optimizer_state_dict'     : optimizer.state_dict(),
        'scheduler_state_dict'      : scheduler.state_dict(),
         'metric'                   : metrics,
         'epoch'                    : epoch},
         path)


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
                                    path='./checkpoints/yiyan_run20241017_224254/best_ret.pth')


# criterion = ArcFaceLoss()

# res 34

scaler = torch.cuda.amp.GradScaler()  # Re-initialize the scaler
e = 0
#e = epoch

best_valid_cls_acc = 0.0
eval_cls = True
best_valid_ret_acc = 0.0
for epoch in range(e, config['epochs']):
        # epoch
        print("\nEpoch {}/{}".format(epoch+1, config['epochs']))

        # train
        train_cls_acc, train_loss = train_epoch(model, train_loader, optimizer, scheduler, scaler, DEVICE, config)

        curr_lr = float(optimizer.param_groups[0]['lr'])
        print("\nEpoch {}/{}: \nTrain Cls. Acc {:.04f}%\t Train Cls. Loss {:.04f}\t Learning Rate {:.04f}".format(epoch + 1, config['epochs'], train_cls_acc, train_loss, curr_lr))
        metrics = {
            'train_cls_acc': train_cls_acc,
            'train_loss': train_loss,
            'learning_rate': curr_lr,
        }
        # classification validation
        if eval_cls:
            valid_cls_acc, valid_loss = valid_epoch_cls(model, val_loader, DEVICE, config)
            print("Val Cls. Acc {:.04f}%\t Val Cls. Loss {:.04f}".format(valid_cls_acc, valid_loss))
            metrics.update({
                'valid_cls_acc': valid_cls_acc,
                'valid_loss': valid_loss,
            })
        

        scheduler.step(valid_loss) # for reduceLRonPlaetau
        #scheduler.step()
        # retrieval validation
        #valid_ret_acc = valid_epoch_ver(model, pair_dataloader, DEVICE, config)
        metric_dict = valid_epoch_ver(model, pair_dataloader, DEVICE, config)
        valid_ret_acc = metric_dict['ACC']
        valid_eer = metric_dict['EER']

        print(f"Val Ret. Acc {valid_ret_acc:.04f}%\tEER: {valid_eer:.04f}%")
        metrics.update({
            'valid_ret_acc': valid_ret_acc,
            # add eer to wandb display here
            'valid_eer': valid_eer 
            # add change of my learning rate here
          

        })

        # save model
        save_model(model, optimizer, scheduler, metrics, epoch, os.path.join(config['checkpoint_dir'], 'last.pth'))
        print("Saved epoch model")

        # save best model
        if eval_cls:
            if valid_cls_acc >= best_valid_cls_acc:
                best_valid_cls_acc = valid_cls_acc
                save_model(model, optimizer, scheduler, metrics, epoch, os.path.join(config['checkpoint_dir'], 'best_cls.pth'))
                wandb.save(os.path.join(config['checkpoint_dir'], 'best_cls.pth'))
                print("Saved best classification model")

        if valid_ret_acc >= best_valid_ret_acc:
            best_valid_ret_acc = valid_ret_acc
            save_model(model, optimizer, scheduler, metrics, epoch, os.path.join(config['checkpoint_dir'], 'best_ret.pth'))
            wandb.save(os.path.join(config['checkpoint_dir'], 'best_ret.pth'))
            print("Saved best retrieval model")

        # log to tracker
        if run is not None:
            run.log(metrics)


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

scores = test_epoch_ver(model, test_pair_dataloader, config)



with open(f"verification_convnext_{run_name}.csv", "w+") as f:
    f.write("ID,Label\n")
    for i in range(len(scores)):
        f.write("{},{}\n".format(i, scores[i]))



## train 150 epoch without arcface using cutmix and label smoothing using my convext 