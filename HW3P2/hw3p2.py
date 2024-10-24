import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchsummaryX import summary
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import torchaudio.transforms as tat

from sklearn.metrics import accuracy_score
import gc

import zipfile
import pandas as pd
from tqdm import tqdm
import os
import datetime

# imports for decoding and distance calculation
import ctcdecode
import Levenshtein
from ctcdecode import CTCBeamDecoder

import warnings
warnings.filterwarnings('ignore')


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Device: ", device)


# get me RAMMM!!!!
import gc
gc.collect()

# # Create objects for the dataset class
# train_data = AudioDataset() #TODO
# val_data = ___ # TODO : You can either use the same class with some modifications or make a new one :)
# test_data = AudioDatasetTest() #TODO

# # Do NOT forget to pass in the collate function as parameter while creating the dataloader
# train_loader = #TODO
# val_loader = #TODO
# test_loader = #TODO

# print("Batch size: ", config['batch_size'])
# print("Train dataset samples = {}, batches = {}".format(train_data.__len__(), len(train_loader)))
# print("Val dataset samples = {}, batches = {}".format(val_data.__len__(), len(val_loader)))
# print("Test dataset samples = {}, batches = {}".format(test_data.__len__(), len(test_loader)))

# # sanity check
# for data in train_loader:
#     x, y, lx, ly = data
#     print(x.shape, y.shape, lx.shape, ly.shape)
#     break

# model = ASRModel(
#     input_size  = #TODO,
#     embed_size  = #TODO
#     output_size = len(PHONEMES)
# ).to(device)
# print(model)
# summary(model, x.to(device), lx)


# #TODO


# criterion = # Define CTC loss as the criterion. How would the losses be reduced?
# # CTC Loss: https://pytorch.org/docs/stable/generated/torch.nn.CTCLoss.html
# # Refer to the handout for hints

# optimizer =  torch.optim.AdamW(...) # What goes in here?

# # Declare the decoder. Use the CTC Beam Decoder to decode phonemes
# # CTC Beam Decoder Doc: https://github.com/parlance/ctcdecode
# decoder = #TODO

# scheduler = #TODO

# # Mixed Precision, if you need it
# scaler = torch.cuda.amp.GradScaler()


# def decode_prediction(output, output_lens, decoder, PHONEME_MAP= LABELS):

#     # TODO: look at docs for CTC.decoder and find out what is returned here. Check the shape of output and expected shape in decode.
#     (...) = decoder.decode(output, seq_lens= output_lens) #lengths - list of lengths

#     pred_strings                    = []

#     for i in range(output_lens.shape[0]):
#         #TODO: Create the prediction from the output of decoder.decode. Don't forget to map it using PHONEMES_MAP.

#     return pred_strings

# def calculate_levenshtein(output, label, output_lens, label_lens, decoder, PHONEME_MAP= LABELS): # y - sequence of integers

#     dist            = 0
#     batch_size      = label.shape[0]

#     pred_strings    = decode_prediction(output, output_lens, decoder, PHONEME_MAP)

#     for i in range(batch_size):
#         # TODO: Get predicted string and label string for each element in the batch
#         pred_string = #TODO
#         label_string = #TODO
#         dist += Levenshtein.distance(pred_string, label_string)

#     #dist /= batch_size # TODO: Uncomment this, but think about why we are doing this
#     raise NotImplemented
#     # return dist


#     # test code to check shapes

# model.eval()
# for i, data in enumerate(val_loader, 0):
#     x, y, lx, ly = data
#     x, y = x.to(device), y.to(device)
#     h, lh = model(x, lx)
#     print(h.shape)
#     h = torch.permute(h, (1, 0, 2))
#     print(h.shape, y.shape)
#     loss = criterion(h, y, lh, ly)
#     print(loss)

#     print(calculate_levenshtein(h, y, lx, ly, decoder, LABELS))

#     break


# import wandb
# wandb.login(key="<replace with your API key here>")


# run = wandb.init(
#     name = "early-submission", ## Wandb creates random run names if you skip this field
#     reinit = True, ### Allows reinitalizing runs when you re-run this cell
#     # run_id = ### Insert specific run id here if you want to resume a previous run
#     # resume = "must" ### You need this to resume previous runs, but comment out reinit = True when using this
#     project = "hw3p2-ablations", ### Project should be created in your wandb account
#     config = config ### Wandb Config for your run
# )


# from tqdm import tqdm

# def train_model(model, train_loader, criterion, optimizer):

#     model.train()
#     batch_bar = tqdm(total=len(train_loader), dynamic_ncols=True, leave=False, position=0, desc='Train')

#     total_loss = 0

#     for i, data in enumerate(train_loader):
#         optimizer.zero_grad()

#         x, y, lx, ly = data
#         x, y = x.to(device), y.to(device)

#         with torch.cuda.amp.autocast():
#             h, lh = model(x, lx)
#             h = torch.permute(h, (1, 0, 2))
#             loss = criterion(h, y, lh, ly)

#         total_loss += loss.item()

#         batch_bar.set_postfix(
#             loss="{:.04f}".format(float(total_loss / (i + 1))),
#             lr="{:.06f}".format(float(optimizer.param_groups[0]['lr'])))

#         batch_bar.update() # Update tqdm bar

#         # Another couple things you need for FP16.
#         scaler.scale(loss).backward() # This is a replacement for loss.backward()
#         scaler.step(optimizer) # This is a replacement for optimizer.step()
#         scaler.update() # This is something added just for FP16

#         del x, y, lx, ly, h, lh, loss
#         torch.cuda.empty_cache()

#     batch_bar.close() # You need this to close the tqdm bar

#     return total_loss / len(train_loader)


# def validate_model(model, val_loader, decoder, phoneme_map= LABELS):

#     model.eval()
#     batch_bar = tqdm(total=len(val_loader), dynamic_ncols=True, position=0, leave=False, desc='Val')

#     total_loss = 0
#     vdist = 0

#     for i, data in enumerate(val_loader):

#         x, y, lx, ly = data
#         x, y = x.to(device), y.to(device)

#         with torch.inference_mode():
#             h, lh = model(x, lx)
#             h = torch.permute(h, (1, 0, 2))
#             loss = criterion(h, y, lh, ly)

#         total_loss += float(loss)
#         vdist += calculate_levenshtein(torch.permute(h, (1, 0, 2)), y, lh, ly, decoder, phoneme_map)

#         batch_bar.set_postfix(loss="{:.04f}".format(float(total_loss / (i + 1))), dist="{:.04f}".format(float(vdist / (i + 1))))

#         batch_bar.update()

#         del x, y, lx, ly, h, lh, loss
#         torch.cuda.empty_cache()

#     batch_bar.close()
#     total_loss = total_loss/len(val_loader)
#     val_dist = vdist/len(val_loader)
#     return total_loss, val_dist


# def save_model(model, optimizer, scheduler, metric, epoch, path):
#     torch.save(
#         {'model_state_dict'         : model.state_dict(),
#          'optimizer_state_dict'     : optimizer.state_dict(),
#          'scheduler_state_dict'     : scheduler.state_dict(),
#          metric[0]                  : metric[1],
#          'epoch'                    : epoch},
#          path
#     )

# def load_model(path, model, metric= 'valid_acc', optimizer= None, scheduler= None):

#     checkpoint = torch.load(path)
#     model.load_state_dict(checkpoint['model_state_dict'])

#     if optimizer != None:
#         optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#     if scheduler != None:
#         scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

#     epoch   = checkpoint['epoch']
#     metric  = checkpoint[metric]

#     return [model, optimizer, scheduler, epoch, metric]


# # This is for checkpointing, if you're doing it over multiple sessions

# last_epoch_completed = 0
# start = last_epoch_completed
# end = config["epochs"]
# best_lev_dist = float("inf") # if you're restarting from some checkpoint, use what you saw there.
# epoch_model_path = #TODO set the model path( Optional, you can just store best one. Make sure to make the changes below )
# best_model_path = #TODO set best model path


# torch.cuda.empty_cache()
# gc.collect()

# #TODO: Please complete the training loop

# for epoch in range(0, config['epochs']):

#     print("\nEpoch: {}/{}".format(epoch+1, config['epochs']))

#     curr_lr = #TODO

#     train_loss              = #TODO
#     valid_loss, valid_dist  = #TODO
#     scheduler.step(valid_dist)

#     print("\tTrain Loss {:.04f}\t Learning Rate {:.07f}".format(train_loss, curr_lr))
#     print("\tVal Dist {:.04f}%\t Val Loss {:.04f}".format(valid_dist, valid_loss))


#     wandb.log({
#         'train_loss': train_loss,
#         'valid_dist': valid_dist,
#         'valid_loss': valid_loss,
#         'lr'        : curr_lr
#     })

#     save_model(model, optimizer, scheduler, ['valid_dist', valid_dist], epoch, epoch_model_path)
#     wandb.save(epoch_model_path)
#     print("Saved epoch model")

#     if valid_dist <= best_lev_dist:
#         best_lev_dist = valid_dist
#         save_model(model, optimizer, scheduler, ['valid_dist', valid_dist], epoch, best_model_path)
#         wandb.save(best_model_path)
#         print("Saved best model")
#       # You may find it interesting to exlplore Wandb Artifcats to version your models
# run.finish()


# #TODO: Make predictions

# # Follow the steps below:
# # 1. Create a new object for CTCBeamDecoder with larger (why?) number of beams
# # 2. Get prediction string by decoding the results of the beam decoder

# TEST_BEAM_WIDTH = #TODO

# test_decoder    = #TODO
# results = []

# model.eval()
# print("Testing")
# for data in tqdm(test_loader):

#     x, lx   = data
#     x       = x.to(device)

#     with torch.no_grad():
#         h, lh = model(x, lx)

#     prediction_string= # TODO call decode_prediction
#     #TODO save the output in results array.

#     del x, lx, h, lh
#     torch.cuda.empty_cache()


# data_dir = f"{root}/test-clean/random_submission.csv"
# df = pd.read_csv(data_dir)
# df.label = results
# df.to_csv('submission.csv', index = False)