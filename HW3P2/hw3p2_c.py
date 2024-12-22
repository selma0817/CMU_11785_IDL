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
from datetime import datetime
# imports for decoding and distance calculation
import ctcdecode
import Levenshtein
from ctcdecode import CTCBeamDecoder

import warnings
warnings.filterwarnings('ignore')


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Device: ", device)



from dataset import AudioDataset, AudioDatasetTest
run_name = f"yiyan_run{datetime.now().strftime('%Y%m%d_%H%M%S')}"

# get me RAMMM!!!!
import gc
gc.collect()

################### PHONEMES and LABELS ###########################
###################################################################
CMUdict_ARPAbet = {
    "" : " ",
    "[SIL]": "-", "NG": "G", "F" : "f", "M" : "m", "AE": "@",
    "R"    : "r", "UW": "u", "N" : "n", "IY": "i", "AW": "W",
    "V"    : "v", "UH": "U", "OW": "o", "AA": "a", "ER": "R",
    "HH"   : "h", "Z" : "z", "K" : "k", "CH": "C", "W" : "w",
    "EY"   : "e", "ZH": "Z", "T" : "t", "EH": "E", "Y" : "y",
    "AH"   : "A", "B" : "b", "P" : "p", "TH": "T", "DH": "D",
    "AO"   : "c", "G" : "g", "L" : "l", "JH": "j", "OY": "O",
    "SH"   : "S", "D" : "d", "AY": "Y", "S" : "s", "IH": "I",
    "[SOS]": "[SOS]", "[EOS]": "[EOS]"
}
CMUdict = list(CMUdict_ARPAbet.keys())
ARPAbet = list(CMUdict_ARPAbet.values())

PHONEMES = CMUdict[:-2]
LABELS = ARPAbet[:-2]

###################################################################
###################################################################


# Create objects for the dataset class
root = "/ihome/hkarim/yip33/11785/HW3P2/11785-f24-hw3p2"
train_data = AudioDataset(root=root, partition='train-clean-100') #TODO
val_data =  AudioDataset(root="/ihome/hkarim/yip33/11785/HW3P2/11785-f24-hw3p2", partition = "dev-clean")  # TODO : You can either use the same class with some modifications or make a new one :)
test_data = AudioDatasetTest(root="/ihome/hkarim/yip33/11785/HW3P2/11785-f24-hw3p2") #TODO


config = {
    'epochs'        : 200, # change this to 10 
    'batch_size'    : 32,
    'init_lr'       : 2e-3,
    'architecture'  : 'high-cutoff-submission',
    # Add more as you need them - e.g dropout values, weight decay, scheduler parameters
    'dropout'       : 0.2, # changed from 0.1 to 0.2
    'weight_decay'  : 1e-5,
    #'scheduler'     : 'ReduceLROnPlateau',
    'scheduler'     : "CosineAnnealingWarmRestarts",
    'T_0'            : 20,
    'batch_norm'    : True,
    'optimizer'     : 'AdamW',
    'activation'    : 'ReLU',
    "beam_width"    : 5,
    'checkpoint_dir': f"./checkpoints/{run_name}",
}
os.makedirs(config['checkpoint_dir'], exist_ok=True)

# # Do NOT forget to pass in the collate function as parameter while creating the dataloader
train_loader = torch.utils.data.DataLoader(
    dataset     = train_data,
    num_workers = 4,
    batch_size  = config['batch_size'],
    pin_memory  = True,
    shuffle     = True,
    collate_fn = train_data.collate_fn,
)

val_loader = torch.utils.data.DataLoader(
    dataset     = val_data,
    num_workers = 2,
    batch_size  = config['batch_size'],
    pin_memory  = True,
    shuffle     = False,
    collate_fn = val_data.collate_fn,
)

test_loader = torch.utils.data.DataLoader(
    dataset     = test_data,
    num_workers = 2,
    batch_size  = config['batch_size'],
    pin_memory  = True,
    shuffle     = False,
    collate_fn = test_data.collate_fn,
)

print("Batch size: ", config['batch_size'])
print("Train dataset samples = {}, batches = {}".format(train_data.__len__(), len(train_loader)))
print("Val dataset samples = {}, batches = {}".format(val_data.__len__(), len(val_loader)))
print("Test dataset samples = {}, batches = {}".format(test_data.__len__(), len(test_loader)))

# sanity check
for data in train_loader:
    x, y, lx, ly = data
    print(x.shape, y.shape, lx.shape, ly.shape)
    print(type(x), type(y), type(lx), type(ly))
    break

#from network import ASRModel, Network
from network_ars import ASRModel

model = ASRModel(
    input_size  = 28,
    embed_size  = 1024,
    output_size = len(PHONEMES)
).to(device)


# model = Network(
#     input_size  = 28,#TODO,
#     encoder_hidden_size = 512, #TODO
#     output_size = len(PHONEMES)
# ).to(device)
print(model)
#summary(model, x.to(device), lx)
# input_size,  encoder_hidden_size, output_size

#TODO

criterion = torch.nn.CTCLoss() # Define CTC loss as the criterion. How would the losses be reduced?

optimizer =  torch.optim.AdamW(model.parameters(), lr=config["init_lr"]) # What goes in here?

decoder = CTCBeamDecoder(labels=LABELS, beam_width=config['beam_width'], log_probs_input=True)

#scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode = 'min', patience = 3, threshold=1e-2)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
#                                                         factor=0.9,
#                                                         patience=3,
#                                                         threshold=1e-2)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, config["T_0"])
scaler = torch.cuda.amp.GradScaler()

torch.cuda.empty_cache()
def decode_prediction(output, output_lens, decoder, PHONEME_MAP= LABELS):

    # TODO: look at docs for CTC.decoder and find out what is returned here. Check the shape of output and expected shape in decode.
    # BATCHSIZE x N_BEAMS X N_TIMESTEPS
    # beam_results， beam_scores, timesteps, out_lens
    # beam_results, beam_scores, timesteps, out_seq_len = decoder.decode(output, seq_lens= output_lens)
    #beam_results, beam_scores, timesteps, out_seq_len = decoder.decode(output, seq_lens = output_lens) #lengths - list of lengths
    beam_results, beam_scores, timesteps, out_lens = decoder.decode(output, seq_lens=output_lens)

    pred_strings                    = []

    # for i in range(output_lens.shape[0]):
    #     best_beam = beam_results[i][0][:out_seq_len[i][0]]
    #     #TODO: Create the prediction from the output of decoder.decode. Don't forget to map it using PHONEMES_MAP.
    #     pred_strings.append(''.join([PHONEME_MAP[c] for c in best_beam]))
    # return pred_strings
    for i in range(output_lens.shape[0]):

        best_beam = beam_results[i][0][:out_lens[i][0]]
        # print(best_beam.shape)

        pred_strings.append("".join([PHONEME_MAP[character] for character in best_beam]))
        # pred_strings.append([PHONEME_MAP[character] for character in best_beam])

        #TODO: Create the prediction from the output of decoder.decode. Don't forget to map it using PHONEMES_MAP.

    return pred_strings




def calculate_levenshtein(output, label, output_lens, label_lens, decoder, PHONEME_MAP= LABELS): # y - sequence of integers

    dist            = 0
    batch_size      = label.shape[0]
    pred_strings    = decode_prediction(output, output_lens, decoder, PHONEME_MAP)

    for i in range(batch_size):
        pred_string = pred_strings[i]
        label_string = ''.join([PHONEME_MAP[n] for n in label[i][:label_lens[i]]])
        dist += Levenshtein.distance(pred_string, label_string)

    dist /= batch_size # TODO: Uncomment this, but think about why we are doing this
    del pred_string, label_string
    gc.collect()
    torch.cuda.empty_cache()
    return dist

#     # test code to check shapes
torch.cuda.empty_cache()
model.eval()
for i, data in enumerate(val_loader, 0):
    x, y, lx, ly = data
    x, y = x.to(device), y.to(device)
    h, lh = model(x, lx)
    print(h.shape)
    print(calculate_levenshtein(h, y, lx, ly, decoder, LABELS))
    h = torch.permute(h, (1, 0, 2))
    print(h.shape, y.shape)
    loss = criterion(h, y, lh, ly)
    print(loss)

    break


import wandb
wandb.login(key="57c916d673703185e1b47000c74bd854db77bcf8")


run = wandb.init(
    name = "asr_1024_embed_size", ## Wandb creates random run names if you skip this field
    reinit = True, ### Allows reinitalizing runs when you re-run this cell
    # run_id = ### Insert specific run id here if you want to resume a previous run
    # resume = "must" ### You need this to resume previous runs, but comment out reinit = True when using this
    project = "hw3p2-yiyan", ### Project should be created in your wandb account
    config = config ### Wandb Config for your run
)


from tqdm import tqdm

def train_model(model, train_loader, criterion, optimizer):

    model.train()
    batch_bar = tqdm(total=len(train_loader), dynamic_ncols=True, leave=False, position=0, desc='Train')

    total_loss = 0

    for i, data in enumerate(train_loader):
        optimizer.zero_grad()

        x, y, lx, ly = data
        x, y = x.to(device), y.to(device)

        with torch.cuda.amp.autocast():
            h, lh = model(x, lx)
            h = torch.permute(h, (1, 0, 2))
            loss = criterion(h, y, lh, ly)

        total_loss += loss.item()

        batch_bar.set_postfix(
            loss="{:.04f}".format(float(total_loss / (i + 1))),
            lr="{:.06f}".format(float(optimizer.param_groups[0]['lr'])))

        batch_bar.update() # Update tqdm bar

        # Another couple things you need for FP16.
        scaler.scale(loss).backward() # This is a replacement for loss.backward()
        scaler.step(optimizer) # This is a replacement for optimizer.step()
        scaler.update() # This is something added just for FP16

        del x, y, lx, ly, h, lh, loss
        torch.cuda.empty_cache()

    batch_bar.close() # You need this to close the tqdm bar

    return total_loss / len(train_loader)


def validate_model(model, val_loader, decoder, phoneme_map= LABELS):

    model.eval()
    batch_bar = tqdm(total=len(val_loader), dynamic_ncols=True, position=0, leave=False, desc='Val')

    total_loss = 0
    vdist = 0

    for i, data in enumerate(val_loader):

        x, y, lx, ly = data
        x, y = x.to(device), y.to(device)

        with torch.inference_mode():
            h, lh = model(x, lx)
            h = torch.permute(h, (1, 0, 2))
            loss = criterion(h, y, lh, ly)

        total_loss += float(loss)
        vdist += calculate_levenshtein(torch.permute(h, (1, 0, 2)), y, lh, ly, decoder, phoneme_map)

        batch_bar.set_postfix(loss="{:.04f}".format(float(total_loss / (i + 1))), dist="{:.04f}".format(float(vdist / (i + 1))))

        batch_bar.update()

        del x, y, lx, ly, h, lh, loss
        torch.cuda.empty_cache()

    batch_bar.close()
    total_loss = total_loss/len(val_loader)
    val_dist = vdist/len(val_loader)
    return total_loss, val_dist


def save_model(model, optimizer, scheduler, metric, epoch, path):
    torch.save(
        {'model_state_dict'         : model.state_dict(),
         'optimizer_state_dict'     : optimizer.state_dict(),
         'scheduler_state_dict'     : scheduler.state_dict(),
         metric[0]                  : metric[1],
         'epoch'                    : epoch},
         path
    )

def load_model(path, model, metric= 'valid_dist', optimizer= None, scheduler= None):

    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer != None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler != None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    epoch   = checkpoint['epoch']
    metric  = checkpoint[metric]

    return [model, optimizer, scheduler, epoch, metric]

# model = model, _, _, _, _ = load_model(path='/ihome/hkarim/yip33/11785/HW3P2/checkpoints/yiyan_run20241108_135156/best_cls.pth',
#                                model = model)
# This is for checkpointing, if you're doing it over multiple sessions

last_epoch_completed = 0
start = last_epoch_completed
end = config["epochs"]
best_lev_dist = float("inf") # if you're restarting from some checkpoint, use what you saw there.
#TODO set the model path( Optional, you can just store best one. Make sure to make the changes below )
#TODO set best model path

# save_model(model, optimizer, scheduler, metrics, epoch, os.path.join(config['checkpoint_dir'], 'last.pth'))
# print("Saved epoch model")

epoch_model_path = os.path.join(config['checkpoint_dir'], 'last.pth')
# save_model(model, optimizer, scheduler, metrics, epoch, os.path.join(config['checkpoint_dir'], 'best_cls.pth'))
best_model_path = os.path.join(config['checkpoint_dir'], 'best_cls.pth')

torch.cuda.empty_cache()
gc.collect()

# #TODO: Please complete the training loop

for epoch in range(0, config['epochs']):

    print("\nEpoch: {}/{}".format(epoch+1, config['epochs']))

    curr_lr = float(optimizer.param_groups[0]['lr'])

    train_loss              = train_model(model, train_loader, criterion, optimizer)
    valid_loss, valid_dist  = validate_model(model, val_loader, decoder, phoneme_map= LABELS)
    #scheduler.step(valid_dist)
    scheduler.step()

    print("\tTrain Loss {:.04f}\t Learning Rate {:.07f}".format(train_loss, curr_lr))
    print("\tVal Dist {:.04f}%\t Val Loss {:.04f}".format(valid_dist, valid_loss))


    wandb.log({
        'train_loss': train_loss,
        'valid_dist': valid_dist,
        'valid_loss': valid_loss,
        'lr'        : curr_lr
    })

    save_model(model, optimizer, scheduler, ['valid_dist', valid_dist], epoch, epoch_model_path)
    wandb.save(epoch_model_path)
    print("Saved epoch model")

    if valid_dist <= best_lev_dist:
        best_lev_dist = valid_dist
        save_model(model, optimizer, scheduler, ['valid_dist', valid_dist], epoch, best_model_path)
        wandb.save(best_model_path)
        print("Saved best model")
      # You may find it interesting to exlplore Wandb Artifcats to version your models
run.finish()


# #TODO: Make predictions

# # Follow the steps below:
# # 1. Create a new object for CTCBeamDecoder with larger (why?) number of beams
# # 2. Get prediction string by decoding the results of the beam decoder

TEST_BEAM_WIDTH = 10

test_decoder    = CTCBeamDecoder(LABELS, beam_width = TEST_BEAM_WIDTH, log_probs_input = True)
results = []

model.eval()
print("Testing")
for data in tqdm(test_loader):

    x, lx   = data
    x       = x.to(device)

    with torch.no_grad():
        h, lh = model(x, lx)

    prediction_string= decode_prediction(h, lh, test_decoder)
    #TODO save the output in results array.
    results.extend(prediction_string)

    del x, lx, h, lh
    torch.cuda.empty_cache()


# data_dir = f"{root}/test-clean/final_submission.csv"
# df = pd.read_csv(data_dir)
# df.label = results
# df.to_csv('final_submission.csv', index = False)
data_dir = f'/ihome/hkarim/yip33/11785/HW3P2/submissions/submission-{run_name}.csv'
# #data_dir = f"{root}/test-clean/random_submission.csv"
# df = pd.read_csv(data_dir)
# df.label = results
# df.to_csv('submission.csv', index = False)

index_column = range(len(results)) 
# Define the output path where the CSV will be saved
output_csv_path = f'/ihome/hkarim/yip33/11785/HW3P2/submissions/high-cutoff-submission_{run_name}.csv'
submission_df = pd.DataFrame({
    'index': index_column,
    'label': results
})

# Save the DataFrame as a CSV file
submission_df.to_csv(data_dir, index=False)
submission_df.to_csv(output_csv_path, index=False)
print(f"Predictions saved to {output_csv_path}")