
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.functional as aF
import torchaudio.transforms as tat
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.utils.data import TensorDataset, Dataset, DataLoader
import gc
import os
from transformers import AutoTokenizer
import yaml
import math
from typing import Literal, List, Optional, Any, Dict, Tuple
import random
import zipfile
import datetime
from torchinfo import summary
import glob
import wandb
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.fftpack import dct
import seaborn as sns
import matplotlib.pyplot as plt
''' Imports for decoding and distance calculation. '''
import json
import warnings
import shutil
warnings.filterwarnings("ignore")
import argparse
# parser = argparse.ArgumentParser(description="run name of job")
# parser.add_argument('--run_log_name', type=str, required=True, help="name of run log")
# args = parser.parse_args()

# run_log_name = args.run_log_name

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device: ", device)

# config
with open("config_high_3.yaml") as file:
    config = yaml.safe_load(file)
print(config)


### check config
root = config['root']
batch_size = config['batch_size']
specaug_conf = config['specaug_conf']

print(f"Root path: {root}")
print(f"Batch size: {batch_size}")
print(f"SpecAugment config: {specaug_conf}")



from dataset import SpeechDataset, TextDataset
from tokenizer import GTokenizer, CharTokenizer
Tokenizer = GTokenizer(config['token_type'])


train_dataset = torch.load("/ix1/hkarim/yip33/kaggle_dataset/hw4_data/pt_feat_70/1k_train_dataset.pt")
val_dataset = torch.load("/ix1/hkarim/yip33/kaggle_dataset/hw4_data/pt_feat_70/1k_val_dataset.pt")
test_dataset = torch.load("/ix1/hkarim/yip33/kaggle_dataset/hw4_data/pt_feat_70/1k_test_dataset.pt")




'''
# UNCOMMENT if pretraining decoder as LM
text_dataset   = TextDataset(
     partition  = config['unpaired_text_partition'],
     config     = config,
     tokenizer  = Tokenizer,
)
'''


train_loader    = DataLoader(
    dataset     = train_dataset,
    batch_size  = config["batch_size"],
    shuffle     = True,
    num_workers = config['NUM_WORKERS'],
    pin_memory  = True,
    collate_fn  = train_dataset.collate_fn
)


val_loader      = DataLoader(
    dataset     = val_dataset,
    batch_size  = 4,
    shuffle     = False,
    num_workers = config['NUM_WORKERS'],
    pin_memory  = True,
    collate_fn  = val_dataset.collate_fn
)


test_loader     = DataLoader(
    dataset     = test_dataset,
    batch_size  = config["batch_size"],
    shuffle     = False,
    num_workers = config['NUM_WORKERS'],
    pin_memory  = True,
    collate_fn  = test_dataset.collate_fn
)

'''
# UNCOMMENT if pretraining decoder as LM
text_loader     = DataLoader(
   dataset       = text_dataset,
   batch_size    = config["batch_size"],
   shuffle       = True,
   num_workers   = config['NUM_WORKERS'],
   pin_memory    = True,
   collate_fn    = text_dataset.collate_fn
)
'''

def verify_dataset(dataloader, partition):
    '''Compute the Maximum MFCC and Transcript sequence length in a dataset'''
    print("Loaded Path: ", partition)
    max_len_feat = 0
    max_len_t    = 0  # To track the maximum length of transcripts

    # Iterate through the dataloader
    for batch in tqdm(dataloader, desc=f"Verifying {partition} Dataset"):
      try:
        x_pad, y_shifted_pad, y_golden_pad, x_len, y_len = batch

        # Update the maximum feat length
        len_x = x_pad.shape[1]
        if len_x > max_len_feat:
            max_len_feat = len_x

        # Update the maximum transcript length
        # transcript length is dim 1 of y_shifted_pad
        if y_shifted_pad is not None:
          len_y = y_shifted_pad.shape[1]
          if len_y > max_len_t:
              max_len_t = len_y

      except Exception as e:
        # The text dataset has no transcripts
        y_shifted_pad, y_golden_pad, y_len = batch

        # Update the maximum transcript length
        # transcript length is dim 1 of y_shifted_pad
        len_y = y_shifted_pad.shape[1]
        if len_y > max_len_t:
            max_len_t = len_y


    print(f"Maximum Feat Length in Dataset       : {max_len_feat}")
    print(f"Maximum Transcript Length in Dataset : {max_len_t}")
    return max_len_feat, max_len_t

print('')
print("Paired Data Stats: ")
print(f"No. of Train Feats   : {train_dataset.__len__()}")
print(f"Batch Size           : {config['batch_size']}")
print(f"Train Batches        : {train_loader.__len__()}")
print(f"Val Batches          : {val_loader.__len__()}")
# print(f"Test Batches         : {test_loader.__len__()}")
print('')
print("Checking the Shapes of the Data --\n")
for batch in train_loader:   #train_loader
    x_pad, y_shifted_pad, y_golden_pad, x_len, y_len, = batch
    print(f"x_pad shape:\t\t{x_pad.shape}")
    print(f"x_len shape:\t\t{x_len.shape}")

    if y_shifted_pad is not None and y_golden_pad is not None and y_len is not None:
      print(f"y_shifted_pad shape:\t{y_shifted_pad.shape}")
      print(f"y_golden_pad shape:\t{y_golden_pad.shape}")
      print(f"y_len shape:\t\t{y_len.shape}\n")
      # convert one transcript to text
      transcript = train_dataset.tokenizer.decode(y_shifted_pad[0].tolist())
      print(f"Transcript Shifted: {transcript}")
      transcript = train_dataset.tokenizer.decode(y_golden_pad[0].tolist())
      print(f"Transcript Golden: {transcript}")
    break
print('')
'''
# UNCOMMENT if pretraining decoder as LM
print("Unpaired Data Stats: ")
print(f"No. of text          : {text_dataset.__len__()}")
print(f"Batch Size           : {config['batch_size']}")
print(f"Train Batches        : {text_loader.__len__()}")
print('')
print("Checking the Shapes of the Data --\n")
for batch in text_loader:
     y_shifted_pad, y_golden_pad, y_len, = batch
     print(f"y_shifted_pad shape:\t{y_shifted_pad.shape}")
     print(f"y_golden_pad shape:\t{y_golden_pad.shape}")
     print(f"y_len shape:\t\t{y_len.shape}\n")

     # convert one transcript to text
     transcript = text_dataset.tokenizer.decode(y_shifted_pad[0].tolist())
     print(f"Transcript Shifted: {transcript}")
     transcript = text_dataset.tokenizer.decode(y_golden_pad[0].tolist())
     print(f"Transcript Golden: {transcript}")
     break
print('')
'''
print("\n\nVerifying Datasets")
max_train_feat, max_train_transcript = verify_dataset(train_loader, config['train_partition'])
max_val_feat, max_val_transcript     = verify_dataset(val_loader,   config['val_partition'])
max_test_feat, max_test_transcript   = verify_dataset(test_loader,  config['test_partition'])
#_, max_text_transcript               = verify_dataset(text_loader,  config['unpaired_text_partition'])

MAX_SPEECH_LEN = max(max_train_feat, max_val_feat, max_test_feat)
MAX_TRANS_LEN  = max(max_train_transcript, max_val_transcript)
print(f"Maximum Feat. Length in Entire Dataset      : {MAX_SPEECH_LEN}")
print(f"Maximum Transcript Length in Entire Dataset : {MAX_TRANS_LEN}")
print('')
gc.collect()

plt.figure(figsize=(10, 6))
plt.imshow(x_pad[0].numpy().T, aspect='auto', origin='lower', cmap='viridis')
plt.xlabel('Time')
plt.ylabel('Features')
plt.title('Feature Representation')

plt.savefig("Feature_Representation.png", dpi=300)
plt.close()  # Close the figure to free memory


from trans import Transformer
model = Transformer(
    input_dim      = x_pad.shape[-1],
    time_stride    = config['time_stride'],
    feature_stride = config['feature_stride'],
    embed_dropout  = config['embed_dropout'],
    d_model        = config['d_model'],
    enc_num_layers = config['enc_num_layers'],
    enc_num_heads  = config['enc_num_heads'],
    speech_max_len = MAX_SPEECH_LEN,
    enc_dropout    = config['enc_dropout'],
    dec_num_layers = config['dec_num_layers'],
    dec_num_heads  = config['dec_num_heads'],
    d_ff           = config['d_ff'],
    dec_dropout    = config['dec_dropout'],
    target_vocab_size = Tokenizer.VOCAB_SIZE,
    trans_max_len     = MAX_TRANS_LEN
)

summary(model.to(device), input_data=[x_pad.to(device), x_len.to(device), y_shifted_pad.to(device), y_len.to(device)])


gc.collect()
torch.cuda.empty_cache()





loss_func   = nn.CrossEntropyLoss(ignore_index = Tokenizer.PAD_TOKEN)
ctc_loss_fn  = None
if config['use_ctc']:
    ctc_loss_fn = nn.CTCLoss(blank=Tokenizer.PAD_TOKEN)
scaler      = torch.cuda.amp.GradScaler()


def get_optimizer():
    optimizer = None
    if config["optimizer"] == "SGD":
        # feel free to change any of the initializations you like to fit your needs
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=config["learning_rate"],
                                    momentum=config["momentum"],
                                    weight_decay=1E-4,
                                    nesterov=config["nesterov"])

    elif config["optimizer"] == "Adam":
        # feel free to change any of the initializations you like to fit your needs
        optimizer = torch.optim.Adam(model.parameters(),
                                    lr=float(config["learning_rate"]),weight_decay=0.01 )

    elif config["optimizer"] == "AdamW":
        # feel free to change any of the initializations you like to fit your needs
        optimizer = torch.optim.AdamW(model.parameters(),
                                    lr=float(config["learning_rate"]),
                                    weight_decay=0.01)
    return optimizer
optimizer = get_optimizer()
assert optimizer!=None

def get_scheduler():
    scheduler  =  None
    if config["scheduler"] == "ReduceLR":
        #Feel Free to change any of the initializations you like to fit your needs
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                        factor=config["factor"], patience=config["patience"], min_lr=1E-8, threshold=1E-1)

    elif config["scheduler"] == "CosineAnnealing":
        #Feel Free to change any of the initializations you like to fit your needs
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                        T_max = config["T_max"], eta_min=1E-8)
    return scheduler

scheduler = get_scheduler()
assert scheduler!=None



# # using WandB? resume training?
USE_WANDB = config['use_wandb']
RESUME_LOGGING = False
from datetime import datetime
# creating your WandB run
curr_time = datetime.now().strftime("%Y%m%d-%H%M%S")
print(f"cur_time: {curr_time}")
# add run_name

run_name = "{}_{}_Transformer_ENC-{}-{}_DEC-{}-{}_{}_{}_{}_{}_token_{}_{}".format(
    config["Name"],
    config['feat_type'],
    config["enc_num_layers"],
    config["enc_num_heads"],
    config["dec_num_layers"],
    config["dec_num_heads"],
    config["d_model"],
    config["d_ff"],
    config["optimizer"],
    config["scheduler"],
    config["token_type"],
    curr_time,
    )

# # change expt_root to my ix1 dir so i don't store my checkpoints and output in ihome
expt_root = os.path.join(os.getcwd(), run_name)
expt_root = "/ix1/hkarim/yip33/HW4P2/runs/" + run_name

print(f"expt_root: {expt_root}")
os.makedirs(expt_root, exist_ok=True)

if USE_WANDB:
    wandb.login(key="57c916d673703185e1b47000c74bd854db77bcf8", relogin=True) # TODO enter your key here

    if RESUME_LOGGING:
        run_id = ""
        run = wandb.init(
            id     = run_id,        ### Insert specific run id here if you want to resume a previous run
            resume = True,          ### You need this to resume previous runs, but comment out reinit=True when using this
            project = "HW4P2-yiyan",  ### Project should be created in your wandb account
        )

    else:
        run = wandb.init(
            name    = "transformer_high_cutoff_300_fulltrain",     ### Wandb creates random run names if you skip this field, we recommend you give useful names
            reinit  = True,         ### Allows reinitalizing runs when you re-run this cell
            project = "HW4P2-yiyan",  ### Project should be created in your wandb account
            config  = config        ### Wandb Config for your run
        )

        ### Save your model architecture as a string with str(model)
        model_arch  = str(model)
        ### Save it in a txt file
        model_path = os.path.join(expt_root, "model_arch.txt")
        arch_file   = open(model_path, "w")
        file_write  = arch_file.write(model_arch)
        arch_file.close()

        ### Log it in your wandb run with wandb.sav


# ### Create a local directory with all the checkpoints
shutil.copy(os.path.join(os.getcwd(), 'config_high_2.yaml'), os.path.join(expt_root, 'config_high_2.yaml'))
e                   = 0
best_loss           = 10.0
best_perplexity     = 10.0
best_dist = 60
RESUME_LOGGING = False
checkpoint_root = os.path.join(expt_root, 'checkpoints')
text_root       = os.path.join(expt_root, 'out_text')
attn_img_root   = os.path.join(expt_root, 'attention_imgs')
os.makedirs(checkpoint_root, exist_ok=True)
os.makedirs(attn_img_root,   exist_ok=True)
os.makedirs(text_root,       exist_ok=True)
checkpoint_best_loss_model_filename     = 'checkpoint-best-loss-modelfull.pth'
checkpoint_last_epoch_filename          = 'checkpoint-epochfull-'
best_loss_model_path                    = os.path.join(checkpoint_root, checkpoint_best_loss_model_filename)


if USE_WANDB:
    wandb.watch(model, log="all")

if RESUME_LOGGING:
    # change if you want to load best test model accordingly
    checkpoint = torch.load(wandb.restore(checkpoint_best_loss_model_filename, run_path=""+run_id).name)

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    e = checkpoint['epoch']

    print("Resuming from epoch {}".format(e+1))
    print("Epochs left: ", config['epochs']-e)
    print("Optimizer: \n", optimizer)

torch.cuda.empty_cache()
gc.collect()

from train_val_test_step import train_step, validate_step, test_step
from misc import save_attention_plot, save_model


# ## Pretrain Approach 2

gc.collect()
torch.cuda.empty_cache()


#set your epochs for this approach
epochs = 20
for epoch in range(e, epochs):

    print("\nEpoch {}/{}".format(epoch+1, epochs))

    curr_lr = float(optimizer.param_groups[0]["lr"])

    train_loss, train_perplexity, attention_weights = train_step(
        model,
        criterion=loss_func, 
        ctc_loss=None,  #ctc_loss_fn
        ctc_weight=0.,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        device=device,
        train_loader=train_loader,
        tokenizer=Tokenizer,
        mode='dec_cond_lm'
    )

    print("\nEpoch {}/{}: \nTrain Loss {:.04f}\t Train Perplexity {:.04f}\t Learning Rate {:.06f}".format(
        epoch + 1, epochs, train_loss, train_perplexity, curr_lr))


    levenshtein_distance, json_out, wer, cer = validate_step(
        model,
        val_loader=val_loader,
        tokenizer=Tokenizer,
        device=device,
        mode='dec_cond_lm',
        threshold=5
    )


    fpath = os.path.join(text_root, f'dec_cond_lm_{epoch+1}_out.json')
    with open(fpath, "w") as f:
        json.dump(json_out, f, indent=4)

    print("Levenshtein Distance : {:.04f}".format(levenshtein_distance))
    print("WER                  : {:.04f}".format(wer))
    print("CER                  : {:.04f}".format(cer))

    attention_keys = list(attention_weights.keys())
    attention_weights_decoder_self   = attention_weights[attention_keys[0]][0].cpu().detach().numpy()
    attention_weights_decoder_cross  = attention_weights[attention_keys[-1]][0].cpu().detach().numpy()

    if USE_WANDB:
        wandb.log({
            "train_loss"       : train_loss,
            "train_perplexity" : train_perplexity,
            "learning_rate"    : curr_lr,
            "lev_dist"         : levenshtein_distance,
            "WER"              : wer,
            "CER"              : cer
        })


    save_attention_plot(str(attn_img_root), attention_weights_decoder_cross, epoch,     mode='dec_cond_lm')
    save_attention_plot(str(attn_img_root), attention_weights_decoder_self,  epoch+100, mode='dec_cond_lm')
    if config["scheduler"] == "ReduceLR":
        scheduler.step(levenshtein_distance)
    else:
        scheduler.step()

    ### Highly Recommended: Save checkpoint in drive and/or wandb if accuracy is better than your current best
    epoch_model_path = os.path.join(checkpoint_root, (checkpoint_last_epoch_filename + str(epoch) + '.pth'))
    save_model(model, optimizer, scheduler, ('CER', cer), epoch, epoch_model_path)

    if best_dist >= levenshtein_distance:
        best_loss = train_loss
        best_dist = levenshtein_distance
        save_model(model, optimizer, scheduler, ('CER', cer), epoch, best_loss_model_path)
        print("Saved best CER model")

# # ### Finish your wandb run
# # if USE_WANDB:
# #     run.finish()
# #### ----------------------------------------------------------------------------------------------------------------------

# ############################## Freeze Unfreeze ########################
# #######################################################################
def freeze(freeze_embedding, freeze_encoder, freeze_decoder):
      # freeze embeddings
      if freeze_embedding:
            for name, param in model.named_parameters():
                  if name.startswith("embedding"):
                        param.requires_grad = False

      # freeze encoder
      if freeze_encoder:
            for name, param in model.named_parameters():
                  if name.startswith("encoder"):
                        param.requires_grad = False

      # freeze decoder
      if freeze_decoder:
            for name, param in model.named_parameters():
                  if name.startswith("decoder"):
                        param.requires_grad = False

def unfreeze(unfreeze_embedding, unfreeze_encoder, unfreeze_decoder):
      
      # unfreeze embedding
      if unfreeze_embedding:
            for name, param in model.named_parameters():
                  if name.startswith("embedding"):
                        param.requires_grad = True

      #unfreeze encoder
      if unfreeze_encoder:
            for name, param in model.named_parameters():
                  if name.startswith("encoder"):
                        param.requires_grad = True

      # unfreeze decoder
      if unfreeze_decoder:
            for name, param in model.named_parameters():
                  if name.startswith("decoder"):
                        param.requires_grad = True

#######################################################################
#######################################################################


#LOAD MODEL HERE
freeze(freeze_embedding=True, freeze_encoder=False, freeze_decoder=True)
# freeze and train encoder for 5 epochs


epochs = 3 
for epoch in range(e, epochs):

    print("\nEpoch {}/{}".format(epoch+1, epochs))

    curr_lr = float(optimizer.param_groups[0]["lr"])

    train_loss, train_perplexity, attention_weights = train_step(
        model,
        criterion=loss_func,
        ctc_loss=ctc_loss_fn,
        ctc_weight=config['ctc_weight'],
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        device=device,
        train_loader=train_loader,
        tokenizer=Tokenizer,
        mode='full'
    )

    print("\nEpoch {}/{}: \nTrain Loss {:.04f}\t Train Perplexity {:.04f}\t Learning Rate {:.06f}".format(
        epoch + 1, epochs, train_loss, train_perplexity, curr_lr))

    levenshtein_distance, json_out, wer, cer = validate_step(
        model,
        val_loader=val_loader,
        tokenizer=Tokenizer,
        device=device,
        mode='full',
        threshold=300
    )


    fpath = os.path.join(text_root, f'full_{epoch+1}_out.json')
    with open(fpath, "w") as f:
        json.dump(json_out, f, indent=4)

    print("Levenshtein Distance : {:.04f}".format(levenshtein_distance))
    print("WER                  : {:.04f}".format(wer))
    print("CER                  : {:.04f}".format(cer))

    attention_keys = list(attention_weights.keys())
    attention_weights_decoder_self   = attention_weights[attention_keys[0]][0].cpu().detach().numpy()
    attention_weights_decoder_cross  = attention_weights[attention_keys[-1]][0].cpu().detach().numpy()

    if USE_WANDB:
        wandb.log({
            "train_loss"       : train_loss,
            "train_perplexity" : train_perplexity,
            "learning_rate"    : curr_lr,
            "lev_dist"         : levenshtein_distance,
            "WER"              : wer,
            "CER"              : cer
        })


    save_attention_plot(str(attn_img_root), attention_weights_decoder_cross, epoch,     mode='full')
    save_attention_plot(str(attn_img_root), attention_weights_decoder_self, epoch+100,  mode='full')
    if config["scheduler"] == "ReduceLR":
        scheduler.step(levenshtein_distance)
    else:
        scheduler.step()

    ### Highly Recommended: Save checkpoint in drive and/or wandb if accuracy is better than your current best
    epoch_model_path = os.path.join(checkpoint_root, (checkpoint_last_epoch_filename + str(epoch) + '-2'+'.pth'))
    save_model(model, optimizer, scheduler, ('CER', cer), epoch, epoch_model_path)

    if best_dist >= levenshtein_distance:
        best_loss = train_loss
        best_dist = levenshtein_distance
        save_model(model, optimizer, scheduler, ('CER', cer), epoch, best_loss_model_path)
        print("Saved best distance model")


unfreeze(unfreeze_embedding=True, unfreeze_encoder=True, unfreeze_decoder=True)

## load and train ### 


from misc import load_checkpoint
checkpoint_path = "/ix1/hkarim/yip33/HW4P2/runs/yiyan_fbank_Transformer_ENC-6-8_DEC-6-8_256_2048_AdamW_CosineAnnealing_token_1k_20241205-222241/checkpoints/checkpoint-epochfull-104-2.pth"

model, optimizer, scheduler= load_checkpoint(checkpoint_path=checkpoint_path,
                                              model = model,
                                              embedding_load=True,
                                              encoder_load=True,
                                              decoder_load=True,
                                              optimizer=optimizer,
                                              scheduler=scheduler,
                                              )

e = 104

# ############################## Full Transformer Training ########################
# #################################################################################
gc.collect()
torch.cuda.empty_cache()

epochs = config['epochs']
for epoch in range(e, epochs):

    print("\nEpoch {}/{}".format(epoch+1, epochs))

    curr_lr = float(optimizer.param_groups[0]["lr"])

    train_loss, train_perplexity, attention_weights = train_step(
        model,
        criterion=loss_func,
        ctc_loss=ctc_loss_fn,
        ctc_weight=config['ctc_weight'],
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        device=device,
        train_loader=train_loader,
        tokenizer=Tokenizer,
        mode='full'
    )

    print("\nEpoch {}/{}: \nTrain Loss {:.04f}\t Train Perplexity {:.04f}\t Learning Rate {:.06f}".format(
        epoch + 1, epochs, train_loss, train_perplexity, curr_lr))


    levenshtein_distance, json_out, wer, cer = validate_step(
        model,
        val_loader=val_loader,
        tokenizer=Tokenizer,
        device=device,
        mode='full',
        threshold=300
    )


    fpath = os.path.join(text_root, f'full_{epoch+1}_out.json')
    with open(fpath, "w") as f:
        json.dump(json_out, f, indent=4)

    print("Levenshtein Distance : {:.04f}".format(levenshtein_distance))
    print("WER                  : {:.04f}".format(wer))
    print("CER                  : {:.04f}".format(cer))

    attention_keys = list(attention_weights.keys())
    attention_weights_decoder_self   = attention_weights[attention_keys[0]][0].cpu().detach().numpy()
    attention_weights_decoder_cross  = attention_weights[attention_keys[-1]][0].cpu().detach().numpy()

    if USE_WANDB:
        wandb.log({
            "train_loss"       : train_loss,
            "train_perplexity" : train_perplexity,
            "learning_rate"    : curr_lr,
            "lev_dist"         : levenshtein_distance,
            "WER"              : wer,
            "CER"              : cer
        })


    save_attention_plot(str(attn_img_root), attention_weights_decoder_cross, epoch,     mode='full')
    save_attention_plot(str(attn_img_root), attention_weights_decoder_self, epoch+100,  mode='full')
    if config["scheduler"] == "ReduceLR":
        scheduler.step(levenshtein_distance)
    else:
        scheduler.step()

    ### Highly Recommended: Save checkpoint in drive and/or wandb if accuracy is better than your current best
    epoch_model_path = os.path.join(checkpoint_root, (checkpoint_last_epoch_filename + str(epoch) + '-2'+'.pth'))
    save_model(model, optimizer, scheduler, ('CER', cer), epoch, epoch_model_path)

    if best_dist >= levenshtein_distance:
        best_loss = train_loss
        best_dist = levenshtein_distance
        save_model(model, optimizer, scheduler, ('CER', cer), epoch, best_loss_model_path)
        print("Saved best distance model")

### Finish your wandb run
if USE_WANDB:
    run.finish()
# #### ----------------------------------------------------------------------------------------------------------------------

# #################################################################################
# #################################################################################


# ### Full Evaluation on Validations Set

levenshtein_distance, json_out, wer, cer = validate_step(
        model,
        val_loader=val_loader,
        tokenizer=Tokenizer,
        device=device,
        mode='full',
        threshold=None
    )

print("Levenshtein Distance : {:.04f}".format(levenshtein_distance))
print("WER                  : {:.04f}".format(wer))
print("CER                  : {:.04f}".format(cer))

fpath = os.path.join(os.getcwd(), f'final_out_{run_name}.json')
with open(fpath, "w") as f:
    json.dump(json_out, f, indent=4)


# ### produce output csv for testing 

predictions = test_step(
        model,
        test_loader=test_loader,
        tokenizer=Tokenizer,
        device=device,
)

import csv

# Sample list
# Specify the CSV file path
csv_file_path = f"submission_{curr_time}_high_cutoff.csv"

# Write the list to the CSV with index as the first column
with open(csv_file_path, mode="w", newline="") as file:
    writer = csv.writer(file)
    # Write the header
    writer.writerow(["Index", "Labels"])
    # Write list items with index
    for idx, item in enumerate(predictions):
        writer.writerow([idx, item])

print(f"CSV file saved to {csv_file_path}")