
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

# get me RAMMM!!!!
import gc
gc.collect()

run_name = f"yiyan_run{datetime.now().strftime('%Y%m%d_%H%M%S')}"
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
# ['B', 'IH', 'K', 'SH', 'AA']
CMUdict = list(CMUdict_ARPAbet.keys())
ARPAbet = list(CMUdict_ARPAbet.values())

PHONEMES = CMUdict[:-2]
LABELS = ARPAbet[:-2]

########################## Config ##################################
config = {
    'epochs'        : 100, # change this to 10 
    'batch_size'    : 64,
    'init_lr'       : 2e-3,
    'architecture'  : 'high-cutoff',
    # Add more as you need them - e.g dropout values, weight decay, scheduler parameters
    'dropout'       : 0.1, # changed from 0.1 to 0.2
    'weight_decay'  : 1e-5,
    'scheduler'     : 'ReduceLROnPlateau',
    'batch_norm'    : True,
    'optimizer'     : 'AdamW',
    'activation'    : 'ReLU',
    "beam_width"    : 2,
}
###################################################################
from dataset import AudioDatasetTest

test_data = AudioDatasetTest(root="/ihome/hkarim/yip33/11785/HW3P2/11785-f24-hw3p2") #TODO
test_loader = torch.utils.data.DataLoader(
    dataset     = test_data,
    num_workers = 4,
    batch_size  = config['batch_size'],
    pin_memory  = True,
    shuffle     = False,
    collate_fn = test_data.collate_fn,
)

from network_ars import ASRModel

# model =  ASRModel(
#     input_size  = 28,#TODO,
#     embed_size = 192, #TODO
#     output_size = len(PHONEMES)
# ).to(device)
# print(model)
model = ASRModel(
    input_size  = 28,
    embed_size  = 1024,
    output_size = len(PHONEMES)
).to(device)
print(model)









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

model = model, _, _, _, _ = load_model(path='/ihome/hkarim/yip33/11785/HW3P2/checkpoints/yiyan_run20241109_162652/best_cls.pth',
                               model = model)

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

data_dir = f'/ihome/hkarim/yip33/11785/HW3P2/submissions/submission-{run_name}.csv'
# #data_dir = f"{root}/test-clean/random_submission.csv"
# df = pd.read_csv(data_dir)
# df.label = results
# df.to_csv('submission.csv', index = False)

index_column = range(len(results)) 
# Define the output path where the CSV will be saved
output_csv_path = f'/ihome/hkarim/yip33/11785/HW3P2/submissions/high-cutoff_{run_name}.csv'
submission_df = pd.DataFrame({
    'index': index_column,
    'label': results
})

# Save the DataFrame as a CSV file
submission_df.to_csv(data_dir, index=False)
submission_df.to_csv(output_csv_path, index=False)
print(f"Predictions saved to {output_csv_path}")