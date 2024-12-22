
import torch
import torch.nn as nn
from train_val_test_step import test_step
from misc import load_checkpoint
from trans import Transformer
from dataset import SpeechDataset
from dataset import SpeechDataset, TextDataset
from tokenizer import GTokenizer, CharTokenizer
import yaml
from torch.utils.data import TensorDataset, Dataset, DataLoader
from tqdm import tqdm
import csv

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

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device: ", device)

# config
with open("config_high_3.yaml") as file:
    config = yaml.safe_load(file)
print(config)

# # config
# with open("/ix1/hkarim/yip33/HW4P2/runs/yiyan_fbank_Transformer_ENC-6-8_DEC-6-8_256_2048_AdamW_CosineAnnealing_token_1k_20241128-194910/config.yaml") as file:
#     config = yaml.safe_load(file)
# print(config)

### check config
root = config['root']
batch_size = config['batch_size']
specaug_conf = config['specaug_conf']

print(f"Root path: {root}")
print(f"Batch size: {batch_size}")
print(f"SpecAugment config: {specaug_conf}")


Tokenizer = GTokenizer(config['token_type'])

#checkpoint_path = "/ix1/hkarim/yip33/HW4P2/runs/yiyan_fbank_Transformer_ENC-6-8_DEC-6-8_256_2048_AdamW_CosineAnnealing_token_1k_20241129-142641/checkpoints/checkpoint-epochfull-43-2.pth"
checkpoint_path = "/ix1/hkarim/yip33/HW4P2/runs/yiyan_fbank_Transformer_ENC-6-8_DEC-6-8_256_2048_AdamW_CosineAnnealing_token_1k_20241205-222241/checkpoints/checkpoint-epochfull-104-2.pth"

##############################################
##############################################
##############################################
train_dataset = torch.load("/ix1/hkarim/yip33/kaggle_dataset/hw4_data/pt_feat_70/1k_train_dataset.pt")
val_dataset = torch.load("/ix1/hkarim/yip33/kaggle_dataset/hw4_data/pt_feat_70/1k_val_dataset.pt")
test_dataset = torch.load("/ix1/hkarim/yip33/kaggle_dataset/hw4_data/pt_feat_70/1k_test_dataset.pt")

##############################################
##############################################
##############################################
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


max_train_feat, max_train_transcript = verify_dataset(train_loader, config['train_partition'])
max_val_feat, max_val_transcript     = verify_dataset(val_loader,   config['val_partition'])
max_test_feat, max_test_transcript   = verify_dataset(test_loader,  config['test_partition'])
MAX_SPEECH_LEN = max(max_train_feat, max_val_feat, max_test_feat)
MAX_TRANS_LEN  = max(max_train_transcript, max_val_transcript)

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
model = model.to(device)

model, _, _ = load_checkpoint(checkpoint_path=checkpoint_path,
                                              model = model,
                                              embedding_load=True,
                                              encoder_load=True,
                                              decoder_load=True,
                                              )

predictions = test_step(
        model,
        test_loader=test_loader,
        tokenizer=Tokenizer,
        device=device,
)




# Sample list

# Specify the CSV file path
csv_file_path = "submission_predict_high_meow_meow.csv"

# Write the list to the CSV with index as the first column
with open(csv_file_path, mode="w", newline="") as file:
    writer = csv.writer(file)
    # Write the header
    writer.writerow(["Index", "Labels"])
    # Write list items with index
    for idx, item in enumerate(predictions):
        writer.writerow([idx, item])
print(f"CSV file saved to {csv_file_path}")