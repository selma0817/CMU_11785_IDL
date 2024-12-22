c# ARPABET PHONEME MAPPING
# DO NOT CHANGE
import numpy as np
import os
import torch
import torch.nn as nn
from torchsummaryX import summary
import sklearn
import gc
import zipfile
import pandas as pd
from tqdm.auto import tqdm
import os
import datetime
import wandb
import torchaudio.transforms as T 
from torch.nn.utils.rnn import pad_sequence

from network_ars import PermuteBlock
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


class AudioDataset(torch.utils.data.Dataset):

    # For this homework, we give you full flexibility to design your data set class.
    # Hint: The data from HW1 is very similar to this HW

    def __init__(self, root, phonemes = PHONEMES, partition= "train-clean-100", limit=None):
        '''
        Initializes the dataset.

        INPUTS: What inputs do you need here?
        '''

        self.partition = partition
        self.mfcc_dir = os.path.join(root, partition, 'mfcc')
        self.transcript_dir = os.path.join(root, partition, 'transcript')

        self.mfcc_names = sorted(os.listdir(self.mfcc_dir))
        self.transcript_names = sorted(os.listdir(self.transcript_dir))

        self.PHONEMES = phonemes

        assert len(self.mfcc_names) == len(self.transcript_names)

        self.length = len(self.transcript_names)
        self.mfccs, self.transcripts = [], []
        
        # Iterate through mfccs and transcripts
        for i in range(len(self.mfcc_names)):
            # normalize
            mfcc        = np.load(os.path.join(self.mfcc_dir, self.mfcc_names[i]))
            mfcc        = (mfcc - np.mean(mfcc, axis=0)) / (np.std(mfcc, axis=0) + 1e-8)
            transcript  = np.load(os.path.join(self.transcript_dir, self.transcript_names[i]))
            transcript  = transcript[1:-1]
            self.mfccs.append(mfcc)
            self.transcripts.append(transcript)

        # map to integer using index of phoneme
        self.transcripts = [[self.PHONEMES.index(phoneme) for phoneme in transcript] for transcript in self.transcripts]
        assert len(self.mfccs) == len(self.transcripts)
        
        '''
        You may decide to do this in __getitem__ if you wish.
        However, doing this here will make the __init__ function take the load of
        loading the data, and shift it away from training.
        '''


    def __len__(self):

        return self.length

    def __getitem__(self, ind):
        '''
        RETURN THE MFCC COEFFICIENTS AND ITS CORRESPONDING LABELS

        If you didn't do the loading and processing of the data in __init__,
        do that here.

        Once done, return a tuple of features and labels.
        '''
        mfcc        = torch.tensor(self.mfccs[ind], dtype=torch.float32)
        transcript  = torch.tensor(self.transcripts[ind], dtype=torch.int32)
        return mfcc, transcript


    def collate_fn(self,batch):
        '''
        1.  Extract the features and labels from 'batch'
        2.  We will additionally need to pad both features and labels,
            look at pytorch's docs for pad_sequence
        3.  This is a good place to perform transforms, if you so wish.
            Performing them on batches will speed the process up a bit.
        4.  Return batch of features, labels, lenghts of features,
            and lengths of labels.
        '''
        # batch of input mfcc coefficients and output phonemes
        batch_mfcc, batch_transcript = zip(*batch)

        # HINT: CHECK OUT -> pad_sequence (imported above)
        # Also be sure to check the input format (batch_first)
        batch_mfcc_pad = pad_sequence(batch_mfcc, batch_first=True, padding_value=0)
        lengths_mfcc = np.array([mfcc.shape[0] for mfcc in batch_mfcc])

        batch_transcript_pad = pad_sequence(batch_transcript, batch_first=True)
        lengths_transcript =  np.array([transcript.shape[0] for transcript in batch_transcript])

        # You may apply some transformation, Time and Frequency masking, here in the collate function;
        # Food for thought -> Why are we applying the transformation here and not in the __getitem__?
        #                  -> Would we apply transformation on the validation set as well?
        #                  -> Is the order of axes / dimensions as expected for the transform functions?

        # Return the following values: padded features, padded labels, actual length of features, actual length of the labels
        audio_transforms = nn.Sequential(
            PermuteBlock(),
            T.FrequencyMasking(freq_mask_param=5),
            T.TimeMasking(time_mask_param=100),
            PermuteBlock(),
        )
        # only add masking to train-data not on val data
        if self.partition == "train-clean-100":
            batch_mfcc_pad = audio_transforms(batch_mfcc_pad)
            
        return batch_mfcc_pad, batch_transcript_pad, torch.tensor(lengths_mfcc), torch.tensor(lengths_transcript)

# Test Dataloader
class AudioDatasetTest(torch.utils.data.Dataset):
    # For this homework, we give you full flexibility to design your data set class.
    # Hint: The data from HW1 is very similar to this HW

    def __init__(self, root, phonemes = PHONEMES, partition= "test-clean", limit=None):
        '''
        Initializes the dataset.

        INPUTS: What inputs do you need here?
        '''

        # Load the directory and all files in them

        self.mfcc_dir = os.path.join(root, partition, 'mfcc')

        self.mfcc_names = sorted(os.listdir(self.mfcc_dir))

        self.PHONEMES = phonemes


        self.length = len(self.mfcc_names)

        self.mfccs = []
        
        if limit is None:
            limit = len(self.mfcc_names)
        for i in range(limit):
            mfcc        = np.load(os.path.join(self.mfcc_dir, self.mfcc_names[i]))
            mfcc        = (mfcc - np.mean(mfcc, axis=0)) / np.std(mfcc, axis=0)
            self.mfccs.append(mfcc)

        # CREATE AN ARRAY OF ALL FEATUERS
        # Use Cepstral Normalization of mfcc
        
        '''
        You may decide to do this in __getitem__ if you wish.
        However, doing this here will make the __init__ function take the load of
        loading the data, and shift it away from training.
        '''


    def __len__(self):

        return self.length

    def __getitem__(self, ind):
        '''
        RETURN THE MFCC COEFFICIENTS

        If you didn't do the loading and processing of the data in __init__,
        do that here.

        Once done, return a tuple of features.
        '''
        mfcc = torch.tensor(self.mfccs[ind], dtype=torch.float32)
        return mfcc

    def collate_fn(self,batch):
        '''
        1.  Extract the features and labels from 'batch'
        2.  We will additionally need to pad both features and labels,
            look at pytorch's docs for pad_sequence
        3.  This is a good place to perform transforms, if you so wish.
            Performing them on batches will speed the process up a bit.
        4.  Return batch of features, labels, lenghts of features,
            and lengths of labels.
        '''
        # batch of input mfcc coefficients
        batch_mfcc = batch

        batch_mfcc_pad = pad_sequence(batch_mfcc, batch_first=True)
        lengths_mfcc = np.array([mfcc.shape[0] for mfcc in batch_mfcc])

        return batch_mfcc_pad, torch.tensor(lengths_mfcc)