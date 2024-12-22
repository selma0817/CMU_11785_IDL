import torch
import torch.nn as nn
from torchsummaryX import summary

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

torch.cuda.empty_cache()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Device: ", device)
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

torch.cuda.empty_cache()

class LockedDropout(torch.nn.Module):
    def __init__(self, dropout_prob):
        super(LockedDropout, self).__init__()
        self.dropout_prob = dropout_prob

    def forward(self, x):
        if not self.training or self.dropout_prob == 0:
            return x
        x, x_lens = pad_packed_sequence(x, batch_first=True)
        # Create a dropout mask with the same size as the feature dimension
        dropout_mask = torch.rand((x.size(0), 1, x.size(2)), device=x.device) >= self.dropout_prob
        dropout_mask = dropout_mask.float() / (1 - self.dropout_prob)
        x = x * dropout_mask
        x_packed = pack_padded_sequence(x, x_lens, batch_first=True, enforce_sorted=False)
        return x_packed



class PermuteBlock(torch.nn.Module):
    def forward(self, x):
        return x.transpose(1, 2)
    
class pBLSTM(torch.nn.Module):

    '''
    Pyramidal BiLSTM
    Read the write up/paper and understand the concepts and then write your implementation here.

    At each step,
    1. Pad your input if it is packed (Unpack it)
    2. Reduce the input length dimension by concatenating feature dimension
        (Tip: Write down the shapes and understand)
        (i) How should  you deal with odd/even length input?
        (ii) How should you deal with input length array (x_lens) after truncating the input?
    3. Pack your input
    4. Pass it into LSTM layer

    To make our implementation modular, we pass 1 layer at a time.
    '''

    def __init__(self, input_size, hidden_size):
        super(pBLSTM, self).__init__()

        # self.blstm = nn.LSTM(input_size=input_size*2, hidden_size=hidden_size, num_layers=1, bidirectional=True, dropout=dropout)
        self.blstm = nn.LSTM(input_size=input_size*2, hidden_size=hidden_size, num_layers=1, bidirectional=True)
        # TODO: Initialize a single layer bidirectional LSTM with the given input_size and hidden_size

    def forward(self, x_packed, f_summary=False): # x_packed is a PackedSequence

        # TODO: Pad Packed Sequence

        # Call self.trunc_reshape() which downsamples the time steps of x and increases the feature dimensions as mentioned above
        # self.trunc_reshape will return 2 outputs. What are they? Think about what quantites are changing.
        # TODO: Pack Padded Sequence. What output(s) would you get?
        # TODO: Pass the sequence through bLSTM

        # What do you return?
        x, x_lens = pad_packed_sequence(x_packed, batch_first=True)
        x, x_lens = self.trunc_reshape(x, x_lens)
        
        x_packed = pack_padded_sequence(x, x_lens, batch_first=True, enforce_sorted=False)

        if f_summary:
            x_packed, _ = self.blstm(x)
        else:
            x_packed, _ = self.blstm(x_packed)

        return x_packed

    def trunc_reshape(self, x, x_lens):
        # TODO: If you have odd number of timesteps, how can you handle it?
        if x.size(1) % 2 != 0:
            x = x[:, :-1, :] # (Hint: You can exclude them)
        # TODO: Reshape x. When reshaping x, you have to reduce number of timesteps by a downsampling factor while increasing number of features by the same factor
        B, T, F = x.size()
        x = x.view(B, T // 2, F * 2)  # (batch, time // 2, feature * 2)

        # TODO: Reduce lengths by the same downsampling factor
        x_lens = x_lens // 2

        return x, x_lens
    
class Encoder(torch.nn.Module):
    '''
    The Encoder takes utterances as inputs and returns latent feature representations
    '''
    def __init__(self, input_size, encoder_hidden_size):
        super(Encoder, self).__init__()


        
        self.embedding = nn.Sequential(
            PermuteBlock(),
            nn.Conv1d(in_channels=input_size, out_channels=encoder_hidden_size//2, kernel_size=3, padding=1),
            nn.BatchNorm1d(encoder_hidden_size//2),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Conv1d(encoder_hidden_size//2, out_channels=encoder_hidden_size//2, kernel_size=3, padding=1),
            nn.BatchNorm1d(encoder_hidden_size//2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),

            nn.Conv1d(encoder_hidden_size//2, out_channels=encoder_hidden_size//2, kernel_size=3, padding=1),
            nn.BatchNorm1d(encoder_hidden_size//2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            PermuteBlock(),
        )
        
        
        
        #TODO: You can use CNNs as Embedding layer to extract features. Keep in mind the Input dimensions and expected dimension of Pytorch CNN.

        self.pBLSTMs = torch.nn.Sequential( # How many pBLSTMs are required?
            # TODO: Fill this up with pBLSTMs - What should the input_size be?
            # Hint: You are downsampling timesteps by a factor of 2, upsampling features by a factor of 2 and the LSTM is bidirectional)
            # Optional: Dropout/Locked Dropout after each pBLSTM (Not needed for early submission)
            # https://github.com/salesforce/awd-lstm-lm/blob/dfd3cb0235d2caf2847a4d53e1cbd495b781b5d2/locked_dropout.py#L5
            # ...
            # ...
            pBLSTM(input_size=encoder_hidden_size//2, hidden_size=encoder_hidden_size//2),
            LockedDropout(dropout_prob=0.1),
            pBLSTM(input_size=encoder_hidden_size, hidden_size=encoder_hidden_size//2),
            LockedDropout(dropout_prob=0.1),
            # pBLSTM(input_size=64*4, hidden_size=96),
            
        )
        # self.locked_dropout = LockedDropout(dropout_prob=0.2)

    def forward(self, x, x_lens, f_summary=False):
        # Where are x and x_lens coming from? The dataloader
        # Remember the number of output(s) each function returns
        # print('shape of x before embed:', x.shape)
        #x = x.transpose(1, 2)
        x = self.embedding(x)
        #x = x.transpose(1, 2)
        # print('shape of x after embed:', x.shape)
        x_packed = pack_padded_sequence(x, x_lens, batch_first=True, enforce_sorted=False)
        
        
        x_packed = self.pBLSTMs(x_packed)

        # x, x_lens = pad_packed_sequence(x_packed, batch_first=True)
        if f_summary:
            x = x_packed
        else:
            x, x_lens = pad_packed_sequence(x_packed, batch_first=True)
        
        encoder_outputs = x
        # print('shape of encoder_outputs:', encoder_outputs.shape)
        encoder_lens = x_lens
        
        return encoder_outputs, encoder_lens
    
class Decoder(torch.nn.Module):

    def __init__(self, embed_size, output_size=41):
        super().__init__()

        self.mlp = torch.nn.Sequential(
            PermuteBlock(), torch.nn.BatchNorm1d(embed_size), PermuteBlock(), # shape: (batch, time, feature), so normalize along time dimension for one feature
            #TODO define your MLP arch. Refer HW1P2
            #Use Permute Block before and after BatchNorm1d() to match the size
            nn.Linear(embed_size, 1024),
            PermuteBlock(), nn.BatchNorm1d(1024), PermuteBlock(),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 2048),
            PermuteBlock(), nn.BatchNorm1d(2048), PermuteBlock(),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(2048, 2048),
            PermuteBlock(), nn.BatchNorm1d(2048), PermuteBlock(),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(2048, 1024),
            PermuteBlock(), nn.BatchNorm1d(1024), PermuteBlock(),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 512),
            PermuteBlock(), nn.BatchNorm1d(512), PermuteBlock(),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, output_size),    
        )
        self.softmax = torch.nn.LogSoftmax(dim=2)

    def forward(self, encoder_out):
        #TODO call your MLP
        #TODO Think what should be the final output of the decoder for the classification
        # print('shape of encoder_out:', encoder_out.shape)
        out = self.mlp(encoder_out)
        out = self.softmax(out)

        return out



# add noise as regularization
# i have added time and frequency masking to dataset so no 
# need to add here

class GaussianNoise(torch.nn.Module):
    def __init__(self, noise_level = 0.005):
        super().__init__()
        self.noise_level = noise_level
    def forward(self, x):
        noise = torch.randn_like(x) * self.noise_level
        return x + noise
    
class ASRModel(torch.nn.Module):

    def __init__(self, input_size, embed_size= 1024, output_size= len(PHONEMES)):
        super().__init__()
        self.add_gaussian_noise = GaussianNoise()

        self.augmentations  = torch.nn.Sequential(
            PermuteBlock(), self.add_gaussian_noise, PermuteBlock(),
        )
        self.encoder = Encoder(input_size=input_size, encoder_hidden_size=embed_size)
        self.decoder = Decoder(embed_size=embed_size, output_size=output_size)



    def forward(self, x, lengths_x):

        if self.training:
            x = self.augmentations(x)

        encoder_out, encoder_lens   = self.encoder(x, lengths_x)
        decoder_out                 = self.decoder(encoder_out)

        return decoder_out, encoder_lens