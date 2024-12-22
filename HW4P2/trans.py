import torch
import math
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from model_mask import PadMask, CausalMask
import numpy as np
from typing import Literal, List, Optional, Any, Dict, Tuple

device = "cuda" if torch.cuda.is_available() else "cpu"

class PositionalEncoding(torch.nn.Module):
    ''' Position Encoding from Attention Is All You Need Paper '''

    def __init__(self, d_model, max_len=512):
        super().__init__()

        # Initialize a tensor to hold the positional encodings
        pe          = torch.zeros(max_len, d_model)

        # Create a tensor representing the positions (0 to max_len-1)
        position    = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # Calculate the division term for the sine and cosine functions
        # This term creates a series of values that decrease geometrically, used to generate varying frequencies for positional encodings
        div_term    = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # Compute the positional encodings using sine and cosine functions
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Reshape the positional encodings tensor and make it a buffer
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
      return x + self.pe[:, :x.size(1)]
    

# 2-Layer BiLSTM
class BiLSTMEmbedding(nn.Module):
    def __init__(self, input_dim, output_dim, dropout):
        super(BiLSTMEmbedding, self).__init__()
        self.bilstm = nn.LSTM(
                input_dim, output_dim // 2,
                num_layers=1,
                batch_first=True,
                bidirectional=True,
                dropout=dropout
        )

    def forward(self, x,  x_len):
        """
        Args:
            x.    : Input tensor (batch_size, seq_len, input_dim)
            x_len : Non-padded lengths (batch_size)
        Returns:
            Output tensor (batch_size, seq_len, output_dim)
        """
        # BiLSTM expects (batch_size, seq_len, input_dim)
        # Pack the padded sequence to avoid computing over padded tokens
        packed_input = pack_padded_sequence(x, x_len.cpu(), batch_first=True, enforce_sorted=False)
        # Pass through the BiLSTM
        packed_output, _ = self.bilstm(packed_input)
        # Unpack the sequence to restore the original padded shape
        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        return output

### DO NOT MODIFY

class Conv2DSubsampling(torch.nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0.0, time_stride=2, feature_stride=2):
        """
        Conv2dSubsampling module that can selectively apply downsampling
        for time and feature dimensions, and calculate cumulative downsampling factor.
        Args:
            time_stride (int): Stride along the time dimension for downsampling.
            feature_stride (int): Stride along the feature dimension for downsampling.
        """
        super(Conv2DSubsampling, self).__init__()

        # decompose to get effective stride across two layers
        tstride1, tstride2 = self.closest_factors(time_stride)
        fstride1, fstride2 = self.closest_factors(feature_stride)

        self.feature_stride = feature_stride
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, output_dim, kernel_size=3, stride=(tstride1, fstride1)),
            torch.nn.ReLU(),
            torch.nn.Conv2d(output_dim, output_dim, kernel_size=3, stride=(tstride2, fstride2)),
            torch.nn.ReLU(),
        )
        self.time_downsampling_factor = tstride1 * tstride2
        # Calculate output dimension for the linear layer
        conv_out_dim = (input_dim - (3 - 1) - 1) // fstride1 + 1
        conv_out_dim = (conv_out_dim - (3 - 1) - 1) // fstride2 + 1
        conv_out_dim = output_dim * conv_out_dim
        self.out = torch.nn.Sequential(
            torch.nn.Linear(conv_out_dim, output_dim),
            torch.nn.Dropout(dropout)
        )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_dim).
            x_mask (torch.Tensor): Optional mask for the input tensor.

        Returns:
            torch.Tensor: Downsampled output of shape (batch_size, new_seq_len, output_dim).
        """
        x = x.unsqueeze(1)  # Add a channel dimension for Conv2D
        x = self.conv(x)
        b, c, t, f = x.size()
        x = self.out(x.transpose(1, 2).contiguous().view(b, t, c * f))
        return x

    def closest_factors(self, n):
        factor = int(n**0.5)
        while n % factor != 0:
            factor -= 1
        # Return the factor pair
        return max(factor, n // factor), min(factor, n // factor)



class SpeechEmbedding(nn.Module):
    def __init__(self, input_dim, output_dim, time_stride, feature_stride, dropout):
        super(SpeechEmbedding, self).__init__()

        self.cnn = Conv2DSubsampling(input_dim, output_dim, dropout=dropout, time_stride=time_stride, feature_stride=feature_stride)
        self.blstm = BiLSTMEmbedding(output_dim, output_dim, dropout)
        self.time_downsampling_factor = self.cnn.time_downsampling_factor

    def forward(self, x, x_len, use_blstm: bool = False):
        """
        Args:
            x    : Input tensor (batch_size, seq_len, input_dim)
            x_len : Non-padded lengths (batch_size)
        Returns:
            Output tensor (batch_size, seq_len // stride, output_dim)
        """
        # First, apply Conv2D subsampling
        x = self.cnn(x)
        # Adjust sequence length based on downsampling factor
        x_len = torch.ceil(x_len.float() / self.time_downsampling_factor).int()
        x_len = x_len.clamp(max=x.size(1))

        # Apply BiLSTM if requested
        if use_blstm:
            x = self.blstm(x, x_len)

        return x, x_len


class EncoderLayer(nn.Module):

    def __init__(self, d_model, num_heads, d_ff, dropout):

        super(EncoderLayer, self).__init__()

        self.pre_norm = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(embed_dim = d_model, num_heads = num_heads, dropout = dropout, batch_first=True)
        self.ffn1 = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        ) # Hint: Linear layer - GELU - dropout - Linear layer
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, pad_mask):
        # Step 1: Apply pre-normalization
        ''' TODO '''
        pre_norm_x = self.pre_norm(x)

        # Step 2: Self-attention with with dropout, and with residual connection
        ''' TODO '''
        attn_output, _ = self.self_attn(pre_norm_x, pre_norm_x, pre_norm_x, key_padding_mask=pad_mask)
        x = x + self.dropout(attn_output)

        # Step 3: Apply normalization
        ''' TODO '''
        norm1_x = self.norm1(x)

        # Step 4: Apply Feed-Forward Network (FFN) with dropout, and residual connection
        ''' TODO '''
        ffn_output = self.ffn1(norm1_x)
        x = x + self.dropout(ffn_output)

        # Step 5: Apply normalization after FFN
        ''' TODO '''
        x = self.norm2(x)

        return x, pad_mask



class Encoder(nn.Module):
    def __init__(self,
                 num_layers,
                 d_model,
                 num_heads,
                 d_ff,
                 max_len,
                 target_vocab_size,
                 dropout=0.1):

        super(Encoder, self).__init__()

        self.pos_encoding = PositionalEncoding(d_model=d_model, max_len=max_len)
        self.dropout =  nn.Dropout(dropout)
        self.enc_layers =  nn.ModuleList([
            EncoderLayer(d_model=d_model, num_heads=num_heads, d_ff=d_ff, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.after_norm =  nn.LayerNorm(d_model)
        self.ctc_head   =  nn.Linear(d_model, target_vocab_size)

    def forward(self, x, x_len):

        # Step 1: Create padding mask for inputs
        ''' TODO '''
        # pad_mask = PadMask(x_len, x.size(1))
        pad_mask = PadMask(x, x_len).to(device)

        # Step 2: Apply positional encoding
        ''' TODO '''
        x = self.pos_encoding(x)

        # Step 3: Apply dropout
        ''' TODO '''
        x_res = self.dropout(x)

        # Step 4: Add the residual connection (before passing through layers)
        ''' TODO '''
        x  = x + x_res

        # Step 5: Pass through all encoder layers
        ''' TODO '''
        for enc_layer in self.enc_layers:
            x,_ = enc_layer(x, pad_mask)

        # Step 6: Apply final normalization
        ''' TODO '''
        x = self.after_norm(x)

        # Step 7: Pass a branch through the CTC head
        ''' TODO '''
        x_ctc = self.ctc_head(x)

        return x, x_len, x_ctc.log_softmax(2).permute(1, 0, 2)
    

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()

        # @TODO: fill in the blanks appropriately (given the modules above)
        self.mha1       = nn.MultiheadAttention(embed_dim = d_model, num_heads = num_heads, dropout = dropout,  batch_first=True)
        self.mha2       = nn.MultiheadAttention(embed_dim = d_model, num_heads = num_heads, dropout = dropout, batch_first=True)
        self.ffn        =  nn.Sequential(
                                nn.Linear(d_model, d_ff),
                                nn.GELU(),
                                nn.Dropout(dropout),
                                nn.Linear(d_ff, d_model),
                            )
        self.identity   = nn.Identity()
        self.pre_norm   = nn.LayerNorm(d_model)
        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)
        self.layernorm3 = nn.LayerNorm(d_model)
        self.dropout1   = nn.Dropout(dropout)
        self.dropout2   = nn.Dropout(dropout)
        self.dropout3   = nn.Dropout(dropout)

    def forward(self, padded_targets, enc_output, pad_mask_enc, pad_mask_dec, slf_attn_mask):

        #   Step 1: Self Attention
        #   (1) pass through the Multi-Head Attention (Hint: you need to store weights here as part of the return value)
        #   (2) add dropout
        #   (3) residual connections
        #   (4) layer normalization

        x_norm = self.pre_norm(padded_targets)

        ''' TODO '''
        #mha1_attn, mha1_attn_weights = self.mha1()
        # need 2 mask
        mha1_attn, mha1_attn_weights = self.mha1(x_norm, x_norm, x_norm, attn_mask=slf_attn_mask, key_padding_mask=pad_mask_dec)
        x = padded_targets + self.dropout1(mha1_attn)
        x = self.layernorm1(x)

        #   Step 2: Cross Attention
        #   (1) pass through the Multi-Head Attention (Hint: you need to store weights here as part of the return value)
              #  think about if key,value,query here are the same as the previous one?
        #   (2) add dropout
        #   (3) residual connections
        #   (4) layer normalization
        if enc_output is None:
            mha2_output       = self.identity(padded_targets)
            mha2_attn_weights = torch.zeros_like(mha1_attn_weights)
        else:
            ''' TODO '''
            mha2_output,  mha2_attn_weights = self.mha2(x, enc_output, enc_output, key_padding_mask=pad_mask_enc)
            x = x + self.dropout2(mha2_output)
        #   Step 3: Feed Forward Network
        #   (1) pass through the FFN
        #   (2) add dropout
        #   (3) residual connections
        #   (4) layer normalization
        ''' TODO '''
        ffn_output = self.ffn(x)
        x = x + self.dropout3(ffn_output)
        ffn_output = self.layernorm3(x)


        return ffn_output, mha1_attn_weights, mha2_attn_weights


class Decoder(torch.nn.Module):
    def __init__(self,
                 num_layers,
                 d_model,
                 num_heads,
                 d_ff, dropout,
                 max_len,
                 target_vocab_size):

        super().__init__()

        self.max_len        = max_len
        self.num_layers     = num_layers
        self.num_heads      = num_heads

        # use torch.nn.ModuleList() with list comprehension looping through num_layers
        # @NOTE: think about what stays constant per each DecoderLayer (how to call DecoderLayer)
        self.dec_layers = nn.ModuleList([
            DecoderLayer(d_model=d_model, num_heads=num_heads, d_ff=d_ff, dropout=dropout) for _ in range(num_layers)
        ])

        self.target_embedding       = nn.Embedding(target_vocab_size, d_model)  # use torch.nn.Embedding
        self.positional_encoding    = PositionalEncoding(d_model=d_model, max_len=max_len)
        self.final_linear           = nn.Linear(d_model, target_vocab_size)
        self.dropout                = dropout


    def forward(self, padded_targets, target_lengths, enc_output, enc_input_lengths):

        # Processing targets
        # create a padding mask for the padded_targets with <PAD_TOKEN>
        # creating an attention mask for the future subsequences (look-ahead mask)
        # computing embeddings for the target sequence
        # computing Positional Encodings with the embedded targets and apply dropout

        pad_mask_dec = None
        if target_lengths is not None:
            pad_mask_dec = PadMask(padded_input=padded_targets, input_lengths=target_lengths).to(padded_targets.device)
        causal_mask = CausalMask(input_tensor=padded_targets).to(padded_targets.device)

        # Step1:  Apply the embedding
        ''' TODO '''
        x = self.target_embedding(padded_targets)

        # Step2:  Apply positional encoding
        ''' TODO '''
        x = self.positional_encoding(x)


        # Step3:  Create attention mask to ignore padding positions in the input sequence during attention calculation
        ''' TODO '''
        pad_mask_enc = None
        if enc_output is not None:
            pad_mask_enc  = PadMask(enc_output, enc_input_lengths).to(device)

        # Step4: Pass through decoder layers
        # @NOTE: store your mha1 and mha2 attention weights inside a dictionary
        # @NOTE: you will want to retrieve these later so store them with a useful name
        ''' TODO '''
        runnint_att = {}
        for i in range(self.num_layers):
            x, runnint_att['layer{}_dec_self'.format(i + 1)], runnint_att['layer{}_dec_cross'.format(i + 1)] = self.dec_layers[i](
                x, enc_output, pad_mask_enc, pad_mask_dec, causal_mask
            )

        # Step5: linear layer (Final Projection) for next character prediction
        ''' TODO '''
        seq_out = self.final_linear(x)

        return seq_out, runnint_att


    def recognize_greedy_search(self, enc_output, enc_input_lengths, tokenizer):
        ''' passes the encoder outputs and its corresponding lengths through autoregressive network
            @NOTE: You do not need to make changes to this method.
        '''
        # start with the <SOS> token for each sequence in the batch
        batch_size = enc_output.size(0)
        target_seq = torch.full((batch_size, 1), tokenizer.SOS_TOKEN, dtype=torch.long).to(enc_output.device)
        finished = torch.zeros(batch_size, dtype=torch.bool).to(enc_output.device)

        for _ in range(self.max_len):

            seq_out, runnint_att = self.forward(target_seq, None, enc_output, enc_input_lengths)
            logits = torch.nn.functional.log_softmax(seq_out[:, -1], dim=1)

            # selecting the token with the highest probability
            # @NOTE: this is the autoregressive nature of the network!
            # appending the token to the sequence
            # checking if <EOS> token is generated
            # or opration, if both or one of them is true store the value of the finished sequence in finished variable
            # end if all sequences have generated the EOS token
            next_token = logits.argmax(dim=-1).unsqueeze(1)
            target_seq = torch.cat([target_seq, next_token], dim=-1)
            eos_mask = next_token.squeeze(-1) == tokenizer.EOS_TOKEN
            finished |= eos_mask
            if finished.all(): break

        # remove the initial <SOS> token and pad sequences to the same length
        target_seq = target_seq[:, 1:]
        max_length = target_seq.size(1)
        target_seq = torch.nn.functional.pad(target_seq,
            (0, self.max_len - max_length), value=tokenizer.PAD_TOKEN)

        return target_seq

    def recognize_beam_search(self, enc_output, enc_input_lengths, tokenizer, beam_width=5):
      # TODO Beam Decoding
        batch_size = enc_output.size(0)
        #target_seq = torch.full((batch_size, 1), tokenizer.SOS_TOKEN, dtype=torch.long).to(enc_output.device)
        finished = torch.zeros(batch_size, dtype=torch.bool).to(enc_output.device)
        #beams = [[(tokenizer.SOS_TOKEN, 0)]for _ in batch_size]
        beams = [ [ ( [tokenizer.SOS_TOKEN], 0.0 ) ] for _ in range(batch_size) ]
        for _ in range(self.max_len):
            new_beams = []
            all_finished = True
            for i in range(batch_size):
                current_beams = beams[i]
                temp_beams = []
                for seq, score in current_beams:
                    if seq[-1] == tokenizer.EOS_TOKEN:
                    # If EOS is already generated, keep the sequence as is
                        temp_beams.append((seq, score))
                        continue  # Skip expanding this beam



class Transformer(nn.Module):
    def __init__(self,
                 input_dim,
                time_stride,
                feature_stride,
                embed_dropout,

                d_model,
                enc_num_layers,
                enc_num_heads,
                speech_max_len,
                enc_dropout,

                dec_num_layers,
                dec_num_heads,
                d_ff,
                dec_dropout,

                target_vocab_size,
                trans_max_len):

      super(Transformer, self).__init__()

      self.embedding = SpeechEmbedding(input_dim, d_model, time_stride, feature_stride, embed_dropout)
      speech_max_len = int(np.ceil(speech_max_len/self.embedding.time_downsampling_factor))

      self.encoder   = Encoder(num_layers=enc_num_layers,
                               d_model=d_model,
                               num_heads=enc_num_heads,
                               d_ff=d_ff,
                               max_len=speech_max_len,
                               target_vocab_size=target_vocab_size,
                               dropout=enc_dropout)

      self.decoder = Decoder(num_layers=dec_num_layers,
                             d_model=d_model, 
                             num_heads=dec_num_heads,
                             d_ff=d_ff, 
                             dropout=dec_dropout, 
                             max_len=trans_max_len,
                             target_vocab_size=target_vocab_size)


    def forward(self, padded_input, input_lengths, padded_target, target_lengths, mode:Literal['full', 'dec_cond_lm', 'dec_lm']='full'):
        '''DO NOT MODIFY'''
        if mode == 'full': # Full transformer training
            encoder_output, encoder_lengths          = self.embedding(padded_input, input_lengths, use_blstm=False)
            encoder_output, encoder_lengths, ctc_out = self.encoder(encoder_output, encoder_lengths)
        if mode == 'dec_cond_lm': # Training Decoder as a conditional LM
            encoder_output, encoder_lengths   = self.embedding(padded_input, input_lengths, use_blstm=True)
            ctc_out = None
        if mode == 'dec_lm': # Training Decoder as an LM
            encoder_output, encoder_lengths, ctc_out = None, None, None

        # passing Encoder output through Decoder
        output, attention_weights = self.decoder(padded_target, target_lengths, encoder_output, encoder_lengths)
        return output, attention_weights, ctc_out


    def recognize(self, inp, inp_len, tokenizer, mode:Literal['full', 'dec_cond_lm', 'dec_lm'], strategy:str='greedy'):
        """ sequence-to-sequence greedy search -- decoding one utterance at a time """
        '''DO NOT MODIFY'''
        if mode == 'full':
            encoder_output, encoder_lengths          = self.embedding(inp, inp_len, use_blstm=False)
            encoder_output, encoder_lengths, ctc_out = self.encoder(encoder_output, encoder_lengths)
        
        if mode == 'dec_cond_lm':
            encoder_output, encoder_lengths,  = self.embedding(inp, inp_len, use_blstm=True)
            ctc_out = None
      
        if mode == 'dec_lm':
            encoder_output, encoder_lengths, ctc_out = None, None, None
        
        if strategy =='greedy':
          out = self.decoder.recognize_greedy_search(encoder_output, encoder_lengths, tokenizer=tokenizer)
        elif strategy == 'beam':
          out = self.decoder.recognize_beam_search(encoder_output, encoder_lengths, tokenizer=tokenizer, beam_width=5)
        return out