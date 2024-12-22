
import torchaudio.functional as aF
import torch
def calculateMetrics(reference, hypothesis):
        # sentence-level edit distance
        dist = aF.edit_distance(reference, hypothesis)
        # split sentences into words
        ref_words = reference.split()
        hyp_words = hypothesis.split()
        # compute edit distance
        dist = aF.edit_distance(ref_words, hyp_words)
        # calculate WER
        wer = dist / len(ref_words)
        # convert sentences into character sequences
        ref_chars = list(reference)
        hyp_chars = list(hypothesis)
        # compute edit distance
        dist = aF.edit_distance(ref_chars, hyp_chars)
        # calculate CER
        cer = dist / len(ref_chars)
        return dist, wer * 100, cer * 100


def calculateBatchMetrics(predictions, y, y_len, tokenizer):
    '''
    Calculate levenshtein distance, WER, CER for a batch
    predictions (Tensor) : the model predictions
    y (Tensor) : the target transcript
    y_len (Tensor) : Length of the target transcript (non-padded positions)
    '''
    batch_size, _  = predictions.shape
    dist, wer, cer = 0., 0., 0.
    for batch_idx in range(batch_size):

        # trim predictons upto the EOS_TOKEN
        pad_indices = torch.where(predictions[batch_idx] == tokenizer.EOS_TOKEN)[0]
        lowest_pad_idx = pad_indices.min().item() if pad_indices.numel() > 0 else len(predictions[batch_idx])
        pred_trimmed = predictions[batch_idx, :lowest_pad_idx]

        # trim target upto EOS_TOKEN
        y_trimmed   = y[batch_idx, 0 : y_len[batch_idx]-1]

        # decodes
        pred_string  = tokenizer.decode(pred_trimmed)
        y_string     = tokenizer.decode(y_trimmed)

        # calculate metrics and update
        curr_dist, curr_wer, curr_cer = calculateMetrics(y_string, pred_string)
        dist += curr_dist
        wer  += curr_wer
        cer  += curr_cer

    # average by batch sizr
    dist /= batch_size
    wer  /= batch_size
    cer  /= batch_size
    return dist, wer, cer, y_string, pred_string