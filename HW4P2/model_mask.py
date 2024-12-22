import torch
import matplotlib.pyplot as plt

def PadMask(padded_input, input_lengths=None, pad_idx=None):
    """ Create a mask to identify non-padding positions.

    Args:
        padded_input: The input tensor with padding, shape (N, T, ...) or (N, T).
        input_lengths: Optional, the actual lengths of each sequence before padding, shape (N,).
        pad_idx: Optional, the index used for padding tokens.

    Returns:
        A mask tensor with shape (N, T), where padding positions are marked with 1 and non-padding positions are marked with 0.
    """

    # If input is a 2D tensor (N, T), add an extra dimension
    # print('padded_input shape', padded_input.shape)
    if padded_input.dim() == 2:
        padded_input = padded_input.unsqueeze(-1)

    # TODO: Initialize the mask variable here. What type should it be and how should it be initialized?

    if input_lengths is not None:
        # TODO: Use the provided input_lengths to create the mask.
        N, T, _ = padded_input.shape
        mask = torch.zeros((N, T), dtype=torch.bool)
        for i in range(N):
            mask[i, input_lengths[i]:] = 1
            # TODO: Set non-padding positions to False based on input_lengths
            # pass
    else:
        # TODO: Infer the mask from the padding index.
        mask = (padded_input.squeeze(-1) == pad_idx)  # Shape (N, T)
    # print('mask shape', mask.shape)
    return mask



def CausalMask(input_tensor):
    """
    Create an attention mask for causal self-attention based on input lengths.

    Args:
        input_tensor (torch.Tensor): The input tensor of shape (N, T, *).

    Returns:
        attn_mask (torch.Tensor): The causal self-attention mask of shape (T, T)
    """
    T = input_tensor.shape[1]  # Sequence length
    # TODO: Initialize attn_mask as a tensor of zeros with the right shape.
    attn_mask = torch.zeros((T, T), dtype=torch.bool) # Shape (T, T)

    # TODO: Create a lower triangular matrix to form the causal mask.
    causal_mask = torch.triu(torch.ones((T, T), dtype=torch.bool), diagonal=1)  # triangular matrix

    # TODO: Combine the initial mask with the causal mask.
    attn_mask = attn_mask | causal_mask
    # print('input_tensor (N, T, *)', input_tensor.shape)
    # print('attn mask shape (T, T)', attn_mask.shape)
    return attn_mask


# Test w/ dummy inputs
enc_inp_tensor     = torch.randn(4, 20, 32)  # (N, T,  *)
dec_inp_tensor     = torch.randn(4, 10)       # (N, T', *)
enc_inp_lengths    = torch.tensor([13, 6, 9, 12])   # Lengths of input sequences before padding
dec_inp_lengths    = torch.tensor([8, 3, 1, 6])   # Lengths of target sequences before padding


enc_padding_mask        = PadMask(padded_input=enc_inp_tensor, input_lengths=enc_inp_lengths)
dec_padding_mask        = PadMask(padded_input=dec_inp_tensor, input_lengths=dec_inp_lengths)
dec_causal_mask         = CausalMask(input_tensor=dec_inp_tensor)

# Black portions are attended to
fig, axs = plt.subplots(1, 3, figsize=(20, 5))
axs[0].imshow(enc_padding_mask, cmap="gray", aspect='auto')
axs[0].set_title("Encoder Padding Mask")
axs[1].imshow(dec_padding_mask, cmap="gray", aspect='auto')
axs[1].set_title("Decoder Padding Mask")
axs[2].imshow(dec_causal_mask, cmap="gray", aspect='auto')
axs[2].set_title("Decoder Causal Self-Attn Mask")


plt.savefig("mask_visualization.png", dpi=300)
plt.close()  # Close the figure to free memory
