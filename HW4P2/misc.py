import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Literal, List, Optional, Any, Dict, Tuple


def save_attention_plot(plot_path, attention_weights, epoch=0, mode: Literal['full', 'dec_cond_lm', 'dec_lm'] = 'full'):
    """
    Saves attention weights plot to a specified path.

    Args:
        plot_path (str): Directory path where the plot will be saved.
        attention_weights (Tensor): Attention weights to plot.
        epoch (int): Current training epoch (default is 0).
        mode (str): Mode of attention - 'full', 'dec_cond_lm', or 'dec_lm'.
    """
    if not isinstance(attention_weights, (np.ndarray, torch.Tensor)):
        raise ValueError("attention_weights must be a numpy array or torch Tensor")

    plt.clf()  # Clear the current figure
    sns.heatmap(attention_weights, cmap="viridis", cbar=True)  # Create heatmap
    plt.title(f"{mode} Attention Weights - Epoch {epoch}")
    plt.xlabel("Target Sequence")
    plt.ylabel("Source Sequence")

    # Save the plot with clearer filename distinction
    attention_type = "cross" if epoch < 100 else "self"
    epoch_label = epoch if epoch < 100 else epoch - 100
    plt.savefig(f"{plot_path}/{mode}_{attention_type}_attention_epoch{epoch_label}.png")




def save_model(model, optimizer, scheduler, metric, epoch, path):
    """
    Saves the model, optimizer, and scheduler states along with a metric to a specified path.

    Args:
        model (nn.Module): Model to be saved.
        optimizer (Optimizer): Optimizer state to save.
        scheduler (Scheduler or None): Scheduler state to save.
        metric (tuple): Metric tuple (name, value) to be saved.
        epoch (int): Current epoch number.
        path (str): File path for saving.
    """
    # Ensure metric is provided as a tuple with correct structure
    if not (isinstance(metric, tuple) and len(metric) == 2):
        raise ValueError("metric must be a tuple in the form (name, value)")

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else {},
            metric[0]: metric[1],  # Unpacks the metric name and value
            "epoch": epoch
        },
        path
    )


def load_checkpoint(
    checkpoint_path,
    model,
    embedding_load: bool,
    encoder_load: bool,
    decoder_load: bool,
    optimizer=None,
    scheduler=None
):
    """
    Loads weights from a checkpoint into the model and optionally returns updated model, optimizer, and scheduler.

    Args:
        checkpoint_path (str): Path to the checkpoint file.
        model (Transformer): Transformer model to load weights into.
        embedding_load (bool): Load embedding weights if True.
        encoder_load (bool): Load encoder weights if True.
        decoder_load (bool): Load decoder weights if True.
        optimizer (Optimizer, optional): Optimizer to load state into (if provided).
        scheduler (Scheduler, optional): Scheduler to load state into (if provided).

    Returns:
        model (Transformer): Model with loaded weights.
        optimizer (Optimizer or None): Optimizer with loaded state if provided.
        scheduler (Scheduler or None): Scheduler with loaded state if provided.
    """
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path)
    model_state_dict = model.state_dict()

    # Define the components to be loaded
    load_map = {
        "embedding": embedding_load,
        "encoder": encoder_load,
        "decoder": decoder_load
    }

    # Filter and load the specified components
    for key, should_load in load_map.items():
        if should_load:
            component_state_dict = {k: v for k, v in checkpoint['model_state_dict'].items() if k.startswith(key)}
            if component_state_dict:
                model_state_dict.update(component_state_dict)
            else:
                print(f"Warning: No weights found for {key} in checkpoint.")

    # Load the updated state_dict into the model
    model.load_state_dict(model_state_dict, strict=False)
    loaded_components = ", ".join([k.capitalize() for k, v in load_map.items() if v])
    print(f"Loaded components: {loaded_components}")

    # Load optimizer and scheduler states if available and provided
    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return model, optimizer, scheduler