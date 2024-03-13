### Original code: https://github.com/WoosukKwon/retraining-free-pruning

import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset
from transformers import (
    DataCollatorWithPadding,
    set_seed,
)

from glue import glue_dataset, max_seq_length, avg_seq_length

TASKS = [
    "mnli",
    "qqp",
    "qnli",
    "sst2",
    "stsb",
    "mrpc",
    "rte",
    "squad"
    ]


def register_mask(module, mask):
    hook = lambda _, inputs: (inputs[0] * mask, inputs[1])
    handle = module.register_forward_pre_hook(hook)
    return handle

def get_backbone(model):
    model_type = model.base_model_prefix
    backbone = getattr(model, model_type)
    return backbone

def get_encoder(model):
    backbone = get_backbone(model)
    encoder = backbone.encoder
    return encoder

def get_layers(model):
    encoder = get_encoder(model)
    layers = encoder.layer
    return layers

def get_ffn2(model, index):
    layer = get_layers(model)[index]
    ffn2 = layer.output
    return ffn2

def apply_neuron_mask(model, neuron_mask):
    num_hidden_layers = neuron_mask.shape[0]
    handles = []
    for layer_idx in range(num_hidden_layers):
        ffn2 = get_ffn2(model, layer_idx)
        handle = register_mask(ffn2, neuron_mask[layer_idx])
        handles.append(handle)
    return handles

def collect_mask_grads(model, head_mask, neuron_mask, dataloader):
    head_mask.requires_grad_(True)
    neuron_mask.requires_grad_(True)

    handles = apply_neuron_mask(model, neuron_mask)

    model.eval()
    head_grads = []
    neuron_grads = []
    for batch in tqdm(dataloader):
        for k, v in batch.items():
            batch[k] = v.to("cuda", non_blocking=True)

        outputs = model(head_mask=head_mask, **batch)
        loss = outputs.loss
        loss.backward()

        head_grads.append(head_mask.grad.detach())
        head_mask.grad = None

        neuron_grads.append(neuron_mask.grad.detach())
        neuron_mask.grad = None

    for handle in handles:
        handle.remove()
    head_mask.requires_grad_(False)
    neuron_mask.requires_grad_(False)

    head_grads = torch.stack(head_grads, dim=0)
    neuron_grads = torch.stack(neuron_grads, dim=0)
    return head_grads, neuron_grads


@torch.no_grad()
def compute_fisher_info(grads):
    fisher_info = grads.pow(2).sum(dim=0)
    return fisher_info

def calculate_fisher(model, tokenizer, task_name, num_samples=2048, device='cuda', seed=0):
    """Calculate the Fisher mask nodes for a model and task. Returns the neuron and head masks."""
    # Set the experiment seed
    set_seed(seed)

    # Check the task name
    assert task_name in TASKS, f"Task name must be one of {TASKS}"

    seq_len = avg_seq_length(task_name)
    training_dataset = glue_dataset(
        task_name,
        tokenizer,
        training=True,
        max_seq_len=max_seq_length(task_name),
        pad_to_max=False,
    )

    if num_samples > len(training_dataset) or num_samples == -1:
        num_samples = len(training_dataset)
    # Sample the examples to be used for search
    collate_fn = DataCollatorWithPadding(tokenizer)
    sample_dataset = Subset(
        training_dataset,
        np.random.choice(len(training_dataset), num_samples).tolist(),
    )
    sample_batch_size = 32
    sample_dataloader = DataLoader(
        sample_dataset,
        batch_size=sample_batch_size,
        collate_fn=collate_fn,
        shuffle=False,
        pin_memory=True,
    )

    # Prepare the model
    model = model.to(device=device)
    model.eval()
    config = model.config
    for param in model.parameters():
        param.requires_grad_(False)

    full_head_mask = torch.ones(config.num_hidden_layers, config.num_attention_heads).to(device=device) # 12 x 12
    full_neuron_mask = torch.ones(config.num_hidden_layers, config.intermediate_size).to(device=device) # 12 x 3072

    # Search the optimal mask
    head_grads, neuron_grads = collect_mask_grads(
        model,
        full_head_mask,
        full_neuron_mask,
        sample_dataloader,
    )

    # Compute the Fisher information
    head_grads = compute_fisher_info(head_grads)
    neuron_grads = compute_fisher_info(neuron_grads)

    return neuron_grads, head_grads