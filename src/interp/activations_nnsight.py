# This file contains utilities for extracting activations using the nnsight library.
import torch
import einops
import transformers
from typing import Optional


def extract_prompt_activations(wrapped_model, input_ids, full_sequence: bool = False, extraction_layer: int = None):
    """Extract the residual stream activations before the first generated token at all layers.
    If full_sequence is True, extract the activations for the full prompt sequence (each token).
    If full_sequence is False, extract the activations for the last token in the prompt.
    """

    # Move input_ids to the same device as the model
    if isinstance(input_ids, torch.Tensor):
        input_ids = input_ids.to(wrapped_model.model.device)

    # Handle Gemma models which have nested config structure
    config = wrapped_model.model.config
    if hasattr(config, "text_config") and hasattr(
        config.text_config, "num_hidden_layers"
    ):
        # print("Detected Gemma model configuration.")
        num_layers = config.text_config.num_hidden_layers
    else:
        num_layers = config.num_hidden_layers

    # Determine the correct path to layers (Gemma3 has language_model.layers)
    if hasattr(wrapped_model.model, "language_model"):
        # print("Detected Gemma model with language_model attribute.")
        layers_module = wrapped_model.model.language_model.layers
    else:
        layers_module = wrapped_model.model.layers

    layer_indices = [extraction_layer] if extraction_layer is not None else range(num_layers)

    all_layer_outputs = []

    with torch.no_grad():
        with wrapped_model.trace(input_ids):
            for layer in layer_indices:
                full_layer_output = layers_module[layer].output # Layer output has shape (batch_size, seq_len, hidden_size)
                if full_sequence:
                    layer_output = full_layer_output[
                        0, :, :
                    ]  # (seq_len, hidden_size)
                else:
                    layer_output = full_layer_output[
                        0, -1, :
                    ]  # (hidden_size,) Only take the residual stream at the last token position.
                
                all_layer_outputs.append(
                    layer_output.cpu()
                )  # Move to CPU immediately to handle multi-GPU

    all_layer_outputs = (
        torch.stack(all_layer_outputs).detach().clone()
    )  # (num_layers, seq_len, hidden_size) if full_sequence else (num_layers, hidden_size)

    if extraction_layer is not None:
        final_outputs = all_layer_outputs[0]  # (seq_len, hidden_size) if full_sequence else (hidden_size,)
    else:
        final_outputs = all_layer_outputs  # (num_layers, seq_len, hidden_size) if full_sequence else (num_layers, hidden_size)

    return final_outputs


def extract_batch_prompt_activations(
    wrapped_model, input_ids, layer, prompt_lengths, move_to_cpu: bool = True,
    past_key_values: Optional[transformers.DynamicCache] = None,
    attention_mask: Optional[torch.Tensor] = None,
):
    """
    Args:
        wrapped_model: The wrapped model.
        input_ids: torch.Tensor of shape (batch_size, seq_len)
        layer: The layer to extract the activations from.
        prompt_lengths: List of actual (unpadded) prompt lengths per batch element.
        attention_mask: Optional attention mask for left-padded inputs.
    Returns:
        The activations at the last real token position for each batch element.
    """
    if isinstance(input_ids, torch.Tensor):
        input_ids = input_ids.to(wrapped_model.model.device)
    if attention_mask is not None and isinstance(attention_mask, torch.Tensor):
        attention_mask = attention_mask.to(wrapped_model.model.device)

    # Determine the correct path to layers (Gemma3 has language_model.layers)
    if hasattr(wrapped_model.model, "language_model"):
        layers_module = wrapped_model.model.language_model.layers
    else:
        layers_module = wrapped_model.model.layers

    trace_kwargs = {}
    if past_key_values is not None:
        trace_kwargs["past_key_values"] = past_key_values
    if attention_mask is not None:
        trace_kwargs["attention_mask"] = attention_mask

    batch_activations = []
    with torch.no_grad():
        with wrapped_model.trace(input_ids, **trace_kwargs):
            full_layer_output = layers_module[layer].output  # Layer output has shape (batch_size, seq_len, hidden_size)

            # For each batch element, take the residual stream at the last token position.
            for batch_idx in range(full_layer_output.shape[0]):
                prompt_length = prompt_lengths[batch_idx]
                batch_element_activation = full_layer_output[
                    batch_idx, prompt_length - 1, :
                ]  # (hidden_size,)
                if move_to_cpu:
                    batch_activations.append(batch_element_activation.cpu())
                else:
                    batch_activations.append(batch_element_activation)
    
    batch_activations = (
        torch.stack(batch_activations).detach().clone()
    )  # (batch_size, hidden_size)

    return batch_activations


def extract_response_activations(wrapped_model, prompt, response, layer_indices=None):
    tokenized_prompt = wrapped_model.tokenizer(prompt, return_tensors="pt").input_ids
    prompt_length = tokenized_prompt.shape[1]

    prompt_and_response = prompt + response
    input_ids = wrapped_model.tokenizer(
        prompt_and_response, return_tensors="pt"
    ).input_ids

    # Move input_ids to the same device as the model
    input_ids = input_ids.to(wrapped_model.model.device)

    if layer_indices is None:
        # Extract all layers
        config = wrapped_model.model.config
        if hasattr(config, "text_config") and hasattr(
            config.text_config, "num_hidden_layers"
        ):
            # print("Detected Gemma model configuration.")
            num_layers = config.text_config.num_hidden_layers
        else:
            num_layers = config.num_hidden_layers
        layer_indices = list(range(num_layers))

    # Determine the correct path to layers (Gemma3 has language_model.layers)
    if hasattr(wrapped_model.model, "language_model"):
        # print("Detected Gemma model with language_model attribute.")
        layers_module = wrapped_model.model.language_model.layers
    else:
        layers_module = wrapped_model.model.layers

    all_layer_outputs = []

    with torch.no_grad():
        with wrapped_model.trace(input_ids):
            for layer in layer_indices:
                full_layer_output = layers_module[layer].output[
                    0
                ]  # (batch_size, seq_len, hidden_size)
                layer_output = full_layer_output[
                    0, prompt_length:, :
                ]  # (response_length, hidden_size)
                all_layer_outputs.append(
                    layer_output.cpu()
                )  # Move to CPU immediately to handle multi-GPU

    # Stack and process
    response_activations = (
        torch.stack(all_layer_outputs).detach().clone()
    )  # (num_layers, response_length, hidden_size)
    return response_activations


def extract_all_prompt_activations(wrapped_model, input_ids):
    """Extract the residual stream activations at all prompt positions and at all layers."""

    num_layers = wrapped_model.model.config.num_hidden_layers
    all_layer_outputs = []

    with torch.no_grad():
        with wrapped_model.trace(input_ids):
            for layer in range(num_layers):
                full_layer_output = wrapped_model.model.layers[layer].output[
                    0
                ]  # Layer output has shape (batch_size, seq_len, hidden_size)
                layer_output = full_layer_output[
                    0, :, :
                ]  # Take the residual stream at all token positions.
                all_layer_outputs.append(layer_output)

    all_layer_outputs = (
        torch.stack(all_layer_outputs).detach().clone().cpu()
    )  # (num_layers, prompt_length, hidden_size)

    return all_layer_outputs


def extract_prompt_all_activations(wrapped_model, input_ids):
    """Extract the activations of each model component at last prompt position and at all layers.
    All components are:
    - head outputs
    - attn block outputs
    - mlp block outputs
    """

    num_layers = wrapped_model.model.config.num_hidden_layers
    head_dim = wrapped_model.model.config.head_dim
    hidden_size = wrapped_model.model.config.hidden_size
    num_heads = int(hidden_size / head_dim)
    attn_outputs = []
    mlp_outputs = []
    head_outputs = []

    with torch.no_grad():
        with wrapped_model.trace(input_ids):
            for layer in range(num_layers):
                # # apply out_proj to each head independently
                head_results = torch.zeros(
                    (num_heads, hidden_size),
                    dtype=wrapped_model.dtype,
                    device=wrapped_model.device,
                )
                pre_output_proj_input = (
                    wrapped_model.model.layers[layer].self_attn.o_proj.input
                )  # (batch_size, seq_len, (num_heads * head_dim))
                output_proj_matrix = wrapped_model.model.layers[
                    layer
                ].self_attn.o_proj.weight  # (hidden_size, (num_heads * head_dim))

                for head_idx in range(num_heads):
                    # Only take the values for this one head
                    pre_output_proj_input_head = pre_output_proj_input[
                        :, :, head_idx * head_dim : (head_idx + 1) * head_dim
                    ]  # (batch_size, seq_len, head_dim)
                    output_proj_matrix_head = output_proj_matrix[
                        :, head_idx * head_dim : (head_idx + 1) * head_dim
                    ]  # (hidden_size, head_dim)
                    head_attn_out = einops.einsum(
                        pre_output_proj_input_head,
                        output_proj_matrix_head,
                        "batch seq head, hidden head -> batch seq hidden",
                    )  # (batch_size, seq_len, hidden_size)
                    # Only take the values for the last token and first element in batch
                    head_output = head_attn_out[0, -1, :]  # (hidden_size)
                    head_results[head_idx, :] = head_output

                attn_output = wrapped_model.model.layers[layer].self_attn.o_proj.output[
                    0
                ]
                attn_output = attn_output[-1, :]
                # assert torch.allclose(head_results.sum(dim=0), attn_output, atol=1e-03)
                mlp_output = wrapped_model.model.layers[layer].mlp.down_proj.output[0]
                mlp_output = mlp_output[-1, :]
                attn_outputs.append(attn_output.to(wrapped_model.device))
                mlp_outputs.append(mlp_output.to(wrapped_model.device))
                head_outputs.append(head_results.to(wrapped_model.device))

    attn_outputs = (
        torch.stack(attn_outputs).detach().clone().cpu()
    )  # (num_layers, hidden_size)
    mlp_outputs = (
        torch.stack(mlp_outputs).detach().clone().cpu()
    )  # (num_layers, hidden_size)
    head_outputs = (
        torch.stack(head_outputs).detach().clone().cpu()
    )  # (num_layers, num_heads, hidden_size)

    return head_outputs, torch.stack([attn_outputs, mlp_outputs]).detach().clone().cpu()
