import contextlib
import torch
import transformers
# from lxt.efficient import monkey_patch
import gc
from vllm import LLM
from vllm.distributed import destroy_distributed_environment, destroy_model_parallel
import ray


def set_chat_template_if_needed(tokenizer: transformers.AutoTokenizer):
    if tokenizer.chat_template is None:
        # ChatML template, following https://huggingface.co/docs/transformers/v4.34.0/en/chat_templating
        tokenizer.chat_template = "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %})"


def get_tokenizer(model_name: str):
    return transformers.AutoTokenizer.from_pretrained(model_name)

def _is_mistral_model(model_name: str) -> bool:
    """Check if the model requires Mistral-specific vLLM loading parameters.

    Official mistralai models need tokenizer_mode/config_format/load_format
    set to "mistral" for correct tokenization and weight loading.
    See: https://docs.vllm.ai/projects/recipes/en/latest/Mistral/Ministral-3-Reasoning.html
    """
    name_lower = model_name.lower()
    return "mistralai/" in name_lower or "ministral" in name_lower


def get_vllm_model(
    model_name: str,
    dtype: str = "bfloat16",
    tensor_parallel_size: int = 1,
    gpu_memory_utilization=None,
    max_model_len=None,
    quantize: bool = False,
) -> LLM:
    """Initialize and return a VLLM model."""
    if "secalign" in model_name.lower():
        # Annoying way SecAlign shared the model requires some special handling.
        model = LLM(
            model="meta-llama/Llama-3.1-8B-Instruct",
            enable_lora=True,
            max_lora_rank=64,
            trust_remote_code=True,
        )
        lora_request = LoRARequest("Meta-SecAlign-8B", 1, "facebook/Meta-SecAlign-8B")
        return model, lora_request

    mistral_kwargs = {}
    if _is_mistral_model(model_name):
        mistral_kwargs = {
            "tokenizer_mode": "mistral",
            "config_format": "mistral",
            "load_format": "mistral",
        }

    llm = LLM(
        model=model_name,
        tensor_parallel_size=tensor_parallel_size,  # torch.cuda.device_count(),
        trust_remote_code=True,
        dtype=dtype,
        gpu_memory_utilization=gpu_memory_utilization,
        distributed_executor_backend="mp",
        max_model_len=max_model_len,
        **mistral_kwargs,
    )
    return llm, None



def get_vllm_model_and_tokenizer(
    model_name: str,
    dtype: str = "bfloat16",
    multi_gpu: bool = False,
    use_less_gpu_memory: bool = False,
    max_model_len = None,
    quantize: bool = False,
):
    if _is_mistral_model(model_name):
        tokenizer_kwargs = {
            "fix_mistral_regex": True,
        }
    else:
        tokenizer_kwargs = {}

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)
    set_chat_template_if_needed(tokenizer)
    tokenizer.padding_side = "left"  # Set padding side to left for decoder-only models
    tokenizer.pad_token = tokenizer.eos_token
    gpu_memory_utilization = 0.5 if use_less_gpu_memory else 0.95
    model, lora_request = get_vllm_model(
        model_name=model_name,
        dtype=dtype,
        tensor_parallel_size=torch.cuda.device_count() if multi_gpu else 1,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        quantize=quantize,
    )

    return model, tokenizer, lora_request


def delete_huggingface_model_and_free_memory(model, tokenizer=None):
    del model
    if tokenizer is not None:
        del tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()


def delete_vllm_model_and_free_memory(vllm_model):
    print("Shutting down VLLM model...")
    vllm_model.llm_engine.engine_core.shutdown()
    destroy_model_parallel()
    destroy_distributed_environment()
    with contextlib.suppress(AssertionError):
        torch.distributed.destroy_process_group()
    del vllm_model
    gc.collect()
    torch.cuda.empty_cache()
    # Reset CUDA device to fully clear memory
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()  # Wait for all streams on the current device

    # CRITICAL: Shutdown Ray to kill all worker processes
    ray.shutdown()


def get_model_and_tokenizer(
    model_name: str,
    device: str,
    half_precision: bool = False,
    multi_gpu: bool = False,
    use_lxt: bool = False,
    use_kernels: bool = False,
):
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    set_chat_template_if_needed(tokenizer)
    tokenizer.padding_side = "left"  # Set padding side to left for decoder-only models
    tokenizer.pad_token = tokenizer.eos_token

    if use_lxt:
        # Check if the model is Qwen or Llama and apply appropriate monkey patch
        if "qwen3" in model_name.lower():
            # Modify the Qwen module to compute LRP in the backward pass
            monkey_patch(transformers.models.qwen3.modeling_qwen3, verbose=True)
            model = transformers.models.qwen3.modeling_qwen3.Qwen3ForCausalLM.from_pretrained(
                model_name,
                device_map="auto" if multi_gpu else device,
                dtype=torch.bfloat16 if half_precision else torch.float32,
            )
        elif "qwen2" in model_name.lower():
            # Modify the Qwen module to compute LRP in the backward pass
            monkey_patch(transformers.models.qwen2.modeling_qwen2, verbose=True)
            model = transformers.models.qwen2.modeling_qwen2.Qwen2ForCausalLM.from_pretrained(
                model_name,
                device_map="auto" if multi_gpu else device,
                dtype=torch.bfloat16 if half_precision else torch.float32,
            )
        else:
            # Assume Llama model (default behavior)
            # Modify the LLaMA module to compute LRP in the backward pass
            monkey_patch(transformers.models.llama.modeling_llama, verbose=True)
            model = transformers.models.llama.modeling_llama.LlamaForCausalLM.from_pretrained(
                model_name,
                device_map="auto" if multi_gpu else device,
                dtype=torch.bfloat16 if half_precision else torch.float32,
            )
    else:
        if "Llama-3.1-70B" in model_name:
            # Special handling for Llama 3.1 70B due to its size
            print("Loading Llama 3.1 70B with 4-bit quantization...")
            quantization_config = transformers.BitsAndBytesConfig(load_in_4bit=True)

            model = transformers.AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto" if multi_gpu else device,
                quantization_config=quantization_config,
            )
            model.config.pad_token_id = tokenizer.pad_token_id

        else:
            model = transformers.AutoModelForCausalLM.from_pretrained(
                model_name,
                dtype=torch.bfloat16 if half_precision else torch.float32,
                device_map="auto" if multi_gpu else device,
                use_kernels=use_kernels,
            )
            model.config.pad_token_id = tokenizer.pad_token_id

    return model, tokenizer
