# This code contains implementation of LWQ paper
import torch
import json
import io
import struct
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import hf_hub_download
import numpy as np
import os
import random
from datasets import load_dataset

MODEL_ID = "/shareddata/dheyo/varunika/dynamic_quants/gsm8k_dheyo_ZD_QAT_12082025"
SHARD_FILENAME = "model-00001-of-00001.safetensors"
SEED = 0
NSAMPLES = 50
SEQLEN = 128

def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    random.seed(seed)

set_seed(SEED)

def get_gsm8k(nsamples, seed, seqlen, model):
    print(f"Loading GSM8K dataset. Using sequence length: {seqlen}")
    traindata = load_dataset("openai/gsm8k", "main", split="train")
    testdata = load_dataset("openai/gsm8k", "main", split="test")
    
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False, local_files_only=True)
    train_text_samples = [d["question"] + " " + d["answer"] for d in traindata]
    random.seed(seed)
    random.shuffle(train_text_samples)

    trainloader = []
    print(f"Tokenizing {nsamples} calibration samples...")
    for text in train_text_samples[:nsamples]:
        enc = tokenizer(
            text,
            return_tensors="pt",
            max_length=seqlen,
            truncation=True,
            padding="max_length"
        )
        inp = enc.input_ids
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    
    val_text_samples = [d["question"] for d in testdata]
    valenc = tokenizer(" ".join(val_text_samples[:256]), return_tensors="pt")
    class TokenizerWrapper:
        def __init__(self, input_ids):
            self.input_ids = input_ids
    valenc = TokenizerWrapper(valenc.input_ids)
    return trainloader, valenc

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    local_files_only=True,
    attention_dropout=0.0,
    hidden_size=1536,
    num_hidden_layers=28,
    num_attention_heads=12,
    num_key_value_heads=2,
    intermediate_size=8960,
    max_position_embeddings=131072,
    rms_norm_eps=1e-06,
    vocab_size=151936,
    bos_token_id=151643,
    eos_token_id=151643,
    use_cache=True
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print(f"LOG_MAIN: Downloading/verifying {SHARD_FILENAME} from {MODEL_ID}...")
local_path = os.path.join(MODEL_ID, SHARD_FILENAME)
print(f"LOG_MAIN: Using local shard {SHARD_FILENAME} at {local_path}")

all_tensor_metadata = {}
print(f"LOG_MAIN: Extracting tensor metadata from {SHARD_FILENAME}...")
try:
    with io.open(local_path, 'rb') as f:
        n_json_header_len = struct.unpack('<Q', f.read(8))[0]
        current_data_payload_start_offset_abs = 8 + n_json_header_len
        json_header_bytes = f.read(n_json_header_len)
        if len(json_header_bytes) < n_json_header_len:
            raise ValueError(f"Incomplete JSON header for {SHARD_FILENAME}")
        parsed_json_header = json.loads(json_header_bytes.decode('utf-8'))
        
        for tensor_key, tensor_info in parsed_json_header.items():
            if tensor_key == "__metadata__": continue
            required_meta_keys = ['dtype', 'shape', 'data_offsets']
            if not all(k in tensor_info for k in required_meta_keys):
                if 'offsets' in tensor_info and all(k in tensor_info for k in ['dtype', 'shape', 'offsets']):
                    tensor_info['data_offsets'] = tensor_info.pop('offsets')
                else:
                    print(f"LOG_MAIN_WARNING: Incomplete metadata for tensor '{tensor_key}'. Skipping.")
                    continue
            
            shape = tuple(tensor_info['shape'])
            if len(shape) == 2:
                all_tensor_metadata[tensor_key] = {
                    "shape": shape,
                    "dtype_str": tensor_info['dtype'],
                    "offsets_in_payload_relative": tuple(tensor_info['data_offsets']),
                    "shard_data_payload_start_offset_abs": current_data_payload_start_offset_abs
                }
except Exception as e:
    print(f"LOG_MAIN_ERROR: Failed to process {SHARD_FILENAME}: {type(e).__name__} - {e}")
    exit(1)

trainloader, valenc = get_gsm8k(NSAMPLES, SEED, SEQLEN, MODEL_ID)
inputs = torch.cat([item[0] for item in trainloader], dim=0).to(device)
print(f"LOG_MAIN: Input shape after tokenization: {inputs.shape}")

def compute_lim_score(layer_idx):
    with torch.no_grad():
        input_embeds = model.get_input_embeddings()(inputs)
        outputs = model(inputs, output_hidden_states=True)
        layer_output = outputs.hidden_states[layer_idx + 1]
        
        # Compute cosine similarity per token pair, average over batch and sequence
        batch_size, seq_len, hidden_size = input_embeds.shape
        cosine_sims = []
        for i in range(batch_size):
            input_seq = input_embeds[i]  # [seq_len, hidden_size]
            output_seq = layer_output[i]  # [seq_len, hidden_size]
            input_norm = torch.norm(input_seq, p=2, dim=-1, keepdim=True)  # [seq_len, 1]
            output_norm = torch.norm(output_seq, p=2, dim=-1, keepdim=True)  # [seq_len, 1]
            dot_product = torch.sum(input_seq * output_seq, dim=-1, keepdim=True)  # [seq_len, 1]
            cosine_sim = dot_product / (input_norm * output_norm + 1e-8)  # [seq_len, 1]
            cosine_sims.append(cosine_sim.mean().item())  # Average over sequence
        lim_score = -np.mean(cosine_sims)  # Negate and average over batch
    return lim_score

def compute_zd_score(layer_idx):
    with torch.no_grad():
        state_dict = model.state_dict()
        layer_weights = []
        layer_prefix = f"model.layers.{layer_idx}"
        for name, param in state_dict.items():
            if name.startswith(layer_prefix) and "weight" in name:
                if name in all_tensor_metadata:
                    meta = all_tensor_metadata[name]
                    if param.shape == meta["shape"] and meta["dtype_str"] == "BF16":
                        layer_weights.append(param.flatten())
        if not layer_weights:
            return 0.0
        weights = torch.cat(layer_weights)
        mean = weights.mean().item()
        std = weights.std().item()
        if std == 0:
            return 0.0
        z_scores = (weights - mean) / std
        zd_score = (z_scores.abs() > 1).float().mean().item()
    return zd_score

lim_scores = {i: compute_lim_score(i) for i in range(28)}
zd_scores = {i: compute_zd_score(i) for i in range(28)}

M_available = 20 * 1024
M_lower = 17 * 1024
M_higher = 34 * 1024
N_layers = 28
Nhigher = int(np.floor((M_available - M_lower) / (M_higher - M_lower) * N_layers))
print(f"LOG_MAIN: Calculated Nhigher = {Nhigher} layers for higher precision (4-bit)")
os.makedirs("final_results", exist_ok=True)
bit_widths = {i: 4 if lim_scores[i] <= sorted(lim_scores.values())[:Nhigher][-1] else 2 for i in range(N_layers)}  # Sort by most negative
with open("final_results/bit_widths_after_qat_for_zd.json", "w") as f:
    json.dump(bit_widths, f, indent=4)
print(f"LOG_MAIN: Bitwidths saved to bit_widths.json")

with open("final_results/lim_scores_after_qat_for_zd.json", "w") as f:
    json.dump(lim_scores, f, indent=4)
with open("final_results/zd_scores_after_qat_for_zd.json", "w") as f:
    json.dump(zd_scores, f, indent=4)

for i in range(28):
    print(f"Layer {i}: LIM Score = {lim_scores[i]:.4f}, ZD Score = {zd_scores[i]:.4f}, Bitwidth = {bit_widths[i]}")