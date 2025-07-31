import time
import torch
import torch.nn as nn
import os
from tqdm import tqdm
import argparse

from gptq_baq import *
from modelutils import *
from quant_baq import *
from datautils import *


def get_deepseek(model_path):
    from transformers import AutoModelForCausalLM, AutoConfig
    print(f"Loading model: {model_path}")
    config = AutoConfig.from_pretrained(model_path)
    if hasattr(config, 'num_attention_heads') and hasattr(config, 'hidden_size'):
        config.head_dim = config.hidden_size // config.num_attention_heads
        print(f"Set head_dim to {config.head_dim} (hidden_size={config.hidden_size}, num_attention_heads={config.num_attention_heads})")
    model = AutoModelForCausalLM.from_pretrained(model_path, config=config, torch_dtype='auto')
    try:
        model.seqlen = model.config.max_position_embeddings
    except AttributeError:
        print("Warning: Could not determine model.seqlen from config. Falling back to 2048.")
        model.seqlen = 2048
    return model

@torch.no_grad()
def opt_sequential_calib(model, dataloader, dev, args):
    print('Starting calibration...')

    use_cache = model.config.use_cache
    model.config.use_cache = False
    
    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    
    # This cache will capture the exact inputs and keyword arguments from the model
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, *args, **kwargs):
            cache['inputs'].append(args[0])
            cache['kwargs'] = kwargs
            raise ValueError
        def __getattr__(self, name):
            try:
                return super().__getattr__(name)
            except AttributeError:
                return getattr(self.module, name)
            
    cache = {'inputs': [], 'kwargs': {}}
    
    print("Dataloader configuration:")
    for i, batch in enumerate(dataloader):
        print(f"Batch {i}: shape={batch[0].shape}")
        if i >= 4:  # Limit to 5 batches
            break
        
    print(f"Original dataloader length: {len(dataloader)}, first batch shape: {dataloader[0][0].shape}")

    from torch.utils.data import DataLoader, TensorDataset

    all_inputs = []
    all_kwargs = []
    samples_collected = 0
    batch_size = 8  # Configurable batch size
    print(f"Creating batched dataloader with batch_size={batch_size}")
    input_tensors = [batch[0] for batch in dataloader]  # Extract inp from (inp, tar)
    if len(input_tensors) < args.nsamples:
        raise ValueError(f"Dataloader has {len(input_tensors)} samples, expected at least {args.nsamples}")
    batched_dataset = TensorDataset(torch.cat(input_tensors[:args.nsamples], dim=0))
    batched_dataloader = DataLoader(batched_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    for batch_idx, batch in enumerate(batched_dataloader):
        if samples_collected >= args.nsamples:
            break
        batch_inputs = batch[0].to(dev)  # Shape: [batch_size, 2048]
        batch_size_actual = batch_inputs.shape[0]
        print(f"Processing batch {batch_idx + 1}, batch shape: {batch_inputs.shape}, samples collected: {samples_collected}/{args.nsamples}")
        print(f"Batch {batch_idx + 1}: batch_inputs shape={batch_inputs.shape}")
        original_layer_0 = model.model.layers[0]
        model.model.layers[0] = Catcher(original_layer_0).to(dev)
        try:
            seq_len = batch_inputs.shape[1]
            batch_position_ids = torch.arange(0, seq_len, dtype=torch.long, device=dev).unsqueeze(0).expand(batch_size_actual, seq_len)
            print(f"Batch {batch_idx + 1}: position_ids shape={batch_position_ids.shape}")
            model(batch_inputs, position_ids=batch_position_ids)
        except ValueError as e:
            print(f"ValueError in model forward: {e}")
            pass
        model.model.layers[0] = original_layer_0
        batch_outputs = cache['inputs'][0]  # Shape: [batch_size_actual, 2048, 1536]
        for i in range(batch_size_actual):
            all_inputs.append(batch_outputs[i:i+1])  # Shape: [1, 2048, 1536]
        all_kwargs.append(cache['kwargs'])
        cache['inputs'] = []
        cache['kwargs'] = {}
        samples_collected += batch_size_actual
        print(f"Collected {len(all_inputs)} input tensors, each of shape: {all_inputs[0].shape if all_inputs else 'N/A'}")
    
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    torch.cuda.empty_cache()

    if not all_inputs:
        raise ValueError("No inputs collected from dataloader")
    inps = torch.cat(all_inputs, dim=0)
    kwargs = all_kwargs[0]
    if inps.shape[0] > args.nsamples:
        inps = inps[:args.nsamples]
    elif inps.shape[0] < args.nsamples:
        raise ValueError(f"Collected {inps.shape[0]} samples, expected {args.nsamples}")
    outs = torch.zeros_like(inps)
    print(f"Combined input shape: {inps.shape}")

    num_linear_layers = len(find_layers(model.model.layers[0]))
    print(f"Found {num_linear_layers} linear layers per block.")
    ele_sum_container = {'value': 0.0}
    R_aver_container = {'value': 0.0}
    R_record_container = {'value': torch.zeros((len(model.model.layers), num_linear_layers))}
    Gain_vec_container = {'value': []}
    loss_vec_container = {'value': []}
    quantizers = {}

    # Process all layers in smaller batches to match input collection
    batch_size = 8  # Match the batch size used in input collection
    for i in tqdm(range(len(model.model.layers))):
        layer = model.model.layers[i].to(dev)
        subset = find_layers(layer)
        gptq = {}
        for name in subset:
            gptq[name] = GPTQ(subset[name])
            gptq[name].quantizer = Quantizer()
            gptq[name].quantizer.configure(
                args.wbits, perchannel=True, sym=args.sym, mse=False, trits=args.trits
            )

        def add_batch(name):
            def tmp(_, inp, out):
                gptq[name].add_batch(inp[0].data, out.data)
            return tmp
        handles = []
        for name in subset:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        # Process inputs in mini-batches
        for j in range(0, inps.shape[0], batch_size):
            batch_inps = inps[j:j+batch_size].to(dev)
            batch_size_actual = batch_inps.shape[0]
            seq_len = batch_inps.shape[1]
            batch_position_ids = torch.arange(0, seq_len, dtype=torch.long, device=dev).unsqueeze(0).expand(batch_size_actual, seq_len)
            current_kwargs = {k: v[j:j+batch_size] if torch.is_tensor(v) and v.shape[0] == inps.shape[0] else v for k, v in kwargs.items()}
            current_kwargs['position_ids'] = batch_position_ids
            print(f"Layer {i}, batch {j//batch_size + 1}: Input shape={batch_inps.shape}, position_ids shape={current_kwargs['position_ids'].shape}")
            layer(batch_inps, **current_kwargs)

        for h in handles:
            h.remove()

        for name in subset:
            col_idx = list(find_layers(layer).keys()).index(name)
            print(f"Quantizing layer {i}, module {name}")
            gptq[name].fasterquant_calib(
                percdamp=args.percdamp, groupsize=args.groupsize, actorder=args.act_order, 
                static_groups=args.static_groups, layer_idx=i, name=name, 
                R_aver_container=R_aver_container, ele_sum_container=ele_sum_container, 
                R_record_container=R_record_container, col_idx=col_idx, 
                Gain_vec_container=Gain_vec_container, loss_vec_container=loss_vec_container, 
                R_ref=args.wbits
            )
            quantizers['model.layers.%d.%s' % (i, name)] = gptq[name].quantizer
            gptq[name].free()

        # Process output in mini-batches
        for j in range(0, inps.shape[0], batch_size):
            batch_inps = inps[j:j+batch_size].to(dev)
            batch_size_actual = batch_inps.shape[0]
            seq_len = batch_inps.shape[1]
            batch_position_ids = torch.arange(0, seq_len, dtype=torch.long, device=dev).unsqueeze(0).expand(batch_size_actual, seq_len)
            current_kwargs = {k: v[j:j+batch_size] if torch.is_tensor(v) and v.shape[0] == inps.shape[0] else v for k, v in kwargs.items()}
            current_kwargs['position_ids'] = batch_position_ids
            batch_outs = layer(batch_inps, **current_kwargs)[0]
            if batch_outs.dim() == 2:
                batch_outs = batch_outs.unsqueeze(0)
            outs[j:j+batch_size] = batch_outs
        print(f"Layer {i}: Output shape={outs.shape}")

        layer.cpu()
        del layer, gptq
        torch.cuda.empty_cache()
        inps = outs

    print("Saving importance scores...")
    if os.path.exists("results/gain_vec_tensor.pt"):
        os.remove("results/gain_vec_tensor.pt")
    gain_vec_tensor = torch.tensor(Gain_vec_container['value'])
    torch.save(gain_vec_tensor, "results/gain_vec_tensor.pt")
    if os.path.exists("results/loss_vec_tensor.pt"):
        os.remove("results/loss_vec_tensor.pt")
    print(f"Number of loss vectors: {len(loss_vec_container['value'])}")
    for idx, loss_vec in enumerate(loss_vec_container['value']):
        print(f"Loss vector {idx} shape: {loss_vec.shape}")
    try:
        loss_vec_tensor = torch.stack(loss_vec_container['value'])
    except RuntimeError as e:
        print(f"Error stacking loss vectors: {e}")
        torch.save(loss_vec_container['value'], "results/loss_vec_tensor.pt")
    else:
        print(f"Loss vector tensor shape: {loss_vec_tensor.shape}")
        torch.save(loss_vec_tensor, "results/loss_vec_tensor.pt")

    model.config.use_cache = use_cache
    return quantizers

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str, help='Model to load')
    parser.add_argument('dataset', type=str, choices=['wikitext2', 'ptb', 'c4', 'gsm8k'])
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--nsamples', type=int, default=128)
    parser.add_argument('--percdamp', type=float, default=0.1)
    parser.add_argument('--wbits', type=int, default=4, choices=[2, 3, 4, 16])
    parser.add_argument('--trits', action='store_true')
    parser.add_argument('--groupsize', type=int, default=-1)
    parser.add_argument('--sym', action='store_true')
    parser.add_argument('--act-order', action='store_true')
    parser.add_argument('--static-groups', action='store_true')

    args = parser.parse_args()
    DEV = torch.device('cuda')

    torch.backends.cuda.matmul.fp32_precision = 'ieee'
    torch.backends.cudnn.conv.fp32_precision = 'ieee'

    model = get_deepseek(args.model)
    model.eval()
    print(f"Model config: {model.config}")

    calibration_seqlen = 2048
    model.seqlen = calibration_seqlen
    print(f"Overriding sequence length for calibration. Using seqlen = {calibration_seqlen}")

    dataloader, testloader = get_loaders(
        args.dataset, nsamples=args.nsamples, seed=args.seed, model=args.model, seqlen=calibration_seqlen
    )

    if args.wbits < 16:
        print("Starting Calibration to generate importance scores.....")
        tick = time.time()
        opt_sequential_calib(model, dataloader, DEV, args)
        print(f"Calibration finished in {time.time() - tick:.2f} seconds.")
        print("Importance scores have been saved to 'results/loss_vec_tensor.pt'.")

    print("Script Finished.")
