import torch
import json
import re
import numpy as np
from collections import defaultdict

# Load LIM scores
def load_lim_scores(file_path):
    with open(file_path, 'r') as f:
        lim_scores = json.load(f)
    return {int(k): v for k, v in lim_scores.items()}

# Assign bit-widths based on LIM scores
def assign_bit_widths_lim(lim_scores):
    lim_scores_array = np.array(list(lim_scores.values()))
    lim_norm = (lim_scores_array.max() - lim_scores_array) / (lim_scores_array.max() - lim_scores_array.min() + 1e-8)
    bit_width_values = { 'Q2_K': 2.5625, 'Q3_K': 3.4375, 'Q4_K': 4.5, 'Q5_K': 5.5, 'Q6_K': 6.25, 'Q4_0': 4.5, 'Q4_1': 4.75, 'Q5_0': 5.5, 'Q5_1': 5.75, 'Q8_0': 8.5,}
    component_bit_widths = {}
    part_map = {
        'q_proj': 'attn_q.weight', 'k_proj': 'attn_k.weight', 'v_proj': 'attn_v.weight',
        'o_proj': 'attn_output.weight', 'up_proj': 'ffn_up.weight', 'gate_proj': 'ffn_gate.weight',
        'down_proj': 'ffn_down.weight'
    }
    components = []
    importance_scores = []
    for layer_idx in range(28):
        for part in ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'up_proj', 'gate_proj', 'down_proj']:
            components.append((layer_idx, part))
            importance_scores.append(lim_norm[layer_idx])
    for i, (layer_idx, part) in enumerate(components):
        imp = importance_scores[i]
        name = f'blk.{layer_idx}.{part_map[part]}'
        if imp > 0.75:
            bit_width = 'Q8_0'
        elif imp > 0.5:
            bit_width = 'Q6_K' if part in ['o_proj', 'up_proj', 'gate_proj', 'down_proj'] else 'Q5_K'
        elif imp > 0.25:
            bit_width = 'Q4_K' if part in ['o_proj', 'up_proj', 'gate_proj', 'down_proj'] else 'Q4_0'
        else:
            bit_width = 'Q3_K' if part in ['q_proj', 'k_proj', 'v_proj'] else 'Q4_0'
        if part in ['q_proj', 'o_proj', 'gate_proj', 'up_proj'] and imp > 0.4:
            bit_width = 'Q8_0'
        elif part == 'down_proj' and imp > 0.6:
            bit_width = 'Q8_0'
        component_bit_widths[name] = bit_width
    avg_bit_width = np.mean([bit_width_values.get(bw, 4) for bw in component_bit_widths.values()])
    if avg_bit_width > 3.5:
        sorted_comps = sorted(zip(components, importance_scores), key=lambda x: x[1])
        for (layer_idx, part), _ in sorted_comps[:int(len(components) * 0.3)]:
            name = f'blk.{layer_idx}.{part_map[part]}'
            if component_bit_widths[name] == 'Q5_K':
                component_bit_widths[name] = 'Q4_K'
            elif component_bit_widths[name] == 'Q4_K':
                component_bit_widths[name] = 'Q3_K'
            elif component_bit_widths[name] == 'Q4_0':
                component_bit_widths[name] = 'Q3_K'
    component_bit_widths['output.weight'] = 'Q6_K'
    component_bit_widths['token_embd.weight'] = 'F16'
    component_bit_widths['output_norm.weight'] = 'F32'
    return component_bit_widths

# Load ZD scores
def load_zd_scores(file_path):
    with open(file_path, 'r') as f:
        zd_scores = json.load(f)
    return {int(k): v for k, v in zd_scores.items()}

# Assign bit-widths based on ZD scores
def assign_bit_widths_zd(zd_scores):
    zd_scores_array = np.array(list(zd_scores.values()))
    zd_norm = (zd_scores_array - zd_scores_array.min()) / (zd_scores_array.max() - zd_scores_array.min() + 1e-8)
    bit_width_values = { 'Q2_K': 2.5625, 'Q3_K': 3.4375, 'Q4_K': 4.5, 'Q5_K': 5.5, 'Q6_K': 6.25, 'Q4_0': 4.5, 'Q4_1': 4.75, 'Q5_0': 5.5, 'Q5_1': 5.75, 'Q8_0': 8.5,}
    component_bit_widths = {}
    part_map = {
        'q_proj': 'attn_q.weight', 'k_proj': 'attn_k.weight', 'v_proj': 'attn_v.weight',
        'o_proj': 'attn_output.weight', 'up_proj': 'ffn_up.weight', 'gate_proj': 'ffn_gate.weight',
        'down_proj': 'ffn_down.weight'
    }
    components = []
    importance_scores = []
    for layer_idx in range(28):
        for part in ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'up_proj', 'gate_proj', 'down_proj']:
            components.append((layer_idx, part))
            importance_scores.append(zd_norm[layer_idx])
    for i, (layer_idx, part) in enumerate(components):
        imp = importance_scores[i]
        name = f'blk.{layer_idx}.{part_map[part]}'
        if imp > 0.75:
            bit_width = 'Q8_0'
        elif imp > 0.5:
            bit_width = 'Q6_K' if part in ['o_proj', 'up_proj', 'gate_proj', 'down_proj'] else 'Q5_K'
        elif imp > 0.25:
            bit_width = 'Q4_K' if part in ['o_proj', 'up_proj', 'gate_proj', 'down_proj'] else 'Q4_0'
        else:
            bit_width = 'Q3_K' if part in ['q_proj', 'k_proj', 'v_proj'] else 'Q4_0'
        if part in ['q_proj', 'o_proj', 'gate_proj', 'up_proj'] and imp > 0.4:
            bit_width = 'Q8_0'
        elif part == 'down_proj' and imp > 0.6:
            bit_width = 'Q8_0'
        component_bit_widths[name] = bit_width
    avg_bit_width = np.mean([bit_width_values.get(bw, 4) for bw in component_bit_widths.values()])
    if avg_bit_width > 3.5:
        sorted_comps = sorted(zip(components, importance_scores), key=lambda x: x[1])
        for (layer_idx, part), _ in sorted_comps[:int(len(components) * 0.3)]:
            name = f'blk.{layer_idx}.{part_map[part]}'
            if component_bit_widths[name] == 'Q5_K':
                component_bit_widths[name] = 'Q4_K'
            elif component_bit_widths[name] == 'Q4_K':
                component_bit_widths[name] = 'Q3_K'
            elif component_bit_widths[name] == 'Q4_0':
                component_bit_widths[name] = 'Q3_K'
    component_bit_widths['output.weight'] = 'Q6_K'
    component_bit_widths['token_embd.weight'] = 'F16'
    component_bit_widths['output_norm.weight'] = 'F32'
    return component_bit_widths

# Parse results/importance_scores2.txt
def parse_importance_scores(file_path):
    layers = defaultdict(list)
    with open(file_path, 'r') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
    current_layer = None
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith('Layer'):
            match = re.match(r'Layer (\d+):', line)
            if match:
                current_layer = int(match.group(1))
                i += 1
            else:
                i += 1
                continue
        elif line.startswith('Part:'):
            match = re.match(r'Part: (\w+) \(index (\d+)\)', line)
            if match:
                part = match.group(1)
                index = int(match.group(2))
                try:
                    gain_line = lines[i + 1]
                    loss_size_line = lines[i + 2]
                    avg_error_line = lines[i + 3]
                    max_error_line = lines[i + 4]
                    gain = float(gain_line.split(': ')[1])
                    loss_size = eval(loss_size_line.split(': ')[1])
                    avg_error = float(avg_error_line.split(': ')[1])
                    max_error = float(max_error_line.split(': ')[1])
                    layers[current_layer].append({
                        'part': part, 'index': index, 'gain': gain, 'avg_error': avg_error,
                        'max_error': max_error, 'loss_size': loss_size
                    })
                    i += 5
                except (IndexError, ValueError):
                    i += 1
                    continue
        else:
            i += 1
    return layers

# Assign bit-widths based on importance
def assign_bit_widths(layers):
    gains = []
    avg_errors = []
    max_errors = []
    components = []
    for layer_idx, comp_list in layers.items():
        for comp in comp_list:
            gains.append(comp['gain'])
            avg_errors.append(comp['avg_error'])
            max_errors.append(comp['max_error'])
            components.append((layer_idx, comp['part']))
    gains = np.array(gains)
    avg_errors = np.array(avg_errors)
    max_errors = np.array(max_errors)
    gain_norm = (gains - gains.min()) / (gains.max() - gains.min() + 1e-8)
    error_norm = (avg_errors - avg_errors.min()) / (avg_errors.max() - avg_errors.min() + 1e-8)
    importance = 0.5 * gain_norm + 0.5 * error_norm
    bit_width_values = { 'Q2_K': 2.5625, 'Q3_K': 3.4375, 'Q4_K': 4.5, 'Q5_K': 5.5, 'Q6_K': 6.25, 'Q4_0': 4.5, 'Q4_1': 4.75, 'Q5_0': 5.5, 'Q5_1': 5.75, 'Q8_0': 8.5,}
    component_bit_widths = {}
    part_map = {
        'q_proj': 'attn_q.weight', 'k_proj': 'attn_k.weight', 'v_proj': 'attn_v.weight',
        'o_proj': 'attn_output.weight', 'up_proj': 'ffn_up.weight', 'gate_proj': 'ffn_gate.weight',
        'down_proj': 'ffn_down.weight'
    }
    for i, (layer_idx, part) in enumerate(components):
        imp = importance[i]
        max_err = max_errors[i]
        name = f'blk.{layer_idx}.{part_map[part]}'
        if max_err > 1000 or imp > 0.75:
            bit_width = 'Q8_0'
        elif imp > 0.5:
            bit_width = 'Q6_K' if part in ['o_proj', 'up_proj', 'gate_proj', 'down_proj'] else 'Q5_K'
        elif imp > 0.25:
            bit_width = 'Q4_K' if part in ['o_proj', 'up_proj', 'gate_proj', 'down_proj'] else 'Q4_0'
        else:
            bit_width = 'Q3_K' if part in ['q_proj', 'k_proj', 'v_proj'] else 'Q4_0'
        if part in ['q_proj', 'o_proj', 'gate_proj', 'up_proj'] and imp > 0.4:
            bit_width = 'Q8_0'
        elif part == 'down_proj' and max_err > 500:
            bit_width = 'Q8_0'
        component_bit_widths[name] = bit_width
    avg_bit_width = np.mean([bit_width_values.get(bw, 4) for bw in component_bit_widths.values()])
    if avg_bit_width > 3.5:
        sorted_comps = sorted(zip(components, importance), key=lambda x: x[1])
        for (layer_idx, part), _ in sorted_comps[:int(len(components) * 0.3)]:
            name = f'blk.{layer_idx}.{part_map[part]}'
            if component_bit_widths[name] == 'Q5_K':
                component_bit_widths[name] = 'Q4_K'
            elif component_bit_widths[name] == 'Q4_K':
                component_bit_widths[name] = 'Q3_K'
            elif component_bit_widths[name] == 'Q4_0':
                component_bit_widths[name] = 'Q3_K'
    component_bit_widths['output.weight'] = 'Q6_K'
    component_bit_widths['token_embd.weight'] = 'F16'
    component_bit_widths['output_norm.weight'] = 'F32'
    return component_bit_widths

# Generate JSON output
def generate_llama_cpp_config(bit_widths, output_file):
    with open(output_file, 'w') as f:
        json.dump(bit_widths, f, indent=4)

# Generate CSV output
def generate_csv_output(lim, zd, baq, output_file):
    with open(output_file, 'w') as f:
        f.write('key,lim,zd,baq\n')
        keys = set(lim.keys()).union(zd.keys(), baq.keys())
        for key in sorted(keys):
            row = [key, lim.get(key, ''), zd.get(key, ''), baq.get(key, '')]
            f.write(','.join(row) + '\n')

# Main execution
if __name__ == "__main__":
    try:
        # Process lim_scores.json
        lim_scores = load_lim_scores('lim_scores.json')
        lim_bit_widths = assign_bit_widths_lim(lim_scores)
        generate_llama_cpp_config(lim_bit_widths, 'final_results/lim.json')

        # Process zd_scores.json
        zd_scores = load_zd_scores('zd_scores.json')
        zd_bit_widths = assign_bit_widths_zd(zd_scores)
        generate_llama_cpp_config(zd_bit_widths, 'final_results/zd.json')

        # Process results/importance_scores2.txt
        layers = parse_importance_scores('results/importance_scores2.txt')
        imp_bit_widths = assign_bit_widths(layers)
        generate_llama_cpp_config(imp_bit_widths, 'final_results/baq.json')

        # Generate CSV
        generate_csv_output(lim_bit_widths, zd_bit_widths, imp_bit_widths, 'final_results/quantization_comparison.csv')
        print("Generated lim.json, zd.json, imp.json, and quantization_comparison.csv.")
    except FileNotFoundError as e:
        print(f"Error: File not found: {e}")
    except Exception as e:
        print(f"Error: Failed to process: {e}")