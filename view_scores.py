import torch

# Load the saved files
loss_vectors = torch.load("results/loss_vec_tensor.pt")
gain_vectors = torch.load("results/gain_vec_tensor.pt")

# The model has 28 layers, each with 7 parts
num_layers = 28
num_parts = 7
part_names = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'up_proj', 'gate_proj', 'down_proj']

# Open a file to save the output
with open("results/importance_scores2.txt", "w") as f:
    # Write and print header
    header = f"Total loss vectors: {len(loss_vectors)}\nTotal gain vectors: {len(gain_vectors)}\n"
    print(header, end="")
    f.write(header)

    # Go through each layer and part
    for layer in range(num_layers):
        layer_header = f"\nLayer {layer}:\n"
        print(layer_header, end="")
        f.write(layer_header)
        for part_idx in range(num_parts):
            vector_idx = layer * num_parts + part_idx  # Calculate which vector
            loss_vec = loss_vectors[vector_idx]
            gain = gain_vectors[vector_idx].item()
            output = (
                f"  Part: {part_names[part_idx]} (index {vector_idx})\n"
                f"    Gain: {gain:.6f}\n"
                f"    Loss Vector Size: {loss_vec.shape}\n"
                f"    Average Error: {loss_vec.mean().item():.6f}\n"
                f"    Biggest Error: {loss_vec.max().item():.6f}\n"
            )
            print(output, end="")
            f.write(output)

print("Output saved to 'results/importance_scores2.txt'")