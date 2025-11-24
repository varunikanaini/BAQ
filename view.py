import torch

# Load the saved gain_vec_tensor.pt
gain_vec_tensor = torch.load("gain_vec_tensor.pt")

# Print the number of gain vectors
print(f"Number of gain vectors: {len(gain_vec_tensor)}")

# Print the shape and some statistics for each gain vector
for idx, gain_vec in enumerate(gain_vec_tensor):
    print(f"gain vector {idx}:")
    print(f"  Shape: {gain_vec.shape}")
    print(f"  Mean: {gain_vec.mean().item():.6f}")
    print(f"  Std: {gain_vec.std().item():.6f}")
    print(f"  Min: {gain_vec.min().item():.6f}")
    print(f"  Max: {gain_vec.max().item():.6f}")

# Optionally, save the statistics to a text file for easier inspection
with open("gain_vec_stats2.txt", "w") as f:
    f.write(f"Number of gain vectors: {len(gain_vec_tensor)}\n")
    for idx, gain_vec in enumerate(gain_vec_tensor):
        f.write(f"gain vector {idx}:\n")
        f.write(f"  Shape: {gain_vec.shape}\n")
        f.write(f"  Mean: {gain_vec.mean().item():.6f}\n")
        f.write(f"  Std: {gain_vec.std().item():.6f}\n")
        f.write(f"  Min: {gain_vec.min().item():.6f}\n")
        f.write(f"  Max: {gain_vec.max().item():.6f}\n")

print("Statistics saved to 'gain_vec_stats2.txt'")