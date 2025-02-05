import torch
import time

# Define tensors
batch_size, dim1, dim2 = 1000, 100, 100
tensor = torch.randn(batch_size, dim1, dim2)  # Shape [100, 100, 100]
dim = 0
indices = torch.randint(0, batch_size, (100,))  # Randomly select 200 indices

# One-liner advanced indexing
def custom_index_select(tensor, dim, indices):
    return tensor[(slice(None),) * dim + (indices,) + (slice(None),) * (tensor.dim() - dim - 1)]


print(tensor.size())
print(indices.size())

# on CPU

# Time `index_select`
start_time = time.time()
for _ in range(100):
    result_index_select = torch.index_select(tensor, dim, indices)
end_time = time.time()
time_index_select = end_time - start_time

# Time custom indexing
start_time = time.time()
for _ in range(100):
    result_custom = custom_index_select(tensor, dim, indices)
end_time = time.time()
time_custom_indexing = end_time - start_time

# Time pytorch indexing
start_time = time.time()
for _ in range(100):
    # only for dim=0
    result_pytorch = tensor[indices]
end_time = time.time()
time_pytorch_indexing = end_time - start_time


# on GPU
tensor = tensor.cuda()
indices = indices.cuda()
# Time `index_select`
torch.cuda.synchronize()
start_time = time.time()
for _ in range(100):
    result_index_select_cuda = torch.index_select(tensor, dim, indices)
torch.cuda.synchronize()
end_time = time.time()
time_index_select_cuda = end_time - start_time

# Time custom indexing
torch.cuda.synchronize()
start_time = time.time()
for _ in range(100):
    result_custom_cuda = custom_index_select(tensor, dim, indices)
torch.cuda.synchronize()
end_time = time.time()
time_custom_indexing_cuda = end_time - start_time

# Time pytorch indexing on CUDA

torch.cuda.synchronize()
start_time = time.time()
for _ in range(100):
    # only for dim=0
    result_pytorch_cuda = tensor[indices]
torch.cuda.synchronize()
end_time = time.time()
time_pytorch_cuda = end_time - start_time


# Print results
print(f"Time taken by index_select: {time_index_select:.6f} seconds")
print(f"Time taken by custom indexing: {time_custom_indexing:.6f} seconds")
print(f"Time taken by pytorch indexing: {time_pytorch_indexing:.6f} seconds")
print(f"Time taken by index_select cuda: {time_index_select_cuda:.6f} seconds")
print(f"Time taken by custom indexing cuda: {time_custom_indexing_cuda:.6f} seconds")
print(f"Time taken by pytorch cuda: {time_pytorch_cuda:.6f} seconds")


"""
torch.Size([1000, 100, 100])
torch.Size([100])
Time taken by index_select: 0.038408 seconds
Time taken by custom indexing: 0.010384 seconds
Time taken by pytorch indexing: 0.009310 seconds
Time taken by index_select cuda: 0.057033 seconds
Time taken by custom indexing cuda: 0.014972 seconds
Time taken by pytorch cuda: 0.000663 seconds

This means that in a BeamSearch context when reducing the large batch x beam tensor to selected indices, it is quicker to perform on CUDA

"""

