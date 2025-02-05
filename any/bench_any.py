import torch
import time
import torch.utils.cpp_extension as cpp_extension
import os

cuda_home = os.environ.get('CUDA_HOME', '/usr/local/cuda')
print(f"Using CUDA_HOME: {cuda_home}")
    
cuda_extension = cpp_extension.load(
    name='any_cuda_kernel',
    sources=['any_cuda.cpp', 'any_cuda_kernel.cu'],
    with_cuda=True,
    extra_cflags=['-O3'],
    extra_cuda_cflags=['-O3'],
    verbose=True
)

def any_cuda(input_tensor):
    return cuda_extension.forward(input_tensor)


# Function to benchmark the 'any' operation on list and tensor
def benchmark_any_operation(size, true_elements_ratio):
    # Create a list and tensor of the same size and with the same number of True elements
    num_true_elements = int(size * true_elements_ratio)

    # Create a list (Python list)
    mylist = [True] * num_true_elements + [False] * (size - num_true_elements)
    # Shuffle the list to mix True and False values
    from random import shuffle
    shuffle(mylist)

    # Create a tensor (PyTorch tensor)
    mytensor_cpu = torch.tensor(mylist)
    mytensor_cuda = mytensor_cpu.cuda() if torch.cuda.is_available() else None

    # Benchmark 'any' on the Python list
    start_time = time.time()
    list_result = any(mylist)
    list_time = time.time() - start_time

    # Benchmark 'any' on the PyTorch tensor (CPU)
    start_time = time.time()
    tensor_cpu_result = mytensor_cpu.any()
    tensor_cpu_time = time.time() - start_time

    # Benchmark 'any' on the PyTorch tensor (CUDA), only if CUDA is available
    if mytensor_cuda is not None:
        torch.cuda.synchronize()
        start_time = time.time()
        tensor_cuda_result = any_cuda(mytensor_cuda)
        #tensor_cuda_result = mytensor_cuda.any()
        torch.cuda.synchronize()
        tensor_cuda_time = time.time() - start_time
        print(f"Tensor 'any()' on CUDA result: {tensor_cuda_result}, Time: {tensor_cuda_time:.6f} seconds")
    else:
        tensor_cuda_result = None
        tensor_cuda_time = 0.0

    # Print results
    print(f"List 'any()' result: {list_result}, Time: {list_time:.6f} seconds")
    print(f"Tensor 'any()' on CPU result: {tensor_cpu_result}, Time: {tensor_cpu_time:.6f} seconds")

# Example usage:
# Size of the list and tensor
size = 1000000
# Ratio of True elements
true_elements_ratio = 0.1  # 10% of elements are True

benchmark_any_operation(size, true_elements_ratio)

