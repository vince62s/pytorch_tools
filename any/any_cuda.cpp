#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>

// Declaration of the CUDA kernel launcher
void any_cuda_kernel_launch(const bool* input, int64_t size, unsigned int* result, cudaStream_t stream);

torch::Tensor any_cuda(torch::Tensor input) {
    TORCH_CHECK(input.device().is_cuda(), "Input tensor must be on GPU");
    TORCH_CHECK(input.scalar_type() == torch::kBool, "Input tensor must be bool");
    
    auto size = input.numel();
    unsigned int* result;
    cudaMallocManaged(&result, sizeof(unsigned int));
    *result = 0;

    // Get cuda stream from current context
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    
    // Launch kernel
    any_cuda_kernel_launch(
        input.data_ptr<bool>(),
        size,
        result,
        stream
    );

    // Wait for GPU to finish
    cudaDeviceSynchronize();
    bool result_value = (*result != 0);
    cudaFree(result);
    
    return torch::tensor(result_value, input.options().dtype(torch::kBool));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &any_cuda, "Any CUDA forward");
}
