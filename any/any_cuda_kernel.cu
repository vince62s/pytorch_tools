#include <cuda_runtime.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

__global__ void any_kernel_warp_vote(const bool* input, int64_t size, unsigned int* result) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    bool found = (idx < size) ? input[idx] : false;
    
    unsigned int vote = __ballot_sync(0xFFFFFFFF, found);  // Warp-wide voting

    if (vote) {  // If any thread in warp found a True, update global memory
        atomicOr(result, 1u);
    }
}

void any_cuda_kernel_launch(const bool* input, int64_t size, unsigned int* result, cudaStream_t stream) {
    int block_size = 256;
    int num_blocks = (size + block_size - 1) / block_size;
    any_kernel_warp_vote<<<num_blocks, block_size, 0, stream>>>(input, size, result);
}

