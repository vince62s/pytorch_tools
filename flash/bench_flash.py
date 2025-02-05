import torch
import torchaudio
import torch.utils.benchmark as benchmark
import time
from einops import rearrange, repeat, reduce, pack, unpack
from flash_attn import flash_attn_func, flash_attn_varlen_func
from xformers.ops import fmha
#
# Parameters
#

n_heads = 16
n_len = 750
q = torch.rand(n_len, 128 * n_heads, dtype=torch.float16, device="cuda")
k = torch.rand(n_len, 128 * n_heads, dtype=torch.float16, device="cuda")
v = torch.rand(n_len, 128 * n_heads, dtype=torch.float16, device="cuda")
lengths = [100, 150, 150, 150, 200]

#
# Implementations
# 

def torch_attention(q, k, v, mask):
    q, k, v = map(lambda t: rearrange(t, 'n (h d) -> 1 h n d', h = n_heads), (q, k, v))
    y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask = mask)
    y = rearrange(y, '1 h n d -> n (h d)')
    return y
def torch_mask(lengths):
    return fmha.BlockDiagonalMask.from_seqlens(lengths).materialize(shape=(n_len, n_len)).cuda().to(torch.float16)
def flash_attention(q, k, v, mask):
    if mask is None:
        q, k, v = map(lambda t: rearrange(t, 'n (h d) -> 1 n h d', h = n_heads), (q, k, v))
        y = flash_attn_func(q, k, v)
        y = rearrange(y, '1 n h d -> n (h d)')
        return y
    else:
        q, k, v = map(lambda t: rearrange(t, 'n (h d) -> n h d', h = n_heads), (q, k, v))
        y = flash_attn_varlen_func(q, k, v, cu_seqlens_q=mask[1], cu_seqlens_k=mask[1], max_seqlen_q=mask[0], max_seqlen_k=mask[0])
        y = rearrange(y, 'n h d -> n (h d)')
        return y
def flash_mask(lengths):
    seq_lens = torch.tensor(lengths, dtype=torch.int32)
    max_seq_len = seq_lens.max().item()
    return (torch.tensor(max_seq_len, dtype=torch.int32).cuda(), torch.concat([torch.tensor([0]), seq_lens.cumsum(0)]).to(torch.int32).cuda())

#
# Check implementations
# 

torch_m = torch_mask(lengths)
flash_m = flash_mask(lengths)
b = torch_attention(q, k, v, torch_m)
c = flash_attention(q, k, v, flash_m)
print((b - c).abs().max())

#
# Benchmarking
# 
torch.cuda.synchronize()
start = time.time()
for i in range(100):
    torch_attention(q, k, v, torch_m)
torch.cuda.synchronize()
print("torch (mask)", time.time() - start)
torch.cuda.synchronize()
start = time.time()
for i in range(100):
    torch_attention(q, k, v, None)
torch.cuda.synchronize()
print("torch (no mask)", time.time() - start)
torch.cuda.synchronize()
start = time.time()
for i in range(100):
    flash_attention(q, k, v, flash_m)
torch.cuda.synchronize()
print("flash (mask)", time.time() - start)
torch.cuda.synchronize()
start = time.time()
for i in range(100):
    flash_attention(q, k, v, None)
torch.cuda.synchronize()
print("flash (no mask) varlen ", time.time() - start)
