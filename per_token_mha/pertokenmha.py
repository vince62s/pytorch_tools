import time
import torch
import torch.nn as nn
import torch.cuda as cuda
import torch.nn.functional as F


def generate_matching_tokens(batch_size, seq_len, embed_dim, match_ratio=0.5, device="cpu"):
    """
    Generates a tensor with random token values and introduces a controlled ratio of duplicate tokens.

    Args:
        batch_size (int): Batch size for the tensor.
        seq_len (int): Sequence length for the tensor.
        embed_dim (int): Embedding dimension for each token.
        match_ratio (float): Proportion of tokens that are duplicates from the previous sequence.
        device (str): Device to create the tensor on (e.g., 'cpu' or 'cuda').

    Returns:
        torch.Tensor: Generated tensor with matching tokens based on match_ratio.
    """
    # Step 1: Generate random token embeddings (random initial sequence)
    tokens = torch.randn(batch_size, seq_len, embed_dim, device=device)

    prompt = torch.rand(seq_len // 2, embed_dim, device=device).unsqueeze(0)
    
    tokens[:, :seq_len // 2, :] = prompt
    

    return tokens


class RefactoredMHA(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # Linear projections for query, key, value
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        # Persistent token and projection caches
        self.token_cache = torch.zeros(0, embed_dim)  # Persistent cache for unique tokens (dim: [N_tokens, embed_dim])
        self.k_cache = torch.zeros(0, embed_dim)  # Persistent cache for k_proj (dim: [N_tokens, embed_dim])
        self.v_cache = torch.zeros(0, embed_dim)  # Persistent cache for v_proj (dim: [N_tokens, embed_dim])
        self.index_cache = torch.zeros(0, dtype=torch.long)  # Per step index mapping (dim: [batch, seq_len])

    def forward(self, query, step=0):
        """
        Forward pass for multi-head attention with deduplicated token cache.

        Args:
            query (torch.Tensor): Input query tensor of shape (batch, seq_len, embed_dim).
            step (int): Current decoding step (0 for the first step).
        """
        batch, seq_len, embed_dim = query.shape

        # Step 1: Deduplicate tokens
        unique_tokens, unique_indices = self._deduplicate_tokens(query)

        # Step 2: Compute Projections for New Tokens
        if step == 0:
            # For the first step, process all unique tokens
            k_new = self.k_proj(unique_tokens)
            v_new = self.v_proj(unique_tokens)

            # Initialize caches
            self.token_cache = unique_tokens
            self.k_cache = k_new
            self.v_cache = v_new
            self.index_cache = unique_indices
        else:
            # For subsequent steps, process only new unique tokens
            new_tokens, new_indices = self._update_token_cache(unique_tokens)
            k_new = self.k_proj(new_tokens)
            v_new = self.v_proj(new_tokens)

            # Update persistent caches
            self.token_cache = torch.cat([self.token_cache, new_tokens], dim=0)
            self.k_cache = torch.cat([self.k_cache, k_new], dim=0)
            self.v_cache = torch.cat([self.v_cache, v_new], dim=0)

            self.index_cache = new_indices

        # Step 3: Reconstruct KV Cache
        kv_cache_k = self.k_cache[self.index_cache]
        kv_cache_v = self.v_cache[self.index_cache]

        # Step 4: Compute Attention
        q = self.q_proj(query)
        attn_weights = torch.matmul(q, kv_cache_k.transpose(-2, -1)) / (self.embed_dim ** 0.5)
        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, kv_cache_v)

        return attn_output

    def _deduplicate_tokens(self, tokens):
        """
        Deduplicate tokens and return unique embeddings with indices.

        Args:
            tokens (torch.Tensor): Input tensor of shape (batch, seq_len, embed_dim).
        
        Returns:
            unique_tokens (torch.Tensor): Unique embeddings (dim: [N_unique, embed_dim]).
            indices (torch.Tensor): Indices pointing to unique tokens (dim: [batch, seq_len]).
        """
        # Flatten batch and seq_len for deduplication
        flat_tokens = tokens.view(-1, tokens.size(-1))
        
        # Find unique tokens and their indices
        unique_tokens, inverse_indices = torch.unique(flat_tokens, dim=0, return_inverse=True)

        # Reshape indices to match the original shape
        indices = inverse_indices.view(tokens.shape[:-1])
        return unique_tokens, indices

    def _update_token_cache(self, unique_tokens):
        """
        Update the persistent token cache with new unique tokens.

        Args:
            unique_tokens (torch.Tensor): Unique tokens from the current step.

        Returns:
            new_tokens (torch.Tensor): New unique tokens not already in the cache.
            new_cache_indices (torch.Tensor): Indices mapping to the updated cache.
        """
        # Compare new unique tokens with the existing token cache
        combined_tokens = torch.cat([self.token_cache, unique_tokens], dim=0)
        unique_tokens, inverse_indices = torch.unique(combined_tokens, dim=0, return_inverse=True)

        # Get the new tokens that weren't in the original cache
        new_tokens = unique_tokens[len(self.token_cache):]
        new_cache_indices = inverse_indices[len(self.token_cache):]
        return new_tokens, new_cache_indices


class LegacyMHA(nn.Module):
    """
    Legacy multi-head attention that does not use deduplication.
    """
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, kv_cache=None):
        """
        Forward pass for the legacy implementation.
        Args:
            query (torch.Tensor): Input query tensor of shape (batch, seq_len, embed_dim).
            kv_cache (dict): Persistent key/value cache.
        """
        if kv_cache is None:
            kv_cache = {'k': None, 'v': None}

        k = self.k_proj(query)
        v = self.v_proj(query)

        if kv_cache['k'] is None:
            kv_cache['k'] = k
            kv_cache['v'] = v
        else:
            kv_cache['k'] = torch.cat([kv_cache['k'], k], dim=1)
            kv_cache['v'] = torch.cat([kv_cache['v'], v], dim=1)

        q = self.q_proj(query)
        attn_weights = torch.matmul(q, kv_cache['k'].transpose(-2, -1)) / (self.embed_dim ** 0.5)
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, kv_cache['v'])

        return attn_output, kv_cache


def benchmark_mha(impl_class, num_steps, batch_size, seq_len, embed_dim, num_heads, device):
    """
    Benchmark runtime and memory for an MHA implementation.

    Args:
        impl_class: The MHA class to benchmark.
        num_steps: Number of incremental steps to simulate.
        batch_size: Batch size for the input.
        seq_len: Sequence length for the initial input.
        embed_dim: Embedding dimension of the input.
        num_heads: Number of attention heads.
        device: The device to run the benchmark on ('cpu' or 'cuda').

    Returns:
        A dictionary containing runtime and memory statistics.
    """
    model = impl_class(embed_dim, num_heads).to(device)
    # query = torch.randn(batch_size, seq_len, embed_dim, device=device)
    query = generate_matching_tokens(batch_size, seq_len, embed_dim, match_ratio=0.1, device=device)
    kv_cache = None

    # Warm-up
    for _ in range(5):
        if isinstance(model, RefactoredMHA):
            model.forward(query, step=0)
        else:
            model.forward(query, kv_cache=kv_cache)

    # Benchmark runtime
    torch.cuda.synchronize()
    start_time = time.time()
    if isinstance(model, RefactoredMHA):
        model.forward(query, step=0)
    else:
        output, kv_cache = model.forward(query, kv_cache=kv_cache)    
    for step in range(num_steps):
        step_query = generate_matching_tokens(batch_size, 1, embed_dim, match_ratio=0.1, device=device)
        if isinstance(model, RefactoredMHA):
            model.forward(step_query, step=step+1)
        else:
            output, kv_cache = model.forward(step_query, kv_cache=kv_cache)
    torch.cuda.synchronize()
    runtime = time.time() - start_time

    # Measure memory usage
    memory_allocated = cuda.memory_allocated(device) if device == 'cuda' else 0

    return {'runtime': runtime, 'memory_allocated': memory_allocated}


def main():
    # Benchmark parameters
    num_steps = 50        # Number of incremental decoding steps
    batch_size = 16       # Batch size
    seq_len = 20          # Initial sequence length
    embed_dim = 4096        # Embedding dimension
    num_heads = 4         # Number of attention heads
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Run benchmarks
    print("Benchmarking LegacyMHA...")
    legacy_results = benchmark_mha(LegacyMHA, num_steps, batch_size, seq_len, embed_dim, num_heads, device)

    print("Benchmarking RefactoredMHA...")
    refactored_results = benchmark_mha(RefactoredMHA, num_steps, batch_size, seq_len, embed_dim, num_heads, device)

    # Print results
    print("\nResults:")
    print(f"LegacyMHA - Runtime: {legacy_results['runtime']:.4f}s, "
          f"Memory Allocated: {legacy_results['memory_allocated'] / 1e6:.2f}MB")
    print(f"RefactoredMHA - Runtime: {refactored_results['runtime']:.4f}s, "
          f"Memory Allocated: {refactored_results['memory_allocated'] / 1e6:.2f}MB")


if __name__ == "__main__":
    main()

