import torch

def apply_rotary_emb_complex(query, key, rope):
    if True:
        cos, sin = rope
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        query_ = query.float().reshape(*query.shape[:-1], -1, 2)
        key_ = key.float().reshape(*key.shape[:-1], -1, 2)
        
        # Complex operations
        rope = torch.polar(cos, sin)
        
        query_ = torch.view_as_complex(query_)
        key_ = torch.view_as_complex(key_)
        
        rope = rope[:, :rope.size(1) // 2].view(1, query_.size(1), 1, query_.size(3))
        
        query_out = torch.view_as_real(query_ * rope).flatten(3)
        key_out = torch.view_as_real(key_ * rope).flatten(3)
        res2 = query_out.transpose(1, 2).type_as(query), key_out.transpose(1, 2).type_as(key)

        return res2


def apply_rotary_emb(query, key, rope):
    if True:
        cos, sin = rope
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        query_ = query.float().reshape(*query.shape[:-1], -1, 2)
        key_ = key.float().reshape(*key.shape[:-1], -1, 2)

        # Reshape cos and sin to match the dimensions of query_ and key_
        cos = cos[:, :cos.size(1) // 2].view(1, query_.size(1), 1, query_.size(3))
        sin = sin[:, :sin.size(1) // 2].view(1, key_.size(1), 1, key_.size(3))

        # Apply the rotary embedding
        query_real = query_[..., 0]
        query_imag = query_[..., 1]
        key_real = key_[..., 0]
        key_imag = key_[..., 1]

        query_rotated = query_real * cos - query_imag * sin
        query_rotated_imag = query_imag * cos + query_real * sin
        query_out = torch.stack((query_rotated, query_rotated_imag), dim=-1).flatten(3)

        key_rotated = key_real * cos - key_imag * sin
        key_rotated_imag = key_imag * cos + key_real * sin
        key_out = torch.stack((key_rotated, key_rotated_imag), dim=-1).flatten(3)

        res1 = query_out.transpose(1, 2).type_as(query), key_out.transpose(1, 2).type_as(key)

        return res1

# Example usage
query = torch.randn(2, 32, 100, 64 * 2)  # Example shape
key = torch.randn(2, 32, 100, 64 * 2)    # Example shape
cos = torch.randn(100, 128)              # Example shape
sin = torch.randn(100, 128)              # Example shape
rope = (cos, sin)

res2 = apply_rotary_emb_complex(query.clone(), key.clone(), rope)
res1 = apply_rotary_emb(query.clone(), key.clone(), rope)

print("Query shapes:", res1[0].shape, res2[0].shape)
print("Key shapes:", res1[1].shape, res2[1].shape)

print(torch.allclose(res1[0], res2[0], atol=1e-6))  # Check for equality with a tolerance
print(torch.allclose(res1[1], res2[1], atol=1e-6))  # Check for equality with a tolerance

