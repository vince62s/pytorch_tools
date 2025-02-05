import torch

def apply_rotary_emb_complex(rope):
    cos, sin = rope
    rope_complex = torch.complex(cos, sin)
    rope_complex = rope_complex[:, :rope_complex.size(1) // 2].view(1, 100, 1, 64)
    return rope_complex
    
def apply_rotary_emb(rope):
    cos, sin = rope
    cos = cos[:, :cos.size(1) // 2].view(1, 100, 1, 64)
    sin = sin[:, :sin.size(1) // 2].view(1, 100, 1, 64)
    return cos, sin

# Example usage
cos = torch.randn(100, 128)              # Example shape
sin = torch.randn(100, 128)              # Example shape
rope = (cos, sin)

rope_complex = apply_rotary_emb_complex(rope)
cos_manual, sin_manual = apply_rotary_emb(rope)

print("Rope complex real:", rope_complex.real)
print("Rope complex imag:", rope_complex.imag)
print("Cos manual:", cos_manual)
print("Sin manual:", sin_manual)

# Check if the rope values are the same
print("Cos allclose:", torch.allclose(rope_complex.real, cos_manual, atol=1e-6))
print("Sin allclose:", torch.allclose(rope_complex.imag, sin_manual, atol=1e-6))

