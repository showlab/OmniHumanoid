import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional
from einops import rearrange
from .utils import hash_state_dict_keys
from .wan_video_camera_controller import SimpleAdapter
from typing import Any, Dict, Optional, Tuple, Union

# try:
#     import flash_attn_interface
#     FLASH_ATTN_3_AVAILABLE = True
# except ModuleNotFoundError:
FLASH_ATTN_3_AVAILABLE = False

# try:
#     import flash_attn
#     FLASH_ATTN_2_AVAILABLE = True
# except ModuleNotFoundError:
FLASH_ATTN_2_AVAILABLE = False

# try:
#     from sageattention import sageattn
#     SAGE_ATTN_AVAILABLE = True
# except ModuleNotFoundError:
SAGE_ATTN_AVAILABLE = False
   
class LoRALinearLayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 64,
        network_alpha: Optional[float] = None,
        device: Optional[Union[torch.device, str]] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.down = nn.Linear(in_features, rank, bias=False, device=device, dtype=dtype)
        self.up = nn.Linear(rank, out_features, bias=False, device=device, dtype=dtype)
        # This value has the same meaning as the `--network_alpha` option in the kohya-ss trainer script.
        # See https://github.com/darkstorm2150/sd-scripts/blob/main/docs/train_network_README-en.md#execute-learning
        self.network_alpha = network_alpha
        self.rank = rank
        self.out_features = out_features
        self.in_features = in_features

        nn.init.normal_(self.down.weight, std=1 / rank)
        nn.init.zeros_(self.up.weight)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        orig_dtype = hidden_states.dtype
        dtype = self.down.weight.dtype
        self.dim = hidden_states.shape[-1]
    
        down_hidden_states = self.down(hidden_states.to(dtype))
        up_hidden_states = self.up(down_hidden_states)

        if self.network_alpha is not None:
            up_hidden_states *= self.network_alpha / self.rank

        return up_hidden_states.to(orig_dtype) 
    

def flash_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, num_heads: int, compatibility_mode=False, mask=None):
    if mask is not None:
        q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        x = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    elif compatibility_mode:
        q = rearrange(q, "b s (n d) -> b n s d", n=num_heads).to(torch.bfloat16)
        k = rearrange(k, "b s (n d) -> b n s d", n=num_heads).to(torch.bfloat16)
        v = rearrange(v, "b s (n d) -> b n s d", n=num_heads).to(torch.bfloat16)
        x = F.scaled_dot_product_attention(q, k, v)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    elif FLASH_ATTN_3_AVAILABLE:
        q = rearrange(q, "b s (n d) -> b s n d", n=num_heads).to(torch.bfloat16)
        k = rearrange(k, "b s (n d) -> b s n d", n=num_heads).to(torch.bfloat16)
        v = rearrange(v, "b s (n d) -> b s n d", n=num_heads).to(torch.bfloat16)
        x = flash_attn_interface.flash_attn_func(q, k, v)
        if isinstance(x,tuple):
            x = x[0]
        x = rearrange(x, "b s n d -> b s (n d)", n=num_heads)
    elif FLASH_ATTN_2_AVAILABLE:
        q = rearrange(q, "b s (n d) -> b s n d", n=num_heads).to(torch.bfloat16)
        k = rearrange(k, "b s (n d) -> b s n d", n=num_heads).to(torch.bfloat16)
        v = rearrange(v, "b s (n d) -> b s n d", n=num_heads).to(torch.bfloat16)
        x = flash_attn.flash_attn_func(q, k, v)
        x = rearrange(x, "b s n d -> b s (n d)", n=num_heads)
    elif SAGE_ATTN_AVAILABLE:
        q = rearrange(q, "b s (n d) -> b n s d", n=num_heads).to(torch.bfloat16)
        k = rearrange(k, "b s (n d) -> b n s d", n=num_heads).to(torch.bfloat16)
        v = rearrange(v, "b s (n d) -> b n s d", n=num_heads).to(torch.bfloat16)
        x = sageattn(q, k, v)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    else:
        q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        x = F.scaled_dot_product_attention(q, k, v)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    return x


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor):
    return (x * (1 + scale) + shift)

def pad_for_3d_conv(x, kernel_size):
    b, c, t, h, w = x.shape
    pt, ph, pw = kernel_size
    pad_t = (pt - (t % pt)) % pt
    pad_h = (ph - (h % ph)) % ph
    pad_w = (pw - (w % pw)) % pw
    return torch.nn.functional.pad(x, (0, pad_w, 0, pad_h, 0, pad_t), mode='replicate')

def sinusoidal_embedding_1d(dim, position):
    sinusoid = torch.outer(position.type(torch.float64), torch.pow(
        10000, -torch.arange(dim//2, dtype=torch.float64, device=position.device).div(dim//2)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x.to(position.dtype)


def precompute_freqs_cis_3d(dim: int, end: int = 1024, theta: float = 10000.0):
    # 3d rope precompute
    f_freqs_cis = precompute_freqs_cis(dim - 2 * (dim // 3), end, theta)
    h_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    w_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    return f_freqs_cis, h_freqs_cis, w_freqs_cis


def precompute_freqs_cis(dim: int, end: int = 1024, theta: float = 10000.0):
    # 1d rope precompute
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)
                   [: (dim // 2)].double() / dim))
    freqs = torch.outer(torch.arange(end, device=freqs.device), freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis

def precompute_cond_freqs_cis_3d(dim: int, end: int = 1024, theta: float = 10000.0, scale=2.0):
    # 3d rope precompute
    f_freqs_cis = precompute_cond_freqs_cis(dim - 2 * (dim // 3), end, theta, scale)
    h_freqs_cis = precompute_cond_freqs_cis(dim // 3, end, theta, scale)
    w_freqs_cis = precompute_cond_freqs_cis(dim // 3, end, theta, scale)
    return f_freqs_cis, h_freqs_cis, w_freqs_cis

def precompute_cond_freqs_cis(dim: int, end: int = 1024, theta: float = 10000.0, scale=2.0):
    # 1d rope precompute
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)
                   [: (dim // 2)].double() / dim))
    positions = torch.arange(0, end, scale, dtype=torch.float64, device=freqs.device)
    freqs = torch.outer(positions, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def rope_apply(x, freqs, num_heads):
    x = rearrange(x, "b s (n d) -> b s n d", n=num_heads)
    x_out = torch.view_as_complex(x.to(torch.float64).reshape(
        x.shape[0], x.shape[1], x.shape[2], -1, 2))
    x_out = torch.view_as_real(x_out * freqs).flatten(2)
    return x_out.to(x.dtype)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def forward(self, x):
        dtype = x.dtype
        return self.norm(x.float()).to(dtype) * self.weight


class AttentionModule(nn.Module):
    def __init__(self, num_heads):
        super().__init__()
        self.num_heads = num_heads
        
    def forward(self, q, k, v, mask=None):
        x = flash_attention(q=q, k=k, v=v, num_heads=self.num_heads, mask=mask)
        return x

class SelfAttention_ref(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6, rank=128, network_alpha=128):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)
        
        self.q_lora = LoRALinearLayer(dim, dim, rank, network_alpha)
        self.k_lora = LoRALinearLayer(dim, dim, rank, network_alpha)
        self.v_lora = LoRALinearLayer(dim, dim, rank, network_alpha)
        self.o_lora = LoRALinearLayer(dim, dim, rank, network_alpha)
        
        self.attn = AttentionModule(self.num_heads)
        
    def forward(self, x, freqs, return_qkv=False):
        q = self.norm_q(self.q(x)+self.q_lora(x))
        k = self.norm_k(self.k(x)+self.k_lora(x))
        v = self.v(x)+self.v_lora(x)
        q = rope_apply(q, freqs, self.num_heads)
        k = rope_apply(k, freqs, self.num_heads)
        if return_qkv is True:
            return q, k, v
        else:
            x = self.attn(q, k, v)
            return self.o(x) + self.o_lora(x)


class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)
        
        self.attn = AttentionModule(self.num_heads)
        
    def forward(self, x, freqs, return_qkv=False):
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(x))
        v = self.v(x)
        q = rope_apply(q, freqs, self.num_heads)
        k = rope_apply(k, freqs, self.num_heads)
        if return_qkv is True:
            return q, k, v
        else:
            x = self.attn(q, k, v)
            return self.o(x)

class CrossAttention_ref(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6, has_image_input: bool = False, rank=128, network_alpha=128):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)
        self.has_image_input = has_image_input
        if has_image_input:
            self.k_img = nn.Linear(dim, dim)
            self.v_img = nn.Linear(dim, dim)
            self.norm_k_img = RMSNorm(dim, eps=eps)
                
        self.q_lora = LoRALinearLayer(dim, dim, rank, network_alpha)
        self.k_lora = LoRALinearLayer(dim, dim, rank, network_alpha)
        self.v_lora = LoRALinearLayer(dim, dim, rank, network_alpha)
        self.o_lora = LoRALinearLayer(dim, dim, rank, network_alpha)
            
        self.attn = AttentionModule(self.num_heads)
        
    def forward(self, x: torch.Tensor, context: torch.Tensor):
        if self.has_image_input:
            img = context[:, :257]
            ctx = context[:, 257:]
        else:
            ctx = context
        q = self.norm_q(self.q(x)+self.q_lora(x))
        k = self.norm_k(self.k(ctx)+self.k_lora(ctx))
        v = self.v(ctx)+self.v_lora(ctx)
        x = self.attn(q, k, v)
        return self.o(x)+self.o_lora(x)
        
class CrossAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6, has_image_input: bool = False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)
        self.has_image_input = has_image_input
        if has_image_input:
            self.k_img = nn.Linear(dim, dim)
            self.v_img = nn.Linear(dim, dim)
            self.norm_k_img = RMSNorm(dim, eps=eps)
 
        self.attn = AttentionModule(self.num_heads)


    def forward(self, x: torch.Tensor, context: torch.Tensor):
        if self.has_image_input:
            img = context[:, :257]
            ctx = context[:, 257:]
        else:
            ctx = context
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(ctx))
        v = self.v(ctx)
        x = self.attn(q, k, v)
        return self.o(x)

class GateModule(nn.Module):
    def __init__(self,):
        super().__init__()

    def forward(self, x, gate, residual):
        return x + gate * residual

class DiTBlock(nn.Module):
    def __init__(self, has_image_input: bool, dim: int, num_heads: int, ffn_dim: int, eps: float = 1e-6, rank=128, network_alpha=128, mode="causal_attn", with_ref=True):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim
        self.with_ref = with_ref
        self.mode = mode

        self.self_attn = SelfAttention(dim, num_heads, eps)
        self.cross_attn = CrossAttention(dim, num_heads, eps, has_image_input=has_image_input)
        self.norm1 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(dim, eps=eps)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn_dim), nn.GELU(
            approximate='tanh'), nn.Linear(ffn_dim, dim))
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)
        self.gate = GateModule()
        
        if self.with_ref:
            self.self_attn_ref = SelfAttention_ref(dim, num_heads, eps,rank=rank, network_alpha=network_alpha)
            self.cross_attn_ref = CrossAttention_ref(dim, num_heads, eps, has_image_input=has_image_input,rank=rank, network_alpha=network_alpha)
            self.norm1_ref = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
            self.norm2_ref = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
            self.norm3_ref = nn.LayerNorm(dim, eps=eps)
            self.ffn_ref = nn.Sequential(nn.Linear(dim, ffn_dim), nn.GELU(
                approximate='tanh'), nn.Linear(ffn_dim, dim))
            
            self.fc1_lora = LoRALinearLayer(dim, ffn_dim, rank, network_alpha)
            self.fc2_lora = LoRALinearLayer(ffn_dim, dim, rank, network_alpha)
    
            self.modulation_ref = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)
            self.gate_ref = GateModule()
            
            self.kv_cache = {"k": None, "v": None, "y": None}
        
    def forward(self, x, y, context, ref_context, t_mod, cond_t_mod, freqs, cond_freqs, f_y, h_y, w_y, f, h, w):
        has_seq = len(t_mod.shape) == 4
        chunk_dim = 2 if has_seq else 1
        # msa: multi-head self-attention  mlp: multi-layer perceptron
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(6, dim=chunk_dim)
        cond_shift_msa, cond_scale_msa, cond_gate_msa, cond_shift_mlp, cond_scale_mlp, cond_gate_mlp = (
            self.modulation_ref.to(dtype=cond_t_mod.dtype, device=cond_t_mod.device) + cond_t_mod).chunk(6, dim=chunk_dim)        
        
        if has_seq:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                shift_msa.squeeze(2), scale_msa.squeeze(2), gate_msa.squeeze(2),
                shift_mlp.squeeze(2), scale_mlp.squeeze(2), gate_mlp.squeeze(2),
            )
            cond_shift_msa, cond_scale_msa, cond_gate_msa, cond_shift_mlp, cond_scale_mlp, cond_gate_mlp = (
                cond_shift_msa.squeeze(2)[:,0,:], cond_scale_msa.squeeze(2)[:,0,:], cond_gate_msa.squeeze(2)[:,0,:],
                cond_shift_mlp.squeeze(2)[:,0,:], cond_scale_mlp.squeeze(2)[:,0,:], cond_gate_mlp.squeeze(2)[:,0,:],
            )

        input_x = modulate(self.norm1(x), shift_msa, scale_msa)
        input_y = modulate(self.norm1_ref(y), cond_shift_msa, cond_scale_msa)
        q,k,v = self.self_attn(input_x, freqs, return_qkv=True)
        q_ref, k_ref, v_ref = self.self_attn_ref(input_y, cond_freqs, return_qkv=True)
        
        q_all = torch.cat([q, q_ref], dim=1)
        k_all = torch.cat([k, k_ref], dim=1)
        v_all = torch.cat([v, v_ref], dim=1)
        
        if self.mode == "causal_attn":
            seq_len_q = q_all.shape[1]
            seq_len_k = k_all.shape[1]
            N_x = q.shape[1]
            attn_mask = torch.zeros((seq_len_q, seq_len_k), device=x.device)  
            attn_mask[N_x:, :N_x] = 1
            attn_mask = attn_mask.to(
                    dtype=q_all.dtype,
                    device=q_all.device
                ) * (-1e9)
            
            q_all = rearrange(q_all, "b s (n d) -> b n s d", n=self.num_heads)
            k_all = rearrange(k_all, "b s (n d) -> b n s d", n=self.num_heads)
            v_all = rearrange(v_all, "b s (n d) -> b n s d", n=self.num_heads)
            tmp_hidden_states = F.scaled_dot_product_attention(
                q_all, k_all, v_all,
                attn_mask=attn_mask, 
                dropout_p=0.0, 
                is_causal=False
            )
        else:
            q_all = rearrange(q_all, "b s (n d) -> b n s d", n=self.num_heads)
            k_all = rearrange(k_all, "b s (n d) -> b n s d", n=self.num_heads)
            v_all = rearrange(v_all, "b s (n d) -> b n s d", n=self.num_heads)
            tmp_hidden_states = F.scaled_dot_product_attention(
                    q_all, k_all, v_all,
                    attn_mask=None, 
                    dropout_p=0.0, 
                    is_causal=False
                )
            
        tmp_hidden_states = tmp_hidden_states.type_as(q)
        tmp_hidden_states = rearrange(tmp_hidden_states, "b n s d -> b s (n d)", n=self.num_heads)
        attn_out_x = tmp_hidden_states[:, :q.shape[1], :]
        attn_out_y = tmp_hidden_states[:, q.shape[1]:, :]  
    
        x = self.gate(x, gate_msa, self.self_attn.o(attn_out_x))
        y = self.gate_ref(y, cond_gate_msa, self.self_attn_ref.o(attn_out_y))
        
        x = x + self.cross_attn(self.norm3(x), context)
        if ref_context is not None:
            y = y + self.cross_attn_ref(self.norm3_ref(y), ref_context)
        else:
            y = y + self.cross_attn_ref(self.norm3_ref(y), context)
        
        input_x = modulate(self.norm2(x), shift_mlp, scale_mlp)
        input_y = modulate(self.norm2_ref(y), cond_shift_mlp, cond_scale_mlp)
        x = self.gate(x, gate_mlp, self.ffn(input_x))
        # y = self.gate_ref(y, cond_gate_mlp, self.ffn_ref(input_y))
        tmp = self.ffn_ref[0](input_y) + self.fc1_lora(input_y)  # fc1 + lora
        tmp = self.ffn_ref[1](tmp)  # GELU
        ffn_ref_out = self.ffn_ref[2](tmp) + self.fc2_lora(tmp)  # fc2 + lora
        y = self.gate_ref(y, cond_gate_mlp, ffn_ref_out)
            
        return x, y


class MLP(torch.nn.Module):
    def __init__(self, in_dim, out_dim, has_pos_emb=False):
        super().__init__()
        self.proj = torch.nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim)
        )
        self.has_pos_emb = has_pos_emb
        if has_pos_emb:
            self.emb_pos = torch.nn.Parameter(torch.zeros((1, 514, 1280)))

    def forward(self, x):
        if self.has_pos_emb:
            x = x + self.emb_pos.to(dtype=x.dtype, device=x.device)
        return self.proj(x)


class Head(nn.Module):
    def __init__(self, dim: int, out_dim: int, patch_size: Tuple[int, int, int], eps: float):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
        self.norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.head = nn.Linear(dim, out_dim * math.prod(patch_size))
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, t_mod):
        if len(t_mod.shape) == 3:
            shift, scale = (self.modulation.unsqueeze(0).to(dtype=t_mod.dtype, device=t_mod.device) + t_mod.unsqueeze(2)).chunk(2, dim=2)
            x = (self.head(self.norm(x) * (1 + scale.squeeze(2)) + shift.squeeze(2)))
        else:
            shift, scale = (self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(2, dim=1)
            x = (self.head(self.norm(x) * (1 + scale) + shift))
        return x


class GroupAttnDown(torch.nn.Module):
    def __init__(self, dim, num_heads=8, compression_ratio=2):
        super().__init__()
        self.num_heads = num_heads
        self.compression_ratio = compression_ratio
        self.q = torch.nn.Linear(dim, dim, bias=False)
        self.k = torch.nn.Linear(dim, dim, bias=False)
        self.v = torch.nn.Linear(dim, dim, bias=False)
        self.compress = torch.nn.Linear(dim*int(compression_ratio**3), dim, bias=False)  # 8->1
        self.proj = torch.nn.Linear(dim, dim, bias=False)
        self.norm = RMSNorm(dim)

    def forward(self, tokens, f, h, w):
        B, N, C = tokens.shape
        x = self.norm(tokens).view(B, f, h, w, C)
        assert f % self.compression_ratio == 0 and h % self.compression_ratio == 0 and w % self.compression_ratio == 0
        f2, h2, w2 = f // self.compression_ratio, h // self.compression_ratio, w // self.compression_ratio
        g = x.unfold(1, self.compression_ratio, self.compression_ratio).unfold(2, self.compression_ratio, self.compression_ratio).unfold(3, self.compression_ratio, self.compression_ratio)  # [B,f2,h2,w2,2,2,2,C]
        g = g.reshape(B, f2*h2*w2, int(self.compression_ratio**3), C)  # [B,G,8,C]
        q = self.q(g)                     # [B,G,8,C]
        k = self.k(g)                     # [B,G,8,C]
        v = self.v(g)                     # [B,G,8,C]
        attn = torch.softmax((q @ k.transpose(-1, -2)) / (C ** 0.5), dim=-1)  # [B,G,8,8]
        out = attn @ v                    # [B,G,8,C]
        pooled = self.compress(out.reshape(B, f2*h2*w2, int(self.compression_ratio**3)*C))  # [B,G,C]
        pooled = self.proj(pooled)      # [B,G,C]
        return pooled, (f2, h2, w2)

class GroupAttnUp(torch.nn.Module):
    def __init__(self, dim, num_heads=8, compression_ratio=2):
        super().__init__()
        self.num_heads = num_heads
        self.compression_ratio = compression_ratio
        self.expand = torch.nn.Linear(dim, dim*int(self.compression_ratio**3), bias=False)  # 1->8
        self.q = torch.nn.Linear(dim, dim, bias=False)
        self.k = torch.nn.Linear(dim, dim, bias=False)
        self.v = torch.nn.Linear(dim, dim, bias=False)
        self.proj = torch.nn.Linear(dim, dim, bias=False)
        self.norm = RMSNorm(dim)

    def forward(self, low_tokens, f2, h2, w2, f, h, w):
        # low_tokens: [B, G, C], G=f2*h2*w2
        B, G, C = low_tokens.shape
        x = self.expand(low_tokens).view(B, G, int(self.compression_ratio**3), C)  # [B,G,8,C]
        x = self.norm(x)
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        attn = torch.softmax((q @ k.transpose(-1, -2)) / (C ** 0.5), dim=-1)
        x = (attn @ v)  # [B,G,8,C]
        x = self.proj(x)
        x = x.view(B, f2, h2, w2, self.compression_ratio, self.compression_ratio, self.compression_ratio, C).permute(0, 7, 1, 4, 2, 5, 3, 6)  # [B,C,f2,2,h2,2,w2,2]
        x = x.reshape(B, C, f, h, w)
        tokens = x.permute(0, 2, 3, 4, 1).reshape(B, f*h*w, C)  # [B,N,C]
        return tokens


class WanControlModel(torch.nn.Module):
    def __init__(
        self,
        dim: int,
        in_dim: int,
        ffn_dim: int,
        out_dim: int,
        text_dim: int,
        freq_dim: int,
        eps: float,
        patch_size: Tuple[int, int, int],
        num_heads: int,
        num_layers: int,
        has_image_input: bool,
        has_image_pos_emb: bool = False,
        has_ref_conv: bool = False,
        add_control_adapter: bool = False,
        in_dim_control_adapter: int = 24,
        seperated_timestep: bool = False,
        require_vae_embedding: bool = True,
        require_clip_embedding: bool = True,
        fuse_vae_embedding_in_latents: bool = False,
        rank: int = 128,
        network_alpha: int = 128,
        mode="causal_attn",
        compression_ratio=2
    ):
        super().__init__()
        self.dim = dim
        self.in_dim = in_dim
        self.freq_dim = freq_dim
        self.has_image_input = has_image_input
        self.patch_size = patch_size
        self.seperated_timestep = seperated_timestep
        self.require_vae_embedding = require_vae_embedding
        self.require_clip_embedding = require_clip_embedding
        self.fuse_vae_embedding_in_latents = fuse_vae_embedding_in_latents
        self.rank = rank
        self.network_alpha = network_alpha

        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim),
            nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim)
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )
        self.time_projection = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, dim * 6))
        self.blocks = nn.ModuleList([
            DiTBlock(has_image_input, dim, num_heads, ffn_dim, eps, rank, network_alpha, mode)
            for _ in range(num_layers)
        ])
        self.head = Head(dim, out_dim, patch_size, eps)
        head_dim = dim // num_heads
        self.freqs = precompute_freqs_cis_3d(head_dim)
        self.cond_freqs = precompute_cond_freqs_cis_3d(head_dim, scale=compression_ratio)

        if has_image_input:
            self.img_emb = MLP(1280, dim, has_pos_emb=has_image_pos_emb)  # clip_feature_dim = 1280
        if has_ref_conv:
            self.ref_conv = nn.Conv2d(16, dim, kernel_size=(2, 2), stride=(2, 2))
        self.has_image_pos_emb = has_image_pos_emb
        self.has_ref_conv = has_ref_conv
        if add_control_adapter:
            self.control_adapter = SimpleAdapter(in_dim_control_adapter, dim, kernel_size=patch_size[1:], stride=patch_size[1:])
        else:
            self.control_adapter = None

        self.attn_down_lora = GroupAttnDown(dim, num_heads=8, compression_ratio=compression_ratio)
        self.attn_up_lora = GroupAttnUp(dim, num_heads=8, compression_ratio=compression_ratio)
        self.compression_ratio = compression_ratio
        print("Using compression ratio:", compression_ratio)
        
    def patchify(self, x: torch.Tensor, control_camera_latents_input: Optional[torch.Tensor] = None):
        x = self.patch_embedding(x)
        if self.control_adapter is not None and control_camera_latents_input is not None:
            y_camera = self.control_adapter(control_camera_latents_input)
            x = [u + v for u, v in zip(x, y_camera)]
            x = x[0].unsqueeze(0)
        return x

    def unpatchify(self, x: torch.Tensor, grid_size: torch.Tensor):
        return rearrange(
            x, 'b (f h w) (x y z c) -> b c (f x) (h y) (w z)',
            f=grid_size[0], h=grid_size[1], w=grid_size[2], 
            x=self.patch_size[0], y=self.patch_size[1], z=self.patch_size[2]
        )

    def forward(self,
                x: torch.Tensor,
                timestep: torch.Tensor,
                context: torch.Tensor,
                ref_context: Optional[torch.Tensor] = None,
                clip_feature: Optional[torch.Tensor] = None,
                y: Optional[torch.Tensor] = None,
                use_gradient_checkpointing: bool = False,
                use_gradient_checkpointing_offload: bool = False,
                **kwargs,
                ):
        t = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, timestep))
        cond_t = torch.zeros_like(t)
        t_mod = self.time_projection(t).unflatten(1, (6, self.dim))
        cond_t_mod = torch.zeros_like(t_mod)
        context = self.text_embedding(context)
        if ref_context is not None:
            ref_context = self.text_embedding(ref_context)
        
        x = self.patchify(x)
        y = self.patchify(y)
        
        if y is not None:
            y_original_shape = y.shape[2:]
            
        y_org = y
        y = pad_for_3d_conv(y, (self.compression_ratio, self.compression_ratio, self.compression_ratio))
        y_tokens = rearrange(y, 'b c f h w -> b (f h w) c')
        f_y, h_y, w_y = y.shape[2:]
        y_down, (f2, h2, w2) = self.attn_down_lora(y_tokens, f_y, h_y, w_y)   
        
        # rec会导致质量变差
        # y_up_tokens = self.attn_up_lora(y_down, f2, h2, w2, f_y, h_y, w_y)   
        # y_rec = rearrange(y_up_tokens, 'b (f h w) c -> b c f h w', f=f_y, h=h_y, w=w_y)
        
        f, h, w = x.shape[2:]        
        x = rearrange(x, 'b c f h w -> b (f h w) c').contiguous()
        y = y_down.contiguous() if y_down is not None else None
        
        freqs = torch.cat([
            self.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            self.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            self.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ], dim=-1).reshape(f * h * w, 1, -1).to(x.device)
        
        cond_freqs = torch.cat([
            self.cond_freqs[0][:f2].view(f2, 1, 1, -1).expand(f2, h2, w2, -1),
            self.cond_freqs[1][:h2].view(1, h2, 1, -1).expand(f2, h2, w2, -1),
            self.cond_freqs[2][:w2].view(1, 1, w2, -1).expand(f2, h2, w2, -1)
        ], dim=-1).reshape(f2 * h2 * w2, 1, -1).to(x.device)
        
        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)
            return custom_forward

        for block in self.blocks:
            if self.training and use_gradient_checkpointing:
                if use_gradient_checkpointing_offload:
                    with torch.autograd.graph.save_on_cpu():
                        x, y = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(block),
                            x, y, context, ref_context, t_mod, cond_t_mod, freqs, cond_freqs, f_y, h_y, w_y, f, h, w, 
                            use_reentrant=False
                        )
                else:
                    x, y = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        x, y, context, ref_context, t_mod, cond_t_mod, freqs, cond_freqs, f_y, h_y, w_y, f, h, w, 
                        use_reentrant=False
                    )
            else:
                x, y = block(x, y, context, ref_context, t_mod, cond_t_mod, freqs, cond_freqs, f_y, h_y, w_y, f, h, w)
        
        x = self.head(x, t)        
        x = self.unpatchify(x, (f, h, w))
    
        return x