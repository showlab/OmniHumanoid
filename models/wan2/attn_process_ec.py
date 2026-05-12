import torch
import torch.nn.functional as F
from typing import Optional
from diffusers.models.attention_processor import Attention

class EasyControlAttnProcessor:
    def __init__(self, cil_module=None):
        # 接收外部传入的 CIL 模块实例
        self.cil_module = cil_module
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("EasyControlAttnProcessor requires PyTorch 2.0.")

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[torch.Tensor] = None,
        rotary_emb2: Optional[torch.Tensor] = None,
        rotary_emb3: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        
        # ================== 1. 基础处理 (I2V Context) ==================
        encoder_hidden_states_img = None
        if attn.add_k_proj is not None:
            # 512 is the context length of the text encoder, hardcoded for now
            image_context_length = encoder_hidden_states.shape[1] - 512
            encoder_hidden_states_img = encoder_hidden_states[:, :image_context_length]
            encoder_hidden_states = encoder_hidden_states[:, image_context_length:]
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        # ================== 2. 基础投影 (Base Projections) ==================
        # 此时形状为 [Batch, Length, Dim]
        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # ================== 3. ★ EasyControl 注入 (CIL) ==================
        # 注意：必须在 unflatten (分头) 之前进行，因为 LoRA 是对 hidden_dim 操作的
        if self.cil_module is not None and rotary_emb is not None:
            # rotary_emb 是 tuple(cos, sin)，cos 形状通常是 [1, 1, L_target, D_head]
            # 我们通过它获取目标视频的长度 L1
            L_target = rotary_emb[0].shape[-2]
            
            # 判断是否存在条件部分（即序列长度是否大于目标长度）
            if query.shape[1] > L_target:
                # 切片：取出 [Target | Condition | FirstFrame] 中的后两部分
                cond_hidden = hidden_states[:, L_target:, :]
                
                # 计算 LoRA 增量
                # cil_module 返回 (delta_q, delta_k, delta_v)
                d_q, d_k, d_v = self.cil_module(cond_hidden)
                
                # 将增量加到 Query/Key/Value 的对应部分
                query[:, L_target:, :] = query[:, L_target:, :] + d_q
                key[:, L_target:, :]   = key[:, L_target:, :]   + d_k
                value[:, L_target:, :] = value[:, L_target:, :] + d_v

        # ================== 4. 分头 (Split Heads) ==================
        # [B, L, D] -> [B, H, L, Dh]
        query = query.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        key = key.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        value = value.unflatten(2, (attn.heads, -1)).transpose(1, 2)

        # ================== 5. 三段式 RoPE (保持原逻辑) ==================
        if rotary_emb is not None:
            def _canonicalize_cos_sin(freqs_cos: torch.Tensor, freqs_sin: torch.Tensor, L: int, C: int, ref: torch.dtype, dev: torch.device):
                cos = freqs_cos.to(device=dev, dtype=ref)
                sin = freqs_sin.to(device=dev, dtype=ref)
                while cos.dim() > 2:
                    cos = cos.select(0, 0)
                    sin = sin.select(0, 0)
                if cos.size(-1) == 2 * C:
                    cos = cos[..., 0::2]
                    sin = sin[..., 1::2]
                elif cos.size(-1) != C:
                    if cos.size(-1) > C:
                        cos = cos[..., :C]
                        sin = sin[..., :C]
                    else:
                        pad = C - cos.size(-1)
                        cos = torch.cat([cos, cos[..., -1:].expand(cos.size(0), pad)], dim=-1)
                        sin = torch.cat([sin, sin[..., -1:].expand(sin.size(0), pad)], dim=-1)
                if cos.size(-2) > L:
                    cos = cos[:L, :]
                    sin = sin[:L, :]
                elif cos.size(-2) < L:
                    pad = L - cos.size(-2)
                    cos = torch.cat([cos, cos[-1:, :].expand(pad, cos.size(1))], dim=-2)
                    sin = torch.cat([sin, sin[-1:, :].expand(pad, sin.size(1))], dim=-2)
                cos = cos.view(1, 1, L, C).contiguous()
                sin = sin.view(1, 1, L, C).contiguous()
                return cos, sin

            def apply_rotary_emb(h: torch.Tensor, freqs_cos: torch.Tensor, freqs_sin: torch.Tensor):
                B, Hh, L, Dh = h.shape
                C = Dh // 2
                x = h.view(B, Hh, L, C, 2)
                x1, x2 = x[..., 0], x[..., 1]
                cos, sin = _canonicalize_cos_sin(freqs_cos, freqs_sin, L=L, C=C, ref=h.dtype, dev=h.device)
                out = torch.empty_like(h)
                out[..., 0::2] = x1 * cos - x2 * sin
                out[..., 1::2] = x1 * sin + x2 * cos
                return out

            # 分段应用 RoPE
            if rotary_emb2 is not None:
                if rotary_emb3 is not None:
                    # 三段：Target, Cond, First
                    L1 = rotary_emb[0].shape[-2]
                    L2 = rotary_emb2[0].shape[-2]
                    L3 = rotary_emb3[0].shape[-2]
                    # Target 段
                    query[:, :, :L1] = apply_rotary_emb(query[:, :, :L1], *rotary_emb)
                    key[:, :, :L1] = apply_rotary_emb(key[:, :, :L1], *rotary_emb)
                    # Cond 段
                    query[:, :, L1:L1 + L2] = apply_rotary_emb(query[:, :, L1:L1 + L2], *rotary_emb2)
                    key[:, :, L1:L1 + L2] = apply_rotary_emb(key[:, :, L1:L1 + L2], *rotary_emb2)
                    # First 段
                    query[:, :, -L3:] = apply_rotary_emb(query[:, :, -L3:], *rotary_emb3)
                    key[:, :, -L3:] = apply_rotary_emb(key[:, :, -L3:], *rotary_emb3)
                else:
                    # 两段
                    half = query.shape[2] // 2
                    query[:, :, :half] = apply_rotary_emb(query[:, :, :half], *rotary_emb)
                    key[:, :, :half] = apply_rotary_emb(key[:, :, :half], *rotary_emb)
                    query[:, :, half:] = apply_rotary_emb(query[:, :, half:], *rotary_emb2)
                    key[:, :, half:] = apply_rotary_emb(key[:, :, half:], *rotary_emb2)
            else:
                # 单段
                query = apply_rotary_emb(query, *rotary_emb)
                key = apply_rotary_emb(key, *rotary_emb)

        # ================== 6. 额外 Context 处理 (I2V Task) ==================
        hidden_states_img = None
        if encoder_hidden_states_img is not None:
            key_img = attn.add_k_proj(encoder_hidden_states_img)
            key_img = attn.norm_added_k(key_img)
            value_img = attn.add_v_proj(encoder_hidden_states_img)

            key_img = key_img.unflatten(2, (attn.heads, -1)).transpose(1, 2)
            value_img = value_img.unflatten(2, (attn.heads, -1)).transpose(1, 2)

            hidden_states_img = F.scaled_dot_product_attention(
                query, key_img, value_img, attn_mask=None, dropout_p=0.0, is_causal=False
            )
            hidden_states_img = hidden_states_img.transpose(1, 2).flatten(2, 3)
            hidden_states_img = hidden_states_img.type_as(query)

        # ================== 7. 主注意力计算 (Attention) ==================
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        hidden_states = hidden_states.transpose(1, 2).flatten(2, 3)
        hidden_states = hidden_states.type_as(query)

        # 叠加 I2V Context 结果
        if hidden_states_img is not None:
            hidden_states = hidden_states + hidden_states_img

        # ================== 8. 输出投影 ==================
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states