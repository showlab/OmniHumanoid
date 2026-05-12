import torch
import torch.nn.functional as F
from typing import Any, Callable, Dict, List, Optional, Union
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.models.attention_processor import Attention
from diffusers.utils import is_torch_xla_available

# 导入你原版的 CustomWanPipeline 和 输出类
from models.wan2.custom_pipeline import CustomWanPipeline, WanPipelineOutput

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm
    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False


class KVCachedEasyControlAttnProcessor:
    """
    带 KV Cache 的 EasyControl Attention Processor。
    【终极修复版】：
    1. 隔离了 Cond 和 Uncond 的缓存字典，确保 CFG 正常生效。
    2. 构造全 True 掩码，防止 FlashAttention 算子突变导致的数值溢出（核爆第一帧）。
    """
    def __init__(self, cil_module=None):
        self.cil_module = cil_module
        # 将缓存改为字典，独立存储 cond 和 uncond
        self.bank_k = {} 
        self.bank_v = {}
        self.use_kv_cache = False  
        self.is_uncond = False     # 标志位，由 Pipeline 控制

    def clear_cache(self):
        """生成新视频前清空上一次的缓存"""
        self.bank_k = {}
        self.bank_v = {}
        self.use_kv_cache = False
        self.is_uncond = False

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
        
        encoder_hidden_states_img = None
        if attn.add_k_proj is not None:
            image_context_length = encoder_hidden_states.shape[1] - 512
            encoder_hidden_states_img = encoder_hidden_states[:, :image_context_length]
            encoder_hidden_states = encoder_hidden_states[:, image_context_length:]
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        if attn.norm_q is not None: query = attn.norm_q(query)
        if attn.norm_k is not None: key = attn.norm_k(key)

        L_target = rotary_emb[0].shape[-2] if rotary_emb is not None else hidden_states.shape[1]
        
        # ★ 决定当前操作哪个缓存键 (解决 CFG 失效问题)
        cache_key = "uncond" if self.is_uncond else "cond"

        def _canonicalize_cos_sin(freqs_cos, freqs_sin, L, C, ref, dev):
            cos, sin = freqs_cos.to(device=dev, dtype=ref), freqs_sin.to(device=dev, dtype=ref)
            while cos.dim() > 2:
                cos, sin = cos.select(0, 0), sin.select(0, 0)
            if cos.size(-1) == 2 * C:
                cos, sin = cos[..., 0::2], sin[..., 1::2]
            elif cos.size(-1) != C:
                if cos.size(-1) > C:
                    cos, sin = cos[..., :C], sin[..., :C]
                else:
                    pad = C - cos.size(-1)
                    cos = torch.cat([cos, cos[..., -1:].expand(cos.size(0), pad)], dim=-1)
                    sin = torch.cat([sin, sin[..., -1:].expand(sin.size(0), pad)], dim=-1)
            if cos.size(-2) > L:
                cos, sin = cos[:L, :], sin[:L, :]
            elif cos.size(-2) < L:
                pad = L - cos.size(-2)
                cos = torch.cat([cos, cos[-1:, :].expand(pad, cos.size(1))], dim=-2)
                sin = torch.cat([sin, sin[-1:, :].expand(pad, sin.size(1))], dim=-2)
            return cos.view(1, 1, L, C).contiguous(), sin.view(1, 1, L, C).contiguous()

        def apply_rotary_emb(h, freqs_cos, freqs_sin):
            B, Hh, L, Dh = h.shape
            C = Dh // 2
            x = h.view(B, Hh, L, C, 2)
            x1, x2 = x[..., 0], x[..., 1]
            cos, sin = _canonicalize_cos_sin(freqs_cos, freqs_sin, L=L, C=C, ref=h.dtype, dev=h.device)
            out = torch.empty_like(h)
            out[..., 0::2] = x1 * cos - x2 * sin
            out[..., 1::2] = x1 * sin + x2 * cos
            return out

        # ==========================================
        # 阶段 A：未开启缓存 (全量计算并写入缓存) 
        # ==========================================
        if not self.use_kv_cache:
            if self.cil_module is not None and rotary_emb2 is not None:
                # ✅ 修复：同时包含 Cond (L2) 和 Ref (L3) tokens 进行 CIL 适配
                cond_hidden = hidden_states[:, L_target:, :]
                d_q, d_k, d_v = self.cil_module(cond_hidden)
                query[:, L_target:, :] += d_q
                key[:, L_target:, :]   += d_k
                value[:, L_target:, :] += d_v

            query = query.unflatten(2, (attn.heads, -1)).transpose(1, 2)
            key = key.unflatten(2, (attn.heads, -1)).transpose(1, 2)
            value = value.unflatten(2, (attn.heads, -1)).transpose(1, 2)

            if rotary_emb is not None:
                if rotary_emb2 is not None:
                    if rotary_emb3 is not None:
                        L1, L2, L3 = L_target, rotary_emb2[0].shape[-2], rotary_emb3[0].shape[-2]
                        query[:, :, :L1] = apply_rotary_emb(query[:, :, :L1], *rotary_emb)
                        key[:, :, :L1] = apply_rotary_emb(key[:, :, :L1], *rotary_emb)
                        query[:, :, L1:L1 + L2] = apply_rotary_emb(query[:, :, L1:L1 + L2], *rotary_emb2)
                        key[:, :, L1:L1 + L2] = apply_rotary_emb(key[:, :, L1:L1 + L2], *rotary_emb2)
                        query[:, :, -L3:] = apply_rotary_emb(query[:, :, -L3:], *rotary_emb3)
                        key[:, :, -L3:] = apply_rotary_emb(key[:, :, -L3:], *rotary_emb3)
                    else:
                        half = query.shape[2] // 2
                        query[:, :, :half] = apply_rotary_emb(query[:, :, :half], *rotary_emb)
                        key[:, :, :half] = apply_rotary_emb(key[:, :, :half], *rotary_emb)
                        query[:, :, half:] = apply_rotary_emb(query[:, :, half:], *rotary_emb2)
                        key[:, :, half:] = apply_rotary_emb(key[:, :, half:], *rotary_emb2)
                else:
                    query = apply_rotary_emb(query, *rotary_emb)
                    key = apply_rotary_emb(key, *rotary_emb)

            # 写入对应的字典
            if rotary_emb2 is not None:
                self.bank_k[cache_key] = key[:, :, L_target:, :].clone()
                self.bank_v[cache_key] = value[:, :, L_target:, :].clone()

        # ==========================================
        # ★ 阶段 B：开启缓存 (目标独立计算并拼贴对应缓存)
        # ==========================================
        else:
            query = query.unflatten(2, (attn.heads, -1)).transpose(1, 2)
            key = key.unflatten(2, (attn.heads, -1)).transpose(1, 2)
            value = value.unflatten(2, (attn.heads, -1)).transpose(1, 2)

            if rotary_emb is not None:
                query = apply_rotary_emb(query, *rotary_emb)
                key = apply_rotary_emb(key, *rotary_emb)

            # 提取正确的对应的缓存并拼接
            if cache_key in self.bank_k and cache_key in self.bank_v:
                key = torch.cat([key, self.bank_k[cache_key]], dim=2)
                value = torch.cat([value, self.bank_v[cache_key]], dim=2)
                
                # ★ 核心修复：构造全 True 掩码，防止 FlashAttention 算子突变
                # L_q = query.shape[2]
                # L_k = key.shape[2]
                # attention_mask = torch.ones(
                #     1, 1, L_q, L_k, 
                #     dtype=torch.bool, 
                #     device=query.device
                # )
                attention_mask = None

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
            hidden_states_img = hidden_states_img.transpose(1, 2).flatten(2, 3).type_as(query)

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        hidden_states = hidden_states.transpose(1, 2).flatten(2, 3).type_as(query)

        if hidden_states_img is not None:
            hidden_states = hidden_states + hidden_states_img

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


class KVCacheWanPipeline(CustomWanPipeline):
    """
    修改了 Denoising loop，精准控制 Processor 的 is_uncond 状态。
    """
    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        negative_prompt: Union[str, List[str]] = None,
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        guidance_scale_2: Optional[float] = None,
        num_videos_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "np",
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
    ):
        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        self.check_inputs(
            prompt, negative_prompt, height, width, prompt_embeds, negative_prompt_embeds,
            callback_on_step_end_tensor_inputs, guidance_scale_2,
        )

        if num_frames % self.vae_scale_factor_temporal != 1:
            num_frames = num_frames // self.vae_scale_factor_temporal * self.vae_scale_factor_temporal + 1
        num_frames = max(num_frames, 1)

        if self.config.boundary_ratio is not None and guidance_scale_2 is None:
            guidance_scale_2 = guidance_scale

        self._guidance_scale = guidance_scale
        self._guidance_scale_2 = guidance_scale_2
        
        attention_kwargs = attention_kwargs or {}

        cond_states = attention_kwargs.get("encoder_contion_states", None)
        if isinstance(cond_states, torch.Tensor):
            attention_kwargs["encoder_contion_states"] = cond_states.to(
                device=self._execution_device, dtype=self.transformer.dtype
            )
        
        first_states = attention_kwargs.get("encoder_first_states", None)
        if isinstance(first_states, torch.Tensor):
            attention_kwargs["encoder_first_states"] = first_states.to(
                device=self._execution_device, dtype=self.transformer.dtype
            )

        self._attention_kwargs = attention_kwargs
        self._current_timestep = None
        self._interrupt = False

        device = self._execution_device

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            num_videos_per_prompt=num_videos_per_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            max_sequence_length=max_sequence_length,
            device=device,
        )

        transformer_dtype = self.transformer.dtype
        prompt_embeds = prompt_embeds.to(transformer_dtype)
        if negative_prompt_embeds is not None:
            negative_prompt_embeds = negative_prompt_embeds.to(transformer_dtype)
        
        def _scheduler_tensors_to_numpy(sched):
            import numpy as np, torch
            _maybe_np = ["betas", "alphas", "alphas_cumprod", "alphas_cumprod_prev", "sigmas", "timesteps"]
            for name in _maybe_np:
                if hasattr(sched, name):
                    val = getattr(sched, name)
                    if isinstance(val, torch.Tensor):
                        setattr(sched, name, val.detach().cpu().float().numpy())
            return sched

        _scheduler_tensors_to_numpy(self.scheduler)
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        num_channels_latents = self.transformer.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_videos_per_prompt, num_channels_latents, height, width, num_frames,
            torch.float32, device, generator, latents,
        )

        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)

        for block in self.transformer.blocks:
            if hasattr(block.attn1.processor, "clear_cache"):
                block.attn1.processor.clear_cache()

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt: continue

                self._current_timestep = t
                current_model = self.transformer
                current_guidance_scale = guidance_scale

                latent_model_input = latents.to(transformer_dtype)
                B, C, F, H, W = latent_model_input.shape
                device = latent_model_input.device

                is_first_step = (i == 0)
                
                if is_first_step:
                    current_attention_kwargs = attention_kwargs
                    for blk in current_model.blocks:
                        if hasattr(blk.attn1.processor, "use_kv_cache"):
                            blk.attn1.processor.use_kv_cache = False
                else:
                    current_attention_kwargs = {
                        k: v for k, v in attention_kwargs.items() 
                        if k not in ["encoder_contion_states", "encoder_first_states"]
                    }
                    for blk in current_model.blocks:
                        if hasattr(blk.attn1.processor, "use_kv_cache"):
                            blk.attn1.processor.use_kv_cache = True

                t_reshaped = t.to(device=device, dtype=transformer_dtype).view(1, 1, 1, 1).expand(B, 1, 1, 1)
                mask_grid = torch.ones((B, 1, F, H, W), device=device, dtype=transformer_dtype)
                kkk = (t_reshaped * mask_grid[:, 0, :, ::2, ::2]).flatten(1)
                
                t_embed_parts = [kkk]
                if current_attention_kwargs.get("encoder_contion_states", None) is not None:
                    t_embed_parts.append(torch.zeros_like(kkk))
                if current_attention_kwargs.get("encoder_first_states", None) is not None:
                    t_embed_parts.append(torch.zeros_like(mask_grid[:, 0, 0, ::2, ::2].flatten(1)))

                current_t_embed = torch.cat(t_embed_parts, dim=-1)

                # ===============================================
                # ★ 1. Cond Pass (告诉 Processor 此时是 Cond)
                # ===============================================
                for blk in current_model.blocks:
                    if hasattr(blk.attn1.processor, "is_uncond"):
                        blk.attn1.processor.is_uncond = False

                with current_model.cache_context("cond"):
                    noise_pred = current_model(
                        hidden_states=latent_model_input,
                        timestep=current_t_embed,
                        encoder_hidden_states=prompt_embeds,
                        attention_kwargs=current_attention_kwargs,
                        return_dict=False,
                    )[0]

                # ===============================================
                # ★ 2. Uncond Pass (告诉 Processor 此时是 Uncond)
                # ===============================================
                if self.do_classifier_free_guidance:
                    for blk in current_model.blocks:
                        if hasattr(blk.attn1.processor, "is_uncond"):
                            blk.attn1.processor.is_uncond = True

                    with current_model.cache_context("uncond"):
                        noise_uncond = current_model(
                            hidden_states=latent_model_input,
                            timestep=current_t_embed,
                            encoder_hidden_states=negative_prompt_embeds,
                            attention_kwargs=current_attention_kwargs,
                            return_dict=False,
                        )[0]
                    noise_pred = noise_uncond + current_guidance_scale * (noise_pred - noise_uncond)

                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

        self._current_timestep = None

        if not output_type == "latent":
            latents = latents.to(self.vae.dtype)
            latents_mean = (
                torch.tensor(self.vae.config.latents_mean)
                .view(1, self.vae.config.z_dim, 1, 1, 1)
                .to(latents.device, latents.dtype)
            )
            latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
                latents.device, latents.dtype
            )
            latents = latents / latents_std + latents_mean
            video = self.vae.decode(latents, return_dict=False)[0]
            video = self.video_processor.postprocess_video(video, output_type=output_type)
        else:
            video = latents

        self.maybe_free_model_hooks()
        if not return_dict: return (video,)
        return WanPipelineOutput(frames=video)