import os, argparse, torch, numpy as np
from omegaconf import OmegaConf
from typing import List, Dict
from transformers import AutoTokenizer, UMT5EncoderModel
from diffusers import AutoencoderKLWan, FlowMatchEulerDiscreteScheduler, UniPCMultistepScheduler
from diffusers.utils import export_to_video
from peft import LoraConfig, get_peft_model

from safetensors.torch import load_file

from models.wan2.transformer_wan import WanTransformer3DModel
from models.wan2.custom_pipeline import CustomWanPipeline as WanPipeline

def _safe_load_state(path):
    ckpt = torch.load(path, map_location="cpu")
    return ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt



def _infer_r_from_lora_state(lora_state: Dict[str, torch.Tensor], fallback: int = 64):
    for k, v in lora_state.items():
        if not isinstance(v, torch.Tensor):
            continue
        if v.ndim != 2:
            continue
        name = k.lower()
        if ("lora_a" in name or "lora_down" in name) and name.endswith("weight"):
            return int(v.shape[0])  # [r, in]
    if fallback is not None:
        return int(fallback)
    raise ValueError("无法从 LoRA 权重中推断 r，请手动指定（例如 --p1_r_text / --p1_r_transf）。")

def _scheduler_tensors_to_numpy(sched):
    import numpy as np, torch
    for name in ["betas","alphas","alphas_cumprod","alphas_cumprod_prev","sigmas","timesteps"]:
        if hasattr(sched, name):
            v = getattr(sched, name)
            if isinstance(v, torch.Tensor):
                setattr(sched, name, v.detach().cpu().float().numpy())
    return sched

def build_pipeline_and_load_lora(model_id: str, lora_ckpt: str, device: str):
    # base
    tokenizer = AutoTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    text_encoder = UMT5EncoderModel.from_pretrained(model_id, subfolder="text_encoder", torch_dtype=torch.float32)
    vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)

    ttt = FlowMatchEulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
    scheduler = UniPCMultistepScheduler.from_config(ttt.config, flow_shift=5)
    _scheduler_tensors_to_numpy(scheduler)

    transformer = WanTransformer3DModel.from_pretrained(model_id, subfolder="transformer", torch_dtype=torch.float32)

    # 注入 LoRA
    state = _safe_load_state(lora_ckpt)
    text_lora_state = state.get("text_encoder_lora", {})
    transf_lora_state = state.get("transformer_lora", {})
    extra_state = state.get("patch_embedding_extra", {})

    if len(text_lora_state) > 0:
        r_text = _infer_r_from_lora_state(text_lora_state, fallback=96)
        text_encoder = get_peft_model(
            text_encoder,
            LoraConfig(r=r_text, lora_alpha=r_text, init_lora_weights=True, target_modules=["q","k","v","o"])
        )
        text_encoder.load_state_dict(text_lora_state, strict=False)

    if len(transf_lora_state) > 0:
        r_transf = _infer_r_from_lora_state(transf_lora_state, fallback=None)
        transformer.add_adapter(
            LoraConfig(r=r_transf, lora_alpha=r_transf, init_lora_weights=True,
                       target_modules=["to_k","to_q","to_v","to_out.0","ffn.net.0.proj", "ffn.net.2"])
        )
        transformer.load_state_dict(transf_lora_state, strict=False)

    # patch_embedding_extra（若有就加载；否则复制一份）
    if not hasattr(transformer, "patch_embedding_extra"):
        import copy
        transformer.patch_embedding_extra = copy.deepcopy(transformer.patch_embedding)
    if len(extra_state) > 0:
        transformer.load_state_dict(extra_state, strict=False)

    pipe = WanPipeline(
        vae=vae, text_encoder=text_encoder, tokenizer=tokenizer,
        transformer=transformer, scheduler=scheduler
    ).to(device)

    # 第一种训练未扩展 timestep
    pipe.config.expand_timesteps_factor = 1
    return pipe

def main(opt):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(opt.output_dir, exist_ok=True)

    pipe = build_pipeline_and_load_lora(opt.model_id, opt.lora_ckpt, device=device)

    # prompts
    if opt.prompts is not None:
        prompts = [p.strip() for p in opt.prompts.split("|") if p.strip()]
    else:
        prompts = ["A person waves to a robot in a lab, cinematic lighting."]

    with torch.no_grad():
        for i, prompt in enumerate(prompts):
            out = pipe(
                prompt=prompt,
                height=opt.height,
                width=opt.width,
                num_frames=opt.num_frames,
                guidance_scale=opt.guidance_scale,
                num_inference_steps=opt.steps,
                output_type="np",
            ).frames[0]
            export_to_video(out, os.path.join(opt.output_dir, f"p1_{i:03d}.mp4"), fps=opt.fps)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/main.yaml")
    parser.add_argument("--lora_ckpt", required=True, help="第一种LoRA ckpt 路径（含 text/transformer/extra）")
    parser.add_argument("--prompts", type=str, default=None, help="用 | 分隔的多条 prompt")
    parser.add_argument("--output_dir", type=str, default="outputs_p1")
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--width", type=int, default=416)
    parser.add_argument("--num_frames", type=int, default=32)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=5.0)
    args, extras = parser.parse_known_args()

    cfg = OmegaConf.load(args.config) if os.path.isfile(args.config) else OmegaConf.create()
    opt = OmegaConf.merge(
        cfg,
        OmegaConf.from_cli(extras),
        OmegaConf.create(vars(args)),
        OmegaConf.create({"model_id": cfg.get("model_id", args.config)})
    )
    # 允许从 yaml 取默认分辨率/帧数
    for k in ["height","width","num_frames","fps"]:
        if k in cfg.get("dataset", {}):
            setattr(opt, k, cfg.dataset[k])
    main(opt)
