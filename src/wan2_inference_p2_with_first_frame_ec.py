import os
import torch
import numpy as np
import argparse
import copy
from PIL import Image
from decord import VideoReader
from torchvision import transforms
from diffusers import AutoencoderKLWan, UniPCMultistepScheduler
from peft import LoraConfig, get_peft_model

# 导入你项目中的模块
from models.wan2.transformer_wan import WanTransformer3DModel
from models.wan2.custom_pipeline import CustomWanPipeline as WanPipeline
from models.wan2.attn_process_ec import EasyControlAttnProcessor
from models.wan2.ec_lora import ConditionInjectionLoRA 

def _safe_load_state(path):
    ckpt = torch.load(path, map_location="cpu")
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    else:
        state = ckpt
    return state

def _infer_r_from_lora_state(state, fallback=96):
    for k, v in state.items():
        if "lora_A" in k:
            return v.shape[0]
    return fallback

def load_video(path, height, width, sample_n_frames=49):
    vr = VideoReader(path)
    total_frames = len(vr)
    indices = np.linspace(0, total_frames - 1, sample_n_frames).astype(int)
    video = vr.get_batch(indices).asnumpy()
    
    transform = transforms.Compose([
        transforms.Resize((height, width)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    
    # [F, C, H, W] -> [C, F, H, W]
    pixel_values = torch.stack([transform(Image.fromarray(f)) for f in video])
    return pixel_values.permute(1, 0, 2, 3).unsqueeze(0) # [1, C, F, H, W]

def load_image(path, height, width):
    img = Image.open(path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((height, width)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    return transform(img).unsqueeze(0).unsqueeze(2) # [1, C, 1, H, W]

@torch.no_grad()
def inference(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    weight_dtype = torch.bfloat16 if args.use_bf16 else torch.float32

    # 1. 加载基础模型
    print(f"Loading Base Model: {args.model_id}")
    vae = AutoencoderKLWan.from_pretrained(args.model_id, subfolder="vae", torch_dtype=torch.float32)
    transformer = WanTransformer3DModel.from_pretrained(args.model_id, subfolder="transformer", torch_dtype=torch.float32)
    scheduler = UniPCMultistepScheduler.from_pretrained(args.model_id, subfolder="scheduler", flow_shift=5.0)
    
    # 临时初始化 pipe 以便挂载 Text Encoder LoRA
    pipe = WanPipeline.from_pretrained(args.model_id, vae=vae, transformer=transformer, scheduler=scheduler, torch_dtype=weight_dtype)

    # 2. 挂载 P1 LoRA (终极 Key 对齐版)
    if args.p1_path:
        print(f"Mounting P1 LoRA from: {args.p1_path}")
        state = _safe_load_state(args.p1_path)
        
        transf_lora_state = state.get("transformer_lora", {})
        if transf_lora_state:
            r_t = _infer_r_from_lora_state(transf_lora_state)
            transf_lora_cfg = LoraConfig(
                r=r_t, lora_alpha=r_t, 
                target_modules=["to_k", "to_q", "to_v", "to_out.0", "ffn.net.0.proj", "ffn.net.2"]
            )
            
            # 1. 直接添加 adapter
            adapter_name = "default" # 建议先用 default，与训练保持一致
            transformer.add_adapter(transf_lora_cfg, adapter_name)
            
            # 2. 预处理 state_dict：只替换 adapter 名称，不加 base_model.model 前缀
            # 如果训练时保存的 key 包含 "default"，而你这里指定了 "p1_adapter"，才需要替换
            new_state = {}
            for k, v in transf_lora_state.items():
                # 去掉可能存在的 base_model.model. 前缀（如果有的话）
                fixed_k = k.replace("base_model.model.", "")
                # 确保 adapter 名称对齐
                fixed_k = fixed_k.replace(".default.", f".{adapter_name}.")
                new_state[fixed_k] = v

            # 3. 载入
            msg = transformer.load_state_dict(new_state, strict=False)
            transformer.set_adapter(adapter_name)
            
            print(f"✅ P1 Transformer LoRA Fixed.")
            print(f"   - Missing keys: {len(msg.missing_keys)}")
            print(f"   - Unexpected keys: {len(msg.unexpected_keys)}")
        else:
            print("⚠️ No Transformer LoRA found in P1.")
        
        # --- Text Encoder P1 ---
        text_lora_state = state.get("text_encoder_lora", {})
        if text_lora_state:
            r_txt = _infer_r_from_lora_state(text_lora_state)
            text_lora_cfg = LoraConfig(r=r_txt, lora_alpha=r_txt, target_modules=["q", "k", "v", "o"])
            pipe.text_encoder = get_peft_model(pipe.text_encoder, text_lora_cfg)
            msg = pipe.text_encoder.load_state_dict(text_lora_state, strict=False)
            print(f"✅ P1 Text Encoder LoRA Loaded. Missing keys: {len(msg.missing_keys)}")
            print(f"✅ P1 Text Encoder LoRA Loaded. Unexpected keys: {len(msg.unexpected_keys)}")
        else:
            print("⚠️ No Text Encoder LoRA found in P1 checkpoint.")

    # 3. 挂载 P2 CIL & Patch Embedding Extra
    print(f"Mounting P2 CIL from: {args.p2_path}")
    p2_state = torch.load(args.p2_path, map_location="cpu")
    if "state_dict" in p2_state: p2_state = p2_state["state_dict"]

    transformer.patch_embedding_extra = copy.deepcopy(transformer.patch_embedding)
    inner_dim = transformer.config.num_attention_heads * transformer.config.attention_head_dim
    cil_modules = torch.nn.ModuleList([
        ConditionInjectionLoRA(dim=inner_dim, r=args.cil_r, alpha=args.cil_alpha)
        for _ in range(len(transformer.blocks))
    ])
    
    transformer.patch_embedding_extra.load_state_dict(p2_state["patch_embedding_extra"])
    cil_modules.load_state_dict(p2_state["cil_modules"])

    for i, blk in enumerate(transformer.blocks):
        processor = EasyControlAttnProcessor(cil_module=cil_modules[i].to(device, weight_dtype))
        blk.attn1.set_processor(processor)

    # 4. 准备数据并推理
    transformer.to(device, weight_dtype)
    pipe.to(device) # pipe 已经包含了更新后的 transformer 和 text_encoder

    cond_video = load_video(args.cond_video_path, args.height, args.width, args.num_frames).to(device)
    ref_image = load_image(args.ref_image_path, args.height, args.width).to(device)

    with torch.no_grad():
        z_dim = vae.config.z_dim
        v_mean = torch.tensor(vae.config.latents_mean[:z_dim]).view(1, z_dim, 1, 1, 1).to(device)
        v_std = torch.tensor(vae.config.latents_std[:z_dim]).view(1, z_dim, 1, 1, 1).to(device)
        
        cond_latents = (vae.encode(cond_video).latent_dist.sample() - v_mean) / v_std
        ref_latents = (vae.encode(ref_image).latent_dist.sample() - v_mean) / v_std

    print(f"Generating: {args.prompt}")
    output = pipe(
        prompt=args.prompt,
        height=args.height, width=args.width, num_frames=args.num_frames,
        num_inference_steps=args.steps, guidance_scale=args.cfg,
        attention_kwargs={
            "encoder_contion_states": cond_latents.to(weight_dtype),
            "encoder_first_states": ref_latents.to(weight_dtype),
        }
    ).frames[0]

    # 5. 保存
    from diffusers.utils import export_to_video
    os.makedirs(args.output_dir, exist_ok=True)
    video_filename = args.output_name if args.output_name.endswith(".mp4") else f"{args.output_name}.mp4"
    out_path = os.path.join(args.output_dir, video_filename)
    export_to_video(output, out_path, fps=args.fps)
    print(f"🚀 Saved to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
    parser.add_argument("--p1_path", type=str, help="P1 LoRA .pt file")
    parser.add_argument("--p2_path", type=str, required=True, help="P2 CIL .ckpt file")
    parser.add_argument("--cond_video_path", type=str, required=True)
    parser.add_argument("--ref_image_path", type=str, required=True)
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--output_name", type=str, default="output_video")
    parser.add_argument("--cil_r", type=int, default=64)
    parser.add_argument("--cil_alpha", type=int, default=64)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--num_frames", type=int, default=49)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--cfg", type=float, default=5.0)
    parser.add_argument("--use_bf16", action="store_true", default=False)
    parser.add_argument("--output_dir", type=str, default="./outputs")
    args = parser.parse_args()
    inference(args)