import torch
from safetensors.torch import load_file
import argparse

def convert_safetensors_to_ckpt(input_path, output_path):
    print(f"正在加载: {input_path}")
    raw_state = load_file(input_path)
    
    # 准备输出结构
    output_dict = {
        "transformer_lora": {},
        "text_encoder_lora": {},
        "patch_embedding_extra": {}
    }
    
    mapped_count = 0
    
    for k, v in raw_state.items():
        new_k = k
        
        # 1. 去掉 transformer 前缀 (如果有)
        if new_k.startswith("transformer."):
            new_k = new_k.replace("transformer.", "")
            
        # 2. 【关键修正】确保有 .default 后缀
        # 如果原始 key 没有 .default，我们要加上它，因为 inference 脚本里 add_adapter 默认叫 "default"
        if "lora_" in new_k and "default" not in new_k:
             new_k = new_k.replace(".weight", ".default.weight")


        # === 核心映射逻辑 ===
        
        # A. Self Attention -> attn1
        if "self_attn" in new_k:
            new_k = new_k.replace("self_attn.q", "attn1.to_q")
            new_k = new_k.replace("self_attn.k", "attn1.to_k")
            new_k = new_k.replace("self_attn.v", "attn1.to_v")
            new_k = new_k.replace("self_attn.o", "attn1.to_out.0")

        # B. Cross Attention -> attn2
        elif "cross_attn" in new_k:
            new_k = new_k.replace("cross_attn.q", "attn2.to_q")
            new_k = new_k.replace("cross_attn.k", "attn2.to_k")
            new_k = new_k.replace("cross_attn.v", "attn2.to_v")
            new_k = new_k.replace("cross_attn.o", "attn2.to_out.0")

        # C. FFN -> ffn.net
        elif "ffn" in new_k:
            # 你的 key 是 ffn.0 和 ffn.2
            # 映射到 diffusers 的 ffn.net.0.proj 和 ffn.net.2.proj
            new_k = new_k.replace("ffn.0", "ffn.net.0.proj")
            new_k = new_k.replace("ffn.2", "ffn.net.2")
            # 部分 LoRA 可能会有 ffn.1，如果有也做相应处理，但你截图里只有 0 和 2

        # 3. 归类存储
        if "blocks." in new_k or "head." in new_k or "patch_embedding." in new_k:
            output_dict["transformer_lora"][new_k] = v
            mapped_count += 1
        elif "text_encoder" in k or "te_" in k:
            clean_k = k.replace("text_encoder.", "")
            output_dict["text_encoder_lora"][clean_k] = v
            mapped_count += 1

    print(f"转换完成！共映射了 {mapped_count} 个权重。")
    print(f"其中包含 Self-Attn, Cross-Attn 和 FFN 全部层级。")
    
    # 保存
    torch.save({"state_dict": output_dict}, output_path)
    print(f"成功保存到: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()
    
    out = args.output if args.output else args.input.replace(".safetensors", "_converted.ckpt")
    convert_safetensors_to_ckpt(args.input, out)


# python /opt/liblibai-models/user-workspace2/users/dxy/robotic/DiffSynth-Studio/first_phrase_output/safetensor2ckpt.py --input /opt/liblibai-models/user-workspace2/users/dxy/robotic/DiffSynth-Studio/first_phrase_output/test_real_world/fourier-gr3/continue/step-4000.safetensors --output /opt/liblibai-models/user-workspace2/users/dxy/robotic/InteractionVideo/p1_lora_library/DS_real/fourier-gr3_9000.ckpt