# train_cond_lora.py (基于 EasyControl 方案修改：CIL 内部注入 + Patch Embedding 训练)
import os
import copy
import warnings
import argparse
import math
import torch
import torch.nn as nn
import pytorch_lightning as L
import numpy as np
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities import rank_zero_only

from diffusers import AutoencoderKLWan, FlowMatchEulerDiscreteScheduler, UniPCMultistepScheduler
from diffusers.utils import export_to_video
from transformers import AutoTokenizer, UMT5EncoderModel

from models.wan2.transformer_wan import WanTransformer3DModel
from models.wan2.custom_pipeline import CustomWanPipeline as WanPipeline
from tools.util import CustomProgressBar, CustomModelCheckpoint, masks_like
from tools.my_schedule import FlowMatchScheduler
from datasets.custom_dataset import CustomDataset  

from safetensors.torch import load_file
from peft import LoraConfig, get_peft_model

# ★ 引入修改后的 Attention Processor (请确保 attn_process.py 已更新为 EasyControl 版本)
from models.wan2.attn_process_ec import EasyControlAttnProcessor 

@rank_zero_only
def silence_warnings():
    warnings.filterwarnings("ignore", category=UserWarning)

os.environ["TOKENIZERS_PARALLELISM"] = "false"


# =========================================================================
# ★ 新增：EasyControl 的 Condition Injection LoRA (CIL) 模块
# =========================================================================
class ConditionInjectionLoRA(nn.Module):
    """
    对应 EasyControl 中的 CIL 模块。
    包含针对 Q, K, V 的 LoRA 适配器，仅作用于条件 Token。
    """
    def __init__(self, dim, r=64, alpha=64):
        super().__init__()
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        # 为 Q, K, V 分别建立 LoRA 分支
        self.lora_A_q = nn.Linear(dim, r, bias=False)
        self.lora_B_q = nn.Linear(r, dim, bias=False)
        
        self.lora_A_k = nn.Linear(dim, r, bias=False)
        self.lora_B_k = nn.Linear(r, dim, bias=False)
        
        self.lora_A_v = nn.Linear(dim, r, bias=False)
        self.lora_B_v = nn.Linear(r, dim, bias=False)

        self._init_weights()

    def _init_weights(self):
        # A 矩阵用 Kaiming 初始化，B 矩阵零初始化
        for layer in [self.lora_A_q, self.lora_A_k, self.lora_A_v]:
            nn.init.kaiming_uniform_(layer.weight, a=math.sqrt(5))
        for layer in [self.lora_B_q, self.lora_B_k, self.lora_B_v]:
            nn.init.zeros_(layer.weight)

    def forward(self, cond_hidden_states):
        """
        输入: 仅包含条件部分的 hidden_states [B, L_cond, D]
        输出: (delta_q, delta_k, delta_v)
        """
        d_q = self.lora_B_q(self.lora_A_q(cond_hidden_states)) * self.scaling
        d_k = self.lora_B_k(self.lora_A_k(cond_hidden_states)) * self.scaling
        d_v = self.lora_B_v(self.lora_A_v(cond_hidden_states)) * self.scaling
        
        return d_q, d_k, d_v


def _safe_load_state(path):
    ckpt = torch.load(path, map_location="cpu")
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    else:
        state = ckpt
    return state


def _infer_r_from_lora_state(lora_state: dict, fallback: int = 64):
    for k, v in lora_state.items():
        if isinstance(v, torch.Tensor) and ("lora_A.weight" in k or k.endswith("lora_A")):
            if v.ndim >= 2:
                return int(v.shape[0])
    return fallback


class CondOnlyLoRASystem(L.LightningModule):
    """
    EasyControl 风格训练系统：
    - 移除原来的外部 LoRAConv3d
    - 使用 CIL (Condition Injection LoRA) 在 Attention 内部注入条件
    - 训练目标：CIL 模块 + Patch Embedding Extra (特征对齐)
    """
    def __init__(self, opt):
        super().__init__()
        self.save_hyperparameters(opt)
        self.is_configured = False

    # ------- 基础构建 -------
    def configure_model(self):
        if self.is_configured:
            return
        self.is_configured = True

        model_id = self.hparams.model_id

        # tokenizer / text encoder（冻结）
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, subfolder="tokenizer")
        self.text_encoder = UMT5EncoderModel.from_pretrained(
            model_id, subfolder="text_encoder", torch_dtype=torch.float32
        )
        self.text_encoder.requires_grad_(False)

        # VAE（冻结）
        self.vae = AutoencoderKLWan.from_pretrained(
            model_id, subfolder="vae", torch_dtype=torch.float32
        )
        self.vae.requires_grad_(False)

        # 调度器
        self.train_scheduler = FlowMatchScheduler(shift=5, sigma_min=0.0, extra_one_step=True)
        self.train_scheduler.set_timesteps(1000, training=True)
        ttt = FlowMatchEulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
        self.sample_scheduler = UniPCMultistepScheduler.from_config(ttt.config, flow_shift=5)

        # 3D Transformer（主体冻结）
        self.transformer = WanTransformer3DModel.from_pretrained(
            model_id, subfolder="transformer", torch_dtype=torch.float32
        )
        self.transformer.requires_grad_(False)
        
        if self.hparams.training.get("gradient_checkpointing", False):
            self.transformer.gradient_checkpointing = True
            self.transformer.enable_gradient_checkpointing()

        # 1. 准备 patch_embedding_extra
        # 在 EasyControl 模式下，我们需要训练它来对齐 VAE Latent 和条件输入
        if not hasattr(self.transformer, "patch_embedding_extra"):
            self.transformer.patch_embedding_extra = copy.deepcopy(self.transformer.patch_embedding)
        
        # ★ 关键修改：解冻 patch_embedding_extra 以便训练
        self.transformer.patch_embedding_extra.requires_grad_(True)

        # 2. 准备 CIL 模块 (Condition Injection LoRA)
        # 每一层 Transformer Block 都需要一个对应的 CIL 模块
        inner_dim = self.transformer.config.num_attention_heads * self.transformer.config.attention_head_dim
        r = int(self.hparams.lora.get("r", 64))
        alpha = int(self.hparams.lora.get("alpha", 64))
        
        self.cil_modules = nn.ModuleList([
            ConditionInjectionLoRA(dim=inner_dim, r=r, alpha=alpha)
            for _ in range(len(self.transformer.blocks))
        ])

        # 3. 注入 EasyControlAttnProcessor
        for i, blk in enumerate(self.transformer.blocks):
            # 将对应的 CIL 模块传入 Processor
            processor = EasyControlAttnProcessor(cil_module=self.cil_modules[i])
            blk.attn1.set_processor(processor)

        # 注册 VAE 统计量
        self.register_buffer(
            'latents_mean',
            torch.tensor(self.vae.config.latents_mean).float().view(1, self.vae.config.z_dim, 1, 1, 1)
        )
        self.register_buffer(
            'latents_std',
            torch.tensor(self.vae.config.latents_std).float().view(1, self.vae.config.z_dim, 1, 1, 1)
        )

        # ★ 挂载第一种 LoRA（若提供了 p1_model）
        self._maybe_mount_p1_lora()

    # ------- 挂载第一种LoRA（逻辑保持不变） -------
    def _maybe_mount_p1_lora(self):
        p1_path = self.hparams.get("p1_model", None)
        if not p1_path:
            return
        if not os.path.isfile(p1_path):
            rank_zero_only(lambda: print(f"[WARN] p1_model 文件不存在：{p1_path}"))()
            return

        state = _safe_load_state(p1_path)
        text_lora_state = state.get("text_encoder_lora", {})
        transf_lora_state = state.get("transformer_lora", {})
        extra_state = state.get("patch_embedding_extra", {})

        if len(text_lora_state) > 0:
            r_text = int(self.hparams.get("p1_r_text", _infer_r_from_lora_state(text_lora_state, fallback=96)))
            alpha_text = int(self.hparams.get("p1_alpha_text", r_text))
            text_lora_cfg = LoraConfig(r=r_text, lora_alpha=alpha_text, init_lora_weights=True, target_modules=["q", "k", "v", "o"])
            self.text_encoder = get_peft_model(self.text_encoder, text_lora_cfg)
            self.text_encoder.requires_grad_(False)
            missing, unexpected = self.text_encoder.load_state_dict(text_lora_state, strict=False)
            rank_zero_only(lambda: print(
                f"[P1-Text] Loaded LoRA (r={r_text}, alpha={alpha_text}). "
                f"missing={len(missing)}, unexpected={len(unexpected)}"
            ))()
        else:
            rank_zero_only(lambda: print("[P1-Text] 未在 p1_model 中找到 text_encoder_lora，跳过。"))()

        if len(transf_lora_state) > 0:
            r_transf = int(self.hparams.get("p1_r_transf", _infer_r_from_lora_state(transf_lora_state, fallback=96)))
            alpha_transf = int(self.hparams.get("p1_alpha_transf", r_transf))
            transf_lora_cfg = LoraConfig(r=r_transf, lora_alpha=alpha_transf, init_lora_weights=True, target_modules=["to_k", "to_q", "to_v", "to_out.0","ffn.net.0.proj", "ffn.net.2"])
            self.transformer.add_adapter(transf_lora_cfg)
            self.transformer.requires_grad_(False)
            missing, unexpected = self.transformer.load_state_dict(transf_lora_state, strict=False)
            rank_zero_only(lambda: print(
                f"[P1-Transformer] Loaded LoRA (r={r_transf}, alpha={alpha_transf}). "
                f"missing={len(missing)}, unexpected={len(unexpected)}"
            ))()
        else:
            rank_zero_only(lambda: print("[P1-Transformer] 未在 p1_model 中找到 transformer_lora，跳过。"))()

        # =========================================================
        # [修改点 2]：严禁 P1 覆盖 Patch Embedding Extra
        # =========================================================
        # 原因：Patch Embedding Extra 现在是 P2 训练的一部分，应该由 Lightning 的 ckpt_path 恢复，
        # 而不是被 P1 的权重（可能是初始化的或旧的）覆盖。
        if len(extra_state) > 0:
            rank_zero_only(lambda: print("[INFO] 检测到 P1 模型包含 patch_embedding_extra，但为了防止覆盖训练进度，已跳过加载。"))()
            # self.transformer.load_state_dict(extra_state, strict=False) # <--- 注释掉这一行
            
        else:
            rank_zero_only(lambda: print("[INFO] P1 模型未包含 patch_embedding_extra，使用默认初始化。"))()

    # ------- Prompt 编码 -------
    def encode_prompt(self, prompt_list):
        max_sequence_length = 512
        tok = self.tokenizer(
            prompt_list, padding="max_length", max_length=max_sequence_length,
            truncation=True, add_special_tokens=True, return_attention_mask=True, return_tensors="pt",
        )
        ids, mask = tok.input_ids.to(self.device), tok.attention_mask.to(self.device)
        with torch.no_grad():
            text_embeds = self.text_encoder(ids, mask).last_hidden_state
        seq_lens = mask.gt(0).sum(dim=1).long()
        text_embeds = [u[:v] for u, v in zip(text_embeds, seq_lens)]
        text_embeds = torch.stack(
            [torch.cat([u, u.new_zeros(max_sequence_length - u.size(0), u.size(1))]) for u in text_embeds], dim=0
        )
        return text_embeds

    # ------- Data 解包 -------
    def process_data(self, batch):
        cond_video = batch["pixel_values"]
        target_video = batch["pixel_values2"]
        first_frames = batch["first_frames"].unsqueeze(2)
        prompts = batch.get("prompts", [""] * cond_video.size(0))
        
        if getattr(self.hparams, "use_drop_text", False):
            p = float(getattr(self.hparams, "drop_prob", 0.1))
            prompts = [pr if np.random.rand() > p else "" for pr in prompts]

        return cond_video, target_video, first_frames, prompts

    # ------- 前向/损失 -------
    def forward(self, cond_latent, first_latent, target_latent, prompt_embeds):
        B = target_latent.size(0)
        device = target_latent.device

        # 1. 加噪 (仅 Target)
        noise = torch.randn_like(target_latent)
        timestep_id = torch.randint(0, self.train_scheduler.num_train_timesteps, (B,), device=device)
        timestep = self.train_scheduler.timesteps[timestep_id].to(device=device, dtype=target_latent.dtype)
        latent_noisy = self.train_scheduler.add_noise(target_latent, noise, timestep)
        v_target = self.train_scheduler.training_target(target_latent, noise, timestep)

        # 2. ★ 移除 Adapter 调用
        # cond_latent = self.cond_adapter(cond_latent) <--- 删除这行
        # 直接使用 VAE Latent，它会进入 patch_embedding_extra (可训练) 并通过 CIL 注入

        # 3. 构造 t_embed (Target 带时间，Cond/First 时间为0)
        _, mask2 = masks_like(noise, zero=False)
        kkk = (timestep.view(B, 1, 1, 1) * mask2[0][:, 0, :, ::2, ::2]).flatten(1)
        zeros_cond = torch.zeros_like(kkk)
        zeros_first = torch.zeros_like(mask2[0][:, 0, 0, ::2, ::2].flatten(1))
        t_embed = torch.cat([kkk, zeros_cond, zeros_first], dim=-1)

        # 4. 传入 Transformer
        attention_kwargs = {
            'encoder_contion_states': cond_latent,
            'encoder_first_states': first_latent,
        }

        v_pred = self.transformer(
            hidden_states=latent_noisy,
            encoder_hidden_states=prompt_embeds,
            timestep=t_embed,
            return_dict=False,
            attention_kwargs=attention_kwargs,
        )[0]

        loss = torch.nn.functional.mse_loss(v_pred.float(), v_target.float(), reduction='none')
        weight = self.train_scheduler.training_weight(timestep).to(loss.device)
        loss = (loss * weight[:, None, None, None, None]).mean()
        return loss

    # ------- Lightning Hooks -------
    def training_step(self, batch, batch_idx):
        self.configure_model()
        cond_video, target_video, first_frames, prompts = self.process_data(batch)

        with torch.no_grad():
            cond_latent = self.vae.encode(cond_video).latent_dist.sample()
            cond_latent = (cond_latent - self.latents_mean) / self.latents_std

            target_latent = self.vae.encode(target_video).latent_dist.sample()
            target_latent = (target_latent - self.latents_mean) / self.latents_std

            first_latent = self.vae.encode(first_frames).latent_dist.sample()
            first_latent = (first_latent - self.latents_mean) / self.latents_std

        prompt_embeds = self.encode_prompt(prompts)
        
        # 直接传入原始 latent，不再经过 adapter
        loss = self.forward(cond_latent, first_latent, target_latent, prompt_embeds)

        self.log("train/loss", loss, prog_bar=True, on_step=True, logger=True, sync_dist=True)
        self.log("lr", self.trainer.optimizers[0].param_groups[0]["lr"], prog_bar=True, on_step=True, logger=True)
        return loss

    def on_validation_epoch_start(self):
        # 验证时需要重新实例化 Pipeline
        self.pipeline = WanPipeline(
            vae=self.vae,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            transformer=self.transformer,
            scheduler=self.sample_scheduler,
        ).to(self.device)
        self.val_path = os.path.join(self.hparams.output_root, self.hparams.experiment_name, 'val_samples_cil')
        os.makedirs(self.val_path, exist_ok=True)

    def validation_step(self, batch, batch_idx):
        cond_video, target_video, first_frames, prompts = self.process_data(batch)

        video_gt = target_video.squeeze(0).permute(1, 0, 2, 3)
        video_gt = ((video_gt + 1) * 0.5).clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy()

        with torch.no_grad():
            cond_latent = self.vae.encode(cond_video).latent_dist.sample()
            cond_latent = (cond_latent - self.latents_mean) / self.latents_std

            first_latent = self.vae.encode(first_frames).latent_dist.sample()
            first_latent = (first_latent - self.latents_mean) / self.latents_std

        # ★ 移除 Adapter 调用 (验证阶段同样如此)
        # cond_latent = self.cond_adapter(cond_latent)

        attention_kwargs = {
            'encoder_contion_states': cond_latent,
            'encoder_first_states': first_latent,
        }

        out = self.pipeline(
            prompt=prompts,
            height=self.hparams.dataset.height,
            width=self.hparams.dataset.width,
            num_frames=self.hparams.dataset.sample_n_frames,
            guidance_scale=5.0,
            attention_kwargs=attention_kwargs,
        )
        video_generate = out.frames[0]

        min_f = min(video_generate.shape[0], video_gt.shape[0])
        video_generate = video_generate[:min_f]
        video_gt = video_gt[:min_f]

        concatenated = np.concatenate([video_generate, video_gt], axis=1)
        val_video_path = os.path.join(self.val_path, f"val_{self.global_step}step-batch_{batch_idx}-rank{self.trainer.global_rank}.mp4")
        export_to_video(concatenated, output_video_path=val_video_path, fps=self.hparams.dataset.fps)

    def on_predict_epoch_start(self):
        self.pred_pipeline = WanPipeline(
            vae=self.vae,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            transformer=self.transformer,
            scheduler=self.sample_scheduler,
        ).to(self.device)
        self.pred_path = os.path.join(self.hparams.output_root, self.hparams.experiment_name, 'pred_samples_cil')
        os.makedirs(self.pred_path, exist_ok=True)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        cond_video, _, first_frames, prompts = self.process_data(batch)
        with torch.no_grad():
            cond_latent = self.vae.encode(cond_video).latent_dist.sample()
            cond_latent = (cond_latent - self.latents_mean) / self.latents_std
            first_latent = self.vae.encode(first_frames).latent_dist.sample()
            first_latent = (first_latent - self.latents_mean) / self.latents_std

        # ★ 移除 Adapter 调用
        # cond_latent = self.cond_adapter(cond_latent)

        attention_kwargs = {
            'encoder_contion_states': cond_latent,
            'encoder_first_states': first_latent,
        }
        out = self.pred_pipeline(
            prompt=prompts,
            height=self.hparams.dataset.height,
            width=self.hparams.dataset.width,
            num_frames=self.hparams.dataset.sample_n_frames,
            guidance_scale=5.0,
            attention_kwargs=attention_kwargs,
        )
        video_generate = out.frames[0]
        pred_video_path = os.path.join(self.pred_path, f"batch_{batch_idx}-rank{self.trainer.global_rank}.mp4")
        export_to_video(video_generate, output_video_path=pred_video_path, fps=self.hparams.dataset.fps)

    # ------- 优化器配置 (训练 CIL + Patch Embedding) -------
    def configure_optimizers(self):
        self.configure_model()
        
        params = []
        # 1. 加入所有 CIL 模块的参数
        for cil in self.cil_modules:
            params.extend([p for p in cil.parameters() if p.requires_grad])
        
        # 2. 加入 Patch Embedding Extra 的参数 (特征对齐)
        params.extend([p for p in self.transformer.patch_embedding_extra.parameters() if p.requires_grad])

        lr = self.hparams.training.learning_rate * \
             (self.hparams.training.accumulate_grad_batches * self.hparams.num_gpus * self.hparams.num_nodes) ** 0.5
        
        optimizer = torch.optim.AdamW(
            [{"params": params, "lr": lr}],
            betas=(0.9, 0.95), eps=1e-8,
            weight_decay=self.hparams.training.weight_decay,
        )

        def lr_fn(step, warmup_steps):
            return 1 if warmup_steps <= 0 else min(step / warmup_steps, 1)

        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda step: lr_fn(step, warmup_steps=self.hparams.training.warmup_steps)
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": lr_scheduler, "interval": "step"}}

    # ------- 保存/加载 -------
    def on_save_checkpoint(self, checkpoint):
        checkpoint.pop('hparams_name', None)
        checkpoint.pop('hparams_type', None)
        # ★ 保存 CIL 模块和 Patch Embedding
        state = {
            "cil_modules": self.cil_modules.state_dict(),
            "patch_embedding_extra": self.transformer.patch_embedding_extra.state_dict()
        }
        checkpoint['state_dict'] = state

    def load_state_dict(self, state_dict, strict: bool = True):
        # 恢复 CIL
        if "cil_modules" in state_dict:
            self.cil_modules.load_state_dict(state_dict["cil_modules"], strict=False)
        # 恢复 Patch Embedding
        if "patch_embedding_extra" in state_dict:
            self.transformer.patch_embedding_extra.load_state_dict(state_dict["patch_embedding_extra"], strict=False)


# ------------------- main (保持大部分不变) -------------------
def main(opt):
    L.seed_everything(opt.seed)

    train_dataset = CustomDataset(
        video_root=opt.dataset.video_root,
        video_root2=opt.dataset.video_root2,
        robot_ref_path=opt.dataset.first_root,
        height=opt.dataset.height,
        width=opt.dataset.width,
        sample_n_frames=opt.dataset.sample_n_frames,
        is_one2three=True,
        training_len=opt.num_nodes * opt.num_gpus * opt.training.accumulate_grad_batches * opt.training.max_steps * opt.training.batch_size
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=opt.training.batch_size,
        num_workers=opt.dataset.num_workers,
        drop_last=opt.dataset.drop_last,
        pin_memory=opt.dataset.pin_memory,
        shuffle=opt.dataset.shuffle,
    )

    val_dataset = CustomDataset(
        video_root=opt.dataset.video_root,
        video_root2=opt.dataset.video_root2,
        robot_ref_path=opt.dataset.first_root,
        height=opt.dataset.height,
        width=opt.dataset.width,
        is_one2three=True,
        sample_n_frames=opt.dataset.sample_n_frames,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=1,
        num_workers=opt.dataset.num_workers,
        drop_last=False, pin_memory=opt.dataset.pin_memory, shuffle=False
    )

    system = CondOnlyLoRASystem(opt)

    logger = None
    if opt.get("use_wandb", False):
        logger = WandbLogger(
            project=opt.experiment_project,
            name=opt.experiment_name,
            save_dir=os.path.join(opt.output_root, opt.experiment_name),
            log_model=False, offline=False
        )

    trainer = L.Trainer(
        logger=logger,
        max_steps=opt.training.max_steps,
        precision=opt.training.precision,
        num_sanity_val_steps=0,
        limit_val_batches=1,
        val_check_interval=opt.training.save_val_interval_steps * opt.training.accumulate_grad_batches,
        accumulate_grad_batches=opt.training.accumulate_grad_batches,
        gradient_clip_val=opt.training.gradient_clip_val,
        gradient_clip_algorithm='value',
        log_every_n_steps=1,
        accelerator=opt.training.accelerator,
        strategy=opt.training.strategy,
        benchmark=opt.training.benchmark,
        callbacks=[
            CustomProgressBar(),
            CustomModelCheckpoint(
                dirpath=os.path.join(opt.output_root, opt.experiment_name, 'checkpoints_cil'),
                filename="{step}",
                every_n_train_steps=opt.training.save_val_interval_steps,
                save_top_k=-1,
                save_weights_only=False,
                verbose=False,
            )
        ],
        num_nodes=opt.num_nodes,
    )

    trainer.fit(system, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=opt.ckpt_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/main.yaml", help="path to the yaml config file")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ckpt_path", type=str, default=None)
    parser.add_argument("--p1_model", type=str, default=None)
    
    args, extras = parser.parse_known_args()
    args = vars(args)

    opt = OmegaConf.merge(
        OmegaConf.load(args['config']),
        OmegaConf.from_cli(extras),
        OmegaConf.create(args),
        OmegaConf.create({"num_nodes": int(os.environ.get("NUM_NODES", 1))}),
        OmegaConf.create({"num_gpus": int(torch.cuda.device_count())}),
        OmegaConf.create({
            # "lora": {"r": 64, "alpha": 64}, # CIL 的 r 值
        }),
    )
    opt.ckpt_path = None if args['ckpt_path'] in ("", "null", "None") else args['ckpt_path']
    opt.p1_model = None if args.get('p1_model') in ("", "null", "None") else args.get('p1_model')

    main(opt)