import torch
import torch.nn as nn
import math

class ConditionInjectionLoRA(nn.Module):
    """
    对应 EasyControl 中的 CIL 模块。
    包含针对 Q, K, V 的 LoRA 适配器。
    """
    def __init__(self, dim, r=64, alpha=64):
        super().__init__()
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        # 为 Q, K, V 分别建立 LoRA 分支
        # EasyControl 论文公式 (2): A_Q, B_Q ...
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
        # 计算增量: B(A(x)) * scale
        d_q = self.lora_B_q(self.lora_A_q(cond_hidden_states)) * self.scaling
        d_k = self.lora_B_k(self.lora_A_k(cond_hidden_states)) * self.scaling
        d_v = self.lora_B_v(self.lora_A_v(cond_hidden_states)) * self.scaling
        
        return d_q, d_k, d_v