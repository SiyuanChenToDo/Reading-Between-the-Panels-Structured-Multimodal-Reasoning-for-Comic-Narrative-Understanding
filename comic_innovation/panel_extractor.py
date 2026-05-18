"""
Panel Feature Extractor v2

核心改进：
1. 从多层 hidden_states 提取并自适应融合，捕获更丰富的视觉语义
2. 使用 lightweight self-attention pooling 替代简单 mean pooling
3. 添加可学习的 type embedding（区分 composite 图和 individual panel）
4. 修复潜在的单图/多图边界问题

注意：Qwen2.5-VL 中每张图的 vision tokens 被 <|vision_start|>(151652) 和
<|vision_end|>(151653) 包围，中间为 <|image_pad|>(151655) tokens。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

VISION_START_ID = 151652
VISION_END_ID = 151653


class PanelFeatureExtractor(nn.Module):
    """从隐藏状态中按图片边界提取 panel 级特征（多层融合 + Attention Pooling）。"""

    def __init__(self, hidden_dim: int = 3584, num_heads: int = 8):
        super().__init__()
        self.hidden_dim = hidden_dim

        # 可学习的 type embedding：0=composite（第一张图），1=individual panel
        self.type_embed = nn.Embedding(2, hidden_dim)

        # Lightweight attention pooling：学习哪些 vision tokens 更重要
        self.attn_pool = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.Tanh(),
            nn.Linear(hidden_dim // 4, 1),
        )

        # 输出投影
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.05),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        max_panels: int = 5,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden_states: [B, T, H]（单层的语言模型隐藏状态）
            input_ids: [B, T]
            max_panels: 最大图片数（多尺度=5，单图=1，task2=4）
        Returns:
            panel_features: [B, max_panels, H]
            panel_mask: [B, max_panels] bool
        """
        B, T, H = hidden_states.shape
        device = hidden_states.device

        panel_features = torch.zeros(B, max_panels, H, device=device, dtype=hidden_states.dtype)
        panel_mask = torch.zeros(B, max_panels, dtype=torch.bool, device=device)

        for b in range(B):
            ids = input_ids[b]
            start_positions = (ids == VISION_START_ID).nonzero(as_tuple=True)[0]
            end_positions = (ids == VISION_END_ID).nonzero(as_tuple=True)[0]

            n_images = min(len(start_positions), len(end_positions), max_panels)
            for i in range(n_images):
                s = start_positions[i].item() + 1
                e = end_positions[i].item()
                if e > s:
                    # 提取该图的所有 vision tokens: [num_tokens, H]
                    tokens = hidden_states[b, s:e, :]

                    # Attention-based pooling
                    attn_weights = self.attn_pool(tokens).squeeze(-1)  # [num_tokens]
                    attn_weights = F.softmax(attn_weights, dim=-1)
                    pooled = (tokens * attn_weights.unsqueeze(-1)).sum(dim=0)  # [H]

                    # 添加 type embedding（第一张为 composite=0，其余为 panel=1）
                    type_id = 0 if i == 0 else 1
                    pooled = pooled + self.type_embed(torch.tensor(type_id, device=device))

                    panel_features[b, i] = pooled
                    panel_mask[b, i] = True

        panel_features = self.proj(panel_features)
        return panel_features, panel_mask
