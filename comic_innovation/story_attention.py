"""
Story-Aware Attention v3.1 -- Sinkhorn-based Permutation Learning (Reviewed)

核心改进（相比 v2）：
1. 引入 Sinkhorn Operator 进行可微分双随机矩阵学习，替代粗糙的24类分类
2. 精简参数量：~10M（原~130M），避免小batch下难以收敛
3. 修复 pairwise loss 对角线污染问题
4. 补全全部24种排列（保留用于兼容评估）
5. 移除有问题的 consistency_loss（训练早期伪标签随机，会误导学习）
6. _build_target 函数全面向量化，效率提升
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# 全部24种排列（中文task1为4! = 24）
ALL_PERMS = [
    (0, 1, 2, 3), (0, 1, 3, 2), (0, 2, 1, 3), (0, 2, 3, 1), (0, 3, 1, 2), (0, 3, 2, 1),
    (1, 0, 2, 3), (1, 0, 3, 2), (1, 2, 0, 3), (1, 2, 3, 0), (1, 3, 0, 2), (1, 3, 2, 0),
    (2, 0, 1, 3), (2, 0, 3, 1), (2, 1, 0, 3), (2, 1, 3, 0), (2, 3, 0, 1), (2, 3, 1, 0),
    (3, 0, 1, 2), (3, 0, 2, 1), (3, 1, 0, 2), (3, 1, 2, 0), (3, 2, 0, 1), (3, 2, 1, 0),
]
PERM_TO_IDX = {p: i for i, p in enumerate(ALL_PERMS)}
NUM_PERMS = len(ALL_PERMS)


def sinkhorn_normalize(log_scores: torch.Tensor, n_iters: int = 20, temp: float = 0.5) -> torch.Tensor:
    """
    对数域 Sinkhorn 归一化，将 log_scores [B, N, N] 转为双随机矩阵。
    输入 log_scores[b, i, j] 表示 panel i 分配到位置 j 的未归一化分数。
    """
    log_alpha = log_scores / temp
    for _ in range(n_iters):
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=2, keepdim=True)
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=1, keepdim=True)
    return torch.exp(log_alpha)


class StoryAwareModule(nn.Module):
    """
    轻量级故事感知模块，使用 Sinkhorn 排列学习。
    参数量约 10M，适合小 batch 稳定训练。
    """

    def __init__(
        self,
        hidden_dim: int = 3584,
        num_panels: int = 4,
        reduced_dim: int = 896,
        sinkhorn_iters: int = 20,
        sinkhorn_temp: float = 0.5,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_panels = num_panels
        self.sinkhorn_iters = sinkhorn_iters
        self.sinkhorn_temp = sinkhorn_temp

        # Panel 特征编码器（轻量）
        self.encoder = nn.Sequential(
            nn.Linear(hidden_dim, reduced_dim),
            nn.LayerNorm(reduced_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )

        # 位置嵌入：4个阅读位置的表示
        self.pos_embed = nn.Parameter(torch.randn(num_panels, reduced_dim) * 0.02)

        # Sinkhorn score: panel i -> position j
        self.score_fn = nn.Sequential(
            nn.Linear(reduced_dim * 2, reduced_dim),
            nn.LayerNorm(reduced_dim),
            nn.GELU(),
            nn.Linear(reduced_dim, 1),
        )

        # Pairwise ordering: panel i 是否应在 panel j 之前
        self.pairwise_fn = nn.Sequential(
            nn.Linear(reduced_dim * 2, reduced_dim),
            nn.LayerNorm(reduced_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(reduced_dim, 1),
        )

        # Position classification (auxiliary): panel i 应在哪个位置
        self.position_fn = nn.Sequential(
            nn.Linear(reduced_dim, reduced_dim // 2),
            nn.LayerNorm(reduced_dim // 2),
            nn.GELU(),
            nn.Linear(reduced_dim // 2, num_panels),
        )

        # 兼容性：保留24类分类头（仅用于评估/日志，不参与主损失）
        self.ordering_head = nn.Sequential(
            nn.Linear(reduced_dim * num_panels, reduced_dim),
            nn.LayerNorm(reduced_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(reduced_dim, NUM_PERMS),
        )

    def forward(
        self,
        panel_features: torch.Tensor,
        correct_order: torch.Tensor | None = None,
    ) -> dict:
        """
        Args:
            panel_features: [B, 4, H]
            correct_order: [B, 4] 正确阅读顺序（训练时提供）
        Returns:
            dict with sinkhorn_loss, pairwise_loss, position_loss, permutation_matrix
        """
        B, N, H = panel_features.shape
        device = panel_features.device
        result = {}

        # 编码 panel 特征
        z = self.encoder(panel_features)  # [B, N, R]

        # ---- Sinkhorn 排列学习 ----
        z_i = z.unsqueeze(2).expand(-1, -1, N, -1)           # [B, N, N, R]
        pos_j = self.pos_embed.unsqueeze(0).unsqueeze(0).expand(B, N, -1, -1)  # [B, N, N, R]
        score_input = torch.cat([z_i, pos_j], dim=-1)         # [B, N, N, 2R]
        log_scores = self.score_fn(score_input).squeeze(-1)   # [B, N, N]

        P = sinkhorn_normalize(log_scores, self.sinkhorn_iters, self.sinkhorn_temp)
        result["permutation_matrix"] = P
        result["order_logits"] = self.ordering_head(z.reshape(B, -1))  # 兼容旧接口

        # ---- Pairwise scores ----
        z_i = z.unsqueeze(2).expand(-1, -1, N, -1)
        z_j = z.unsqueeze(1).expand(-1, N, -1, -1)
        pairwise_input = torch.cat([z_i, z_j], dim=-1)
        pairwise_scores = self.pairwise_fn(pairwise_input).squeeze(-1)  # [B, N, N]
        result["pairwise_scores"] = pairwise_scores

        # ---- Position logits ----
        position_logits = self.position_fn(z)  # [B, N, N]
        result["position_logits"] = position_logits

        if correct_order is not None:
            # ---- Sinkhorn Loss ----
            # correct_order[b, pos] = panel_index，即位置 pos 上放的是哪个 panel
            # 我们希望 P[b, panel_index, pos] 接近 1
            # 向量化实现：gather 所有 target probabilities
            batch_idx = torch.arange(B, device=device).unsqueeze(1).expand(B, N)  # [B, N]
            pos_idx = torch.arange(N, device=device).unsqueeze(0).expand(B, N)    # [B, N]
            target_panel = correct_order.to(device)  # [B, N]
            target_probs = P[batch_idx, target_panel, pos_idx]  # [B, N]
            sinkhorn_loss = -torch.log(target_probs + 1e-8).mean()
            result["sinkhorn_loss"] = sinkhorn_loss

            # ---- Pairwise Loss（忽略对角线） ----
            pairwise_target = self._build_pairwise_target(correct_order, device)
            diag_mask = (1.0 - torch.eye(N, device=device)).unsqueeze(0)  # [1, N, N]
            pairwise_loss = F.binary_cross_entropy_with_logits(
                pairwise_scores, pairwise_target, weight=diag_mask, reduction="sum"
            ) / (diag_mask.sum() * B + 1e-8)
            result["pairwise_loss"] = pairwise_loss

            # ---- Position Loss ----
            position_target = self._build_position_target(correct_order, device)
            position_loss = F.cross_entropy(
                position_logits.view(B * N, N),
                position_target.view(B * N),
            )
            result["position_loss"] = position_loss

        return result

    def _build_position_target(self, correct_order: torch.Tensor, device: torch.device) -> torch.Tensor:
        """panel i 在正确顺序中的位置。向量化实现。"""
        B, N = correct_order.shape
        # correct_order[b, pos] = panel_idx
        # 我们需要 target[b, panel_idx] = pos
        batch_idx = torch.arange(B, device=device).unsqueeze(1).expand(B, N)
        pos_idx = torch.arange(N, device=device).unsqueeze(0).expand(B, N)
        target = torch.zeros(B, N, dtype=torch.long, device=device)
        target[batch_idx, correct_order] = pos_idx
        return target

    def _build_pairwise_target(self, correct_order: torch.Tensor, device: torch.device) -> torch.Tensor:
        """pairwise_target[b, i, j] = 1 if panel i should come before panel j。向量化实现。"""
        B, N = correct_order.shape
        # correct_order[b, pos] = panel_idx
        # 先构建 order_pos: order_pos[b, panel_idx] = pos
        batch_idx = torch.arange(B, device=device).unsqueeze(1).expand(B, N)
        pos_idx = torch.arange(N, device=device).unsqueeze(0).expand(B, N)
        order_pos = torch.zeros(B, N, dtype=torch.long, device=device)
        order_pos[batch_idx, correct_order] = pos_idx  # [B, N]

        # 广播比较: order_pos_i < order_pos_j
        pos_i = order_pos.unsqueeze(2)  # [B, N, 1]
        pos_j = order_pos.unsqueeze(1)  # [B, 1, N]
        target = (pos_i < pos_j).float()
        return target
