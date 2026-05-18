"""
Panel-wise Contrastive Learning v2

核心改进：
1. 正样本扩展：不仅相邻 panel，还包含叙事距离 <= 2 的 panel 对（起-转、承-合）
2. Hard Negative Mining：只保留最困难的负样本，提升学习效率
3. 温度参数下限提高到 0.03，避免数值不稳定
4. 负样本队列改为 FIFO ring buffer，更新更高效
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PanelContrastiveLoss(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 3584,
        proj_dim: int = 128,
        temperature: float = 0.07,
        queue_size: int = 128,
        max_negatives: int = 32,
    ):
        super().__init__()
        # 可学习温度，log 空间中初始化
        self.log_temperature = nn.Parameter(torch.tensor(temperature).log())
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, proj_dim),
        )
        # 负样本队列
        self.queue_size = queue_size
        self.max_negatives = max_negatives
        self.register_buffer("queue", torch.zeros(queue_size, proj_dim))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self._queue_initialized = False

    @property
    def temperature(self) -> torch.Tensor:
        # 下限 0.03，避免温度过小导致梯度爆炸
        return self.log_temperature.exp().clamp(min=0.03, max=1.0)

    def _enqueue(self, z: torch.Tensor) -> None:
        """将当前 batch 的投影向量推入循环队列。z: [N, proj_dim]"""
        n = z.shape[0]
        ptr = int(self.queue_ptr)
        end = ptr + n
        if end <= self.queue_size:
            self.queue[ptr:end] = z.detach()
        else:
            tail = self.queue_size - ptr
            self.queue[ptr:] = z.detach()[:tail]
            self.queue[: end - self.queue_size] = z.detach()[tail:]
        self.queue_ptr[0] = end % self.queue_size
        self._queue_initialized = True

    def forward(
        self,
        panel_features: torch.Tensor,
        correct_order: torch.Tensor,
        panel_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            panel_features: [B, N, H]
            correct_order:  [B, 4]
            panel_mask:     [B, N]
        Returns:
            contrastive loss scalar
        """
        B, N, H = panel_features.shape
        device = panel_features.device

        if N > 4:
            panel_features = panel_features[:, 1:5, :]
            panel_mask = panel_mask[:, 1:5]
            N = 4
        elif N < 4:
            return torch.tensor(0.0, device=device, requires_grad=True)

        z = self.projection(panel_features)  # [B, 4, proj_dim]
        z = F.normalize(z, dim=-1)

        # 收集所有有效 panel 向量，用于更新队列
        all_z_list = []
        for b in range(B):
            if panel_mask[b].sum().item() >= 4:
                all_z_list.append(z[b])  # [4, proj_dim]

        total_loss = torch.tensor(0.0, device=device)
        n_valid = 0
        temp = self.temperature

        for b in range(B):
            if panel_mask[b].sum().item() < 4:
                continue

            order = correct_order[b]  # [4]
            sorted_z = z[b][order]  # 按正确阅读顺序排列

            # 正样本对：叙事距离 <= 2 的 panel 对
            pos_pairs = []
            for i in range(4):
                for j in range(i + 1, 4):
                    if j - i <= 2:  # 距离1或2：相邻或隔一个
                        pos_pairs.append((i, j))

            for i, j in pos_pairs:
                anchor = sorted_z[i]
                positive = sorted_z[j]

                # 收集负样本
                negatives = []
                # 同样本内：非 anchor、非 positive
                for k in range(4):
                    if k != i and k != j:
                        negatives.append(sorted_z[k])
                # 跨样本
                for b2 in range(B):
                    if b2 != b and panel_mask[b2].sum().item() >= 4:
                        negatives.extend(z[b2])
                # 历史队列
                if self._queue_initialized:
                    negatives.extend(self.queue)

                if not negatives:
                    continue

                neg_stack = torch.stack(negatives, dim=0)  # [K, proj_dim]

                # Hard Negative Mining：只保留相似度最高的 top-k 负样本
                neg_sims_full = neg_stack @ anchor  # [K]
                k = min(self.max_negatives, len(neg_sims_full))
                if k < len(neg_sims_full):
                    _, top_idx = torch.topk(neg_sims_full, k)
                    neg_stack = neg_stack[top_idx]

                # InfoNCE
                pos_sim = torch.dot(anchor, positive) / temp
                neg_sims = (neg_stack @ anchor) / temp  # [K]

                logits = torch.cat([pos_sim.unsqueeze(0), neg_sims], dim=0)  # [K+1]
                target = torch.zeros(1, dtype=torch.long, device=device)
                total_loss = total_loss + F.cross_entropy(logits.unsqueeze(0), target)
                n_valid += 1

        if n_valid > 0:
            total_loss = total_loss / n_valid

        # 更新队列
        if all_z_list:
            all_z = torch.cat(all_z_list, dim=0)  # [B*4, proj_dim]
            self._enqueue(all_z)

        return total_loss
