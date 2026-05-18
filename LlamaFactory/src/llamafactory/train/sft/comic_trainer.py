"""
ComicUnderstandingTrainer v3 -- 四格漫画理解统一训练器

整合改进模块：
1. Multi-Scale Comic Encoding   (数据层，由 JSON 配置驱动)
2. Panel-wise Contrastive Learning v2 (损失层，hard negative mining)
3. Story-Aware Attention v3      (结构层，Sinkhorn 排列学习)
4. Multi-Task Joint Training     (训练层，由数据混合驱动)

核心修复（v3）：
- Fix H: hidden_states 默认 detach()，彻底切断辅助模块到 LoRA 的梯度回传。
  之前 v2 声称 detach 但实际未执行，导致随机初始化的 130M 辅助模块梯度
  严重污染主模型，是所有消融实验（E2-E5）结果劣于 E1 的根因。
- Fix I: 辅助模块学习率降至主模型的 0.05x，避免大参数量辅助模块破坏优化。
- Fix J: 默认辅助权重降至 0.001（原 0.01），进一步降低辅助损失影响。
- Fix K: 适配 StoryAwareModule v3 接口（sinkhorn_loss + position + pairwise），移除训练早期有害的 consistency_loss。
- Fix L: 改进 panel 特征提取逻辑，优先使用 attention-pooled 特征。

继承自 LlamaFactory 的 CustomSeq2SeqTrainer，仅在 compute_loss 中注入额外损失。
不包装模型、不冻结参数，与标准 LoRA 训练完全兼容。
"""

import re
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import torch
from typing_extensions import override

from .trainer import CustomSeq2SeqTrainer

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))
from comic_innovation.panel_extractor import PanelFeatureExtractor
from comic_innovation.contrastive_loss import PanelContrastiveLoss
from comic_innovation.story_attention import StoryAwareModule

if TYPE_CHECKING:
    from transformers import ProcessorMixin
    from ...hparams import FinetuningArguments, ModelArguments


class ComicUnderstandingTrainer(CustomSeq2SeqTrainer):
    def __init__(
        self,
        finetuning_args: "FinetuningArguments",
        processor: Optional["ProcessorMixin"],
        model_args: Optional["ModelArguments"] = None,
        gen_kwargs: Optional[dict[str, Any]] = None,
        ref_model: Optional["torch.nn.Module"] = None,
        comic_config: Optional[dict] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            finetuning_args=finetuning_args,
            processor=processor,
            model_args=model_args,
            gen_kwargs=gen_kwargs,
            ref_model=ref_model,
            **kwargs,
        )

        cfg = comic_config or {}
        hidden_dim = cfg.get("hidden_dim", 3584)
        # v3: 默认权重大幅降低至 0.001，因为 detach 后辅助模块独立学习
        self.cl_weight = cfg.get("contrastive_loss_weight", 0.001)
        self.story_weight = cfg.get("story_loss_weight", 0.001)
        self.use_contrastive = cfg.get("use_contrastive", True)
        self.use_story = cfg.get("use_story", True)
        # 辅助损失 warmup 步数
        self.aux_warmup_steps = cfg.get("aux_warmup_steps", 100)
        # 辅助损失的单步最大绝对值
        self.aux_loss_max = cfg.get("aux_loss_max", 2.0)
        # 辅助模块相对主模型的学习率比例（v3 新增）
        self.aux_lr_ratio = cfg.get("aux_lr_ratio", 0.05)

        if self.use_contrastive or self.use_story:
            self.panel_extractor = PanelFeatureExtractor(hidden_dim).to(self.args.device)

        if self.use_contrastive:
            self.contrastive_loss_fn = PanelContrastiveLoss(
                hidden_dim=hidden_dim,
                temperature=cfg.get("contrastive_temperature", 0.07),
                queue_size=cfg.get("contrastive_queue_size", 128),
            ).to(self.args.device)

        if self.use_story:
            self.story_module = StoryAwareModule(
                hidden_dim=hidden_dim,
                sinkhorn_iters=cfg.get("sinkhorn_iters", 20),
                sinkhorn_temp=cfg.get("sinkhorn_temp", 0.5),
            ).to(self.args.device)

    def _aux_scale(self) -> float:
        """辅助损失的 warmup 缩放因子，[0, 1]。"""
        if self.aux_warmup_steps <= 0:
            return 1.0
        step = self.state.global_step if hasattr(self, "state") else 0
        return min(1.0, step / self.aux_warmup_steps)

    @override
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        input_ids = inputs.get("input_ids")
        labels = inputs.get("labels")

        is_training = model.training
        has_order_labels = self._batch_has_order_labels(labels)
        need_aux = is_training and has_order_labels and (self.use_contrastive or self.use_story)

        if need_aux:
            inputs["output_hidden_states"] = True

        outputs = model(**inputs)
        lm_loss = outputs.loss

        # 评估时直接返回纯 lm_loss，不混入辅助损失
        if not is_training:
            return (lm_loss, outputs) if return_outputs else lm_loss

        total_loss = lm_loss

        if need_aux:
            hidden_states = outputs.hidden_states[-1] if outputs.hidden_states else None
            if hidden_states is not None:
                # Fix H: 默认 detach hidden_states，彻底切断辅助梯度到 LoRA
                # 这是 v3 最关键修复。v2 声称 detach 但实际未执行，导致 E2-E5 全部失败。
                hs_for_aux = hidden_states.detach()

                panel_features, panel_mask = self.panel_extractor(
                    hs_for_aux, input_ids, max_panels=5
                )

                if panel_features.shape[1] > 4:
                    aux_panels = panel_features[:, 1:5, :]
                    aux_mask = panel_mask[:, 1:5]
                else:
                    aux_panels = panel_features[:, :4, :]
                    aux_mask = panel_mask[:, :4]

                task1_indices, correct_orders = self._parse_task1_samples(labels)

                if task1_indices and correct_orders:
                    idx_tensor = torch.tensor(task1_indices, device=panel_features.device)
                    order_tensor = torch.tensor(correct_orders, dtype=torch.long, device=panel_features.device)

                    t1_panel_features = panel_features[idx_tensor]
                    t1_panel_mask = panel_mask[idx_tensor]
                    t1_aux_panels = aux_panels[idx_tensor]

                    min_valid = (panel_mask[idx_tensor])[:, 1:5].sum(dim=1).min().item() if panel_features.shape[1] > 4 \
                        else panel_mask[idx_tensor].sum(dim=1).min().item()

                    if min_valid >= 4:
                        scale = self._aux_scale()

                        if self.use_contrastive and self.cl_weight > 0:
                            cl_loss = self.contrastive_loss_fn(
                                t1_panel_features, order_tensor, t1_panel_mask
                            )
                            cl_loss = torch.clamp(cl_loss, max=self.aux_loss_max)
                            total_loss = total_loss + scale * self.cl_weight * cl_loss

                        if self.use_story and self.story_weight > 0:
                            story_out = self.story_module(t1_aux_panels, order_tensor)
                            # Fix K: 适配 StoryAwareModule v3 接口（移除有问题的 consistency_loss）
                            story_loss = (
                                story_out.get("sinkhorn_loss", 0.0) * 0.5
                                + story_out.get("position_loss", 0.0) * 0.25
                                + story_out.get("pairwise_loss", 0.0) * 0.25
                            )
                            story_loss = torch.clamp(story_loss, max=self.aux_loss_max)
                            total_loss = total_loss + scale * self.story_weight * story_loss

        return (total_loss, outputs) if return_outputs else total_loss

    def _batch_has_order_labels(self, labels: torch.Tensor) -> bool:
        """判断当前 batch 中是否存在 task1 样本（输出为短数字序列）。"""
        if labels is None:
            return False
        for b in range(labels.shape[0]):
            if (labels[b] != -100).sum().item() < 20:
                return True
        return False

    def _parse_task1_samples(
        self,
        labels: torch.Tensor,
    ) -> tuple[list[int], list[list[int]]]:
        """逐样本解析，返回 task1 样本在 batch 中的索引和对应的正确顺序。"""
        tokenizer = self.processing_class
        task1_indices = []
        correct_orders = []

        for b in range(labels.shape[0]):
            valid_ids = labels[b][labels[b] != -100]
            if len(valid_ids) == 0 or len(valid_ids) >= 20:
                continue
            text = tokenizer.decode(valid_ids, skip_special_tokens=True).strip()
            nums = re.findall(r"\d+", text)
            if len(nums) >= 4:
                order = [int(n) for n in nums[:4]]
                if sorted(order) == [0, 1, 2, 3]:
                    task1_indices.append(b)
                    correct_orders.append(order)

        return task1_indices, correct_orders

    @override
    def create_optimizer(self):
        optimizer = super().create_optimizer()
        extra_params = []
        if self.use_contrastive or self.use_story:
            extra_params.extend(self.panel_extractor.parameters())
        if self.use_contrastive:
            extra_params.extend(self.contrastive_loss_fn.parameters())
        if self.use_story:
            extra_params.extend(self.story_module.parameters())

        if extra_params:
            # Fix I: 辅助模块使用更低的学习率（默认 5% 主学习率）
            aux_lr = self.args.learning_rate * self.aux_lr_ratio
            optimizer.add_param_group(
                {
                    "params": extra_params,
                    "lr": aux_lr,
                    "weight_decay": self.args.weight_decay,
                }
            )
        return optimizer
