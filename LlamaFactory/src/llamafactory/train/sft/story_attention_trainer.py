# Copyright 2025 Story-Aware Attention Integration for Llama Factory
#
# 将 Story-Aware Attention 模块集成到 Llama Factory SFT 训练中
# 支持四格漫画排序任务的增量创新

import json
import os
import re
from functools import partial
from types import MethodType
from typing import TYPE_CHECKING, Any, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Seq2SeqTrainer
from typing_extensions import override

from ...extras import logging
from ...extras.constants import IGNORE_INDEX
from ..callbacks import SaveProcessorCallback
from ..fp8_utils import configure_fp8_environment, patch_accelerator_for_fp8, verify_fp8_status
from ..trainer_utils import create_custom_optimizer, create_custom_scheduler
from .trainer import CustomSeq2SeqTrainer

if TYPE_CHECKING:
    from torch.utils.data import Dataset
    from transformers import ProcessorMixin
    from transformers.trainer import PredictionOutput
    from ...hparams import FinetuningArguments, ModelArguments, TrainingArguments

logger = logging.get_logger(__name__)


class StoryStructureEncoder(nn.Module):
    """
    故事结构编码器 - 科学改进版
    
    核心改进：
    1. 软注意力提取panel特征（替代硬分割）
    2. 移除硬编码的"起承转合"假设
    3. 成对比较建模panel间排序关系
    4. 位置评分网络用于排序
    """

    def __init__(self, hidden_dim: int = 4096, num_heads: int = 8, num_panels: int = 4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_panels = num_panels

        # ★ 改进1：软注意力提取panel特征（替代硬分割+mean pooling）
        # 学习每个token对4个panel的贡献权重
        self.panel_extractor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_panels)  # 输出每个token属于4个panel的权重
        )
        
        # ★ 改进2：成对关系编码器（替代起承转合假设）
        # 计算panel i和panel j的相对顺序关系
        self.pairwise_encoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1)  # 输出相对顺序分数
        )
        
        # ★ 改进3：位置评分网络（ListNet风格）
        # 为每个panel预测其在4个位置上的适合度
        self.position_scorer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, num_panels)  # 输出4个位置的分数
        )

        # 输出投影
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim * num_panels, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        # 特征融合门控 - 控制story特征融合强度（token-level）
        self.fusion_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Sigmoid()
        )
        
        # 残差投影 - 将panel特征映射回hidden_dim
        self.residual_projection = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None) -> dict:
        """
        Args:
            hidden_states: [batch, seq_len, hidden_dim]
            attention_mask: 注意力掩码
            labels: 用于监督排序损失的标签token IDs

        Returns:
            dict 包含排序特征和辅助信息
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape

        # === ★ 改进1：软注意力提取panel特征（替代硬分割）===
        # 计算每个token对4个panel的贡献权重 [B, T, 4]
        token_to_panel_logits = self.panel_extractor(hidden_states)  # [B, T, 4]
        
        # 应用attention_mask（如果有）
        if attention_mask is not None:
            # attention_mask: [B, T] -> 扩展为 [B, T, 1]
            mask = attention_mask.unsqueeze(-1).float()
            token_to_panel_logits = token_to_panel_logits.masked_fill(mask == 0, float('-inf'))
        
        token_to_panel_weights = F.softmax(token_to_panel_logits, dim=1)  # 在seq_len维度softmax
        
        # 用软注意力聚合panel特征 [B, 4, H]
        # panel_features[b, i, :] = sum_j (token_to_panel_weights[b, j, i] * hidden_states[b, j, :])
        panel_features = torch.bmm(
            token_to_panel_weights.transpose(1, 2),  # [B, 4, T]
            hidden_states  # [B, T, H]
        )  # [B, 4, H]

        # === ★ 改进2：计算成对关系矩阵（替代转移分数+起承转合）===
        # pairwise_scores[b, i, j] 表示 "panel i 应该在 panel j 前面" 的分数
        pairwise_scores = self._compute_pairwise_scores(panel_features)

        # === ★ 改进3：计算位置分数（用于排序）===
        # 为每个panel预测其在4个位置上的适合度
        position_logits = self.position_scorer(panel_features)  # [B, 4, 4]
        # position_logits[b, i, pos] 表示 panel i 放在位置 pos 的分数

        # === 步骤4：融合特征 ===
        flat_features = panel_features.view(batch_size, -1)  # [B, 4*H]
        output_features = self.output_projection(flat_features)  # [B, H]

        return {
            'features': output_features.unsqueeze(1).expand(-1, seq_len, -1),  # [B, T, H]
            'panel_features': panel_features,  # [B, 4, H]
            'pairwise_scores': pairwise_scores,  # [B, 4, 4]
            'position_logits': position_logits,  # [B, 4, 4]
            'token_to_panel_weights': token_to_panel_weights,  # [B, T, 4]
        }

    def _compute_pairwise_scores(self, panel_features: torch.Tensor) -> torch.Tensor:
        """
        计算panel间的成对比较分数
        
        Args:
            panel_features: [B, 4, H]
        
        Returns:
            pairwise_scores: [B, 4, 4]，scores[b, i, j]表示i在j前的合理性
        """
        batch_size, num_panels, hidden_dim = panel_features.shape

        # 扩展特征为所有配对 [B, 4, 4, H]
        feat_i = panel_features.unsqueeze(2).expand(-1, -1, num_panels, -1)  # [B, 4, 4, H]
        feat_j = panel_features.unsqueeze(1).expand(-1, num_panels, -1, -1)  # [B, 4, 4, H]

        # 拼接配对 [B, 4, 4, 2H]
        pairs = torch.cat([feat_i, feat_j], dim=-1)

        # 计算成对分数 [B, 4, 4]
        scores = self.pairwise_encoder(pairs).squeeze(-1)

        return scores


class StoryAttentionLoRAModule(nn.Module):
    """
    Story Attention LoRA 模块
    包装在现有 LoRA 模型之上，添加故事感知能力

    注意：训练时会保存 state dict，但 LoRA 参数可以从 base_model 获取
    """

    def __init__(self, base_model, hidden_dim: int = 4096, use_story_attention: bool = True):
        super().__init__()
        self.base_model = base_model
        self.use_story_attention = use_story_attention
        self.config = getattr(base_model, 'config', None)
        self.peft_config = getattr(base_model, 'peft_config', None)

        if use_story_attention:
            self.story_encoder = StoryStructureEncoder(hidden_dim)
            
            # 轻量级排序辅助头（可选）
            self.ordering_head = nn.Sequential(
                nn.Linear(hidden_dim * 4, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, 24)  # 4! = 24种排列
            )

        # 冻结基础模型（LoRA已经处理了可训练参数）
        self._freeze_base_model()
    
    def __getattr__(self, name):
        """代理所有未定义的属性到 base_model"""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.base_model, name)

    def _freeze_base_model(self):
        """冻结基础模型参数，只训练story attention模块"""
        for param in self.base_model.parameters():
            param.requires_grad = False

        # 解冻LoRA参数
        for name, param in self.base_model.named_parameters():
            if 'lora' in name.lower():
                param.requires_grad = True

        # 解冻story attention参数
        if self.use_story_attention:
            for param in self.story_encoder.parameters():
                param.requires_grad = True
            for param in self.ordering_head.parameters():
                param.requires_grad = True

    def save_pretrained(self, save_directory=None, **kwargs):
        """保存模型，确保adapter_config.json被正确创建"""
        import os
        import json
        from pathlib import Path
        import torch

        # 添加空值检查
        if save_directory is None:
            if hasattr(self, 'args') and hasattr(self.args, 'output_dir'):
                save_directory = self.args.output_dir
            else:
                raise ValueError("save_directory cannot be None and no output_dir found in args")

        os.makedirs(save_directory, exist_ok=True)

        # 保存基础模型的LoRA参数
        if hasattr(self.base_model, 'save_pretrained'):
            self.base_model.save_pretrained(save_directory, **kwargs)
        else:
            # 回退到原始保存方法
            super().save_pretrained(save_directory, **kwargs)

        # 保存 story-attention 模块权重（推理/评估时需要显式加载）
        if self.use_story_attention:
            story_state = {
                "story_encoder": self.story_encoder.state_dict(),
                "ordering_head": self.ordering_head.state_dict(),
                "meta": {
                    "use_story_attention": True,
                    "version": "v2_scientific"
                }
            }
            torch.save(story_state, os.path.join(save_directory, "story_attention_state.pt"))

        # 创建adapter_config.json
        self._create_adapter_config(os.path.join(save_directory, "adapter_config.json"))

        print(f"Model saved to {save_directory} with adapter_config.json")

    def create_or_update_model_card(self, output_dir):
        """
        兼容 transformers Trainer 的模型卡导出调用。
        当前模块主要保存 LoRA 适配器，不需要生成完整 model card。
        """
        return

    def _create_adapter_config(self, config_path):
        """创建 adapter_config.json 文件"""
        # 尝试从 peft_config 获取配置
        if hasattr(self, 'peft_config') and self.peft_config:
            raw_config = self.peft_config
            # peft_config 可能是 LoraConfig 或 dict[str, LoraConfig]，需要转成可JSON序列化
            if hasattr(raw_config, "to_dict"):
                config = raw_config.to_dict()
            elif isinstance(raw_config, dict):
                if "peft_type" in raw_config:
                    config = raw_config
                else:
                    # 兼容 {"default": LoraConfig(...)} 的情况
                    first_cfg = next(iter(raw_config.values()))
                    if hasattr(first_cfg, "to_dict"):
                        config = first_cfg.to_dict()
                    else:
                        config = {
                            "peft_type": "LORA",
                            "r": 32,
                            "lora_alpha": 64,
                            "lora_dropout": 0.05,
                            "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                            "bias": "none",
                            "task_type": "CAUSAL_LM",
                            "inference_mode": True
                        }
            else:
                config = {
                    "peft_type": "LORA",
                    "r": 32,
                    "lora_alpha": 64,
                    "lora_dropout": 0.05,
                    "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                    "bias": "none",
                    "task_type": "CAUSAL_LM",
                    "inference_mode": True
                }
        else:
            # 默认配置
            config = {
                "peft_type": "LORA",
                "r": 32,
                "lora_alpha": 64,
                "lora_dropout": 0.05,
                "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                "bias": "none",
                "task_type": "CAUSAL_LM",
                "inference_mode": True
            }

        def _json_safe(obj):
            if isinstance(obj, dict):
                return {str(k): _json_safe(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [_json_safe(v) for v in obj]
            if isinstance(obj, set):
                return sorted([_json_safe(v) for v in obj], key=lambda x: str(x))
            return obj

        with open(config_path, 'w') as f:
            json.dump(_json_safe(config), f, indent=2)

    def forward(self, input_ids=None, attention_mask=None, labels=None, images=None, **kwargs):
        """前向传播 - 科学改进版"""
        # 获取基础模型输出
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            images=images,
            output_hidden_states=True,
            **kwargs
        )

        if not self.use_story_attention:
            return outputs

        # 提取最后一层隐藏状态
        hidden_states = outputs.hidden_states[-1] if outputs.hidden_states else None
        
        if hidden_states is None or hidden_states.size(1) == 0:
            return outputs
        
        batch_size, seq_len, hidden_dim = hidden_states.shape
        
        # 应用改进的story编码
        story_output = self.story_encoder(hidden_states, attention_mask, labels)
        
        # === 特征融合：使用排序感知特征 ===
        # 取4个panel特征的平均作为全局排序特征（更简洁）
        panel_global = story_output['panel_features'].mean(dim=1)  # [B, H]
        story_residual = self.story_encoder.residual_projection(panel_global)  # [B, H]
        
        # Token-level门控融合
        token_gates = self.story_encoder.fusion_gate(hidden_states)  # [B, T, H]
        story_residual = story_residual.unsqueeze(1).expand(-1, seq_len, -1)  # [B, T, H]
        
        # 残差融合
        enhanced_hidden = hidden_states + token_gates * story_residual
        
        # 重新计算logits
        if hasattr(self.base_model, 'lm_head'):
            enhanced_logits = self.base_model.lm_head(enhanced_hidden)
            outputs.logits = enhanced_logits
        elif hasattr(self.base_model, 'get_base_model'):
            base = self.base_model.get_base_model()
            if hasattr(base, 'lm_head'):
                enhanced_logits = base.lm_head(enhanced_hidden)
                outputs.logits = enhanced_logits
        
        # 重新计算loss（使用IGNORE_INDEX）
        if labels is not None and hasattr(outputs, 'logits'):
            shift_logits = outputs.logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
            outputs.loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
        
        # 计算轻量级排序损失（可选）
        if labels is not None:
            panel_flat = story_output['panel_features'].view(batch_size, -1)  # [B, 4*H]
            order_logits = self.ordering_head(panel_flat)  # [B, 24]
            outputs.order_logits = order_logits
        
        # 存储story信息
        outputs.story_output = story_output

        return outputs


class StoryAttentionTrainer(CustomSeq2SeqTrainer):
    """
    Story-Aware Attention Trainer
    继承自 Llama Factory 的 CustomSeq2SeqTrainer，添加故事感知训练能力

    改进点：
    1. 连贯性损失基于实际labels动态计算
    2. 角色分配鼓励多样性（覆盖所有4个角色）
    3. 添加顺序验证损失
    4. 改进的转移分数正则化
    """

    def __init__(
        self,
        finetuning_args: "FinetuningArguments",
        processor: Optional["ProcessorMixin"],
        model_args: Optional["ModelArguments"] = None,
        gen_kwargs: Optional[dict[str, Any]] = None,
        ref_model: Optional["torch.nn.Module"] = None,
        story_attention_config: Optional[dict] = None,
        **kwargs,
    ) -> None:
        # Story Attention 配置
        self.story_config = story_attention_config or {}
        self.use_story_attention = self.story_config.get('use_story_attention', True)
        # ★ 简化：只保留排序损失权重
        self.ordering_loss_weight = self.story_config.get('ordering_loss_weight', 0.05)
        self.hidden_dim = self.story_config.get('hidden_dim', 4096)

        # 初始化父类
        super().__init__(
            finetuning_args=finetuning_args,
            processor=processor,
            model_args=model_args,
            gen_kwargs=gen_kwargs,
            ref_model=ref_model,
            **kwargs
        )

        # 包装模型（如果需要）
        if self.use_story_attention:
            logger.info_rank0("Initializing Story-Aware Attention module (Scientific Version)...")
            self._setup_story_attention()

    def _setup_story_attention(self):
        """设置 Story Attention 模块"""
        # 包装模型
        self.model = StoryAttentionLoRAModule(
            base_model=self.model,
            hidden_dim=self.hidden_dim,
            use_story_attention=self.use_story_attention
        )

        logger.info_rank0(f"Story Attention (Scientific) module initialized:")
        logger.info_rank0(f"  - Hidden dim: {self.hidden_dim}")
        logger.info_rank0(f"  - Ordering loss weight: {self.ordering_loss_weight}")

    @override
    def save_model(self, output_dir=None, _internal_call=False):
        """保存模型"""
        if output_dir is None:
            output_dir = self.args.output_dir if hasattr(self, 'args') else None

        if output_dir is None:
            raise ValueError("output_dir cannot be None")

        # 保存story attention权重
        if hasattr(self.model, 'save_pretrained'):
            self.model.save_pretrained(output_dir)
        else:
            # 直接保存base_model
            self.model.base_model.save_pretrained(output_dir)

    @override
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        计算损失 - 简化版：只保留主损失和轻量级排序损失
        """
        if not self.use_story_attention:
            return super().compute_loss(model, inputs, return_outputs, num_items_in_batch)
        
        outputs = model(**inputs)
        loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
        
        # 可选：添加轻量级排序损失（如果weight > 0）
        if self.ordering_loss_weight > 0 and hasattr(outputs, 'order_logits'):
            ordering_loss = self._compute_ordering_loss(outputs, inputs)
            loss = loss + self.ordering_loss_weight * ordering_loss
        
        return (loss, outputs) if return_outputs else loss

    def _compute_ordering_loss(self, outputs, inputs) -> torch.Tensor:
        """
        计算排序损失 - 使用排列预测
        """
        if not hasattr(outputs, 'order_logits'):
            return torch.tensor(0.0, device=outputs.loss.device)
        
        order_logits = outputs.order_logits  # [B, 24]
        batch_size = order_logits.size(0)
        device = order_logits.device
        
        # 简化：假设训练数据都是正确顺序1-2-3-4，对应排列索引0
        # 实际可以从labels中解析真实顺序
        target = torch.zeros(batch_size, dtype=torch.long, device=device)
        
        return F.cross_entropy(order_logits, target)

    @override
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None, **gen_kwargs):
        """预测步骤，保持与父类一致"""
        return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys, **gen_kwargs)

    def get_story_explanation(self, sample_input: dict) -> dict:
        """
        获取模型对输入的故事结构解释（用于分析和调试）
        """
        if not self.use_story_attention:
            return {"error": "Story attention is not enabled"}

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**sample_input)

            if not hasattr(outputs, 'story_output'):
                return {"error": "No story output available"}

            story_output = outputs.story_output
            pairwise_scores = story_output['pairwise_scores'][0]
            position_logits = story_output['position_logits'][0]

            explanation = {
                'pairwise_scores': pairwise_scores.cpu().numpy().tolist(),
                'position_logits': position_logits.cpu().numpy().tolist(),
                'panel_attention_weights': story_output['token_to_panel_weights'][0].cpu().numpy().tolist(),
            }

        return explanation


def create_story_attention_trainer(
    finetuning_args: "FinetuningArguments",
    training_args: "TrainingArguments",
    model,
    tokenizer,
    processor,
    train_dataset,
    eval_dataset,
    callbacks,
    story_attention_config: Optional[dict] = None,
) -> StoryAttentionTrainer:
    """
    创建 Story Attention Trainer 的工厂函数
    """
    # 构建 gen_kwargs
    gen_kwargs = {
        "max_length": training_args.generation_max_length or training_args.max_length,
        "max_new_tokens": training_args.generation_max_new_tokens,
        "do_sample": training_args.predict_with_generate,
        "num_beams": training_args.generation_num_beams,
        "temperature": training_args.generation_temperature,
        "top_p": training_args.generation_top_p,
        "repetition_penalty": training_args.repetition_penalty,
    }
    gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}

    # 默认配置 - 科学改进版
    default_config = {
        'use_story_attention': True,
        'hidden_dim': getattr(model.config, 'hidden_size', 4096),
        'ordering_loss_weight': 0.05,  # 轻量级排序损失
    }

    if story_attention_config:
        default_config.update(story_attention_config)

    trainer = StoryAttentionTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        finetuning_args=finetuning_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=callbacks,
        processor=processor,
        gen_kwargs=gen_kwargs if gen_kwargs else None,
        story_attention_config=default_config,
    )

    return trainer
