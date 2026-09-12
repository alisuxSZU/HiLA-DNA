"""Caduceus + HiLA model for DNA sequence classification.

中文说明：
支持 Stage 5 消融实验的完整 HiLA 模型，包含：
  - 结构开关：use_local_latent, use_global_latent 控制 A1/A2
  - mask reconstruction 辅助损失 (A3)
  - RC consistency 辅助损失 (A4)
  - Full HiLA = A2 + mask rec + RC consistency (A5)

消融变体：
  A1: use_local_latent=True,  use_global_latent=False
      → classifier_input_dim = hidden_dim + latent_dim
  A2: use_local_latent=True,  use_global_latent=True  (HiLA-Core)
      → classifier_input_dim = hidden_dim + latent_dim + latent_dim
  A3: A2 + mask_reconstruction
  A4: A2 + rc_consistency
  A5: A2 + mask_reconstruction + rc_consistency (Full HiLA)
"""

from typing import Dict, List, Optional, Sequence

import torch
from torch import nn

from src.models.caduceus_wrapper import (
    _get_hidden_size,
    resolve_caduceus_model_name,
)
from src.models.hila_adapter import HiLAAdapter


class CaduceusHiLA(nn.Module):
    """Caduceus backbone with HiLA adapter and classification head.

    中文说明：
    组合 Caduceus backbone（不含 baseline 分类头）和 HiLA adapter，
    用分层 pooling + global latent 增强序列表示后进行分类。
    支持 mask reconstruction 和 RC consistency 辅助损失。
    """

    def __init__(
        self,
        backbone: str = "caduceus_ps_1k",
        num_labels: int = 2,
        max_length: int = 1024,
        block_size: int = 50,
        latent_dim: int = 128,
        dropout: float = 0.1,
        use_local_latent: bool = True,
        use_global_latent: bool = True,
        global_pool_type: str = "mean_mlp",
        global_num_heads: int = 4,
        global_attention_dropout: float = 0.1,
        global_statistics: Sequence[str] = ("std", "transition_rms"),
        global_bottleneck_dim: int = 64,
        global_statistics_eps: float = 1e-6,
        global_num_blocks: Optional[int] = None,
        global_region_rank: int = 4,
        global_channel_rank: int = 32,
        global_residual_bottleneck_dim: int = 64,
        use_mask_reconstruction: bool = False,
        mask_ratio: float = 0.15,
        reconstruction_loss_weight: float = 0.05,
        use_rc_consistency: bool = False,
        rc_loss_weight: float = 0.05,
    ):
        """
        Args:
            backbone: Caduceus 模型短名称或 HuggingFace 模型 ID。
            num_labels: 分类类别数（二分类为 2）。
            max_length: 最大序列长度。
            block_size: HiLA 每个 block 的 token 数。
            latent_dim: HiLA local/global latent 维度。
            dropout: 分类头和 HiLA MLP 中的 dropout 概率。
            use_local_latent: 是否启用 block pooling + local latent。
            use_global_latent: 是否启用 global latent。
            global_pool_type: ``mean_mlp``（旧版）、``attention``（Stage 8）
                              或 ``statistical``（Stage 9）。
            global_num_heads: attention global pooling 的注意力头数。
            global_attention_dropout: attention 权重 dropout。
            global_statistics: Stage 9 固定统计量列表。
            global_bottleneck_dim: Stage 9 global MLP bottleneck 维度。
            global_statistics_eps: Stage 9 统计量数值稳定项。
            global_num_blocks: Stage 11 固定区域数 Kmax。
            global_region_rank: Stage 11 区域轴低秩维度。
            global_channel_rank: Stage 11 通道轴低秩维度。
            global_residual_bottleneck_dim: Stage 12 残差 MLP bottleneck。
            use_mask_reconstruction: 是否启用 mask reconstruction 辅助损失。
            mask_ratio: mask reconstruction 中被 mask 的 block 比例。
            reconstruction_loss_weight: reconstruction loss 权重。
            use_rc_consistency: 是否启用 reverse-complement consistency 辅助损失。
            rc_loss_weight: RC consistency loss 权重。
        """
        super().__init__()

        from transformers import AutoModel, AutoTokenizer

        model_name = resolve_caduceus_model_name(backbone)
        self.max_length = max_length
        self.num_labels = num_labels

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True,
        )
        self.backbone = AutoModel.from_pretrained(
            model_name, trust_remote_code=True,
        )

        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
                self.backbone.resize_token_embeddings(len(self.tokenizer))

        hidden_dim = _get_hidden_size(self.backbone.config)

        # HiLA adapter（带消融开关）
        self.hila = HiLAAdapter(
            hidden_dim=hidden_dim,
            block_size=block_size,
            latent_dim=latent_dim,
            dropout=dropout,
            use_local_latent=use_local_latent,
            use_global_latent=use_global_latent,
            global_pool_type=global_pool_type,
            global_num_heads=global_num_heads,
            global_attention_dropout=global_attention_dropout,
            global_statistics=global_statistics,
            global_bottleneck_dim=global_bottleneck_dim,
            global_statistics_eps=global_statistics_eps,
            global_num_blocks=global_num_blocks,
            global_region_rank=global_region_rank,
            global_channel_rank=global_channel_rank,
            global_residual_bottleneck_dim=global_residual_bottleneck_dim,
        )

        # 动态 classifier_input_dim
        classifier_input_dim = self.hila.output_dim
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_labels),
        )

        self.loss_fn = nn.CrossEntropyLoss()

        # mask reconstruction 设置
        self.use_mask_reconstruction = use_mask_reconstruction
        self.mask_ratio = mask_ratio
        self.reconstruction_loss_weight = reconstruction_loss_weight

        # mask reconstruction 预测头: latent_dim → hidden_dim
        if use_mask_reconstruction:
            self.reconstruction_head = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim),
            )

        # RC consistency 设置
        self.use_rc_consistency = use_rc_consistency
        self.rc_loss_weight = rc_loss_weight

    def _extract_hidden_states(self, outputs) -> torch.Tensor:
        """Extract token hidden states from model outputs."""
        if hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
            return outputs.last_hidden_state
        if hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
            return outputs.hidden_states[-1]
        if isinstance(outputs, (tuple, list)) and outputs:
            return outputs[0]
        raise ValueError("Could not extract hidden states from model outputs.")

    def _mean_pool(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor]) -> torch.Tensor:
        """Mask-aware mean pooling."""
        if attention_mask is None:
            return hidden_states.mean(dim=1)
        mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
        summed = (hidden_states * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1.0)
        return summed / counts

    def _reverse_complement(self, sequences: List[str]) -> List[str]:
        """Generate reverse-complement of DNA sequences.

        中文说明：
        对 DNA 序列生成反向互补序列。
        标准碱基互补：A↔T, C↔G。N 保持不变。
        """
        complement = {"A": "T", "T": "A", "C": "G", "G": "C", "N": "N",
                       "a": "t", "t": "a", "c": "g", "g": "c", "n": "n"}
        rc_seqs = []
        for seq in sequences:
            rc = "".join(complement.get(base, "N") for base in reversed(seq))
            rc_seqs.append(rc)
        return rc_seqs

    def _compute_reconstruction_loss(
        self,
        block_pooled: torch.Tensor,
        local_latents: torch.Tensor,
    ) -> torch.Tensor:
        """Compute mask reconstruction auxiliary loss.

        中文说明：
        对 block_pooled 随机 mask 一部分 block，
        用未 mask 的 local_latents 聚合后预测被 mask 的 block。
        target 使用 stop_gradient(block_pooled)。
        loss 使用 MSE。

        Args:
            block_pooled: [B, num_blocks, D] 原始 block pooling 结果。
            local_latents: [B, num_blocks, latent_dim] local latent 表示。

        Returns:
            reconstruction_loss: scalar
        """
        B, num_blocks, D = block_pooled.shape

        # 生成 mask：每个 block 有 mask_ratio 的概率被 mask
        num_masked = max(1, int(num_blocks * self.mask_ratio))
        # 随机选择要 mask 的 block 索引
        mask_indices = torch.randperm(num_blocks)[:num_masked]

        # target: stop_gradient(block_pooled)
        target = block_pooled[:, mask_indices, :].detach()  # [B, num_masked, D]

        # 用未 mask 的 local_latents 均值预测被 mask 的 block
        # unmask indices
        unmask_indices = torch.tensor([i for i in range(num_blocks) if i not in mask_indices.tolist()],
                                       device=block_pooled.device)
        if len(unmask_indices) == 0:
            # 如果所有 block 都被 mask（极端情况），用全部 local_latents
            context = local_latents.mean(dim=1)  # [B, latent_dim]
        else:
            context = local_latents[:, unmask_indices, :].mean(dim=1)  # [B, latent_dim]

        # 预测被 mask 的 block
        pred = self.reconstruction_head(context)  # [B, D]
        pred = pred.unsqueeze(1).expand(-1, num_masked, -1)  # [B, num_masked, D]

        # MSE loss
        loss = nn.functional.mse_loss(pred, target)
        return loss

    def _compute_rc_consistency_loss(
        self,
        sequences: List[str],
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute reverse-complement consistency auxiliary loss.

        中文说明：
        对原序列和反向互补序列分别 forward，
        计算两者概率分布的 MSE 一致性损失。
        只在训练时启用。

        Args:
            sequences: 原始 DNA 序列列表。
            labels: 可选标签（不使用，但保持接口一致）。

        Returns:
            rc_consistency_loss: scalar
        """
        device = next(self.parameters()).device

        # 生成 RC 序列
        rc_sequences = self._reverse_complement(sequences)

        # 原序列 forward（不递归调用，直接走 backbone + adapter）
        normalized = [s.upper() for s in sequences]
        batch = self.tokenizer(
            normalized,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        batch = {k: v.to(device) for k, v in batch.items()}
        attention_mask = batch.get("attention_mask")

        with torch.no_grad():
            outputs = self.backbone(**batch, output_hidden_states=True)
            hidden_states = self._extract_hidden_states(outputs)
            backbone_pooled = self._mean_pool(hidden_states, attention_mask)

        adapter_out = self.hila(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            backbone_pooled=backbone_pooled,
        )
        logits_orig = self.classifier(adapter_out["adapter_output"])
        probs_orig = torch.softmax(logits_orig.float(), dim=-1)

        # RC 序列 forward
        normalized_rc = [s.upper() for s in rc_sequences]
        batch_rc = self.tokenizer(
            normalized_rc,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        batch_rc = {k: v.to(device) for k, v in batch_rc.items()}
        attention_mask_rc = batch_rc.get("attention_mask")

        outputs_rc = self.backbone(**batch_rc, output_hidden_states=True)
        hidden_states_rc = self._extract_hidden_states(outputs_rc)
        backbone_pooled_rc = self._mean_pool(hidden_states_rc, attention_mask_rc)

        adapter_out_rc = self.hila(
            hidden_states=hidden_states_rc,
            attention_mask=attention_mask_rc,
            backbone_pooled=backbone_pooled_rc,
        )
        logits_rc = self.classifier(adapter_out_rc["adapter_output"])
        probs_rc = torch.softmax(logits_rc.float(), dim=-1)

        # MSE consistency loss
        loss = nn.functional.mse_loss(probs_orig.detach(), probs_rc)
        return loss

    def forward(
        self,
        sequences: List[str],
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass.

        中文说明：
        输入 DNA 序列列表，通过 Caduceus + HiLA 得到分类结果。
        如果启用辅助损失且在训练模式，会附加 reconstruction_loss 和/或 rc_consistency_loss。

        Args:
            sequences: DNA 序列字符串列表。
            labels: 可选，标签张量 [batch_size]。

        Returns:
            dict with keys:
                logits:         [B, num_labels]
                probabilities:  [B, num_labels] (softmax)
                loss:           scalar (仅当 labels 不为 None)
                reconstruction_loss: scalar (仅当 use_mask_reconstruction 且 training)
                rc_consistency_loss: scalar (仅当 use_rc_consistency 且 training)
        """
        device = next(self.parameters()).device

        # tokenize
        normalized = [s.upper() for s in sequences]
        batch = self.tokenizer(
            normalized,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        batch = {k: v.to(device) for k, v in batch.items()}
        attention_mask = batch.get("attention_mask")

        # backbone forward
        outputs = self.backbone(**batch, output_hidden_states=True)
        hidden_states = self._extract_hidden_states(outputs)

        # backbone pooled
        backbone_pooled = self._mean_pool(hidden_states, attention_mask)

        # HiLA adapter
        adapter_out = self.hila(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            backbone_pooled=backbone_pooled,
        )

        # classify
        logits = self.classifier(adapter_out["adapter_output"])
        probabilities = torch.softmax(logits, dim=-1)

        result = {
            "logits": logits,
            "probabilities": probabilities,
        }
        for key in (
            "global_latent",
            "global_base_latent",
            "global_residual_delta",
            "global_attention_weights",
            "global_std",
            "global_transition_rms",
            "global_channel_factors",
            "global_region_factors",
            "block_valid_mask",
        ):
            if key in adapter_out:
                result[key] = adapter_out[key]

        if labels is not None:
            # 主分类损失
            classification_loss = self.loss_fn(logits, labels)

            total_loss = classification_loss

            # mask reconstruction 辅助损失（仅训练时）
            if self.use_mask_reconstruction and self.training:
                if "block_pooled" in adapter_out and "local_latents" in adapter_out:
                    rec_loss = self._compute_reconstruction_loss(
                        adapter_out["block_pooled"],
                        adapter_out["local_latents"],
                    )
                    result["reconstruction_loss"] = rec_loss
                    total_loss = total_loss + self.reconstruction_loss_weight * rec_loss

            # RC consistency 辅助损失（仅训练时）
            if self.use_rc_consistency and self.training:
                rc_loss = self._compute_rc_consistency_loss(sequences, labels)
                result["rc_consistency_loss"] = rc_loss
                total_loss = total_loss + self.rc_loss_weight * rc_loss

            result["loss"] = total_loss

        return result
