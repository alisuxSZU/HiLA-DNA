"""HiLA/HiLA hierarchical latent adapter module.

中文说明：
支持消融实验的 HiLA adapter，通过配置开关控制各组件，并提供：
  - use_local_latent: 是否使用 block pooling + local latent MLP
  - use_global_latent: 是否使用 global latent
  - global_pool_type: mean_mlp（旧版）、attention（Stage 8）、
    statistical（Stage 9）、factorized_regional（Stage 11）或
    factorized_regional_residual（Stage 12）

消融变体对应的配置：
  A1 (+ Block pooling):   use_local_latent=True,  use_global_latent=False
  A2-v1 (Mean global):    use_global_latent=True, global_pool_type="mean_mlp"
  A2-v2 (Attention):      use_global_latent=True, global_pool_type="attention"
  A2-S/A2-SD (Stats):     use_global_latent=True, global_pool_type="statistical"
  A2-FR (Factorized):      use_global_latent=True, global_pool_type="factorized_regional"
  A2-FR-R64 (Residual):    use_global_latent=True,
                           global_pool_type="factorized_regional_residual"
  未设置 global_pool_type 时默认 mean_mlp，以兼容旧配置和 checkpoint。

结构：
    hidden_states [B, L, D]
      → block pooling（block_size=50）
      → per-block mask-aware mean pooling → MLP → local_latents [B, num_blocks, latent_dim]
      → mean_mlp: mean(local_latents) → MLP → global_latent
        或 attention: backbone_pooled 查询 local_latents → global_latent
        或 statistical: block dispersion/transition → MLP → global_latent
        或 factorized_regional: block_pooled 的通道/区域轴可分离投影 → global_latent
        或 factorized_regional_residual: 在因子化表示后增加残差瓶颈 MLP

输出 adapter_output：
  A1: concat(backbone_pooled, local_pooled) → dim = hidden_dim + latent_dim
  A2: concat(backbone_pooled, local_pooled, global_latent) → dim = hidden_dim + latent_dim + latent_dim
"""

import math
from typing import Dict, Optional, Sequence

import torch
from torch import nn


class AttentionGlobalPool(nn.Module):
    """Mask-aware multi-head attention from token summary to local latents.

    ``h_token`` is projected to one query per head. Block-level local latents
    provide keys and values. Returned attention weights are averaged over
    heads and are taken before attention dropout, so valid-block weights remain
    normalized and padding-block weights remain exactly zero.
    """

    def __init__(
        self,
        hidden_dim: int,
        latent_dim: int,
        num_heads: int = 4,
        attention_dropout: float = 0.1,
        mlp_dropout: float = 0.1,
    ):
        super().__init__()
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}.")
        if latent_dim % num_heads != 0:
            raise ValueError(
                "latent_dim must be divisible by num_heads: "
                f"latent_dim={latent_dim}, num_heads={num_heads}."
            )

        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        self.head_dim = latent_dim // num_heads

        self.query_proj = nn.Linear(hidden_dim, latent_dim)
        self.key_proj = nn.Linear(latent_dim, latent_dim)
        self.value_proj = nn.Linear(latent_dim, latent_dim)
        self.attention_dropout = nn.Dropout(attention_dropout)
        self.output_proj = nn.Linear(latent_dim, latent_dim)
        self.output_norm = nn.LayerNorm(latent_dim)
        self.global_mlp = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Dropout(mlp_dropout),
            nn.Linear(latent_dim, latent_dim),
        )

    def forward(
        self,
        h_token: torch.Tensor,
        local_latents: torch.Tensor,
        block_valid_mask: torch.Tensor,
    ):
        """Return ``(global_latent, attention_weights)``.

        Args:
            h_token: [B, hidden_dim] token-level sequence summary.
            local_latents: [B, K, latent_dim] block-level representations.
            block_valid_mask: [B, K], True for blocks containing valid tokens.

        Returns:
            global_latent: [B, latent_dim].
            attention_weights: [B, K], averaged across attention heads.
        """
        if h_token.ndim != 2:
            raise ValueError(f"h_token must have shape [B, D], got {tuple(h_token.shape)}.")
        if local_latents.ndim != 3:
            raise ValueError(
                "local_latents must have shape [B, K, D], "
                f"got {tuple(local_latents.shape)}."
            )
        if block_valid_mask.shape != local_latents.shape[:2]:
            raise ValueError(
                "block_valid_mask must match local_latents[:2]: "
                f"mask={tuple(block_valid_mask.shape)}, "
                f"latents={tuple(local_latents.shape)}."
            )
        if h_token.shape[0] != local_latents.shape[0]:
            raise ValueError("h_token and local_latents must have the same batch size.")
        if h_token.shape[-1] != self.hidden_dim:
            raise ValueError(
                f"Expected h_token hidden_dim={self.hidden_dim}, got {h_token.shape[-1]}."
            )
        if local_latents.shape[-1] != self.latent_dim:
            raise ValueError(
                "Expected local_latents latent_dim="
                f"{self.latent_dim}, got {local_latents.shape[-1]}."
            )

        batch_size, num_blocks, _ = local_latents.shape
        block_valid_mask = block_valid_mask.to(device=local_latents.device, dtype=torch.bool)

        query = self.query_proj(h_token).view(
            batch_size, self.num_heads, 1, self.head_dim
        )
        keys = self.key_proj(local_latents).view(
            batch_size, num_blocks, self.num_heads, self.head_dim
        ).transpose(1, 2)
        values = self.value_proj(local_latents).view(
            batch_size, num_blocks, self.num_heads, self.head_dim
        ).transpose(1, 2)

        scores = torch.matmul(query, keys.transpose(-2, -1))
        scores = scores / math.sqrt(self.head_dim)

        expanded_mask = block_valid_mask[:, None, None, :]
        # Avoid softmax(-inf, ..., -inf) for a pathological all-padding sample.
        any_valid = block_valid_mask.any(dim=-1, keepdim=True)
        safe_mask = expanded_mask | (~any_valid[:, None, :, None])
        scores = scores.masked_fill(~safe_mask, torch.finfo(scores.dtype).min)

        weights = torch.softmax(scores.float(), dim=-1).to(scores.dtype)
        weights = weights * expanded_mask.to(weights.dtype)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(
            torch.finfo(weights.dtype).tiny
        )

        context_weights = self.attention_dropout(weights)
        context = torch.matmul(context_weights, values)
        context = context.transpose(1, 2).contiguous().view(batch_size, self.latent_dim)
        context = self.output_norm(self.output_proj(context))
        global_latent = self.global_mlp(context)

        attention_weights = weights.squeeze(2).mean(dim=1)
        return global_latent, attention_weights


class ComplementaryStatisticalGlobalPool(nn.Module):
    """Build a global latent from fixed block-level statistics.

    The local mean is already exposed to the classifier as ``local_pooled``.
    This module therefore uses complementary information: per-channel block
    dispersion and, optionally, the RMS change between adjacent valid blocks.
    It never learns per-block weights.
    """

    _SUPPORTED_STATISTICS = {
        ("std",),
        ("std", "transition_rms"),
    }

    def __init__(
        self,
        latent_dim: int,
        statistics: Sequence[str] = ("std", "transition_rms"),
        bottleneck_dim: int = 64,
        dropout: float = 0.1,
        eps: float = 1e-6,
    ):
        super().__init__()
        statistics = tuple(statistics)
        if statistics not in self._SUPPORTED_STATISTICS:
            raise ValueError(
                "statistics must be ('std',) or "
                "('std', 'transition_rms'), "
                f"got {statistics!r}."
            )
        if latent_dim <= 0:
            raise ValueError(f"latent_dim must be positive, got {latent_dim}.")
        if bottleneck_dim <= 0:
            raise ValueError(
                f"bottleneck_dim must be positive, got {bottleneck_dim}."
            )
        if eps <= 0:
            raise ValueError(f"eps must be positive, got {eps}.")

        self.latent_dim = latent_dim
        self.statistics = statistics
        self.bottleneck_dim = bottleneck_dim
        self.eps = eps

        input_dim = latent_dim * len(statistics)
        self.global_mlp = nn.Sequential(
            nn.Linear(input_dim, bottleneck_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck_dim, latent_dim),
        )

    def forward(
        self,
        local_latents: torch.Tensor,
        block_valid_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Return the statistical global latent and its source statistics."""
        if local_latents.ndim != 3:
            raise ValueError(
                "local_latents must have shape [B, K, D], "
                f"got {tuple(local_latents.shape)}."
            )
        if local_latents.shape[-1] != self.latent_dim:
            raise ValueError(
                f"Expected latent_dim={self.latent_dim}, "
                f"got {local_latents.shape[-1]}."
            )
        if block_valid_mask.shape != local_latents.shape[:2]:
            raise ValueError(
                "block_valid_mask must match local_latents[:2]: "
                f"mask={tuple(block_valid_mask.shape)}, "
                f"latents={tuple(local_latents.shape)}."
            )

        mask = block_valid_mask.to(
            device=local_latents.device,
            dtype=torch.bool,
        )
        latents_fp32 = local_latents.float()
        mask_fp32 = mask.unsqueeze(-1).to(dtype=torch.float32)
        valid_count = mask_fp32.sum(dim=1)
        safe_valid_count = valid_count.clamp_min(1.0)

        block_mean = (latents_fp32 * mask_fp32).sum(dim=1) / safe_valid_count
        centered = (latents_fp32 - block_mean.unsqueeze(1)) * mask_fp32
        variance = centered.square().sum(dim=1) / safe_valid_count
        global_std = torch.sqrt(variance.clamp_min(0.0) + self.eps)
        has_dispersion = valid_count > 1
        global_std = torch.where(
            has_dispersion,
            global_std,
            torch.zeros_like(global_std),
        )

        result = {"global_std": global_std.to(dtype=local_latents.dtype)}
        features = [global_std]

        if "transition_rms" in self.statistics:
            if local_latents.shape[1] < 2:
                transition_rms = torch.zeros_like(global_std)
            else:
                pair_mask = mask[:, :-1] & mask[:, 1:]
                pair_mask_fp32 = pair_mask.unsqueeze(-1).to(dtype=torch.float32)
                pair_count = pair_mask_fp32.sum(dim=1)
                safe_pair_count = pair_count.clamp_min(1.0)
                adjacent_difference = (
                    latents_fp32[:, 1:, :] - latents_fp32[:, :-1, :]
                )
                mean_squared_difference = (
                    adjacent_difference.square() * pair_mask_fp32
                ).sum(dim=1) / safe_pair_count
                transition_rms = torch.sqrt(
                    mean_squared_difference.clamp_min(0.0) + self.eps
                )
                transition_rms = torch.where(
                    pair_count > 0,
                    transition_rms,
                    torch.zeros_like(transition_rms),
                )
            result["global_transition_rms"] = transition_rms.to(
                dtype=local_latents.dtype
            )
            features.append(transition_rms)

        statistical_features = torch.cat(features, dim=-1).to(
            dtype=local_latents.dtype
        )
        global_latent = self.global_mlp(statistical_features)
        any_valid = mask.any(dim=1, keepdim=True).to(global_latent.dtype)
        global_latent = global_latent * any_valid

        result["global_latent"] = global_latent
        return result


class FactorizedRegionalGlobalPool(nn.Module):
    """Low-parameter, position-preserving projection of regional summaries.

    Stage 11 treats mask-aware mean-pooled backbone regions as a matrix
    ``[K, hidden_dim]``. A shared channel projection maps ``hidden_dim`` to
    ``channel_rank``. A shared position projection then maps the fixed block
    axis ``K`` to ``region_rank``. The resulting
    ``[region_rank, channel_rank]`` core is flattened directly to the global
    latent, avoiding a dense ``(K * hidden_dim) -> hidden_dim`` layer.
    """

    def __init__(
        self,
        hidden_dim: int,
        latent_dim: int,
        num_blocks: int,
        region_rank: int = 4,
        channel_rank: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        for name, value in (
            ("hidden_dim", hidden_dim),
            ("latent_dim", latent_dim),
            ("num_blocks", num_blocks),
            ("region_rank", region_rank),
            ("channel_rank", channel_rank),
        ):
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value!r}.")
        if region_rank * channel_rank != latent_dim:
            raise ValueError(
                "region_rank * channel_rank must equal latent_dim: "
                f"{region_rank} * {channel_rank} != {latent_dim}."
            )
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}.")

        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_blocks = num_blocks
        self.region_rank = region_rank
        self.channel_rank = channel_rank

        self.region_norm = nn.LayerNorm(hidden_dim)
        self.channel_projection = nn.Linear(hidden_dim, channel_rank)
        self.region_projection = nn.Linear(num_blocks, region_rank)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        block_pooled: torch.Tensor,
        block_valid_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Return factorized channel/region features and a global latent."""
        if block_pooled.ndim != 3:
            raise ValueError(
                "block_pooled must have shape [B, K, D], "
                f"got {tuple(block_pooled.shape)}."
            )
        if block_pooled.shape[-1] != self.hidden_dim:
            raise ValueError(
                f"Expected hidden_dim={self.hidden_dim}, "
                f"got {block_pooled.shape[-1]}."
            )
        if block_valid_mask.shape != block_pooled.shape[:2]:
            raise ValueError(
                "block_valid_mask must match block_pooled[:2]: "
                f"mask={tuple(block_valid_mask.shape)}, "
                f"regions={tuple(block_pooled.shape)}."
            )

        batch_size, actual_blocks, _ = block_pooled.shape
        if actual_blocks > self.num_blocks:
            raise ValueError(
                "Actual block count exceeds configured num_blocks; "
                "truncation is forbidden: "
                f"actual={actual_blocks}, configured={self.num_blocks}."
            )

        mask = block_valid_mask.to(
            device=block_pooled.device,
            dtype=torch.bool,
        )
        if actual_blocks < self.num_blocks:
            pad_blocks = self.num_blocks - actual_blocks
            block_pooled = torch.nn.functional.pad(
                block_pooled,
                (0, 0, 0, pad_blocks),
                value=0.0,
            )
            mask = torch.nn.functional.pad(
                mask,
                (0, pad_blocks),
                value=False,
            )

        mask_values = mask.unsqueeze(-1).to(dtype=block_pooled.dtype)
        normalized_regions = self.region_norm(block_pooled) * mask_values
        channel_factors = torch.nn.functional.gelu(
            self.channel_projection(normalized_regions)
        )
        # Mask again because channel_projection.bias is otherwise non-zero.
        channel_factors = channel_factors * mask_values

        region_factors = self.region_projection(
            channel_factors.transpose(1, 2)
        ).transpose(1, 2)
        region_factors = torch.nn.functional.gelu(region_factors)
        global_latent = self.dropout(region_factors).reshape(
            batch_size,
            self.latent_dim,
        )
        any_valid = mask.any(dim=1, keepdim=True).to(global_latent.dtype)
        global_latent = global_latent * any_valid

        return {
            "global_channel_factors": channel_factors,
            "global_region_factors": region_factors,
            "global_latent": global_latent,
        }


class FactorizedRegionalResidualGlobalPool(FactorizedRegionalGlobalPool):
    """Stage 12 factorized regional pool with a compact residual interaction.

    The position-preserving Stage 11 representation is kept as the direct
    path. A ``latent_dim -> residual_bottleneck_dim -> latent_dim`` MLP learns
    nonlinear interactions between the flattened region/channel factors. The
    residual sum is normalized and finally masked so an all-padding sample is
    exactly zero even when residual linear layers have non-zero biases.
    """

    def __init__(
        self,
        hidden_dim: int,
        latent_dim: int,
        num_blocks: int,
        region_rank: int = 4,
        channel_rank: int = 32,
        residual_bottleneck_dim: int = 64,
        dropout: float = 0.1,
    ):
        if (
            not isinstance(residual_bottleneck_dim, int)
            or isinstance(residual_bottleneck_dim, bool)
            or residual_bottleneck_dim <= 0
        ):
            raise ValueError(
                "residual_bottleneck_dim must be a positive integer, "
                f"got {residual_bottleneck_dim!r}."
            )
        super().__init__(
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            num_blocks=num_blocks,
            region_rank=region_rank,
            channel_rank=channel_rank,
            dropout=dropout,
        )
        self.residual_bottleneck_dim = residual_bottleneck_dim
        self.residual_mlp = nn.Sequential(
            nn.Linear(latent_dim, residual_bottleneck_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(residual_bottleneck_dim, latent_dim),
        )
        self.output_norm = nn.LayerNorm(latent_dim)

    def forward(
        self,
        block_pooled: torch.Tensor,
        block_valid_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        result = super().forward(block_pooled, block_valid_mask)
        base_latent = result["global_latent"]
        residual_delta = self.residual_mlp(base_latent)
        global_latent = self.output_norm(base_latent + residual_delta)
        any_valid = block_valid_mask.to(
            device=global_latent.device,
            dtype=torch.bool,
        ).any(dim=1, keepdim=True)
        global_latent = global_latent * any_valid.to(global_latent.dtype)

        result["global_base_latent"] = base_latent
        result["global_residual_delta"] = residual_delta
        result["global_latent"] = global_latent
        return result


class HiLAAdapter(nn.Module):
    """Hierarchical Complementary Latent Adapter for DNA sequence models.

    中文说明：
    将 backbone 的 token-level hidden states 通过分层结构转换为增强的序列表示。
    block pooling 将序列切分为固定大小的块，每块独立池化后过 MLP 得到 local latent，
    再对所有 local latent 取均值过 MLP 得到 global latent。
    最终拼接 backbone 整体池化、local 均值池化和（可选）global latent 作为增强表示。
    """

    def __init__(
        self,
        hidden_dim: int,
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
    ):
        """
        Args:
            hidden_dim: backbone hidden states 的特征维度（Caduceus RCPS 输出为 d_model*2=512）。
            block_size: 每个 block 包含的 token 数。
            latent_dim: local/global latent 的维度。
            dropout: MLP 中的 dropout 概率。
            use_local_latent: 是否启用 block pooling + local latent MLP。
                              若为 False，adapter_output = backbone_pooled（退化为 A0）。
            use_global_latent: 是否启用 global latent MLP。
                               若为 False，adapter_output = concat(backbone_pooled, local_pooled)（A1）。
            global_pool_type: ``mean_mlp`` 保持旧版行为；``attention`` 使用
                              token summary 查询 block-level local latents；
                              ``statistical`` 使用 Stage 9 固定统计量。
            global_num_heads: AttentionGlobalPool 的注意力头数。
            global_attention_dropout: attention 权重 dropout。
            global_statistics: Stage 9 统计量，支持 ``("std",)`` 或
                               ``("std", "transition_rms")``。
            global_bottleneck_dim: Stage 9 global MLP bottleneck 维度。
            global_statistics_eps: Stage 9 sqrt 数值稳定项。
            global_num_blocks: Stage 11 固定区域数 Kmax。
            global_region_rank: Stage 11 区域轴低秩维度。
            global_channel_rank: Stage 11 通道轴低秩维度。
            global_residual_bottleneck_dim: Stage 12 残差 MLP bottleneck。
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.block_size = block_size
        self.latent_dim = latent_dim
        self.use_local_latent = use_local_latent
        self.use_global_latent = use_global_latent
        self.global_pool_type = global_pool_type

        if use_global_latent and not use_local_latent:
            raise ValueError("use_global_latent=True requires use_local_latent=True.")
        if global_pool_type not in {
            "mean_mlp",
            "attention",
            "statistical",
            "factorized_regional",
            "factorized_regional_residual",
        }:
            raise ValueError(
                "global_pool_type must be 'mean_mlp', 'attention', "
                "'statistical', 'factorized_regional', or "
                "'factorized_regional_residual', "
                f"got {global_pool_type!r}."
            )

        if use_local_latent:
            # local latent MLP: D → latent_dim → latent_dim
            self.local_mlp = nn.Sequential(
                nn.Linear(hidden_dim, latent_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(latent_dim, latent_dim),
            )

        if use_global_latent and global_pool_type == "mean_mlp":
            # global latent MLP: latent_dim → latent_dim → latent_dim
            self.global_mlp = nn.Sequential(
                nn.Linear(latent_dim, latent_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(latent_dim, latent_dim),
            )
        elif use_global_latent and global_pool_type == "attention":
            self.attention_global_pool = AttentionGlobalPool(
                hidden_dim=hidden_dim,
                latent_dim=latent_dim,
                num_heads=global_num_heads,
                attention_dropout=global_attention_dropout,
                mlp_dropout=dropout,
            )
        elif use_global_latent and global_pool_type == "statistical":
            self.statistical_global_pool = ComplementaryStatisticalGlobalPool(
                latent_dim=latent_dim,
                statistics=global_statistics,
                bottleneck_dim=global_bottleneck_dim,
                dropout=dropout,
                eps=global_statistics_eps,
            )
        elif use_global_latent and global_pool_type == "factorized_regional":
            if global_num_blocks is None:
                raise ValueError(
                    "global_num_blocks is required for "
                    "global_pool_type='factorized_regional'."
                )
            self.factorized_regional_global_pool = FactorizedRegionalGlobalPool(
                hidden_dim=hidden_dim,
                latent_dim=latent_dim,
                num_blocks=global_num_blocks,
                region_rank=global_region_rank,
                channel_rank=global_channel_rank,
                dropout=dropout,
            )
        elif use_global_latent:
            if global_num_blocks is None:
                raise ValueError(
                    "global_num_blocks is required for "
                    "global_pool_type='factorized_regional_residual'."
                )
            self.factorized_regional_residual_global_pool = (
                FactorizedRegionalResidualGlobalPool(
                    hidden_dim=hidden_dim,
                    latent_dim=latent_dim,
                    num_blocks=global_num_blocks,
                    region_rank=global_region_rank,
                    channel_rank=global_channel_rank,
                    residual_bottleneck_dim=global_residual_bottleneck_dim,
                    dropout=dropout,
                )
            )

    @property
    def output_dim(self) -> int:
        """Return adapter_output dimension based on current config.

        中文说明：
        根据 use_local_latent 和 use_global_latent 返回 adapter_output 的维度。
        A0: hidden_dim（只有 backbone_pooled）
        A1: hidden_dim + latent_dim（backbone_pooled + local_pooled）
        A2: hidden_dim + latent_dim + latent_dim（backbone_pooled + local_pooled + global_latent）
        """
        dim = self.hidden_dim
        if self.use_local_latent:
            dim += self.latent_dim
        if self.use_global_latent:
            dim += self.latent_dim
        return dim

    def _block_pool(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Partition hidden states into blocks and compute mask-aware mean pooling per block.

        中文说明：
        将 hidden_states 按块大小划分为固定大小的块，不丢弃最后不足一个块的部分。
        对每个块做 mask-aware 均值池化（padding 位置不参与计算）。

        Args:
            hidden_states: [B, L, D]
            attention_mask: [B, L]（1=有效，0=padding）

        Returns:
            block_pooled: [B, num_blocks, D]
        """
        B, L, D = hidden_states.shape
        num_blocks = math.ceil(L / self.block_size)

        # 补齐到 num_blocks * block_size
        padded_len = num_blocks * self.block_size
        if padded_len > L:
            pad_size = padded_len - L
            hidden_states = torch.nn.functional.pad(hidden_states, (0, 0, 0, pad_size))
            if attention_mask is not None:
                attention_mask = torch.nn.functional.pad(attention_mask, (0, pad_size), value=0)

        # reshape 为 [B, num_blocks, block_size, D]
        hidden_states = hidden_states.view(B, num_blocks, self.block_size, D)

        if attention_mask is not None:
            # [B, num_blocks, block_size]
            block_mask = attention_mask.view(B, num_blocks, self.block_size).unsqueeze(-1).to(hidden_states.dtype)
            # mask-aware mean pooling
            summed = (hidden_states * block_mask).sum(dim=2)
            counts = block_mask.sum(dim=2).clamp(min=1.0)
            block_pooled = summed / counts
        else:
            block_pooled = hidden_states.mean(dim=2)

        return block_pooled

    def _build_block_valid_mask(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Return [B, num_blocks] mask for blocks with at least one valid token."""
        batch_size, seq_len, _ = hidden_states.shape
        num_blocks = math.ceil(seq_len / self.block_size)
        if attention_mask is None:
            return torch.ones(
                batch_size,
                num_blocks,
                dtype=torch.bool,
                device=hidden_states.device,
            )

        padded_len = num_blocks * self.block_size
        if padded_len > seq_len:
            attention_mask = torch.nn.functional.pad(
                attention_mask,
                (0, padded_len - seq_len),
                value=0,
            )
        block_mask = attention_mask.view(batch_size, num_blocks, self.block_size)
        return block_mask.to(dtype=torch.bool).any(dim=-1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        backbone_pooled: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Run HiLA adapter forward pass.

        中文说明：
        根据 use_local_latent 和 use_global_latent 配置，输出不同维度的 adapter_output。

        Args:
            hidden_states: [B, L, D] backbone token-level 输出。
            attention_mask: [B, L] 可选，padding mask。
            backbone_pooled: [B, D] 可选，backbone 整体均值池化结果。
                             如果未提供，则在此处自行计算。

        Returns:
            dict with keys:
                block_pooled:     [B, num_blocks, D]（仅当 use_local_latent=True）
                local_latents:    [B, num_blocks, latent_dim]（仅当 use_local_latent=True）
                local_pooled:     [B, latent_dim]（仅当 use_local_latent=True）
                global_latent:    [B, latent_dim]（仅当 use_global_latent=True）
                block_valid_mask: [B, num_blocks]（仅当 use_local_latent=True）
                global_attention_weights: [B, num_blocks]（仅 attention 模式）
                global_std: [B, latent_dim]（仅 statistical 模式）
                global_transition_rms: [B, latent_dim]（仅 A2-SD）
                global_channel_factors: [B, Kmax, channel_rank]（仅 Stage 11）
                global_region_factors: [B, region_rank, channel_rank]（仅 Stage 11）
                global_base_latent: [B, latent_dim]（仅 Stage 12）
                global_residual_delta: [B, latent_dim]（仅 Stage 12）
                adapter_output:   [B, output_dim]
        """
        # 如果没有传入 backbone_pooled，自行计算
        if backbone_pooled is None:
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
                backbone_pooled = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
            else:
                backbone_pooled = hidden_states.mean(dim=1)

        result = {}

        if not self.use_local_latent:
            # A0 退化为只用 backbone_pooled（不应走到这里，A0 不用 HiLA）
            result["adapter_output"] = backbone_pooled
            return result

        # block pooling → [B, num_blocks, D]
        block_pooled = self._block_pool(hidden_states, attention_mask)
        result["block_pooled"] = block_pooled
        block_valid_mask = self._build_block_valid_mask(hidden_states, attention_mask)
        result["block_valid_mask"] = block_valid_mask

        # local latent: per-block MLP → [B, num_blocks, latent_dim]
        local_latents = self.local_mlp(block_pooled)
        result["local_latents"] = local_latents

        # local pooled: mean over blocks → [B, latent_dim]
        local_pooled = local_latents.mean(dim=1)
        result["local_pooled"] = local_pooled

        if self.use_global_latent:
            if self.global_pool_type == "attention":
                global_latent, attention_weights = self.attention_global_pool(
                    h_token=backbone_pooled,
                    local_latents=local_latents,
                    block_valid_mask=block_valid_mask,
                )
                result["global_attention_weights"] = attention_weights
            elif self.global_pool_type == "statistical":
                statistical_result = self.statistical_global_pool(
                    local_latents=local_latents,
                    block_valid_mask=block_valid_mask,
                )
                result.update(statistical_result)
                global_latent = statistical_result["global_latent"]
            elif self.global_pool_type == "factorized_regional":
                factorized_result = self.factorized_regional_global_pool(
                    block_pooled=block_pooled,
                    block_valid_mask=block_valid_mask,
                )
                result.update(factorized_result)
                global_latent = factorized_result["global_latent"]
            elif self.global_pool_type == "factorized_regional_residual":
                residual_result = self.factorized_regional_residual_global_pool(
                    block_pooled=block_pooled,
                    block_valid_mask=block_valid_mask,
                )
                result.update(residual_result)
                global_latent = residual_result["global_latent"]
            else:
                # Legacy global latent: mean over blocks → MLP.
                global_latent = self.global_mlp(local_latents.mean(dim=1))
            result["global_latent"] = global_latent

            # A2: concat(backbone_pooled, local_pooled, global_latent)
            adapter_output = torch.cat([backbone_pooled, local_pooled, global_latent], dim=-1)
        else:
            # A1: concat(backbone_pooled, local_pooled)
            adapter_output = torch.cat([backbone_pooled, local_pooled], dim=-1)

        result["adapter_output"] = adapter_output
        return result
