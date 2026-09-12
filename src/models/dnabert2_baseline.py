"""DNABERT-2 baseline model for DNA sequence classification.

中文说明：
第四阶段外部 baseline，结构为：
    DNA 序列 → DNABERT-2 backbone → mean pooling → Dropout → Linear → logits
不做 HiLA、不做冻结/解冻。

DNABERT-2 使用 BPE tokenizer，模型通过 AutoModel + trust_remote_code 加载。
hidden_size = 768（BERT-base 架构，117M 参数）。
"""

from typing import Dict, List, Optional

import torch
from torch import nn


class DNABERT2Baseline(nn.Module):
    """DNABERT-2-117M baseline with cross-entropy loss.

    中文说明：
    封装 DNABERT-2 backbone，加入 mean pooling、分类头和 CrossEntropyLoss。
    训练时 forward(sequences, labels) 返回 loss + logits + probabilities；
    推理时 forward(sequences) 只返回 logits + probabilities。
    """

    def __init__(
        self,
        backbone: str = "zhihan1996/DNABERT-2-117M",
        num_labels: int = 2,
        max_length: int = 1024,
        dropout: float = 0.1,
        trust_remote_code: bool = True,
    ):
        """
        Args:
            backbone: HuggingFace 模型 ID。
            num_labels: 分类类别数（二分类为 2）。
            max_length: 最大序列长度，超过则截断。
            dropout: 分类头前的 dropout 概率。
            trust_remote_code: 是否信任远程代码（DNABERT-2 必须为 True）。
        """
        super().__init__()
        from transformers import AutoConfig, AutoTokenizer

        self.max_length = max_length
        self.num_labels = num_labels

        self.tokenizer = AutoTokenizer.from_pretrained(
            backbone, trust_remote_code=trust_remote_code,
        )

        # 加载 DNABERT-2 backbone
        # 中文：transformers 4.40+ 的 AutoModel.register 会检查 config_class 一致性，
        # DNABERT-2 的自定义 BertConfig 与标准 BertConfig 不匹配导致报错。
        # 解决方案：先通过 AutoConfig 加载自定义代码，再直接用自定义 BertModel 加载权重，
        # 绕过 AutoModel 的注册检查。
        config = AutoConfig.from_pretrained(backbone, trust_remote_code=trust_remote_code)

        try:
            from transformers import AutoModel
            self.backbone = AutoModel.from_pretrained(
                backbone, trust_remote_code=trust_remote_code,
            )
        except ValueError as e:
            if "config_class" not in str(e):
                raise
            # 回退方案：直接导入自定义 BertModel 类
            import importlib
            config_module = config.__class__.__module__
            base_module = config_module.rsplit(".configuration_bert", 1)[0]
            # DNABERT-2 的 auto_map 指向 bert_layers.BertModel
            modeling_module = importlib.import_module(f"{base_module}.bert_layers")
            CustomBertModel = modeling_module.BertModel
            self.backbone = CustomBertModel.from_pretrained(backbone, config=config)

        # 修复 DNABERT-2 Triton flash attention 兼容性
        # 中文：DNABERT-2 的自定义 flash_attn_triton 使用 tl.dot(q, k, trans_b=True)，
        # 在 Triton 3.x 中已移除 trans_b 参数。用 PyTorch SDPA 替换，不需要改 Triton。
        self._patch_flash_attention()

        # 处理 pad token
        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
                self.backbone.resize_token_embeddings(len(self.tokenizer))

        # 从 config 获取 hidden_size
        hidden_size = self._get_hidden_size(self.backbone.config)

        self.dropout_layer = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.loss_fn = nn.CrossEntropyLoss()

    @staticmethod
    def _sdpa_flash_attn(qkv, bias=None):
        """PyTorch SDPA fallback for DNABERT-2 flash attention.

        中文说明：
        替换 DNABERT-2 的 Triton flash attention（Triton 3.x 不兼容）。
        使用 PyTorch 内置 scaled_dot_product_attention。
        输入格式：qkv [batch, seqlen, 3, heads, headdim]
        输出格式：[batch, seqlen, heads, headdim]
        """
        import torch.nn.functional as F
        q, k, v = qkv.unbind(dim=2)  # each [B, S, H, D]
        q = q.transpose(1, 2)  # [B, H, S, D]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
        return out.transpose(1, 2)  # [B, S, H, D]

    def _patch_flash_attention(self):
        """Monkey-patch DNABERT-2's Triton flash attention with PyTorch SDPA.

        中文说明：在 sys.modules 中找到 DNABERT-2 的 bert_layers 模块，
        将 flash_attn_qkvpacked_func 替换为 PyTorch SDPA 实现。
        """
        import sys
        for key in list(sys.modules.keys()):
            if "transformers_modules" in key and "DNABERT" in key and "bert_layers" in key:
                sys.modules[key].flash_attn_qkvpacked_func = self._sdpa_flash_attn
                print("[DNABERT-2] Patched flash attention with PyTorch SDPA (Triton 3.x compat)")
                return
        # If not found in sys.modules yet, it'll be patched on first use
        print("[DNABERT-2] Warning: bert_layers module not found for flash attention patch")

    @staticmethod
    def _get_hidden_size(config) -> int:
        """Infer hidden dimension from model config.

        中文说明：依次检查 hidden_size / d_model / n_embd / dim 字段。
        """
        for name in ("hidden_size", "d_model", "n_embd", "dim"):
            value = getattr(config, name, None)
            if value is not None:
                return int(value)
        raise ValueError("Could not infer hidden size from DNABERT-2 config.")

    def _extract_hidden_states(self, outputs) -> torch.Tensor:
        """Extract token hidden states from model outputs.

        中文说明：兼容 last_hidden_state / hidden_states / tuple 三种输出格式。
        """
        if hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
            return outputs.last_hidden_state
        if hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
            return outputs.hidden_states[-1]
        if isinstance(outputs, (tuple, list)) and outputs:
            return outputs[0]
        raise ValueError("Could not extract hidden states from DNABERT-2 outputs.")

    def _mean_pool(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Mask-aware mean pooling.

        中文说明：对 token hidden states 做平均池化，忽略 padding 位置。
        """
        if attention_mask is None:
            return hidden_states.mean(dim=1)
        mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
        summed = (hidden_states * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1.0)
        return summed / counts

    def forward(
        self,
        sequences: List[str],
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass.

        中文说明：
        输入 DNA 序列列表和可选标签，返回包含 logits、probabilities 的字典。
        如果提供了 labels，还会计算并返回 cross-entropy loss。

        Args:
            sequences: DNA 序列字符串列表，如 ["ACGT...", "TGCA..."]。
            labels: 可选，整型标签张量，shape [batch_size]。

        Returns:
            dict with keys:
                logits:        [batch_size, num_labels]
                probabilities: [batch_size, num_labels]  (softmax)
                loss:          scalar (仅当 labels 不为 None 时)
        """
        device = next(self.parameters()).device
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

        outputs = self.backbone(**batch)
        hidden_states = self._extract_hidden_states(outputs)

        pooled = self._mean_pool(hidden_states, attention_mask)
        logits = self.classifier(self.dropout_layer(pooled))
        probabilities = torch.softmax(logits, dim=-1)

        result = {
            "logits": logits,
            "probabilities": probabilities,
        }

        if labels is not None:
            result["loss"] = self.loss_fn(logits, labels)

        return result
