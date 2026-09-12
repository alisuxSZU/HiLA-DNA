"""Caduceus backbone wrapper for DNA sequence classification.

中文说明：
这个模块封装 Caduceus backbone，使后续训练代码可以统一调用：

    outputs = model(sequences)

返回字典包含：

    logits: [batch_size, num_labels]
    hidden_states: [batch_size, seq_len, hidden_dim]

第一阶段只实现最基础结构：Caduceus backbone -> mean pooling -> classifier。
"""

from typing import Dict, List, Optional

import torch
from torch import nn


CADUCEUS_MODEL_ALIASES = {
    "caduceus_ps_1k": "kuleshov-group/caduceus-ps_seqlen-1k_d_model-256_n_layer-4_lr-8e-3",
    "caduceus_ph_1k": "kuleshov-group/caduceus-ph_seqlen-1k_d_model-256_n_layer-4_lr-8e-3",
    "caduceus_ps_131k": "kuleshov-group/caduceus-ps_seqlen-131k_d_model-256_n_layer-16",
    "caduceus_ph_131k": "kuleshov-group/caduceus-ph_seqlen-131k_d_model-256_n_layer-16",
}


def resolve_caduceus_model_name(backbone: str) -> str:
    """Resolve a short backbone alias to a Hugging Face model ID.

    中文说明：
    将配置文件中的简短名称转换成 Hugging Face 模型 ID。
    如果传入的本来就是完整模型 ID，则原样返回。
    """
    return CADUCEUS_MODEL_ALIASES.get(backbone, backbone)


def _get_hidden_size(config) -> int:
    """Infer hidden dimension from a Hugging Face config object.

    中文说明：
    不同模型配置中隐藏层维度字段名可能不同，这里按常见字段依次查找。
    Caduceus 使用 RCPS（Reverse-Complement Parameter Sharing）结构，
    RCPSWrapper 在 forward 时将 forward 和 RC 输出 concat，导致实际输出维度翻倍。
    因此当检测到 rcps=True 时返回 d_model * 2。
    """
    # Caduceus 特殊处理：RCPS 将通道维度翻倍
    if getattr(config, "rcps", False) or getattr(config, "model_type", "") == "caduceus":
        d_model = getattr(config, "d_model", None)
        if d_model is not None:
            return int(d_model) * 2

    for name in ("hidden_size", "d_model", "n_embd", "dim"):
        value = getattr(config, name, None)
        if value is not None:
            return int(value)
    raise ValueError("Could not infer hidden size from Caduceus config.")


class CaduceusSequenceClassifier(nn.Module):
    """Caduceus backbone with mean pooling and a linear classifier.

    中文说明：
    这是第一阶段 baseline 使用的模型封装：
    DNA 序列 -> Caduceus -> token hidden states -> mask-aware mean pooling -> 分类头。
    """

    def __init__(
        self,
        backbone: str = "caduceus_ps_1k",
        num_labels: int = 2,
        max_length: int = 1024,
        dropout: float = 0.1,
        trust_remote_code: bool = True,
    ):
        super().__init__()

        # Import lazily so unit tests can import this module without downloading the model.
        # 中文：延迟导入，避免仅做语法检查时就下载大模型。
        from transformers import AutoModel, AutoTokenizer

        self.model_name_or_path = resolve_caduceus_model_name(backbone)
        self.max_length = max_length
        self.num_labels = num_labels

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path,
            trust_remote_code=trust_remote_code,
        )
        self.backbone = AutoModel.from_pretrained(
            self.model_name_or_path,
            trust_remote_code=trust_remote_code,
        )

        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
                self.backbone.resize_token_embeddings(len(self.tokenizer))

        hidden_size = _get_hidden_size(self.backbone.config)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def tokenize(self, sequences: List[str], device: Optional[torch.device] = None) -> Dict[str, torch.Tensor]:
        """Tokenize DNA sequences for the Caduceus backbone.

        中文说明：
        将 DNA 序列转为模型输入张量。序列会统一转成大写，并按 max_length 截断/补齐。
        """
        normalized_sequences = [sequence.upper() for sequence in sequences]
        batch = self.tokenizer(
            normalized_sequences,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        if device is not None:
            batch = {key: value.to(device) for key, value in batch.items()}
        return batch

    def _extract_hidden_states(self, outputs) -> torch.Tensor:
        """Extract token hidden states from Hugging Face model outputs.

        中文说明：
        优先使用 last_hidden_state；如果模型只返回 hidden_states，则取最后一层。
        """
        if hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
            return outputs.last_hidden_state
        if hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
            return outputs.hidden_states[-1]
        if isinstance(outputs, (tuple, list)) and outputs:
            return outputs[0]
        raise ValueError("Could not extract hidden states from Caduceus outputs.")

    def _mean_pool(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor]) -> torch.Tensor:
        """Mean-pool token hidden states, ignoring padding positions when possible.

        中文说明：
        对 token hidden states 做平均池化。如果有 attention_mask，则不把 padding 位置计入平均值。
        """
        if attention_mask is None:
            return hidden_states.mean(dim=1)

        mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
        summed = (hidden_states * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1.0)
        return summed / counts

    def forward(self, sequences: List[str]) -> Dict[str, torch.Tensor]:
        """Run Caduceus classification forward pass.

        中文说明：
        输入 DNA 序列列表，输出 logits 和 backbone hidden_states。
        """
        device = next(self.parameters()).device
        batch = self.tokenize(sequences, device=device)
        outputs = self.backbone(**batch, output_hidden_states=True)
        hidden_states = self._extract_hidden_states(outputs)
        pooled = self._mean_pool(hidden_states, batch.get("attention_mask"))
        logits = self.classifier(self.dropout(pooled))
        return {
            "logits": logits,
            "hidden_states": hidden_states,
        }

