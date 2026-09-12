"""HyenaDNA baseline model for DNA sequence classification.

中文说明：
第四阶段外部 baseline，结构为：
    DNA 序列 → HyenaDNA backbone → mean pooling → Dropout → Linear → logits
不做 HiLA、不做冻结/解冻。

HyenaDNA 使用自定义 CharacterTokenizer（单字符级：A/C/G/T/N），
模型权重以 PyTorch Lightning .ckpt 格式存储在 HuggingFace。
需要从 .cache/hyenadna/ 加载预训练权重。
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch import nn


# ---------------------------------------------------------------------------
# Weight loading helpers (adapted from HyenaDNA's huggingface.py)
# 中文：权重加载辅助函数，从 HyenaDNA 官方代码适配。
# ---------------------------------------------------------------------------

def _inject_substring(orig_str: str) -> str:
    """Handle matching keys between models with/without gradient checkpointing."""
    modified = re.sub(r"\.mixer", ".mixer.layer", orig_str)
    modified = re.sub(r"\.mlp", ".mlp.layer", modified)
    return modified


def _load_weights(scratch_dict, pretrained_dict, checkpointing=False):
    """Loads pretrained (backbone only) weights into the scratch state dict."""
    for key, value in scratch_dict.items():
        if "backbone" in key:
            key_loaded = "model." + key
            if checkpointing:
                key_loaded = _inject_substring(key_loaded)
            scratch_dict[key] = pretrained_dict[key_loaded]
    return scratch_dict


# ---------------------------------------------------------------------------
# Minimal DNA character tokenizer (compatible with HyenaDNA's vocab)
# 中文：最小 DNA 字符 tokenizer，与 HyenaDNA 的 CharacterTokenizer 词表兼容。
# 避免依赖 transformers.PreTrainedTokenizer（4.40+ 存在兼容性问题）。
# 词表：[CLS]=0, [SEP]=1, [BOS]=2, [MASK]=3, [PAD]=4, [RESERVED]=5, [UNK]=6,
#        A=7, C=8, G=9, T=10, N=11
# ---------------------------------------------------------------------------

_DNA_VOCAB = {
    "[CLS]": 0, "[SEP]": 1, "[BOS]": 2, "+": 3,
    "[PAD]": 4, "[RESERVED]": 5, "[UNK]": 6,
    "A": 7, "C": 8, "G": 9, "T": 10, "N": 11,
}


class _DNACharTokenizer:
    """Minimal character-level tokenizer compatible with HyenaDNA vocab.

    中文说明：
    将 DNA 序列逐字符映射为整数 ID，与 HyenaDNA 的 CharacterTokenizer 词表一致。
    默认左侧补齐（HyenaDNA 是因果模型）。
    """

    def __init__(self, pad_side: str = "left"):
        self.cls_id = _DNA_VOCAB["[CLS]"]
        self.sep_id = _DNA_VOCAB["[SEP]"]
        self.pad_id = _DNA_VOCAB["[PAD]"]
        self.unk_id = _DNA_VOCAB["[UNK]"]
        self.pad_side = pad_side

    def __call__(
        self,
        sequences: List[str],
        max_length: int = 1024,
        device: torch.device = torch.device("cpu"),
    ):
        """Tokenize a batch of DNA sequences.

        Returns:
            input_ids:      [B, L] LongTensor
            attention_mask:  [B, L] LongTensor (1=valid, 0=pad)
        """
        all_ids = []
        for seq in sequences:
            # 字符映射
            char_ids = [_DNA_VOCAB.get(c, self.unk_id) for c in seq]
            # 加 special tokens: [CLS] + tokens + [SEP]
            ids = [self.cls_id] + char_ids[: max_length - 2] + [self.sep_id]
            all_ids.append(ids)

        # 找到 batch 内最大长度
        max_len = min(max(len(ids) for ids in all_ids), max_length)
        padded = []
        for ids in all_ids:
            ids = ids[:max_len]
            pad_len = max_len - len(ids)
            if self.pad_side == "left":
                padded.append([self.pad_id] * pad_len + ids)
            else:
                padded.append(ids + [self.pad_id] * pad_len)

        input_ids = torch.tensor(padded, dtype=torch.long, device=device)
        attention_mask = (input_ids != self.pad_id).long()
        return input_ids, attention_mask


class HyenaDNABaseline(nn.Module):
    """HyenaDNA-small-32k baseline with cross-entropy loss.

    中文说明：
    封装 HyenaDNA backbone，加入 mean pooling、分类头和 CrossEntropyLoss。
    训练时 forward(sequences, labels) 返回 loss + logits + probabilities；
    推理时 forward(sequences) 只返回 logits + probabilities。
    """

    def __init__(
        self,
        backbone: str = "LongSafari/hyenadna-small-32k-seqlen",
        num_labels: int = 2,
        max_length: int = 1024,
        dropout: float = 0.1,
        trust_remote_code: bool = True,
    ):
        """
        Args:
            backbone: HuggingFace 模型 ID（如 LongSafari/hyenadna-small-32k-seqlen）。
            num_labels: 分类类别数（二分类为 2）。
            max_length: 最大序列长度，超过则截断。
            dropout: 分类头前的 dropout 概率。
            trust_remote_code: 未使用，保留接口兼容。
        """
        super().__init__()

        # 解析模型名和本地缓存路径
        model_name = backbone.split("/")[-1]
        cache_dir = Path(__file__).resolve().parent.parent.parent / ".cache" / "hyenadna"
        model_path = cache_dir / model_name

        # 如果本地没有缓存，自动从 HuggingFace 下载
        if not model_path.exists():
            import subprocess
            cache_dir.mkdir(parents=True, exist_ok=True)
            hf_url = f"https://huggingface.co/LongSafari/{model_name}"
            subprocess.run(
                f"cd {cache_dir} && git lfs install && git clone {hf_url}",
                shell=True, check=True,
            )

        # 读取模型配置
        with open(model_path / "config.json") as f:
            config = json.load(f)

        d_model = config["d_model"]

        # 创建 HyenaDNA 模型并加载预训练权重
        from src.models.hyenadna_standalone import HyenaDNAModel

        self.backbone = HyenaDNAModel(**config, use_head=False, n_classes=num_labels)
        ckpt = torch.load(
            model_path / "weights.ckpt",
            map_location="cpu",
            weights_only=False,
        )
        checkpointing = config.get("checkpoint_mixer", False)
        state_dict = _load_weights(
            self.backbone.state_dict(), ckpt["state_dict"],
            checkpointing=checkpointing,
        )
        self.backbone.load_state_dict(state_dict)
        print(f"[HyenaDNA] Loaded pretrained weights from {model_path}")

        # 创建字符级 tokenizer（自实现，避免 PreTrainedTokenizer 兼容性问题）
        self.max_length = max_length
        self._char_tokenizer = _DNACharTokenizer(pad_side="left")

        # 分类头
        self.dropout_layer = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, num_labels)
        self.loss_fn = nn.CrossEntropyLoss()

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

        # Tokenize（使用自实现字符 tokenizer，与 HyenaDNA 的 CharacterTokenizer 兼容）
        input_ids, attention_mask = self._char_tokenizer(
            normalized, max_length=self.max_length, device=device,
        )

        # Backbone forward（HyenaDNAModel 返回 tensor [B, L, d_model]）
        hidden_states = self.backbone(input_ids)

        # Mask-aware mean pooling（忽略左侧 padding 位置）
        mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
        pooled = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)

        # 分类
        logits = self.classifier(self.dropout_layer(pooled))
        probabilities = torch.softmax(logits, dim=-1)

        result = {
            "logits": logits,
            "probabilities": probabilities,
        }

        if labels is not None:
            result["loss"] = self.loss_fn(logits, labels)

        return result
