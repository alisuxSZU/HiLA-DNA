"""Caduceus baseline model for DNA sequence classification.

中文说明：
第一阶段 baseline 模型，结构为：
    DNA 序列 → Caduceus backbone → mean pooling → Dropout → Linear → logits
不做 HiLA、不做 block pooling、不做辅助损失、不做冻结/解冻。
默认 full fine-tuning（backbone + classifier 全部训练）。

训练脚本通过 forward 传入序列和标签即可获得 loss、logits、probabilities。
"""

from typing import Dict, List, Optional

import torch
from torch import nn

from src.models.caduceus_wrapper import CaduceusSequenceClassifier


class CaduceusBaseline(nn.Module):
    """Caduceus-PS-1k baseline with cross-entropy loss.

    中文说明：
    封装 CaduceusSequenceClassifier，加入 CrossEntropyLoss 和概率计算。
    训练时 forward(sequences, labels) 返回 loss + logits + probabilities；
    推理时 forward(sequences) 只返回 logits + probabilities。
    """

    def __init__(
        self,
        backbone: str = "caduceus_ps_1k",
        num_labels: int = 2,
        max_length: int = 1024,
        dropout: float = 0.1,
    ):
        """
        Args:
            backbone: Caduceus 模型短名称或 HuggingFace 模型 ID。
            num_labels: 分类类别数（二分类为 2）。
            max_length: 最大序列长度，超过则截断。
            dropout: 分类头前的 dropout 概率。
        """
        super().__init__()
        self.classifier = CaduceusSequenceClassifier(
            backbone=backbone,
            num_labels=num_labels,
            max_length=max_length,
            dropout=dropout,
        )
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
        outputs = self.classifier(sequences)
        logits = outputs["logits"]
        probabilities = torch.softmax(logits, dim=-1)

        result = {
            "logits": logits,
            "probabilities": probabilities,
        }

        if labels is not None:
            result["loss"] = self.loss_fn(logits, labels)

        return result
