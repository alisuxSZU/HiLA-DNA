"""Dataset wrapper for prepared DNA CSV files.

中文说明：
这个模块定义 DNACsvDataset，用于读取统一格式的 DNA CSV 数据文件。
Dataset 只负责返回原始 sequence 和 label，不在这里做 tokenizer，方便不同模型共用同一数据层。
"""

from pathlib import Path

import pandas as pd
from torch.utils.data import Dataset


class DNACsvDataset(Dataset):
    """Dataset for CSV files with sequence and label columns.

    中文说明：
    读取包含 sequence 和 label 两列的 CSV 文件，并按 PyTorch Dataset 接口返回样本。
    """

    def __init__(self, csv_path):
        """Load the CSV file and validate required columns.

        中文说明：
        加载 CSV 文件，并检查必须存在 sequence 和 label 两列。
        """
        self.csv_path = Path(csv_path)
        self.data = pd.read_csv(self.csv_path)
        required_columns = {"sequence", "label"}
        missing_columns = required_columns.difference(self.data.columns)
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"{self.csv_path} is missing required columns: {missing}")

    def __len__(self):
        """Return the number of samples.

        中文说明：返回样本数量。
        """
        return len(self.data)

    def __getitem__(self, index):
        """Return one sample as a dictionary.

        中文说明：
        返回一个字典，包含 DNA 序列字符串 sequence 和整数标签 label。
        """
        row = self.data.iloc[index]
        return {
            "sequence": str(row["sequence"]),
            "label": int(row["label"]),
        }
