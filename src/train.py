"""Unified training script for Caduceus baseline and HiLA experiments.

中文说明：
训练脚本，支持通过 --config 指定 YAML 配置文件来运行不同实验。
用法：python src/train.py --config configs/caduceus_baseline_human_nontata.yaml

训练流程：
1. 读取 yaml config
2. 设置随机种子
3. 加载 train/val csv；正式评估时再加载 test csv
4. 构建 DataLoader（含自定义 collate_fn）
5. 初始化模型（baseline 或 HiLA）
6. 初始化 optimizer + scheduler
7. 逐 epoch 训练 + 验证
8. 基于 val metric 保存 best checkpoint + early stopping
9. 可选：加载 best checkpoint → 测试集评估
10. 正式评估时保存 test_metrics.json 和 test_predictions.csv
"""

import argparse
import csv
import json
import os
import time
from pathlib import Path

import torch
from torch.cuda.amp import GradScaler, autocast as cuda_autocast
try:
    from torch.amp import autocast as amp_autocast
except ImportError:
    amp_autocast = None
from torch.optim import AdamW
from torch.utils.data import DataLoader

import yaml


def _autocast(enabled, dtype):
    """兼容新旧 PyTorch 的 autocast 封装。

    中文说明：
    优先使用 torch.amp.autocast（支持 device_type 和 dtype），
    回退到 torch.cuda.amp.autocast（只支持 enabled）。
    """
    if amp_autocast is not None:
        return amp_autocast(device_type="cuda", enabled=enabled, dtype=dtype)
    return cuda_autocast(enabled=enabled)

from src.datasets.dna_csv_dataset import DNACsvDataset
from src.metrics.classification import compute_binary_classification_metrics
from src.models.caduceus_baseline import CaduceusBaseline
from src.utils import set_seed


# ---------------------------------------------------------------------------
# Collate function
# 中文：DataLoader 的 collate 函数，将样本列表整理为模型可接受的格式。
# 不在这里做 tokenize，tokenize 由模型内部完成。
# ---------------------------------------------------------------------------

def collate_fn(batch):
    """Collect a list of dataset samples into a batch dict.

    中文说明：
    DNACsvDataset 返回 {"sequence": str, "label": int}。
    collate_fn 把它们拼成 {"sequences": list[str], "labels": tensor[int]}。
    """
    sequences = [item["sequence"] for item in batch]
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
    return {"sequences": sequences, "labels": labels}


# ---------------------------------------------------------------------------
# Model factory
# 中文：根据配置中的 use_hila 标志选择对应的模型类。
# ---------------------------------------------------------------------------

def build_model(cfg):
    """Instantiate model based on config.

    中文说明：
    优先检查 model.architecture 字段选择模型类：
      - hyenadna_baseline  → HyenaDNABaseline
      - dnabert2_baseline  → DNABERT2Baseline
    向后兼容旧配置：
      - model.use_hila=true → CaduceusHiLA
      - 其他                  → CaduceusBaseline
    """
    model_cfg = cfg["model"]
    dataset_cfg = cfg["dataset"]

    architecture = model_cfg.get("architecture", None)
    backbone = model_cfg.get("backbone", "caduceus_ps_1k")
    num_labels = model_cfg.get("num_labels", 2)
    max_length = dataset_cfg.get("max_length", 1024)
    trust_remote_code = model_cfg.get("trust_remote_code", False)

    if architecture == "hyenadna_baseline":
        from src.models.hyenadna_baseline import HyenaDNABaseline

        model = HyenaDNABaseline(
            backbone=backbone,
            num_labels=num_labels,
            max_length=max_length,
            trust_remote_code=trust_remote_code,
        )
    elif architecture == "dnabert2_baseline":
        from src.models.dnabert2_baseline import DNABERT2Baseline

        model = DNABERT2Baseline(
            backbone=backbone,
            num_labels=num_labels,
            max_length=max_length,
            trust_remote_code=trust_remote_code,
        )
    elif model_cfg.get("use_hila", False):
        from src.models.caduceus_hila import CaduceusHiLA

        hila_cfg = cfg.get("hila", {})
        model = CaduceusHiLA(
            backbone=backbone,
            num_labels=num_labels,
            max_length=max_length,
            block_size=hila_cfg.get("block_size", 50),
            latent_dim=hila_cfg.get("latent_dim", 128),
            dropout=hila_cfg.get("dropout", 0.1),
            # Stage 5 消融开关
            use_local_latent=hila_cfg.get("use_local_latent", True),
            use_global_latent=hila_cfg.get("use_global_latent", True),
            # Stage 8/9 global pooling；旧配置默认 mean_mlp。
            global_pool_type=hila_cfg.get("global_pool_type", "mean_mlp"),
            global_num_heads=hila_cfg.get("global_num_heads", 4),
            global_attention_dropout=hila_cfg.get(
                "global_attention_dropout", 0.1
            ),
            global_statistics=hila_cfg.get(
                "global_statistics", ["std", "transition_rms"]
            ),
            global_bottleneck_dim=hila_cfg.get("global_bottleneck_dim", 64),
            global_statistics_eps=hila_cfg.get("global_statistics_eps", 1e-6),
            global_num_blocks=hila_cfg.get("global_num_blocks"),
            global_region_rank=hila_cfg.get("global_region_rank", 4),
            global_channel_rank=hila_cfg.get("global_channel_rank", 32),
            global_residual_bottleneck_dim=hila_cfg.get(
                "global_residual_bottleneck_dim", 64
            ),
            use_mask_reconstruction=hila_cfg.get("use_mask_reconstruction", False),
            mask_ratio=hila_cfg.get("mask_ratio", 0.15),
            reconstruction_loss_weight=hila_cfg.get("reconstruction_loss_weight", 0.05),
            use_rc_consistency=hila_cfg.get("use_rc_consistency", False),
            rc_loss_weight=hila_cfg.get("rc_loss_weight", 0.05),
        )
    else:
        model = CaduceusBaseline(
            backbone=backbone,
            num_labels=num_labels,
            max_length=max_length,
        )
    return model


def freeze_backbone_if_requested(model, cfg):
    """Freeze backbone parameters when model.freeze_backbone is true.

    中文说明：
    如果配置中写了 model.freeze_backbone: true，则冻结 Caduceus backbone，
    只训练 HiLA adapter 和分类头。这样可以用于阶段 2 的轻量调参实验。
    """
    model_cfg = cfg.get("model", {})
    if not model_cfg.get("freeze_backbone", False):
        return 0, 0

    backbone_modules = []
    if hasattr(model, "backbone"):
        backbone_modules.append(model.backbone)
    if hasattr(model, "classifier") and hasattr(model.classifier, "backbone"):
        backbone_modules.append(model.classifier.backbone)

    if not backbone_modules:
        raise ValueError("model.freeze_backbone=true, but no backbone module was found.")

    seen_params = set()
    frozen_tensors = 0
    frozen_params = 0
    for backbone in backbone_modules:
        for param in backbone.parameters():
            param_id = id(param)
            if param_id in seen_params:
                continue
            seen_params.add(param_id)
            param.requires_grad = False
            frozen_tensors += 1
            frozen_params += param.numel()

    return frozen_tensors, frozen_params


def count_parameters(model):
    """Return total and trainable parameter counts.

    中文说明：
    返回模型总参数量和当前可训练参数量，便于确认冻结策略是否生效。
    """
    total_params = sum(param.numel() for param in model.parameters())
    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    return total_params, trainable_params


# ---------------------------------------------------------------------------
# Evaluation helper
# 中文：在一个 DataLoader 上跑完整评估，返回 loss 和所有指标。
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, dataloader, device, fp16=True, amp_dtype=torch.float16):
    """Run evaluation and return avg loss + classification metrics.

    中文说明：
    遍历整个 dataloader，收集所有预测概率和标签，计算平均 loss 和分类指标。
    """
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_labels = []
    all_probs = []

    for batch in dataloader:
        sequences = batch["sequences"]
        labels = batch["labels"].to(device)

        with _autocast(fp16, amp_dtype):
            outputs = model(sequences, labels=labels)

        total_loss += outputs["loss"].item() * len(sequences)
        total_samples += len(sequences)

        # fp32 softmax 避免 NaN
        logits_fp32 = outputs["logits"].float()
        probs = torch.softmax(logits_fp32, dim=-1)[:, 1]
        nan_mask = ~torch.isfinite(probs)
        if nan_mask.any():
            probs = probs.clone()
            probs[nan_mask] = 0.5
        all_probs.extend(probs.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / max(total_samples, 1)
    metrics = compute_binary_classification_metrics(all_labels, all_probs)
    return avg_loss, metrics


# ---------------------------------------------------------------------------
# Main training loop
# 中文：主训练流程。
# ---------------------------------------------------------------------------

def train_one_epoch(model, dataloader, optimizer, scaler, device, fp16=True, amp_dtype=torch.float16, grad_accum_steps=1):
    """Train for one epoch and return average training loss.

    中文说明：
    遍历一遍训练集，返回该 epoch 的平均 loss。
    支持 gradient accumulation：每 grad_accum_steps 个 batch 才执行一次 optimizer step。
    """
    model.train()
    total_loss = 0.0
    total_samples = 0
    optimizer.zero_grad()

    for i, batch in enumerate(dataloader):
        sequences = batch["sequences"]
        labels = batch["labels"].to(device)

        with _autocast(fp16, amp_dtype):
            outputs = model(sequences, labels=labels)
            loss = outputs["loss"]

        # 检测 NaN/Inf loss，跳过该 batch 避免污染权重
        if not torch.isfinite(loss):
            continue

        scaled_loss = loss / grad_accum_steps
        scaler.scale(scaled_loss).backward()

        total_loss += loss.item() * len(sequences)
        total_samples += len(sequences)

        if (i + 1) % grad_accum_steps == 0 or (i + 1) == len(dataloader):
            scaler.unscale_(optimizer)
            # 梯度裁剪，防止梯度爆炸导致权重溢出
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

    return total_loss / max(total_samples, 1)


def main():
    parser = argparse.ArgumentParser(description="Train Caduceus baseline or HiLA model.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file.")
    parser.add_argument("--no-early-stop", action="store_true", help="Disable early stopping, run all epochs.")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 1. Load config
    # 中文：读取 YAML 配置文件。
    # ------------------------------------------------------------------
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    print(f"[Config] experiment: {cfg['experiment_name']}")
    print(f"[Config] config file: {args.config}")

    # ------------------------------------------------------------------
    # 2. Set seed
    # 中文：设置随机种子，确保实验可复现。
    # ------------------------------------------------------------------
    seed = cfg["training"]["seed"]
    set_seed(seed)
    print(f"[Seed] {seed}")

    # ------------------------------------------------------------------
    # 3. Prepare output directory
    # 中文：创建输出目录，并保存一份当前配置的副本。
    # ------------------------------------------------------------------
    output_dir = Path(cfg["output"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    config_save_path = output_dir / "config.yaml"
    with open(config_save_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)
    print(f"[Output] directory: {output_dir}")

    # ------------------------------------------------------------------
    # 4. Device & fp16
    # 中文：检测 GPU 和 fp16 设置。
    # ------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fp16 = cfg["training"].get("fp16", True) and device.type == "cuda"
    print(f"[Device] {device}, fp16={fp16}")

    # ------------------------------------------------------------------
    # 5. Load datasets
    # 中文：pilot 只加载 train/val；formal 保持旧行为并加载 test。
    # ------------------------------------------------------------------
    dataset_cfg = cfg["dataset"]
    run_test = cfg.get("evaluation", {}).get("run_test", True)
    train_ds = DNACsvDataset(dataset_cfg["train_csv"])
    val_ds = DNACsvDataset(dataset_cfg["val_csv"])
    test_ds = (
        DNACsvDataset(dataset_cfg["test_csv"])
        if run_test
        else None
    )
    if run_test:
        print(
            f"[Data] train={len(train_ds)}, val={len(val_ds)}, "
            f"test={len(test_ds)}"
        )
    else:
        print(
            f"[Data] train={len(train_ds)}, val={len(val_ds)}, "
            "test=not loaded (validation-only)"
        )

    # ------------------------------------------------------------------
    # 6. Build DataLoaders
    # 中文：构建 DataLoader，使用自定义 collate_fn。
    # ------------------------------------------------------------------
    batch_size = cfg["training"]["batch_size"]
    num_workers = 0  # 序列数据不大，0 避免多进程 tokenize 麻烦

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=num_workers,
    )
    test_loader = (
        DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=num_workers,
        )
        if run_test
        else None
    )

    # ------------------------------------------------------------------
    # 7. Build model
    # 中文：根据配置初始化模型，并移到 GPU。
    # ------------------------------------------------------------------
    model = build_model(cfg)
    model.to(device)
    frozen_tensors, frozen_params = freeze_backbone_if_requested(model, cfg)
    total_params, trainable_params = count_parameters(model)
    print(f"[Model] {model.__class__.__name__}")
    if frozen_tensors > 0:
        print(f"[Freeze] backbone frozen: tensors={frozen_tensors}, params={frozen_params:,}")
    print(f"[Params] total={total_params:,}, trainable={trainable_params:,}")
    if trainable_params == 0:
        raise ValueError("No trainable parameters left after applying freeze policy.")

    # ------------------------------------------------------------------
    # 8. Optimizer
    # 中文：使用 AdamW 优化器，区分 backbone 和 classifier 的 weight decay。
    # ------------------------------------------------------------------
    lr = cfg["training"]["learning_rate"]
    weight_decay = cfg["training"]["weight_decay"]

    # backbone 参数加 weight_decay，bias 和 LayerNorm 不加
    no_decay = {"bias", "LayerNorm.weight", "layernorm"}
    decay_params = [
        p for n, p in model.named_parameters()
        if p.requires_grad and not any(nd in n for nd in no_decay)
    ]
    no_decay_params = [
        p for n, p in model.named_parameters()
        if p.requires_grad and any(nd in n for nd in no_decay)
    ]
    optimizer_groups = []
    if decay_params:
        optimizer_groups.append({"params": decay_params, "weight_decay": weight_decay})
    if no_decay_params:
        optimizer_groups.append({"params": no_decay_params, "weight_decay": 0.0})
    if not optimizer_groups:
        raise ValueError("No trainable parameters found for optimizer.")
    optimizer = AdamW(optimizer_groups, lr=lr)
    print(f"[Optimizer] AdamW lr={lr}, weight_decay={weight_decay}")

    # ------------------------------------------------------------------
    # 9. GradScaler for mixed precision
    # 中文：混合精度训练需要 GradScaler 来防止梯度下溢。
    #       bf16 动态范围与 fp32 相同，不需要 scaler，因此仅 fp16 时启用。
    # ------------------------------------------------------------------
    use_bf16 = fp16 and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    scaler = GradScaler(enabled=(fp16 and not use_bf16))
    print(f"[AMP] dtype={amp_dtype}, bf16={use_bf16}")

    # ------------------------------------------------------------------
    # 10. Training loop
    # 中文：主训练循环，每个 epoch 训练 + 验证，保存 best checkpoint，
    #       支持 early stopping。
    # ------------------------------------------------------------------
    epochs = cfg["training"]["epochs"]
    patience = cfg["training"]["early_stopping_patience"]
    metric_for_best = cfg["training"]["metric_for_best_model"]
    grad_accum_steps = cfg["training"].get("gradient_accumulation_steps", 1)

    best_metric = -float("inf")
    best_epoch = -1
    best_val_record = None
    epochs_no_improve = 0

    # 训练日志 CSV
    train_log_path = output_dir / "train_log.csv"
    with open(train_log_path, "w", newline="") as f:
        log_writer = csv.writer(f)
        log_writer.writerow([
            "epoch", "train_loss", "val_loss",
            "val_accuracy", "val_f1", "val_mcc", "val_auroc", "val_auprc",
        ])

    print(f"\n{'='*60}")
    print(f"Training started: {epochs} epochs, early stopping patience={patience}")
    print(f"Best model selected by val_{metric_for_best}")
    print(f"{'='*60}\n")

    t_start = time.time()

    for epoch in range(1, epochs + 1):
        # --- train ---
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, device, fp16, amp_dtype, grad_accum_steps)

        # --- validate ---
        val_loss, val_metrics = evaluate(model, val_loader, device, fp16, amp_dtype)

        # --- log ---
        current_metric = val_metrics[metric_for_best]
        log_row = [
            epoch, f"{train_loss:.4f}", f"{val_loss:.4f}",
            f"{val_metrics['accuracy']:.4f}", f"{val_metrics['f1']:.4f}",
            f"{val_metrics['mcc']:.4f}", f"{val_metrics['auroc']:.4f}",
            f"{val_metrics['auprc']:.4f}",
        ]
        with open(train_log_path, "a", newline="") as f:
            log_writer = csv.writer(f)
            log_writer.writerow(log_row)

        improved = current_metric > best_metric
        if improved:
            best_metric = current_metric
            best_epoch = epoch
            best_val_record = {
                "best_epoch": epoch,
                "loss": val_loss,
                **val_metrics,
            }
            epochs_no_improve = 0
            best_model_path = output_dir / "best_model.pt"
            torch.save(model.state_dict(), best_model_path)
        else:
            epochs_no_improve += 1

        # 每个 epoch 打印训练状态
        print(
            f"Epoch {epoch}/{epochs}  "
            f"train_loss={train_loss:.4f}  "
            f"val_loss={val_loss:.4f}  "
            f"val_f1={val_metrics['f1']:.4f}  "
            f"val_mcc={val_metrics['mcc']:.4f}  "
            f"val_auroc={val_metrics['auroc']:.4f}  "
            f"val_auprc={val_metrics['auprc']:.4f}  "
            f"best_{metric_for_best}={best_metric:.4f}"
            f"{'  *saved*' if improved else ''}"
        )

        # --- early stopping ---
        if not args.no_early_stop and epochs_no_improve >= patience:
            print(f"\n[EarlyStopping] No improvement for {patience} epochs. Stopping.")
            break

    elapsed = time.time() - t_start
    print(f"\nTraining finished in {elapsed:.1f}s. Best epoch={best_epoch}, best val_{metric_for_best}={best_metric:.4f}")

    # ------------------------------------------------------------------
    # 11. Save val metrics
    # 中文：将最佳验证集指标保存为 val_metrics.json。
    # ------------------------------------------------------------------
    if best_val_record is None:
        best_val_record = {
            "best_epoch": best_epoch,
            "loss": val_loss,
            **val_metrics,
        }
    val_metrics_path = output_dir / "val_metrics.json"
    with open(val_metrics_path, "w") as f:
        json.dump(best_val_record, f, indent=2)
    print(f"[Saved] {val_metrics_path}")

    if not run_test:
        print("[Evaluation] validation-only run complete; test set was not loaded.")
        print(f"\n{'='*60}")
        print(f"All done. Validation results saved to {output_dir}")
        print(f"{'='*60}")
        return

    # ------------------------------------------------------------------
    # 12. Load best checkpoint & test evaluation
    # 中文：加载 best checkpoint，在测试集上评估，保存结果。
    # ------------------------------------------------------------------
    best_model_path = output_dir / "best_model.pt"
    if best_model_path.exists():
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        print(f"[Loaded] best checkpoint from epoch {best_epoch}")
    else:
        print("[Warning] No best_model.pt found, using last epoch model for test.")

    test_loss, test_metrics = evaluate(model, test_loader, device, fp16, amp_dtype)
    print(f"\n--- Test Results ---")
    print(f"  accuracy = {test_metrics['accuracy']:.4f}")
    print(f"  f1       = {test_metrics['f1']:.4f}")
    print(f"  mcc      = {test_metrics['mcc']:.4f}")
    print(f"  auroc    = {test_metrics['auroc']:.4f}")
    print(f"  auprc    = {test_metrics['auprc']:.4f}")

    # ------------------------------------------------------------------
    # 13. Save test_metrics.json
    # 中文：保存测试集指标。
    # ------------------------------------------------------------------
    test_metrics_path = output_dir / "test_metrics.json"
    with open(test_metrics_path, "w") as f:
        json.dump(test_metrics, f, indent=2)
    print(f"[Saved] {test_metrics_path}")

    # ------------------------------------------------------------------
    # 14. Save test_predictions.csv
    # 中文：保存每条测试序列的预测结果。
    # ------------------------------------------------------------------
    pred_path = output_dir / "test_predictions.csv"
    all_seqs = []
    all_labels = []
    all_probs = []

    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            sequences = batch["sequences"]
            labels = batch["labels"].to(device)
            with _autocast(fp16, amp_dtype):
                outputs = model(sequences)
            # fp32 softmax 避免 NaN
            probs = torch.softmax(outputs["logits"].float(), dim=-1)
            all_seqs.extend(sequences)
            all_labels.extend(labels.cpu().tolist())
            all_probs.extend(probs.cpu().tolist())

    with open(pred_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sequence", "label", "prob_0", "prob_1", "pred"])
        for seq, label, prob in zip(all_seqs, all_labels, all_probs):
            pred = int(prob[1] >= 0.5)
            writer.writerow([seq, label, f"{prob[0]:.6f}", f"{prob[1]:.6f}", pred])

    print(f"[Saved] {pred_path}")
    print(f"\n{'='*60}")
    print(f"All done. Results saved to {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
