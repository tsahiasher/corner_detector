import time
import logging
import numpy as np
import torch
from typing import Dict, Any, List, Optional

class TrainingTracker:
    """Unified tracker for training/validation metrics, timing, and logging."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.epoch_start = 0.0
        self.train_start = 0.0
        self.val_start = 0.0
        self.train_time = 0.0
        self.val_time = 0.0
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        self.train_components: Dict[str, List[float]] = {}
        self.val_components: Dict[str, List[float]] = {}
        self.train_batch_times: List[float] = []
        self.val_batch_times: List[float] = []

    def start_epoch(self) -> None:
        self.epoch_start = time.time()
        self.train_losses = []
        self.val_losses = []
        self.train_components = {}
        self.val_components = {}
        self.train_batch_times = []
        self.val_batch_times = []

    def start_train_phase(self) -> None:
        self.train_start = time.time()

    def end_train_phase(self) -> None:
        self.train_time = time.time() - self.train_start

    def start_val_phase(self) -> None:
        self.val_start = time.time()

    def end_val_phase(self) -> None:
        self.val_time = time.time() - self.val_start

    def record_batch(self, phase: str, loss: float, batch_time: float, 
                     components: Optional[Dict[str, float]] = None) -> None:
        if phase == 'train':
            self.train_losses.append(loss)
            self.train_batch_times.append(batch_time)
            if components:
                for k, v in components.items():
                    if k not in self.train_components:
                        self.train_components[k] = []
                    self.train_components[k].append(float(v))
        else:
            self.val_losses.append(loss)
            self.val_batch_times.append(batch_time)
            if components:
                for k, v in components.items():
                    if k not in self.val_components:
                        self.val_components[k] = []
                    self.val_components[k].append(float(v))

    def log_epoch_summary(self, epoch: int, max_epochs: int, lr: float, 
                          metrics: Dict[str, Any], recall: Optional[Dict[str, Any]] = None) -> None:
        """Prints a rich, structured summary for the epoch."""
        total_time = time.time() - self.epoch_start
        avg_train_loss = np.mean(self.train_losses) if self.train_losses else 0.0
        avg_val_loss = np.mean(self.val_losses) if self.val_losses else 0.0
        avg_train_batch = np.mean(self.train_batch_times) if self.train_batch_times else 0.0
        avg_val_batch = np.mean(self.val_batch_times) if self.val_batch_times else 0.0

        self.logger.info("-" * 60)
        self.logger.info(f"EPOCH {epoch}/{max_epochs} SUMMARY")
        self.logger.info(f"Time: Total: {total_time:.1f}s | Train: {self.train_time:.1f}s | Val: {self.val_time:.1f}s")
        self.logger.info(f"Batch Avg: Train: {avg_train_batch:.3f}s | Val: {avg_val_batch:.3f}s")
        self.logger.info(f"Loss: Train: {avg_train_loss:.5f} | Val: {avg_val_loss:.5f} | LR: {lr:.6f}")
        
        # Loss Components (v7.0 boost)
        if self.train_components:
            self.logger.info("Loss Components (Avg Train):")
            sorted_keys = sorted(self.train_components.keys())
            for k in sorted_keys:
                avg_val = np.mean(self.train_components[k])
                self.logger.info(f"  {k:20s}: {avg_val:.5f}")
        
        if self.val_components:
            self.logger.info("Loss Components (Avg Val):")
            sorted_keys = sorted(self.val_components.keys())
            for k in sorted_keys:
                avg_val = np.mean(self.val_components[k])
                self.logger.info(f"  {k:20s}: {avg_val:.5f}")
        
        # Accuracy Metrics
        if metrics:
            if 'mean_iou' in metrics:
                # IoU-based metrics (Typical for BoundingBox / Global Regressor)
                me = metrics.get('mean_iou', 0.0)
                md = metrics.get('median_iou', 0.0)
                self.logger.info(f"IoU Metrics: Mean: {me:.4f} | Med: {md:.4f}")
                
                # Handling IoU thresholds (e.g. iou_05, iou_75)
                iou_keys = sorted([k for k in metrics.keys() if k.startswith('iou_')])
                items = []
                for k in iou_keys:
                    if k in ['mean_iou', 'median_iou']: continue
                    val = k[4:] # '05' -> '0.5'
                    label = f"0.{val}" if len(val) == 2 and val.isdigit() else val
                    if val == '05': label = '0.5'
                    if val == '75': label = '0.75'
                    items.append(f"IoU@{label}: {metrics[k]*100:.1f}%")
                
                if items:
                    self.logger.info(f"Precision: {' | '.join(items)}")
            
            elif 'mean' in metrics:
                # Pixel-based metrics (Typical for Refiner / Keypoints)
                me = metrics.get('mean', 0.0)
                md = metrics.get('median', 0.0)
                self.logger.info(f"Pixel Error: Mean: {me:.3f} px | Med: {md:.3f} px")
                
                # Per-corner if available
                if 'tl' in metrics:
                    self.logger.info(f"Per-Corner (Mean): TL: {metrics.get('tl',0):.2f} | TR: {metrics.get('tr',0):.2f} | BR: {metrics.get('br',0):.2f} | BL: {metrics.get('bl',0):.2f}")
                
                # Outliers
                p90 = metrics.get('p90', 0.0)
                p95 = metrics.get('p95', 0.0)
                mx = metrics.get('max', 0.0)
                if any(v > 0 for v in [p90, p95, mx]):
                    self.logger.info(f"Outliers: P90: {p90:.2f} | P95: {p95:.2f} | Max: {mx:.2f}")
                
                # Thresholds
                acc_keys = sorted([k for k in metrics.keys() if k.startswith('acc_')])
                if acc_keys:
                    acc_str = " | ".join([f"<{k[4:]}: {metrics[k]:.1f}%" for k in acc_keys])
                    self.logger.info(f"Precision: {acc_str}")

        # Patch Recall
        if recall:
            items = []
            for k in sorted(recall.keys()):
                val = recall[k]
                if k == 'recall' and val == 0.0: continue # Skip dummy zero
                if not isinstance(val, (int, float)) or val <= 0: continue # Skip empty/zero
                
                label = k[7:] if k.startswith('recall_') else k
                if not label.endswith('px') and label.isdigit():
                    label = f"{label}px"
                items.append(f"{label}: {val:.1f}%")
            
            if items:
                rec_str = " | ".join(items)
                self.logger.info(f"Patch Recall: {rec_str}")


class TopLossTracker:
    """Tracks the worst performing samples for visualization."""
    def __init__(self, k: int = 5):
        self.k = k
        self.top_samples: List[Dict[str, Any]] = []

    def update(self, loss: float, sample: Dict[str, Any]):
        """Detach and move all tensor data to CPU to prevent memory leaks."""
        cpu_sample = {}
        for k, v in sample.items():
            if isinstance(v, torch.Tensor):
                cpu_sample[k] = v.detach().cpu()
            else:
                cpu_sample[k] = v
        
        self.top_samples.append({'loss': float(loss), **cpu_sample})
        # Keep only top K, sorted by loss descending
        self.top_samples.sort(key=lambda x: x['loss'], reverse=True)
        if len(self.top_samples) > self.k:
            self.top_samples = self.top_samples[:self.k]

    def get_samples(self) -> List[Dict[str, Any]]:
        return self.top_samples


class HardExampleMiner:
    """Tracks per-sample errors and provides weights for hard-example oversampling."""
    def __init__(self, num_samples: int):
        self.num_samples = num_samples
        # Initialize errors with a small positive value (e.g., 5.0 px)
        self.errors = np.full(num_samples, 5.0, dtype=np.float32)
        self.counts = np.zeros(num_samples, dtype=np.int32)

    def update(self, indices: List[int], batch_errors: List[float]):
        """Updates error tracking for the given indices."""
        for idx, err in zip(indices, batch_errors):
            if idx < self.num_samples:
                # Running average of error or just latest? Latest is more sensitive to recent changes.
                self.errors[idx] = err
                self.counts[idx] += 1

    def get_weights(self, exponent: float = 1.5) -> torch.Tensor:
        """Returns normalized weights for WeightedRandomSampler.
        
        Weight = error ^ exponent. Higher exponent = more aggressive mining.
        """
        # Ensure errors are positive and handle outliers
        safe_errors = np.clip(self.errors, 1e-3, 100.0)
        weights = safe_errors ** exponent
        # Normalize
        weights = weights / (np.sum(weights) + 1e-8)
        return torch.from_numpy(weights).float()

    def state_dict(self) -> Dict[str, Any]:
        return {'errors': self.errors.tolist(), 'counts': self.counts.tolist()}

    def load_state_dict(self, state: Dict[str, Any]):
        if 'errors' in state:
            self.errors = np.array(state['errors'], dtype=np.float32)
        if 'counts' in state:
            self.counts = np.array(state['counts'], dtype=np.int32)
