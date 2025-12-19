#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Streaming model training for Overnight Finance Challenge.
Features computed row-by-row (no future leakage) via StatefulFeatureBuilder.

Pipeline:
- Ensemble: XGBoost + CatBoost + LightGBM (soft blend + stacking)
- Rare Class Detector (binary classifier: common vs rare)
- RareClassBooster for rare class probability correction
- DiagonalTemperatureCalibrator with per-class temperature
- OVR Threshold Optimizer
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import logging
import os
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import optuna
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from scipy.interpolate import Akima1DInterpolator
from scipy.special import softmax
from sklearn.metrics import f1_score, log_loss, confusion_matrix, roc_auc_score
from sklearn.preprocessing import KBinsDiscretizer
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
warnings.filterwarnings("ignore", message="X does not have valid feature names")

from features import StatefulFeatureBuilder, feature_order_from_dict


# ===================== Logging Setup =====================
def setup_logging(output_dir: Path, verbose: bool = True):
    """Setup logging to file and console."""
    log_file = output_dir / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO if verbose else logging.WARNING)
    
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return log_file


def log_step(step_name: str, step_num: int = None, total_steps: int = None):
    """Log step header."""
    if step_num and total_steps:
        header = f"STEP {step_num}/{total_steps}: {step_name}"
    else:
        header = f"STEP: {step_name}"
    logging.info("=" * 70)
    logging.info(header)
    logging.info("=" * 70)


def log_metrics(metrics: dict, prefix: str = ""):
    """Log metrics dictionary."""
    for key, value in metrics.items():
        if isinstance(value, float):
            logging.info(f"{prefix}{key}: {value:.6f}")
        elif isinstance(value, (list, np.ndarray)):
            if len(value) <= 10:
                logging.info(f"{prefix}{key}: {value}")
            else:
                logging.info(f"{prefix}{key}: [{value[0]:.4f}, ..., {value[-1]:.4f}] (len={len(value)})")
        else:
            logging.info(f"{prefix}{key}: {value}")


def log_class_distribution(y: np.ndarray, name: str = ""):
    """Log class distribution with visual bars."""
    n_classes = int(y.max() + 1)
    counts = np.bincount(y, minlength=n_classes)
    total = len(y)
    logging.info(f"Class distribution ({name}, n={total}):")
    for c in range(n_classes):
        pct = 100 * counts[c] / total
        bar = "█" * int(pct / 2)
        logging.info(f"  Class {c}: {counts[c]:7d} ({pct:5.2f}%) {bar}")


def log_model_comparison(results: list):
    """Log model comparison table."""
    logging.info("-" * 50)
    logging.info("MODEL COMPARISON (F1-macro):")
    logging.info("-" * 50)
    sorted_results = sorted(results, key=lambda x: x.get('f1_macro', 0), reverse=True)
    for i, r in enumerate(sorted_results):
        rank = "1st" if i == 0 else "2nd" if i == 1 else "3rd" if i == 2 else "   "
        name = r.get('name', 'Unknown')
        f1 = r.get('f1_macro', 0)
        time_s = r.get('training_time', 0)
        logging.info(f"{rank} {name:20s}: F1={f1:.4f}, Time={time_s:.1f}s")
    logging.info("-" * 50)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ensemble model for Overnight Finance Challenge")
    parser.add_argument("--train-path", type=Path, default=Path("../data/train.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("./"))
    parser.add_argument("--val-fraction", type=float, default=0.1, help="Validation set fraction")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-gpu", action="store_true", help="Enable GPU (CUDA/OpenCL)")
    parser.add_argument("--n-estimators", type=int, default=1000, help="Max boosting rounds")
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--xgb-weight", type=float, default=0.4, help="XGBoost weight in ensemble")
    parser.add_argument("--cat-weight", type=float, default=0.3, help="CatBoost weight in ensemble")
    parser.add_argument("--lgb-weight", type=float, default=0.3, help="LightGBM weight in ensemble")
    parser.add_argument("--optuna-trials", type=int, default=60, help="Optuna HPO trials (0=disable)")
    parser.add_argument("--oversample", action="store_true", default=True, help="Enable SMOTE for rare classes")
    parser.add_argument("--no-oversample", dest="oversample", action="store_false")
    parser.add_argument("--oversample-ratio", type=float, default=0.15, help="Target ratio for rare classes")
    parser.add_argument("--downsample", action="store_true", default=True, help="Enable downsampling of majority class")
    parser.add_argument("--no-downsample", dest="downsample", action="store_false")
    parser.add_argument("--downsample-ratio", type=float, default=0.3, help="Downsample ratio for class 0")
    parser.add_argument("--focal-loss", action="store_true", help="Use Focal Loss for XGBoost")
    parser.add_argument("--focal-gamma", type=float, default=2.0, help="Focal Loss gamma parameter")
    return parser.parse_args()


def stream_build_features(csv_path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Stream-read CSV, build features row-by-row, return X, y, feature_cols."""
    builder = StatefulFeatureBuilder()
    X_rows = []
    y_rows = []
    feature_cols: list[str] = []

    print(f"[train] Stream reading CSV: {csv_path}")
    with csv_path.open("r") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            numeric_row = {k: float(v) for k, v in row.items() if k not in ("y",)}
            feats = builder.build_row(numeric_row)
            if not feature_cols:
                feature_cols = feature_order_from_dict(feats)
            X_rows.append([feats[c] for c in feature_cols])
            y_rows.append(int(row["y"]))
            if (idx + 1) % 200000 == 0:
                print(f"[train] Processed {idx + 1} rows...")

    X = np.array(X_rows, dtype=np.float32)
    y = np.array(y_rows, dtype=np.int32)
    
    # Free memory
    del X_rows, y_rows
    gc.collect()
    
    return X, y, feature_cols


# ===================== Custom Focal Loss for XGBoost =====================
def softmax_numpy(x):
    """Softmax for multiclass classification."""
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)


def focal_loss_multiclass(y_true, y_pred_raw, gamma=2.0, alpha=None, n_classes=5):
    """
    Focal Loss for multiclass classification.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
    Args:
        y_true: true labels
        y_pred_raw: raw logits (n_samples * n_classes)
        gamma: focusing parameter (typically 2.0)
        alpha: class weights (None = equal)
        n_classes: number of classes
    """
    n_samples = len(y_true)
    y_pred_raw = y_pred_raw.reshape(n_samples, n_classes)
    
    # Softmax
    probs = softmax_numpy(y_pred_raw)
    probs = np.clip(probs, 1e-7, 1 - 1e-7)
    
    # One-hot
    y_onehot = np.zeros((n_samples, n_classes))
    y_onehot[np.arange(n_samples), y_true.astype(int)] = 1
    
    # p_t = probability of correct class
    pt = np.sum(probs * y_onehot, axis=1, keepdims=True)
    
    # Focal weight: (1 - p_t)^gamma
    focal_weight = (1 - pt) ** gamma
    
    # Alpha weights (inverse frequency weighting)
    if alpha is None:
        class_counts = np.bincount(y_true.astype(int), minlength=n_classes)
        alpha = n_samples / (n_classes * class_counts + 1)
        # Extra boost for rare classes 3 and 4
        alpha[3] *= 2.0
        alpha[4] *= 2.0
    
    alpha_t = alpha[y_true.astype(int)].reshape(-1, 1)
    
    # Gradient: focal_weight * alpha * (probs - y_onehot)
    grad = focal_weight * alpha_t * (probs - y_onehot)
    
    # Hessian: simplified for stability
    hess = focal_weight * alpha_t * probs * (1 - probs)
    hess = np.clip(hess, 1e-7, None)
    
    # XGBoost >= 2.1 requires shape (n_samples, n_classes)
    return grad, hess


def f1_macro_objective(y_true, y_pred_raw, n_classes=5):
    """Custom objective for XGBoost approximating F1-macro with focal-like weighting."""
    n_samples = len(y_true)
    y_pred_raw = y_pred_raw.reshape(n_samples, n_classes)
    
    # Softmax to get probabilities
    probs = softmax_numpy(y_pred_raw)
    probs = np.clip(probs, 1e-7, 1 - 1e-7)
    
    # One-hot encoding
    y_onehot = np.zeros((n_samples, n_classes))
    y_onehot[np.arange(n_samples), y_true.astype(int)] = 1
    
    # Weights for rare classes (3 and 4)
    class_counts = np.bincount(y_true.astype(int), minlength=n_classes)
    class_weights = n_samples / (n_classes * class_counts + 1)
    # Boost weights for rare classes
    class_weights[3] *= 3.0
    class_weights[4] *= 3.0
    
    # Focal-like weighting
    gamma = 2.0
    pt = np.sum(probs * y_onehot, axis=1)
    focal_weight = (1 - pt) ** gamma
    
    # Gradient: (probs - y_onehot) * weights
    sample_weights = class_weights[y_true.astype(int)] * focal_weight
    grad = (probs - y_onehot) * sample_weights.reshape(-1, 1)
    
    # Hessian: probs * (1 - probs) * weights
    hess = probs * (1 - probs) * sample_weights.reshape(-1, 1)
    hess = np.clip(hess, 1e-7, None)
    
    # XGBoost >= 2.1 requires shape (n_samples, n_classes)
    return grad, hess


def f1_macro_eval(y_true, y_pred_raw, n_classes=5):
    """F1-macro evaluation metric for XGBoost."""
    n_samples = len(y_true)
    y_pred_raw = y_pred_raw.reshape(n_samples, n_classes)
    y_pred = np.argmax(y_pred_raw, axis=1)
    score = f1_score(y_true, y_pred, average='macro')
    return 'f1_macro', -score  # Negative because XGBoost minimizes


# ===================== TargetExpandingMeanEncoder =====================
class TargetExpandingMeanEncoder:
    """
    Target Encoding using strictly past values (expanding mean).
    Bins features and computes target mean per bin using only past data.
    """
    def __init__(self, smoothing: float = 10.0, n_bins: int = 10):
        self.smoothing = smoothing
        self.n_bins = n_bins
        self.bin_edges = {}
        self.global_mean = None
        self.encoding_maps = {}
    
    def fit_transform(self, X: np.ndarray, y: np.ndarray, feature_names: list, 
                      features_to_encode: list) -> tuple:
        """
        Fit and transform with expanding mean (strictly past data).
        
        Returns:
            X_new: array with new features
            new_feature_names: names of new features
        """
        self.global_mean = float(np.mean(y))
        new_features = []
        new_names = []
        
        for feat_name in features_to_encode:
            if feat_name not in feature_names:
                continue
            
            feat_idx = feature_names.index(feat_name)
            feat_values = X[:, feat_idx]
            
            # Bin the feature
            valid_mask = ~np.isnan(feat_values)
            if valid_mask.sum() < 100:
                continue
            
            percentiles = np.linspace(0, 100, self.n_bins + 1)
            edges = np.percentile(feat_values[valid_mask], percentiles)
            edges = np.unique(edges)
            if len(edges) < 3:
                continue
            self.bin_edges[feat_name] = edges
            
            # Convert to bins
            bins = np.digitize(feat_values, edges[1:-1])
            
            # Expanding mean encoding (strictly past data)
            encoded = np.zeros(len(y), dtype=np.float32)
            bin_sums = {}
            bin_counts = {}
            
            for i in range(len(y)):
                b = int(bins[i])
                # Use statistics up to current point
                if b in bin_counts and bin_counts[b] > 0:
                    prior_mean = bin_sums[b] / bin_counts[b]
                    n = bin_counts[b]
                    encoded[i] = (n * prior_mean + self.smoothing * self.global_mean) / (n + self.smoothing)
                else:
                    encoded[i] = self.global_mean
                
                # Update statistics AFTER computing
                if b not in bin_sums:
                    bin_sums[b] = 0.0
                    bin_counts[b] = 0
                bin_sums[b] += y[i]
                bin_counts[b] += 1
            
            # Save final statistics for inference
            self.encoding_maps[feat_name] = {}
            for b in bin_counts:
                if bin_counts[b] > 0:
                    mean_val = bin_sums[b] / bin_counts[b]
                    n = bin_counts[b]
                    self.encoding_maps[feat_name][int(b)] = (n * mean_val + self.smoothing * self.global_mean) / (n + self.smoothing)
                else:
                    self.encoding_maps[feat_name][int(b)] = self.global_mean
            
            new_features.append(encoded)
            new_names.append(f"{feat_name}_te")
            print(f"[TargetEnc] {feat_name}: {len(np.unique(bins))} bins, mean={np.mean(encoded):.4f}")
        
        if new_features:
            X_new = np.column_stack([X] + new_features)
            new_feature_names = feature_names + new_names
        else:
            X_new = X
            new_feature_names = feature_names
        
        return X_new, new_feature_names
    
    def get_state(self) -> dict:
        """Save state for inference."""
        return {
            'smoothing': self.smoothing,
            'n_bins': self.n_bins,
            'global_mean': self.global_mean,
            'bin_edges': {k: v.tolist() for k, v in self.bin_edges.items()},
            'encoding_maps': {k: {str(kk): vv for kk, vv in v.items()} for k, v in self.encoding_maps.items()}
        }


# ===================== RareClassBooster =====================
class RareClassBooster:
    """
    Probability calibrator for improving rare class classification.
    Uses signal from binary classifier (common vs rare).
    """
    
    def __init__(self, rare_classes=[3, 4], common_classes=[0, 1, 2]):
        self.rare_classes = rare_classes
        self.common_classes = common_classes
        self.alpha = None  # weight of model2 signal
        self.beta = None   # boost factor for rare classes
        
    def fit(self, y_true, probs_model1, probs_model2, 
            alpha_range=(0.01, 50), beta_range=(0.01, 50), 
            n_trials=200):
        """
        Find optimal parameters with Optuna.
        
        probs_model1: (n_samples, 5) - main model probabilities
        probs_model2: (n_samples, 2) - binary probs [common, rare]
        """
        
        def objective(trial):
            alpha = trial.suggest_float('alpha', alpha_range[0], alpha_range[1], log=True)
            beta = trial.suggest_float('beta', beta_range[0], beta_range[1], log=True)
            
            adjusted_probs = self._adjust_probs(probs_model1, probs_model2, alpha, beta)
            y_pred = np.argmax(adjusted_probs, axis=1)
            return f1_score(y_true, y_pred, average='macro')
        
        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=42)
        )
        
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        self.alpha = study.best_params['alpha']
        self.beta = study.best_params['beta']
        
        print(f"[RareBoost] Best params: alpha={self.alpha:.4f}, beta={self.beta:.4f}")
        print(f"[RareBoost] F1-macro: {study.best_value:.4f}")
        
        return self
    
    def _adjust_probs(self, probs_model1, probs_model2, alpha, beta):
        """Adjust probabilities using rare class signal."""
        adjusted = probs_model1.copy()
        
        # Probability that sample is rare class (from model2)
        rare_signal = probs_model2[:, 1].reshape(-1, 1)
        
        # Mask for rare classes
        rare_mask = np.zeros_like(adjusted)
        for rare_cls in self.rare_classes:
            rare_mask[:, rare_cls] = 1
        
        # Boost rare classes based on model2 signal
        boost_factor = 1 + alpha * rare_signal * rare_mask
        adjusted = adjusted * boost_factor
        
        # Additional boost for rare classes
        for rare_cls in self.rare_classes:
            adjusted[:, rare_cls] *= beta
        
        # Normalize
        adjusted = adjusted / adjusted.sum(axis=1, keepdims=True)
        return adjusted
    
    def predict_proba(self, probs_model1, probs_model2):
        if self.alpha is None or self.beta is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self._adjust_probs(probs_model1, probs_model2, self.alpha, self.beta)
    
    def predict(self, probs_model1, probs_model2):
        probs = self.predict_proba(probs_model1, probs_model2)
        return np.argmax(probs, axis=1)


# ===================== DiagonalTemperatureCalibrator =====================
class DiagonalTemperatureCalibrator:
    """Temperature calibration with per-class temperatures."""

    def __init__(self, n_trials: int = 100):
        self.temps = None
        self.thresholds = None
        self.n_classes = None
        self.n_trials = n_trials

    def fit(self, probs: np.ndarray, y_true: np.ndarray):
        self.n_classes = probs.shape[1]
        log_probs = np.log(np.clip(probs, 1e-15, 1.0))

        best_temps = np.ones(self.n_classes)
        best_score = -1
        
        print("[DiagTemp] Optimizing per-class temperatures...")
        for _ in range(self.n_trials):
            temps = np.random.uniform(0.5, 5.0, self.n_classes)
            scaled_logits = log_probs / temps
            scaled_probs = softmax(scaled_logits, axis=1)
            preds = np.argmax(scaled_probs, axis=1)
            score = f1_score(y_true, preds, average='macro')
            if score > best_score:
                best_score = score
                best_temps = temps.copy()
        
        self.temps = best_temps
        print(f"[DiagTemp] Best temps: {self.temps}, F1={best_score:.4f}")

        calibrated = self.predict_proba(probs)
        self.thresholds = self._optimize_thresholds(calibrated, y_true)
        return self

    def _optimize_thresholds(self, probs: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        print("[DiagTemp] Optimizing thresholds...")
        best_thresholds = np.zeros(self.n_classes)
        best_f1 = f1_score(y_true, probs.argmax(axis=1), average='macro')
        
        for _ in range(self.n_trials):
            thresholds = np.random.uniform(-0.3, 0.3, self.n_classes)
            adjusted = probs - thresholds
            preds = np.argmax(adjusted, axis=1)
            score = f1_score(y_true, preds, average='macro')
            if score > best_f1:
                best_f1 = score
                best_thresholds = thresholds.copy()
        
        print(f"[DiagTemp] Best thresholds: {best_thresholds}, F1={best_f1:.4f}")
        return best_thresholds

    def predict_proba(self, probs: np.ndarray) -> np.ndarray:
        log_probs = np.log(np.clip(probs, 1e-15, 1.0))
        scaled_logits = log_probs / self.temps
        return softmax(scaled_logits, axis=1)

    def predict(self, probs: np.ndarray) -> np.ndarray:
        calibrated = self.predict_proba(probs)
        adjusted = calibrated - self.thresholds
        return np.argmax(adjusted, axis=1)


# ===================== ThresholdOptimizerOVR =====================
class ThresholdOptimizerOVR:
    """One-vs-Rest threshold optimization for F1-macro."""

    def __init__(self, n_steps: int = 100, n_iters: int = 5):
        self.n_steps = n_steps
        self.n_iters = n_iters
        self.thresholds = None

    def fit(self, probs: np.ndarray, y_true: np.ndarray):
        n_classes = probs.shape[1]
        self.thresholds = np.full(n_classes, 1.0 / n_classes)
        best_f1 = f1_score(y_true, np.argmax(probs, axis=1), average='macro')
        
        print(f"[OVR] Initial F1={best_f1:.4f}, optimizing thresholds...")
        
        for iteration in range(self.n_iters):
            improved = False
            for c in range(n_classes):
                min_prob = probs[:, c].min()
                max_prob = probs[:, c].max()
                threshold_range = np.linspace(min_prob, max_prob, self.n_steps)
                
                best_t_for_c = self.thresholds[c]
                
                for t in threshold_range:
                    temp_thresholds = self.thresholds.copy()
                    temp_thresholds[c] = t
                    preds = self._predict_with_thresholds(probs, temp_thresholds)
                    f1 = f1_score(y_true, preds, average='macro')
                    
                    if f1 > best_f1:
                        best_f1 = f1
                        best_t_for_c = t
                        improved = True
                
                self.thresholds[c] = best_t_for_c
            
            if not improved:
                break
        
        print(f"[OVR] Final F1={best_f1:.4f}, thresholds={self.thresholds}")
        return self

    def _predict_with_thresholds(self, probs: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
        adjusted = probs.copy()
        for c in range(len(thresholds)):
            adjusted[:, c] = np.where(probs[:, c] >= thresholds[c], probs[:, c], -np.inf)
        mask = np.all(adjusted == -np.inf, axis=1)
        adjusted[mask] = probs[mask]
        return np.argmax(adjusted, axis=1)

    def predict(self, probs: np.ndarray) -> np.ndarray:
        return self._predict_with_thresholds(probs, self.thresholds)


# ===================== AKIMA Calibration =====================
def akima_calibration(probs: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """AKIMA interpolation-based calibration per class."""
    calibrated = np.zeros_like(probs)
    n_classes = probs.shape[1]
    for c in range(n_classes):
        true = (y_true == c).astype(int)
        bins = np.linspace(0, 1, 15)
        bin_ids = np.digitize(probs[:, c], bins) - 1
        bin_centers = []
        frac_pos = []
        for b in range(len(bins) - 1):
            mask = bin_ids == b
            if mask.sum() == 0:
                continue
            bin_centers.append((bins[b] + bins[b + 1]) / 2)
            frac_pos.append(true[mask].mean())
        if len(bin_centers) < 4:
            calibrated[:, c] = probs[:, c]
            continue
        ak = Akima1DInterpolator(bin_centers, frac_pos)
        interp_vals = ak(probs[:, c])
        interp_vals = np.nan_to_num(interp_vals, nan=probs[:, c])
        calibrated[:, c] = np.clip(interp_vals, 0.0, 1.0)
    
    calibrated = np.nan_to_num(calibrated, nan=1e-15, posinf=1.0, neginf=1e-15)
    calibrated = np.clip(calibrated, 1e-15, 1.0)
    row_sums = calibrated.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1.0, row_sums)
    return calibrated / row_sums


def clean_probs(probs: np.ndarray, n_classes: int) -> np.ndarray:
    """Clean and normalize probabilities."""
    probs = np.asarray(probs, dtype=np.float64)
    probs = probs.reshape(-1, n_classes)
    probs = np.nan_to_num(probs, nan=1e-15, posinf=1.0, neginf=1e-15)
    probs = np.clip(probs, 1e-15, 1.0)
    row_sums = probs.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1.0, row_sums)
    return probs / row_sums


def resample_classes(X: np.ndarray, y: np.ndarray, 
                     rare_classes: list = [3, 4],
                     target_ratio: float = 0.15,
                     downsample_majority: bool = True,
                     downsample_ratio: float = 0.3,
                     seed: int = 42) -> tuple:
    """
    Combined resampling: Downsampling + SMOTE.
    
    1. Downsample majority class (0) to downsample_ratio of max
    2. SMOTE for rare classes (3, 4) up to target_ratio of max
    
    CRITICAL: After resampling, calibration must use ONLY original data!
    
    Args:
        X: features
        y: target
        rare_classes: classes to oversample
        target_ratio: target ratio of rare classes to majority
        downsample_majority: whether to downsample class 0
        downsample_ratio: fraction to reduce class 0 to
        seed: random seed
        
    Returns:
        X_aug, y_aug: augmented data
        n_original: original dataset size (for calibration!)
    """
    n_original = len(y)
    class_counts = np.bincount(y)
    n_classes = len(class_counts)
    
    print(f"[Resample] Original class distribution: {class_counts}")
    
    X_work, y_work = X, y
    
    # ===== STEP 1: Downsample majority class =====
    if downsample_majority and class_counts[0] > class_counts[1:].max() * 2:
        # Target count for class 0: mean of classes 1,2
        target_majority = int(np.mean(class_counts[1:3]) * (1 + downsample_ratio))
        
        undersample_strategy = {0: target_majority}
        print(f"[Resample] Downsampling Class 0: {class_counts[0]} -> {target_majority}")
        
        rus = RandomUnderSampler(
            sampling_strategy=undersample_strategy,
            random_state=seed
        )
        X_work, y_work = rus.fit_resample(X_work, y_work)
        
        class_counts_after_down = np.bincount(y_work)
        print(f"[Resample] After downsampling: {class_counts_after_down}")
    
    # ===== STEP 2: SMOTE for rare classes =====
    class_counts_current = np.bincount(y_work)
    max_count = class_counts_current.max()
    target_count = int(max_count * target_ratio)
    
    # Strategy: increase only rare classes
    sampling_strategy = {}
    for cls in rare_classes:
        if cls < n_classes and class_counts_current[cls] < target_count:
            sampling_strategy[cls] = max(target_count, class_counts_current[cls] * 3)  # min x3
    
    if sampling_strategy:
        print(f"[Resample] SMOTE strategy: {sampling_strategy}")
        
        try:
            # Try SMOTE first
            smote = SMOTE(
                sampling_strategy=sampling_strategy,
                random_state=seed,
                k_neighbors=min(5, min(class_counts_current[list(sampling_strategy.keys())]) - 1)
            )
            X_aug, y_aug = smote.fit_resample(X_work, y_work)
            print(f"[Resample] SMOTE: {len(y_work)} -> {len(y_aug)} samples")
        except ValueError as e:
            # Fallback to RandomOverSampler if SMOTE fails
            print(f"[Resample] SMOTE failed ({e}), using RandomOverSampler")
            ros = RandomOverSampler(sampling_strategy=sampling_strategy, random_state=seed)
            X_aug, y_aug = ros.fit_resample(X_work, y_work)
            print(f"[Resample] RandomOverSampler: {len(y_work)} -> {len(y_aug)} samples")
    else:
        X_aug, y_aug = X_work, y_work
        print("[Resample] No oversampling needed")
    
    # Free intermediate memory
    del X_work, y_work
    gc.collect()
    
    aug_counts = np.bincount(y_aug)
    print(f"[Resample] Augmented class distribution: {aug_counts}")
    
    return X_aug, y_aug, n_original


def optuna_xgb_objective(trial, X_train, y_train, X_val, y_val, use_gpu, n_classes, seed):
    """Optuna objective for XGBoost hyperparameter optimization.
    
    Note: We don't use XGBoostPruningCallback because:
    - Study direction is 'maximize' (F1)
    - XGBoost monitors 'mlogloss' which should be minimized
    - This causes direction conflict
    
    Instead, we use early_stopping_rounds which effectively prunes bad trials.
    """
    params = {
        "objective": "multi:softprob",
        "num_class": n_classes,
        "tree_method": "hist",
        "eval_metric": "mlogloss",
        "random_state": seed,
        "early_stopping_rounds": 50,
        "n_estimators": 500,  # Reduced for HPO speed
        
        # Optimized parameters
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "min_child_weight": trial.suggest_float("min_child_weight", 1, 50, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "gamma": trial.suggest_float("gamma", 1e-4, 1.0, log=True),
        "max_delta_step": trial.suggest_int("max_delta_step", 0, 5),
    }
    
    if use_gpu:
        params["device"] = "cuda"
    else:
        params["device"] = "cpu"
        params["n_jobs"] = os.cpu_count() or -1
    
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    
    # Predictions
    probs = model.predict_proba(X_val)
    preds = probs.argmax(axis=1)
    f1 = f1_score(y_val, preds, average='macro')
    
    return f1


def optuna_catboost_objective(trial, X_train, y_train, X_val, y_val, use_gpu, n_classes, seed):
    """Optuna objective for CatBoost hyperparameter optimization.
    
    Note: We don't use CatBoostPruningCallback because:
    - Study direction is 'maximize' (F1)
    - CatBoost monitors 'MultiClass' loss which should be minimized
    - This causes direction conflict
    
    Instead, we use od_type='Iter' with od_wait which effectively prunes bad trials.
    """
    params = {
        "iterations": 300,  # Reduced for HPO speed
        "random_seed": seed,
        "loss_function": "MultiClass",
        "eval_metric": "MultiClass",
        "od_type": "Iter",  # Overfitting detector acts as implicit pruning
        "od_wait": 50,
        "verbose": 0,
        
        # Optimized parameters
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "depth": trial.suggest_int("depth", 4, 10),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 50.0, log=True),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
        "random_strength": trial.suggest_float("random_strength", 0.0, 10.0),
        "border_count": trial.suggest_int("border_count", 32, 255),
    }
    
    if use_gpu:
        params["task_type"] = "GPU"
        params["devices"] = "0"
    else:
        params["task_type"] = "CPU"
        params["thread_count"] = os.cpu_count() or -1
    
    model = CatBoostClassifier(**params)
    model.fit(
        X_train, y_train, 
        eval_set=(X_val, y_val), 
        verbose=0
    )
    
    probs = model.predict_proba(X_val)
    preds = probs.argmax(axis=1)
    f1 = f1_score(y_val, preds, average='macro')
    
    return f1


def optuna_lgb_objective(trial, X_train, y_train, X_val, y_val, use_gpu, n_classes, seed):
    """Optuna objective for LightGBM hyperparameter optimization.
    
    Note: We don't use LightGBMPruningCallback because:
    - Study direction is 'maximize' (F1)
    - LightGBM monitors 'multi_logloss' which should be minimized
    - This causes direction conflict
    
    Instead, we use early_stopping which effectively prunes bad trials.
    """
    max_depth = trial.suggest_int("max_depth", 4, 10)
    # num_leaves should be <= 2^max_depth for stability
    max_leaves = min(127, 2 ** max_depth - 1)
    num_leaves = trial.suggest_int("num_leaves", 15, max_leaves)
    
    params = {
        "objective": "multiclass",
        "num_class": n_classes,
        "n_estimators": 300,  # Reduced for HPO speed
        "random_state": seed,
        "verbose": -1,
        "max_depth": max_depth,
        "num_leaves": num_leaves,
        
        # Optimized parameters
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 50, 200),  # Increased for GPU stability
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 100, 500),  # Critical for GPU
        "subsample": trial.suggest_float("subsample", 0.7, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-2, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 10.0, log=True),
    }
    
    # LightGBM: GPU (OpenCL) or CPU
    if use_gpu:
        params["device"] = "gpu"
        params["gpu_use_dp"] = False
    else:
        params["device"] = "cpu"
        params["n_jobs"] = os.cpu_count() or -1
    
    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False)
        ]
    )
    
    probs = model.predict_proba(X_val)
    preds = probs.argmax(axis=1)
    f1 = f1_score(y_val, preds, average='macro')
    
    return f1


def run_optuna_hpo(X_train, y_train, X_val, y_val, use_gpu, n_classes, seed, n_trials):
    """Run Optuna HPO for all boosting models with pruning and error handling."""
    import warnings
    # Suppress Optuna experimental warnings (multivariate TPE is stable in practice)
    warnings.filterwarnings("ignore", message=".*multivariate.*")
    warnings.filterwarnings("ignore", message=".*ExperimentalWarning.*")
    # Suppress XGBoost device mismatch warning (predict uses CPU which is fine)
    warnings.filterwarnings("ignore", message=".*Falling back to prediction using DMatrix.*")
    
    best_params = {"xgb": {}, "cat": {}, "lgb": {}}
    trials_per_model = max(10, n_trials // 3)  # Split trials between models
    
    # Optimized sampler settings
    sampler_kwargs = {
        "seed": seed,
        "n_startup_trials": 10,  # Random trials before TPE
        "multivariate": True,    # Better correlated parameter search
        "warn_independent_sampling": False,  # Suppress warnings for dynamic search spaces (num_leaves depends on max_depth)
    }
    
    # Optimized pruner: prune unpromising trials early
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=5,      # Trials before pruning starts
        n_warmup_steps=20,       # Steps before pruning within trial
        interval_steps=10,       # Check every N steps
    )
    
    # LightGBM HPO (first to test GPU)
    print(f"\n{'='*70}")
    print(f"[Optuna LGB] Starting HPO with {trials_per_model} trials (GPU test)...")
    print(f"{'='*70}")
    
    study_lgb = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(**sampler_kwargs),
        pruner=pruner
    )
    study_lgb.optimize(
        lambda trial: optuna_lgb_objective(trial, X_train, y_train, X_val, y_val, use_gpu, n_classes, seed),
        n_trials=trials_per_model, 
        show_progress_bar=True, 
        n_jobs=1,
        catch=(Exception,),  # Handle crashes gracefully
        gc_after_trial=True  # Memory cleanup
    )
    best_params["lgb"] = study_lgb.best_params if study_lgb.best_trial else {}
    lgb_best = study_lgb.best_value if study_lgb.best_trial else 0.0
    print(f"[Optuna LGB] Best F1: {lgb_best:.4f}")
    
    # Memory cleanup
    gc.collect()
    
    # XGBoost HPO
    print(f"\n{'='*70}")
    print(f"[Optuna XGB] Starting HPO with {trials_per_model} trials...")
    print(f"{'='*70}")
    
    sampler_kwargs["seed"] = seed + 1
    study_xgb = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(**sampler_kwargs),
        pruner=pruner
    )
    study_xgb.optimize(
        lambda trial: optuna_xgb_objective(trial, X_train, y_train, X_val, y_val, use_gpu, n_classes, seed),
        n_trials=trials_per_model, 
        show_progress_bar=True, 
        n_jobs=1,
        catch=(Exception,),
        gc_after_trial=True
    )
    best_params["xgb"] = study_xgb.best_params if study_xgb.best_trial else {}
    xgb_best = study_xgb.best_value if study_xgb.best_trial else 0.0
    print(f"[Optuna XGB] Best F1: {xgb_best:.4f}")
    
    # Memory cleanup
    gc.collect()
    
    # CatBoost HPO
    print(f"\n{'='*70}")
    print(f"[Optuna CAT] Starting HPO with {trials_per_model} trials...")
    print(f"{'='*70}")
    
    sampler_kwargs["seed"] = seed + 2
    study_cat = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(**sampler_kwargs),
        pruner=pruner
    )
    study_cat.optimize(
        lambda trial: optuna_catboost_objective(trial, X_train, y_train, X_val, y_val, use_gpu, n_classes, seed),
        n_trials=trials_per_model, 
        show_progress_bar=True, 
        n_jobs=1,
        catch=(Exception,),
        gc_after_trial=True
    )
    best_params["cat"] = study_cat.best_params if study_cat.best_trial else {}
    cat_best = study_cat.best_value if study_cat.best_trial else 0.0
    print(f"[Optuna CAT] Best F1: {cat_best:.4f}")
    
    # Final cleanup
    gc.collect()
    
    print("\n" + "="*70)
    print("[Optuna] HPO Complete!")
    print("="*70)
    print(f"XGB Best F1: {xgb_best:.4f}")
    print(f"CAT Best F1: {cat_best:.4f}")
    print(f"LGB Best F1: {lgb_best:.4f}")
    
    return best_params


def evaluate_model(name: str, probs: np.ndarray, y_true: np.ndarray, n_classes: int) -> dict:
    """Detailed model evaluation with logging."""
    preds = probs.argmax(axis=1)
    
    # Metrics
    f1_macro = f1_score(y_true, preds, average='macro')
    f1_weighted = f1_score(y_true, preds, average='weighted')
    f1_per_class = f1_score(y_true, preds, average=None)
    logloss = log_loss(y_true, probs)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, preds)
    
    # Prediction distribution
    pred_dist = np.bincount(preds, minlength=n_classes)
    true_dist = np.bincount(y_true, minlength=n_classes)
    
    print("\n" + "="*60)
    print(f"[{name}] DETAILED EVALUATION")
    print("="*60)
    print(f"F1-macro:    {f1_macro:.4f}")
    print(f"F1-weighted: {f1_weighted:.4f}")
    print(f"Log-loss:    {logloss:.4f}")
    print("\nF1 per class:")
    for i, f1 in enumerate(f1_per_class):
        print(f"  Class {i}: F1={f1:.4f}, True={true_dist[i]:6d}, Pred={pred_dist[i]:6d}")
    print("\nConfusion Matrix:")
    print(cm)
    print("="*60 + "\n")
    
    return {
        "name": name,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "f1_per_class": f1_per_class.tolist(),
        "logloss": logloss,
        "pred_distribution": pred_dist.tolist(),
        "true_distribution": true_dist.tolist(),
    }


def train():
    import warnings
    # Suppress common warnings that don't affect functionality
    warnings.filterwarnings("ignore", message=".*Falling back to prediction using DMatrix.*")
    warnings.filterwarnings("ignore", message=".*multivariate.*")
    
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    log_file = setup_logging(args.output_dir)
    
    # Start time
    start_time = time.time()
    
    logging.info("#" * 70)
    logging.info("# OVERNIGHT FINANCE CHALLENGE - MODEL TRAINING")
    logging.info("#" * 70)
    logging.info("")
    logging.info("CONFIGURATION:")
    logging.info(f"  Train path:      {args.train_path}")
    logging.info(f"  Output dir:      {args.output_dir}")
    logging.info(f"  Log file:        {log_file}")
    logging.info(f"  GPU:             {args.use_gpu}")
    logging.info(f"  n_estimators:    {args.n_estimators}")
    logging.info(f"  learning_rate:   {args.learning_rate}")
    logging.info(f"  val_fraction:    {args.val_fraction}")
    logging.info(f"  seed:            {args.seed}")
    logging.info(f"  Oversampling:    {args.oversample} (ratio={args.oversample_ratio})")
    logging.info(f"  Downsampling:    {args.downsample} (ratio={args.downsample_ratio})")
    logging.info(f"  Focal Loss:      {args.focal_loss} (gamma={args.focal_gamma})")
    logging.info(f"  Optuna trials:   {args.optuna_trials}")
    logging.info("")
    
    # ==================== STEP 1: Load Data ====================
    log_step("LOADING DATA & BUILDING FEATURES", 1, 10)
    
    logging.info(f"Loading CSV from: {args.train_path}")
    X, y, feature_cols = stream_build_features(args.train_path)
    
    logging.info("Dataset loaded:")
    logging.info(f"  X shape: {X.shape}")
    logging.info(f"  y shape: {y.shape}")
    logging.info(f"  Features: {len(feature_cols)}")
    logging.info(f"  Memory: {X.nbytes / 1024 / 1024:.1f} MB")
    log_class_distribution(y, "original")

    # ==================== STEP 2: KBinsDiscretizer ====================
    log_step("KBINS DISCRETIZER", 2, 10)
    
    features_to_bin = [
        "mid_price_ratio_3", "mid_price_ratio_5", "mid_price_ratio_10",
        "mid_price_ratio_20", "mid_price_ratio_50", "mid_price_ratio_100",
        "mid_price_position_3", "mid_price_position_5", "mid_price_position_10",
        "mid_price_position_20", "mid_price_position_50", "mid_price_position_100",
        "mid_price_diff_2", "mid_price_diff_3", "mid_price_diff_4", "mid_price_diff_5",
        "spread_pct_change_1", "spread_pct_change_2", "spread_pct_change_3",
        "volume_ratio", "keltner_position_10", "keltner_position_20", "keltner_position_50",
    ]
    
    feature_idx_map = {name: idx for idx, name in enumerate(feature_cols)}
    bin_features = [f for f in features_to_bin if f in feature_idx_map]
    
    logging.info(f"Binning {len(bin_features)} features with KBinsDiscretizer")
    logging.info("  Strategy: quantile, n_bins=5")
    logging.info("  Reason: Discretization helps capture non-linear relationships")
    
    binners = {}
    X_binned_cols = []
    binned_feature_names = []
    
    for feat_name in bin_features:
        idx = feature_idx_map[feat_name]
        col_data = X[:, idx].reshape(-1, 1)
        
        binner = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile', subsample=None, quantile_method='averaged_inverted_cdf')
        binner.fit(col_data)
        binned_col = binner.transform(col_data).flatten()
        
        X_binned_cols.append(binned_col)
        binned_feature_names.append(f"{feat_name}_bin")
        binners[feat_name] = binner.bin_edges_[0].tolist()
    
    if X_binned_cols:
        X_binned = np.column_stack(X_binned_cols)
        X = np.hstack([X, X_binned])
        feature_cols = feature_cols + binned_feature_names
        logging.info(f"Added {len(binned_feature_names)} binned features")
        logging.info(f"Total features: {len(feature_cols)}")

    # ==================== STEP 3: TargetExpandingMeanEncoder ====================
    log_step("TARGET EXPANDING MEAN ENCODER", 3, 10)
    
    te_features = [
        "mid_price_mean", "microprice_1", "spread_abs_mean",
        "volume_imbalance", "mid_price_diff_2", "mid_price_ratio_3",
        "mid_price_position_5", "mid_price_position_10", "mid_price_position_20",
    ]
    te_features_exist = [f for f in te_features if f in feature_cols]
    
    logging.info(f"Target encoding {len(te_features_exist)} features")
    logging.info("  Method: Expanding mean (uses only PAST data - streaming compliant)")
    logging.info("  Smoothing: 10.0, n_bins: 10")
    logging.info("  Reason: Encodes relationship between feature values and target")
    
    if te_features_exist:
        target_encoder = TargetExpandingMeanEncoder(smoothing=10.0, n_bins=10)
        X, feature_cols = target_encoder.fit_transform(X, y, feature_cols, te_features_exist)
        te_state = target_encoder.get_state()
        logging.info(f"Added {len(te_state['encoding_maps'])} target-encoded features")
        logging.info(f"Total features: {len(feature_cols)}")
    else:
        te_state = {}
        logging.warning("No features available for target encoding")

    # ==================== STEP 4: Hybrid Feature Selection ====================
    log_step("HYBRID FEATURE SELECTION", 4, 10)
    
    logging.info("Selection strategy: correlation + redundancy filtering")
    logging.info("1. Compute feature-target correlation (mean across classes)")
    logging.info("2. Compute feature-feature correlation (redundancy)")
    logging.info("3. Keep features with high target-corr OR low redundancy")
    logging.info("Reason: Remove useless features, keep informative ones")
    
    n_features_orig = X.shape[1]
    
    # 1. Compute feature-target correlation (mean across classes)
    print("[train] Computing feature-target correlations...")
    mean_corrs = []
    for i, col in enumerate(feature_cols):
        col_data = X[:, i]
        corrs_per_class = []
        for cls in range(5):
            y_binary = (y == cls).astype(float)
            std_col = np.std(col_data)
            std_y = np.std(y_binary)
            if std_col > 1e-9 and std_y > 1e-9:
                with np.errstate(divide='ignore', invalid='ignore'):
                    corr = np.corrcoef(col_data, y_binary)[0, 1]
                if not np.isnan(corr):
                    corrs_per_class.append(abs(corr))
        mean_corr = np.mean(corrs_per_class) if corrs_per_class else 0.0
        mean_corrs.append(mean_corr)
    
    mean_corrs = np.array(mean_corrs)
    
    # 2. Compute feature-feature correlations (redundancy)
    print("[train] Computing feature-feature correlations (sample for speed)...")
    # Sample for speed
    sample_size = min(50000, X.shape[0])
    sample_idx = np.random.choice(X.shape[0], sample_size, replace=False)
    X_sample = X[sample_idx]
    
    # Feature correlation matrix
    with np.errstate(divide='ignore', invalid='ignore'):
        corr_matrix = np.corrcoef(X_sample.T)
    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
    np.fill_diagonal(corr_matrix, 0)  # Remove diagonal
    sum_corr_all = np.sum(np.abs(corr_matrix), axis=1)
    
    # Free memory
    del X_sample
    gc.collect()
    
    # 3. Hybrid selection: mean_corr >= 0.05 OR sum_corr <= threshold
    MEAN_CORR_THRESHOLD = 0.05
    SUM_CORR_THRESHOLD = 0.5 * n_features_orig  # Relative threshold
    
    selected_mask = (mean_corrs >= MEAN_CORR_THRESHOLD) | (sum_corr_all <= SUM_CORR_THRESHOLD)
    
    # Always keep top-50 by correlation
    top_50_idx = np.argsort(mean_corrs)[-50:]
    selected_mask[top_50_idx] = True
    
    selected_indices = np.where(selected_mask)[0]
    removed_count = n_features_orig - len(selected_indices)
    
    print(f"[train] Hybrid selection: {len(selected_indices)} features kept, {removed_count} removed")
    print(f"[train] Mean corr >= {MEAN_CORR_THRESHOLD}: {np.sum(mean_corrs >= MEAN_CORR_THRESHOLD)}")
    print(f"[train] Low redundancy (sum_corr <= {SUM_CORR_THRESHOLD:.1f}): {np.sum(sum_corr_all <= SUM_CORR_THRESHOLD)}")
    
    # Top-10 features by correlation
    top_10_idx = np.argsort(mean_corrs)[-10:][::-1]
    print("[train] Top-10 features by target correlation:")
    for idx in top_10_idx:
        print(f"  {feature_cols[idx]}: mean_corr={mean_corrs[idx]:.4f}")
    
    # Apply selection
    if len(selected_indices) < n_features_orig:
        X = X[:, selected_indices]
        feature_cols = [feature_cols[i] for i in selected_indices]
        print(f"[train] After selection: {X.shape[1]} features")

    # ==================== STEP 5: Train/Val Split ====================
    log_step("TRAIN/VALIDATION SPLIT", 5, 10)
    
    n_total = len(y)
    n_val = max(1, int(n_total * args.val_fraction))
    X_train_orig, y_train_orig = X[:-n_val], y[:-n_val]
    X_val, y_val = X[-n_val:], y[-n_val:]
    n_classes = int(np.unique(y_train_orig).max() + 1)
    
    logging.info(f"Split strategy: temporal (last {args.val_fraction*100:.0f}% for validation)")
    logging.info("Reason: Preserves time ordering, prevents future leakage")
    logging.info("")
    logging.info(f"Train set: {len(y_train_orig):,} samples")
    logging.info(f"Val set:   {len(y_val):,} samples")
    logging.info(f"Classes:   {n_classes}")
    logging.info("")
    log_class_distribution(y_train_orig, "train (original)")
    log_class_distribution(y_val, "validation")
    
    # ==================== STEP 6: Resampling (Downsample + SMOTE) ====================
    log_step("RESAMPLING (DOWNSAMPLE + SMOTE)", 6, 10)
    
    n_train_original = len(y_train_orig)  # Save for calibration
    
    logging.info(f"Oversampling enabled: {args.oversample}")
    logging.info(f"Downsampling enabled: {args.downsample}")
    if args.oversample or args.downsample:
        logging.info("Method: RandomUnderSampler + SMOTE")
        logging.info(f"Downsample ratio: {args.downsample_ratio} (Class 0)")
        logging.info(f"Oversample ratio: {args.oversample_ratio} (Classes 3, 4)")
        logging.info("")
        logging.info("⚠️  CRITICAL: Calibration will use ORIGINAL data only!")
        logging.info(f"n_train_original = {n_train_original}")
        logging.info("Reason: Calibrators must learn REAL class distribution")
        
        X_train, y_train, _ = resample_classes(
            X_train_orig, y_train_orig, 
            rare_classes=[3, 4],
            target_ratio=args.oversample_ratio,
            downsample_majority=args.downsample,
            downsample_ratio=args.downsample_ratio,
            seed=args.seed
        )
        logging.info("")
        log_class_distribution(y_train, "train (after oversampling)")
    else:
        X_train, y_train = X_train_orig, y_train_orig
        logging.info("Oversampling disabled, using original training data")
    
    # Save sizes for logging (before potential gc)
    n_train_samples = len(y_train)
    n_val_samples = len(y_val)
    
    # Model results
    model_results = []

    # ==================== STEP 7: Optuna HPO ====================
    optuna_best_params = {"xgb": {}, "cat": {}, "lgb": {}}
    
    if args.optuna_trials > 0:
        log_step("OPTUNA HYPERPARAMETER OPTIMIZATION", 7, 10)
        logging.info(f"Total trials: {args.optuna_trials}")
        logging.info(f"Trials per model: {args.optuna_trials // 3}")
        logging.info("Sampler: TPE (Tree-Parzen Estimator)")
        logging.info("Reason: Find optimal hyperparameters for each booster")
        
        optuna_best_params = run_optuna_hpo(
            X_train, y_train, X_val, y_val, 
            args.use_gpu, n_classes, args.seed, args.optuna_trials
        )
        
        logging.info("Optuna best parameters:")
        for model_name, params in optuna_best_params.items():
            if params:
                logging.info(f"  {model_name.upper()}:")
                for k, v in params.items():
                    logging.info(f"    {k}: {v}")
    else:
        logging.info("Optuna HPO disabled (--optuna-trials=0), using default parameters")
    
    # ==================== STEP 8: Model Training ====================
    log_step("MODEL TRAINING (XGBoost + CatBoost + LightGBM + Stacking)", 8, 10)
    
    logging.info("Training pipeline:")
    logging.info("  1. XGBoost (gradient boosting with histogram)")
    logging.info("  2. CatBoost (gradient boosting with ordered boosting)")
    logging.info("  3. LightGBM (gradient boosting with GOSS)")
    logging.info("  4. Binary Rare Class Detector")
    logging.info("  5. Stacking Meta-Model")
    logging.info("")
    
    # XGBoost
    logging.info("-" * 50)
    logging.info("TRAINING: XGBoost")
    logging.info("-" * 50)
    xgb_start = time.time()
    
    # Base parameters
    xgb_params = {
        "objective": "multi:softprob",
        "num_class": n_classes,
        "learning_rate": args.learning_rate,
        "max_depth": 4,
        "subsample": 0.8,
        "colsample_bytree": 0.7,
        "colsample_bylevel": 0.8,
        "reg_alpha": 0.5,
        "reg_lambda": 2.0,
        "min_child_weight": 10.0,
        "gamma": 0.1,
        "max_delta_step": 1,
        "n_estimators": args.n_estimators,
        "tree_method": "hist",
        "eval_metric": "mlogloss",
        "random_state": args.seed,
        "n_jobs": os.cpu_count() or -1,
        "early_stopping_rounds": 100,
    }
    
    # Apply Optuna parameters if available
    if optuna_best_params.get("xgb"):
        print("[XGB] Using Optuna-optimized parameters:")
        for k, v in optuna_best_params["xgb"].items():
            xgb_params[k] = v
            print(f"  {k}: {v:.6f}" if isinstance(v, float) else f"  {k}: {v}")
        # Increase n_estimators for final training
        xgb_params["n_estimators"] = args.n_estimators
        xgb_params["early_stopping_rounds"] = 100
    
    if args.use_gpu:
        xgb_params["device"] = "cuda"
        print("[XGB] Device: CUDA (GPU)")
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        xgb_params["device"] = "cpu"
        print(f"[XGB] Device: CPU ({os.cpu_count()} threads)")

    # Focal Loss (if enabled)
    if args.focal_loss:
        print(f"[XGB] Using Focal Loss with gamma={args.focal_gamma}")
        # For Focal Loss use custom objective via native XGBoost API
        xgb_params.pop("objective")
        xgb_params.pop("early_stopping_rounds", None)
        
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        
        def focal_obj(y_pred, dtrain):
            y_true = dtrain.get_label()
            return focal_loss_multiclass(y_true, y_pred, gamma=args.focal_gamma, n_classes=n_classes)
        
        def focal_eval(y_pred, dtrain):
            y_true = dtrain.get_label()
            return f1_macro_eval(y_true, y_pred, n_classes=n_classes)
        
        native_params = {
            "num_class": n_classes,
            "learning_rate": xgb_params.get("learning_rate", 0.01),
            "max_depth": xgb_params.get("max_depth", 4),
            "subsample": xgb_params.get("subsample", 0.8),
            "colsample_bytree": xgb_params.get("colsample_bytree", 0.7),
            "reg_alpha": xgb_params.get("reg_alpha", 0.5),
            "reg_lambda": xgb_params.get("reg_lambda", 2.0),
            "seed": args.seed,
        }
        if args.use_gpu:
            native_params["device"] = "cuda"
            native_params["tree_method"] = "hist"
        
        xgb_booster = xgb.train(
            native_params,
            dtrain,
            num_boost_round=args.n_estimators,
            evals=[(dtrain, 'train'), (dval, 'val')],
            obj=focal_obj,
            custom_metric=focal_eval,
            early_stopping_rounds=100,
            verbose_eval=50
        )
        
        # Create wrapper for API compatibility
        class XGBFocalWrapper:
            def __init__(self, booster, n_classes):
                self.booster = booster
                self.n_classes = n_classes
                self.best_iteration = booster.best_iteration
            
            def predict_proba(self, X):
                dmat = xgb.DMatrix(X)
                # output_margin=True returns raw scores (logits) for custom objective
                raw = self.booster.predict(dmat, output_margin=True)
                return softmax_numpy(raw.reshape(-1, self.n_classes))
            
            def get_booster(self):
                return self.booster
            
            def save_model(self, path):
                self.booster.save_model(str(path))
        
        xgb_model = XGBFocalWrapper(xgb_booster, n_classes)
    else:
        xgb_model = xgb.XGBClassifier(**xgb_params)
        xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=50)
    
    xgb_time = time.time() - xgb_start
    print(f"[XGB] Training time: {xgb_time:.1f}s")
    print(f"[XGB] Best iteration: {xgb_model.best_iteration}")

    # XGB val probs
    xgb_val_probs = xgb_model.predict_proba(X_val)
    xgb_val_probs = clean_probs(xgb_val_probs, n_classes)
    
    xgb_result = evaluate_model("XGBoost", xgb_val_probs, y_val, n_classes)
    xgb_result["training_time"] = xgb_time
    xgb_result["best_iteration"] = xgb_model.best_iteration
    model_results.append(xgb_result)
    xgb_f1 = xgb_result["f1_macro"]

    # ==================== CatBoost ====================
    # Don't use early_stopping with TotalF1 (stops too early)
    # Use fixed iterations and logloss for stability
    print("\n" + "="*70)
    print("[CatBoost] Starting training...")
    print("="*70)
    cat_start = time.time()
    
    cat_iterations = min(1200, args.n_estimators)  # CatBoost converges faster
    cat_params = {
        "iterations": cat_iterations,
        "learning_rate": 0.03,  # Better than 0.01
        "depth": 6,  # Deeper trees
        "l2_leaf_reg": 20.0,  # Stronger regularization
        "random_seed": args.seed,
        "loss_function": "MultiClass",
        "eval_metric": "MultiClass",  # More stable than F1 for early stopping
        "od_type": "Iter",  # Overfitting detector
        "od_wait": 100,
        "verbose": 100,
    }
    
    # Apply Optuna parameters if available
    if optuna_best_params.get("cat"):
        print("[CAT] Using Optuna-optimized parameters:")
        for k, v in optuna_best_params["cat"].items():
            cat_params[k] = v
            print(f"  {k}: {v:.6f}" if isinstance(v, float) else f"  {k}: {v}")
        cat_params["iterations"] = cat_iterations  # Full iterations for final
    
    if args.use_gpu:
        cat_params["task_type"] = "GPU"
        cat_params["devices"] = "0"
        print(f"[CAT] Device: GPU (device 0), iterations={cat_iterations}")
    else:
        cat_params["task_type"] = "CPU"
        cat_params["thread_count"] = os.cpu_count() or -1
        print(f"[CAT] Device: CPU ({os.cpu_count()} threads), iterations={cat_iterations}")

    cat_model = CatBoostClassifier(**cat_params)
    cat_model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=100)
    
    cat_time = time.time() - cat_start
    print(f"[CAT] Training time: {cat_time:.1f}s")
    print(f"[CAT] Best iteration: {cat_model.best_iteration_}")

    cat_val_probs = cat_model.predict_proba(X_val)
    cat_val_probs = clean_probs(cat_val_probs, n_classes)
    
    cat_result = evaluate_model("CatBoost", cat_val_probs, y_val, n_classes)
    cat_result["training_time"] = cat_time
    cat_result["best_iteration"] = cat_model.best_iteration_
    model_results.append(cat_result)
    cat_f1 = cat_result["f1_macro"]

    # ==================== LightGBM ====================
    print("\n" + "="*70)
    print("[LightGBM] Starting training...")
    print("="*70)
    lgb_start = time.time()
    
    lgb_params = {
        "objective": "multiclass",
        "num_class": n_classes,
        "learning_rate": args.learning_rate,
        "max_depth": 6,
        "num_leaves": 31,
        "subsample": 0.8,
        "colsample_bytree": 0.7,
        "reg_alpha": 0.5,
        "reg_lambda": 2.0,
        "min_child_samples": 100,  # Increased for GPU stability
        "min_data_in_leaf": 200,   # Critical for GPU
        "n_estimators": args.n_estimators,
        "random_state": args.seed,
        "n_jobs": os.cpu_count() or -1,
        "verbose": -1,
    }
    
    # Apply Optuna parameters if available
    if optuna_best_params.get("lgb"):
        print("[LGB] Using Optuna-optimized parameters:")
        for k, v in optuna_best_params["lgb"].items():
            lgb_params[k] = v
            print(f"  {k}: {v:.6f}" if isinstance(v, float) else f"  {k}: {v}")
        lgb_params["n_estimators"] = args.n_estimators  # Full iterations for final
    
    # LightGBM: GPU (OpenCL) or CPU
    if args.use_gpu:
        lgb_params["device"] = "gpu"
        lgb_params["gpu_use_dp"] = False
        print("[LGB] Device: GPU (OpenCL)")
    else:
        lgb_params["device"] = "cpu"
        print(f"[LGB] Device: CPU ({os.cpu_count()} threads)")

    lgb_model = lgb.LGBMClassifier(**lgb_params)
    lgb_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=100, verbose=False),
            lgb.log_evaluation(period=100)
        ]
    )
    
    lgb_time = time.time() - lgb_start
    print(f"[LGB] Training time: {lgb_time:.1f}s")
    print(f"[LGB] Best iteration: {lgb_model.best_iteration_}")

    # Use numpy array to avoid feature names warning
    lgb_val_probs = lgb_model.predict_proba(np.ascontiguousarray(X_val))
    lgb_val_probs = clean_probs(lgb_val_probs, n_classes)
    
    lgb_result = evaluate_model("LightGBM", lgb_val_probs, y_val, n_classes)
    lgb_result["training_time"] = lgb_time
    lgb_result["best_iteration"] = lgb_model.best_iteration_
    model_results.append(lgb_result)
    lgb_f1 = lgb_result["f1_macro"]

    # ==================== Rare Class Detector (Binary Model2) ====================
    print("\n" + "="*70)
    print("[Rare Class Detector] Training binary classifier (common vs rare)...")
    print("="*70)
    rare_start = time.time()
    
    # Binary labels: 0 = common (classes 0,1,2), 1 = rare (classes 3,4)
    y_train_binary = np.where(y_train <= 2, 0, 1)
    y_val_binary = np.where(y_val <= 2, 0, 1)
    
    rare_in_train = y_train_binary.sum()
    rare_in_val = y_val_binary.sum()
    print(f"[Binary] Train: {len(y_train_binary)} samples, rare={rare_in_train} ({100*rare_in_train/len(y_train_binary):.2f}%)")
    print(f"[Binary] Val:   {len(y_val_binary)} samples, rare={rare_in_val} ({100*rare_in_val/len(y_val_binary):.2f}%)")
    
    # CatBoost for binary classification
    binary_params = {
        "iterations": 500,
        "learning_rate": 0.03,
        "depth": 6,
        "l2_leaf_reg": 10.0,
        "random_seed": args.seed,
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "od_type": "Iter",
        "od_wait": 50,
        "verbose": 100,
    }
    
    if args.use_gpu:
        binary_params["task_type"] = "GPU"
        binary_params["devices"] = "0"
        print("[Binary] Device: GPU")
    else:
        binary_params["task_type"] = "CPU"
        binary_params["thread_count"] = os.cpu_count() or -1
        print(f"[Binary] Device: CPU ({os.cpu_count()} threads)")
    
    binary_model = CatBoostClassifier(**binary_params)
    binary_model.fit(X_train, y_train_binary, eval_set=(X_val, y_val_binary), verbose=100)
    
    rare_time = time.time() - rare_start
    print(f"[Binary] Training time: {rare_time:.1f}s")
    print(f"[Binary] Best iteration: {binary_model.best_iteration_}")
    
    # Binary model predictions
    binary_train_probs = binary_model.predict_proba(X_train)
    binary_val_probs = binary_model.predict_proba(X_val)
    
    # AUC for binary model evaluation
    auc_val = roc_auc_score(y_val_binary, binary_val_probs[:, 1])
    print(f"[Binary] AUC on val: {auc_val:.4f}")
    
    # How many rare samples model predicts
    binary_preds = (binary_val_probs[:, 1] > 0.5).astype(int)
    binary_rare_preds = binary_preds.sum()
    print(f"[Binary] Predicted rare on val: {binary_rare_preds} (true: {rare_in_val})")
    
    binary_result = {
        "name": "Binary_Rare_Detector",
        "auc": auc_val,
        "training_time": rare_time,
        "best_iteration": binary_model.best_iteration_,
        "true_rare": int(rare_in_val),
        "pred_rare": int(binary_rare_preds),
    }
    model_results.append(binary_result)

    # ==================== Stacking Meta-Model ====================
    print("\n" + "="*70)
    print("[Stacking] Training meta-model on base model predictions...")
    print("="*70)
    stack_start = time.time()
    
    # Collect OOF predictions from base models for train
    # For XGBoost on GPU use predict_proba directly
    xgb_train_probs = xgb_model.predict_proba(X_train)
    xgb_train_probs = clean_probs(xgb_train_probs, n_classes)
    cat_train_probs = cat_model.predict_proba(X_train)
    cat_train_probs = clean_probs(cat_train_probs, n_classes)
    
    lgb_train_probs = lgb_model.predict_proba(np.ascontiguousarray(X_train))
    lgb_train_probs = clean_probs(lgb_train_probs, n_classes)
    
    # Stack features: probabilities from all base models + binary model
    stack_train = np.hstack([
        xgb_train_probs, cat_train_probs, lgb_train_probs, 
        binary_train_probs
    ])
    stack_val = np.hstack([
        xgb_val_probs, cat_val_probs, lgb_val_probs,
        binary_val_probs
    ])
    
    # Free intermediate train probabilities
    del xgb_train_probs, cat_train_probs, lgb_train_probs, binary_train_probs
    gc.collect()
    
    print(f"[Stack] Stack features shape: train={stack_train.shape}, val={stack_val.shape}")
    
    # Meta-model: lightweight CatBoost
    stack_params = {
        "iterations": 300,
        "learning_rate": 0.05,
        "depth": 4,
        "l2_leaf_reg": 5.0,
        "random_seed": args.seed,
        "loss_function": "MultiClass",
        "eval_metric": "MultiClass",
        "od_type": "Iter",
        "od_wait": 50,
        "verbose": 50,
    }
    
    if args.use_gpu:
        stack_params["task_type"] = "GPU"
        stack_params["devices"] = "0"
    else:
        stack_params["task_type"] = "CPU"
        stack_params["thread_count"] = os.cpu_count() or -1
    
    stack_model = CatBoostClassifier(**stack_params)
    stack_model.fit(stack_train, y_train, eval_set=(stack_val, y_val), verbose=50)
    
    stack_time = time.time() - stack_start
    print(f"[Stack] Training time: {stack_time:.1f}s")
    print(f"[Stack] Best iteration: {stack_model.best_iteration_}")
    
    # Meta-model predictions
    stack_val_probs = stack_model.predict_proba(stack_val)
    stack_val_probs = clean_probs(stack_val_probs, n_classes)
    
    stack_result = evaluate_model("Stacking", stack_val_probs, y_val, n_classes)
    stack_result["training_time"] = stack_time
    stack_result["best_iteration"] = stack_model.best_iteration_
    model_results.append(stack_result)

    # ==================== Ensemble (Soft Blend) ====================
    print("\n[train] Ensemble blend optimization (with stacking)...")
    
    # Grid search for optimal weights (including stacking)
    best_weights = (0.3, 0.2, 0.2, 0.3)  # xgb, cat, lgb, stack
    best_blend_f1 = -1
    
    for xgb_w in np.linspace(0.1, 0.4, 7):
        for cat_w in np.linspace(0.1, 0.3, 5):
            for lgb_w in np.linspace(0.1, 0.3, 5):
                stack_w = 1.0 - xgb_w - cat_w - lgb_w
                if stack_w < 0.1 or stack_w > 0.5:
                    continue
                blend_probs = (xgb_w * xgb_val_probs + cat_w * cat_val_probs + 
                              lgb_w * lgb_val_probs + stack_w * stack_val_probs)
                blend_f1 = f1_score(y_val, blend_probs.argmax(axis=1), average='macro')
                if blend_f1 > best_blend_f1:
                    best_blend_f1 = blend_f1
                    best_weights = (xgb_w, cat_w, lgb_w, stack_w)
    
    xgb_weight, cat_weight, lgb_weight, stack_weight = best_weights
    print(f"[Ensemble] Best weights: XGB={xgb_weight:.2f}, CAT={cat_weight:.2f}, LGB={lgb_weight:.2f}, STACK={stack_weight:.2f}, F1={best_blend_f1:.4f}")
    
    val_probs = (xgb_weight * xgb_val_probs + cat_weight * cat_val_probs + 
                lgb_weight * lgb_val_probs + stack_weight * stack_val_probs)
    baseline_f1 = f1_score(y_val, val_probs.argmax(axis=1), average='macro')
    print(f"[Ensemble] Baseline F1-macro on val: {baseline_f1:.4f}")
    
    # ==================== STEP 9: Calibration ====================
    log_step("CALIBRATION & THRESHOLD OPTIMIZATION", 9, 10)
    
    logging.info("⚠️  CRITICAL: ALL CALIBRATORS USE ORIGINAL DATA ONLY!")
    logging.info(f"Validation set: {len(y_val):,} samples (NEVER augmented)")
    logging.info(f"Training set: {len(y_train):,} samples (original: {n_train_original:,})")
    logging.info("Reason: Calibrators must learn REAL class distribution, not augmented!")
    logging.info("")
    
    # RareClassBooster
    logging.info("-" * 50)
    logging.info("CALIBRATOR 1: RareClassBooster")
    logging.info("-" * 50)
    logging.info("  Purpose: Boost probabilities for rare classes (3, 4)")
    logging.info("  Method: Uses binary classifier signal + Optuna optimization")
    logging.info("  Parameters: alpha (boost strength), beta (rare class multiplier)")
    
    rare_booster = RareClassBooster(rare_classes=[3, 4], common_classes=[0, 1, 2])
    rare_booster.fit(y_val, val_probs, binary_val_probs, 
                     alpha_range=(0.01, 100), beta_range=(0.01, 100), n_trials=500)
    
    val_probs_boosted = rare_booster.predict_proba(val_probs, binary_val_probs)
    boosted_f1 = f1_score(y_val, val_probs_boosted.argmax(axis=1), average='macro')
    
    logging.info(f"  Result: alpha={rare_booster.alpha:.4f}, beta={rare_booster.beta:.4f}")
    logging.info(f"  F1-macro: {baseline_f1:.4f} → {boosted_f1:.4f} ({(boosted_f1-baseline_f1)*100:+.2f}%)")
    
    val_probs = val_probs_boosted

    # DiagonalTemperatureCalibrator
    logging.info("")
    logging.info("-" * 50)
    logging.info("CALIBRATOR 2: DiagonalTemperatureCalibrator")
    logging.info("-" * 50)
    logging.info("  Purpose: Scale logits per-class for better calibration")
    logging.info("  Method: Temperature scaling with class-specific temperatures")
    
    diag_calibrator = DiagonalTemperatureCalibrator(n_trials=1000)
    diag_calibrator.fit(val_probs, y_val)
    
    val_probs_diag = diag_calibrator.predict_proba(val_probs)
    diag_f1 = f1_score(y_val, diag_calibrator.predict(val_probs), average='macro')
    
    logging.info(f"  Temperatures: {diag_calibrator.temps}")
    logging.info(f"  F1-macro: {boosted_f1:.4f} → {diag_f1:.4f} ({(diag_f1-boosted_f1)*100:+.2f}%)")

    # OVR Threshold Optimizer
    logging.info("")
    logging.info("-" * 50)
    logging.info("CALIBRATOR 3: OVR Threshold Optimizer")
    logging.info("-" * 50)
    logging.info("  Purpose: Per-class threshold optimization")
    logging.info("  Method: One-vs-Rest grid search")
    
    ovr_opt = ThresholdOptimizerOVR(n_steps=300, n_iters=15)
    ovr_opt.fit(val_probs_diag, y_val)
    
    ovr_f1 = f1_score(y_val, ovr_opt.predict(val_probs_diag), average='macro')
    logging.info(f"  Thresholds: {ovr_opt.thresholds}")
    logging.info(f"  F1-macro: {diag_f1:.4f} → {ovr_f1:.4f} ({(ovr_f1-diag_f1)*100:+.2f}%)")
    
    # Final F1 = OVR result
    final_f1 = ovr_f1
    
    logging.info("")
    logging.info("=" * 50)
    logging.info(f"FINAL F1-MACRO ON VALIDATION: {final_f1:.4f}")
    logging.info("=" * 50)

    # ==================== STEP 11: Save Models ====================
    log_step("SAVING MODELS & FINAL SUMMARY", 10, 10)
    
    xgb_out = args.output_dir / "model_xgb.json"
    cat_out = args.output_dir / "model_cat.cbm"
    lgb_out = args.output_dir / "model_lgb.txt"
    binary_out = args.output_dir / "model_binary.cbm"
    stack_out = args.output_dir / "model_stack.cbm"
    feature_out = args.output_dir / "feature_order.json"
    calib_out = args.output_dir / "calibration.json"

    logging.info("Saving models:")
    xgb_model.save_model(xgb_out)
    logging.info(f"  ✓ XGBoost: {xgb_out}")
    
    cat_model.save_model(cat_out)
    logging.info(f"  ✓ CatBoost: {cat_out}")
    
    lgb_model.booster_.save_model(lgb_out)
    logging.info(f"  ✓ LightGBM: {lgb_out}")
    
    binary_model.save_model(binary_out)
    logging.info(f"  ✓ Binary Detector: {binary_out}")
    
    stack_model.save_model(stack_out)
    logging.info(f"  ✓ Stacking: {stack_out}")
    
    feature_out.write_text(json.dumps(feature_cols, indent=2))
    logging.info(f"  ✓ Feature order: {feature_out}")
    
    calib_data = {
        "xgb_weight": xgb_weight,
        "cat_weight": cat_weight,
        "lgb_weight": lgb_weight,
        "stack_weight": stack_weight,
        "lgb_available": True,  # LGB always available
        "rare_booster_alpha": rare_booster.alpha,
        "rare_booster_beta": rare_booster.beta,
        "diag_temps": diag_calibrator.temps.tolist(),
        "diag_thresholds": diag_calibrator.thresholds.tolist(),
        "ovr_thresholds": ovr_opt.thresholds.tolist(),
        "bin_edges": binners,  # KBinsDiscretizer edges
        "target_encoding": te_state,  # TargetExpandingMeanEncoder state
        # Metadata for calibration validation
        "n_train_original": n_train_original,
        "n_train_augmented": len(y_train),
        "n_val": len(y_val),
        "oversample_enabled": args.oversample,
        "focal_loss_enabled": args.focal_loss,
        "calibration_note": "All calibrators trained on ORIGINAL validation data only (not augmented!)"
    }
    calib_out.write_text(json.dumps(calib_data, indent=2))
    logging.info(f"  ✓ Calibration: {calib_out}")
    
    # Free training data from memory
    del X_train, y_train, X_val, y_val
    del stack_train, stack_val
    gc.collect()
    
    # ==================== STEP 12: Training Summary ====================
    
    total_time = time.time() - start_time
    
    # Save results to log file
    log_out = args.output_dir / "training_log.json"
    training_log = {
        "timestamp": datetime.now().isoformat(),
        "total_time_sec": total_time,
        "use_gpu": args.use_gpu,
        "n_estimators": args.n_estimators,
        "learning_rate": args.learning_rate,
        "n_features": len(feature_cols),
        "n_train": n_train_samples,
        "n_val": n_val_samples,
        "n_classes": n_classes,
        "optuna_trials": args.optuna_trials,
        "optuna_best_params": optuna_best_params if optuna_best_params else None,
        "models": model_results,
        "ensemble": {
            "xgb_weight": xgb_weight,
            "cat_weight": cat_weight,
            "lgb_weight": lgb_weight,
            "blend_f1": best_blend_f1,
            "rare_boost_f1": boosted_f1,
            "rare_boost_alpha": rare_booster.alpha,
            "rare_boost_beta": rare_booster.beta,
            "diag_temp_f1": diag_f1,
            "final_f1": final_f1,
        },
        "calibration": calib_data,
    }
    log_out.write_text(json.dumps(training_log, indent=2))
    logging.info(f"  ✓ Training log: {log_out}")
    
    # Print final summary
    logging.info("")
    logging.info("#" * 70)
    logging.info("# TRAINING SUMMARY")
    logging.info("#" * 70)
    logging.info(f"Total training time: {total_time/60:.1f} min ({total_time:.0f} sec)")
    logging.info("")
    
    # Model comparison
    logging.info("INDIVIDUAL MODEL PERFORMANCE:")
    logging.info("-" * 50)
    log_model_comparison(model_results)
    
    if args.optuna_trials > 0:
        logging.info("OPTUNA HPO RESULTS:")
        logging.info("-" * 50)
        for model_name in ["xgb", "cat", "lgb"]:
            if optuna_best_params.get(model_name):
                logging.info(f"  {model_name.upper()}:")
                for k, v in optuna_best_params[model_name].items():
                    if isinstance(v, float):
                        logging.info(f"    {k}: {v:.6f}")
                    else:
                        logging.info(f"    {k}: {v}")
        logging.info("")
    
    logging.info("ENSEMBLE PIPELINE:")
    logging.info("-" * 50)
    logging.info(f"  Weights: XGB={xgb_weight:.2f}, CAT={cat_weight:.2f}, LGB={lgb_weight:.2f}, STACK={stack_weight:.2f}")
    logging.info("")
    logging.info("  F1-macro progression:")
    logging.info(f"    Blend:       {best_blend_f1:.4f}")
    logging.info(f"    + RareBoost: {boosted_f1:.4f} ({(boosted_f1-best_blend_f1)*100:+.2f}%)")
    logging.info(f"    + DiagTemp:  {diag_f1:.4f} ({(diag_f1-boosted_f1)*100:+.2f}%)")
    logging.info(f"    + Threshold: {final_f1:.4f} ({(final_f1-diag_f1)*100:+.2f}%)")
    logging.info("")
    
    logging.info("RARE CLASS DETECTOR:")
    logging.info("-" * 50)
    logging.info(f"  AUC: {auc_val:.4f}")
    logging.info(f"  RareBooster alpha: {rare_booster.alpha:.4f}")
    logging.info(f"  RareBooster beta: {rare_booster.beta:.4f}")
    logging.info("")
    
    best_single = max(xgb_f1, cat_f1, lgb_f1)
    
    logging.info("=" * 70)
    logging.info("🏆 FINAL RESULTS")
    logging.info("=" * 70)
    logging.info(f"  Final F1-macro: {final_f1:.4f}")
    logging.info(f"  Best single model: {best_single:.4f}")
    logging.info(f"  Ensemble improvement: {(final_f1-best_single)*100:+.2f}%")
    logging.info("")
    logging.info("FILES SAVED:")
    logging.info(f"  - {xgb_out}")
    logging.info(f"  - {cat_out}")
    logging.info(f"  - {lgb_out}")
    logging.info(f"  - {binary_out}")
    logging.info(f"  - {stack_out}")
    logging.info(f"  - {feature_out}")
    logging.info(f"  - {calib_out}")
    logging.info(f"  - {log_out}")
    logging.info(f"  - {log_file}")
    logging.info("")
    logging.info("=" * 70)
    logging.info("TRAINING COMPLETE!")
    logging.info("=" * 70)


if __name__ == "__main__":
    train()
