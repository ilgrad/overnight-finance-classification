#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Streaming inference for Overnight Finance Challenge.
Ensemble: XGBoost + CatBoost + LightGBM + RareClassBooster with calibration.
All transformations performed row-by-row (organizers' requirement).
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import xgboost as xgb
from catboost import CatBoostClassifier
import lightgbm as lgb
from scipy.special import softmax
from tqdm.auto import tqdm
from sklearn.metrics import f1_score

from features import StatefulFeatureBuilder


# ===================== RareClassBooster (inference) =====================
class RareClassBoosterInference:
    """RareClassBooster for inference."""
    
    def __init__(self, alpha: float, beta: float, rare_classes=[3, 4]):
        self.alpha = alpha
        self.beta = beta
        self.rare_classes = rare_classes
    
    def transform(self, probs_model1: np.ndarray, probs_model2: np.ndarray) -> np.ndarray:
        adjusted = probs_model1.copy()
        
        # Rare class probability from binary model
        rare_signal = probs_model2[:, 1].reshape(-1, 1)
        
        # Mask for rare classes
        rare_mask = np.zeros_like(adjusted)
        for rare_cls in self.rare_classes:
            rare_mask[:, rare_cls] = 1
        
        # Boosting
        boost_factor = 1 + self.alpha * rare_signal * rare_mask
        adjusted = adjusted * boost_factor
        
        for rare_cls in self.rare_classes:
            adjusted[:, rare_cls] *= self.beta
        
        # Normalize
        adjusted = adjusted / adjusted.sum(axis=1, keepdims=True)
        return adjusted


def apply_binning(feats: dict, bin_edges: dict) -> dict:
    """Apply KBinsDiscretizer using saved bin_edges."""
    for feat_name, edges in bin_edges.items():
        if feat_name in feats:
            val = feats[feat_name]
            edges_arr = np.array(edges)
            # np.digitize returns bin index (1-based)
            bin_idx = np.digitize(val, edges_arr[1:-1])  # edges[1:-1] = inner boundaries
            feats[f"{feat_name}_bin"] = float(bin_idx)
    return feats


def apply_target_encoding(feats: dict, te_state: dict) -> dict:
    """
    Apply TargetExpandingMeanEncoder using saved state.
    On inference use statistics collected during training.
    """
    if not te_state or 'encoding_maps' not in te_state:
        return feats
    
    bin_edges = te_state.get('bin_edges', {})
    encoding_maps = te_state.get('encoding_maps', {})
    global_mean = te_state.get('global_mean', 0.0)
    
    for feat_name, edges in bin_edges.items():
        if feat_name not in feats:
            continue
        
        val = feats[feat_name]
        edges_arr = np.array(edges)
        
        # Determine bin
        bin_idx = int(np.digitize(val, edges_arr[1:-1]))
        
        # Get encoded value
        enc_map = encoding_maps.get(feat_name, {})
        encoded_val = enc_map.get(str(bin_idx), global_mean)
        
        feats[f"{feat_name}_te"] = float(encoded_val)
    
    return feats


class EvalSolution:
    def __init__(self, model, builder: StatefulFeatureBuilder, feature_order: list[str], 
                 bin_edges: dict = None, te_state: dict = None):
        self.model = model
        self.builder = builder
        self.feature_order = feature_order
        self.bin_edges = bin_edges or {}
        self.te_state = te_state or {}
        self.pred_list = []

    def eval_model(self, row: dict):
        """All transformations done here, row-by-row."""
        feats = self.builder.build_row(row)
        
        # Apply binning if bin_edges available
        if self.bin_edges:
            feats = apply_binning(feats, self.bin_edges)
        
        # Apply target encoding if te_state available
        if self.te_state:
            feats = apply_target_encoding(feats, self.te_state)
        
        x_vec = np.array([feats.get(c, 0.0) for c in self.feature_order], dtype=np.float32)
        pred = self.model.predict(x_vec.reshape(1, -1))
        if isinstance(pred, (list, np.ndarray)):
            pred = pred[0]
        self.pred_list.append(int(pred))

    def eval_score(self, y: np.ndarray):
        y = np.array(y)
        f1 = f1_score(y, self.pred_list, average="macro")
        print(f"F1 score (macro): {f1:.4f}")


class EnsembleModel:
    """
    Ensemble XGBoost + CatBoost + LightGBM + Stacking + RareClassBooster with calibration.
    """

    def __init__(
        self, 
        xgb_model: xgb.XGBClassifier,
        cat_model: CatBoostClassifier,
        lgb_model,  # lgb.Booster or None
        binary_model: CatBoostClassifier | None,  # Binary rare class detector
        stack_model: CatBoostClassifier | None,  # Stacking meta-model
        xgb_weight: float,
        cat_weight: float,
        lgb_weight: float,
        stack_weight: float,
        rare_booster: RareClassBoosterInference | None,
        diag_temps: list[float],
        diag_thresholds: list[float],
        ovr_thresholds: list[float]
    ):
        self.xgb_model = xgb_model
        self.cat_model = cat_model
        self.lgb_model = lgb_model  # Can be None
        self.binary_model = binary_model
        self.stack_model = stack_model
        self.xgb_weight = xgb_weight
        self.cat_weight = cat_weight
        self.lgb_weight = lgb_weight
        self.stack_weight = stack_weight
        self.rare_booster = rare_booster
        self.diag_temps = np.array(diag_temps, dtype=np.float64)
        self.diag_thresholds = np.array(diag_thresholds, dtype=np.float64)
        self.ovr_thresholds = np.array(ovr_thresholds, dtype=np.float64)
        self.n_classes = len(diag_temps)

    def _clean_probs(self, probs: np.ndarray) -> np.ndarray:
        probs = np.asarray(probs, dtype=np.float64)
        probs = probs.reshape(-1, self.n_classes)
        probs = np.nan_to_num(probs, nan=1e-15, posinf=1.0, neginf=1e-15)
        probs = np.clip(probs, 1e-15, 1.0)
        row_sums = probs.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1.0, row_sums)
        return probs / row_sums

    def predict_proba_batch(self, X: np.ndarray) -> np.ndarray:
        """Ensemble probs with calibration and stacking."""
        # XGBoost probs
        xgb_probs = self.xgb_model.predict_proba(X)
        xgb_probs = self._clean_probs(xgb_probs)
        
        # CatBoost probs
        cat_probs = self.cat_model.predict_proba(X)
        cat_probs = self._clean_probs(cat_probs)
        
        # LightGBM probs (or fallback to XGB+CAT average if not available)
        if self.lgb_model is not None:
            lgb_probs = self.lgb_model.predict(X)
            lgb_probs = self._clean_probs(lgb_probs)
        else:
            # Fallback: use XGB + CAT average
            lgb_probs = (xgb_probs + cat_probs) / 2
        
        # Binary model probs (for stacking and RareBooster)
        binary_probs = None
        if self.binary_model is not None:
            binary_probs = self.binary_model.predict_proba(X)
        
        # Stacking meta-model
        stack_probs = None
        if self.stack_model is not None and binary_probs is not None:
            stack_features = np.hstack([xgb_probs, cat_probs, lgb_probs, binary_probs])
            stack_probs = self.stack_model.predict_proba(stack_features)
            stack_probs = self._clean_probs(stack_probs)
        
        # Blend (including stacking if available)
        if stack_probs is not None and self.stack_weight > 0:
            probs = (self.xgb_weight * xgb_probs + self.cat_weight * cat_probs + 
                    self.lgb_weight * lgb_probs + self.stack_weight * stack_probs)
        else:
            probs = self.xgb_weight * xgb_probs + self.cat_weight * cat_probs + self.lgb_weight * lgb_probs
        
        # Apply RareClassBooster if available
        if self.rare_booster is not None and binary_probs is not None:
            probs = self.rare_booster.transform(probs, binary_probs)
        
        # Diagonal temperature scaling
        log_probs = np.log(np.clip(probs, 1e-15, 1.0))
        scaled_logits = log_probs / self.diag_temps
        probs = softmax(scaled_logits, axis=1)
        
        # Apply diagonal thresholds
        probs = probs - self.diag_thresholds
        
        return probs

    def predict_batch(self, X: np.ndarray) -> np.ndarray:
        """Predict classes with OVR thresholds."""
        probs = self.predict_proba_batch(X)
        
        # OVR thresholds - best method based on experiments
        adj = probs.copy()
        for c, t in enumerate(self.ovr_thresholds):
            adj[:, c] = np.where(probs[:, c] >= t, probs[:, c], -np.inf)
        
        mask = np.all(adj == -np.inf, axis=1)
        adj[mask] = probs[mask]
        
        return adj.argmax(axis=1)

    def predict(self, x_vec: np.ndarray) -> int:
        preds = self.predict_batch(x_vec.reshape(1, -1))
        return int(preds[0])


def load_xgb_model(model_path: Path) -> xgb.XGBClassifier:
    model = xgb.XGBClassifier()
    model.load_model(model_path)
    return model


def load_cat_model(model_path: Path) -> CatBoostClassifier:
    model = CatBoostClassifier()
    model.load_model(model_path)
    return model


def load_lgb_model(model_path: Path) -> lgb.Booster:
    model = lgb.Booster(model_file=str(model_path))
    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", type=Path, default=Path("eval_sample.csv"))
    parser.add_argument("--output-path", type=Path, default=Path("submission.csv"))
    return parser.parse_args()


def main():
    args = parse_args()
    root = Path(__file__).resolve().parent
    
    # Paths
    xgb_path = root / "model_xgb.json"
    cat_path = root / "model_cat.cbm"
    lgb_path = root / "model_lgb.txt"
    binary_path = root / "model_binary.cbm"
    stack_path = root / "model_stack.cbm"
    feature_order_path = root / "feature_order.json"
    calib_path = root / "calibration.json"
    
    # Fallback to old single-model format
    old_model_path = root / "model.json"
    
    # Load feature order
    feature_order = json.loads(feature_order_path.read_text())
    
    # Load calibration
    if calib_path.exists():
        calib = json.loads(calib_path.read_text())
    else:
        calib = {}
    
    # Check if ensemble or single model
    lgb_available = calib.get("lgb_available", True)  # Default True for backward compatibility
    
    if xgb_path.exists() and cat_path.exists():
        print("[main] Loading ensemble (XGBoost + CatBoost + LightGBM + Stacking)...")
        xgb_model = load_xgb_model(xgb_path)
        cat_model = load_cat_model(cat_path)
        
        # LightGBM (optional - may have failed during training)
        lgb_model = None
        if lgb_available and lgb_path.exists():
            lgb_model = load_lgb_model(lgb_path)
            print("[main] LightGBM loaded")
        else:
            print("[main] ⚠️ LightGBM not available, using XGB+CAT fallback")
        
        # Load binary model for RareClassBooster
        binary_model = None
        rare_booster = None
        if binary_path.exists():
            print("[main] Loading RareClassBooster (binary model)...")
            binary_model = load_cat_model(binary_path)
            rare_alpha = calib.get("rare_booster_alpha", None)
            rare_beta = calib.get("rare_booster_beta", None)
            if rare_alpha is not None and rare_beta is not None:
                rare_booster = RareClassBoosterInference(rare_alpha, rare_beta)
                print(f"[main] RareBooster: alpha={rare_alpha:.4f}, beta={rare_beta:.4f}")
        
        # Load stacking meta-model
        stack_model = None
        if stack_path.exists():
            print("[main] Loading Stacking meta-model...")
            stack_model = load_cat_model(stack_path)
        
        xgb_weight = calib.get("xgb_weight", 0.3)
        cat_weight = calib.get("cat_weight", 0.2)
        lgb_weight = calib.get("lgb_weight", 0.2)
        stack_weight = calib.get("stack_weight", 0.3)
        diag_temps = calib.get("diag_temps", [1.0] * 5)
        diag_thresholds = calib.get("diag_thresholds", [0.0] * 5)
        ovr_thresholds = calib.get("ovr_thresholds", [0.0] * 5)
        
        print(f"[main] Using OVR thresholds (best method)")
        
        model = EnsembleModel(
            xgb_model, cat_model, lgb_model, binary_model, stack_model,
            xgb_weight, cat_weight, lgb_weight, stack_weight,
            rare_booster,
            diag_temps, diag_thresholds, ovr_thresholds
        )
        print(f"[main] Weights: XGB={xgb_weight:.2f}, CAT={cat_weight:.2f}, LGB={lgb_weight:.2f}, STACK={stack_weight:.2f}")
    elif old_model_path.exists():
        # Fallback to old single-model format
        print("[main] Loading single XGBoost model (old format)...")
        xgb_model = load_xgb_model(old_model_path)
        
        # Create a simple wrapper
        class SingleModel:
            def __init__(self, xgb_model, diag_temps, diag_thresholds, ovr_thresholds):
                self.xgb_model = xgb_model
                self.diag_temps = np.array(diag_temps)
                self.diag_thresholds = np.array(diag_thresholds)
                self.ovr_thresholds = np.array(ovr_thresholds)
                self.n_classes = len(diag_temps)
            
            def predict(self, x_vec):
                probs = self.xgb_model.predict_proba(x_vec.reshape(1, -1))
                probs = np.clip(probs, 1e-15, 1.0)
                probs = probs / probs.sum()
                log_probs = np.log(probs)
                scaled = softmax(log_probs / self.diag_temps)
                scaled = scaled - self.diag_thresholds
                adj = scaled.copy()
                for c, t in enumerate(self.ovr_thresholds):
                    adj[0, c] = scaled[0, c] if scaled[0, c] >= t else -np.inf
                if np.all(adj == -np.inf):
                    adj = scaled
                return int(adj.argmax())
        
        diag_temps = calib.get("diag_temps", [1.0] * 5)
        diag_thresholds = calib.get("diag_thresholds", [0.0] * 5)
        ovr_thresholds = calib.get("ovr_thresholds", [0.0] * 5)
        model = SingleModel(xgb_model, diag_temps, diag_thresholds, ovr_thresholds)
    else:
        raise FileNotFoundError("No model files found!")

    # Load bin_edges for KBinsDiscretizer
    bin_edges = calib.get("bin_edges", {})
    if bin_edges:
        print(f"[main] KBinsDiscretizer: {len(bin_edges)} features to bin")
    
    # Load target encoding state
    te_state = calib.get("target_encoding", {})
    if te_state and te_state.get("encoding_maps"):
        print(f"[main] TargetEncoding: {len(te_state.get('encoding_maps', {}))} features to encode")
    
    builder = StatefulFeatureBuilder()
    evaluator = EvalSolution(model, builder, feature_order, bin_edges, te_state)

    y_true = []
    preds_out = []

    print(f"[main] Starting stream inference from {args.input_path}")
    with args.input_path.open("r") as f:
        reader = csv.DictReader(f)
        for row in tqdm(reader):
            numeric_row = {k: float(v) for k, v in row.items() if k != "y"}
            evaluator.eval_model(numeric_row)
            preds_out.append(evaluator.pred_list[-1])
            if "y" in row:
                y_true.append(int(row["y"]))

    if y_true:
        evaluator.eval_score(np.array(y_true))

    with args.output_path.open("w", newline="") as fw:
        writer = csv.writer(fw)
        writer.writerow(["pred"])
        for p in preds_out:
            writer.writerow([p])
    print(f"[main] Saved predictions to {args.output_path}")


if __name__ == "__main__":
    main()
