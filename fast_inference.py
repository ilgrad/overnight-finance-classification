#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fast inference for local testing.
Uses batch processing (NOT for final submission!).
"""

import csv
import json
from pathlib import Path
import numpy as np
import xgboost as xgb
from catboost import CatBoostClassifier
import lightgbm as lgb
from scipy.special import softmax
from tqdm.auto import tqdm
import argparse

from features import StatefulFeatureBuilder, feature_order_from_dict


def load_models(root: Path, lgb_available: bool = True):
    """Load all trained models."""
    print("[fast] Loading models...")
    
    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(root / "model_xgb.json")
    
    cat_model = CatBoostClassifier()
    cat_model.load_model(root / "model_cat.cbm")
    
    # LightGBM (optional - may have failed during training)
    lgb_model = None
    lgb_path = root / "model_lgb.txt"
    if lgb_available and lgb_path.exists():
        lgb_model = lgb.Booster(model_file=str(lgb_path))
        print("[fast] LightGBM loaded")
    else:
        print("[fast] ⚠️ LightGBM not available, using XGB+CAT fallback")
    
    binary_model = None
    if (root / "model_binary.cbm").exists():
        binary_model = CatBoostClassifier()
        binary_model.load_model(root / "model_binary.cbm")
    
    stack_model = None
    if (root / "model_stack.cbm").exists():
        stack_model = CatBoostClassifier()
        stack_model.load_model(root / "model_stack.cbm")
    
    return xgb_model, cat_model, lgb_model, binary_model, stack_model


def count_lines(csv_path: Path) -> int:
    """Fast line counting in file."""
    with csv_path.open("r") as f:
        return sum(1 for _ in f) - 1  # minus header


def apply_target_encoding(feats: dict, te_state: dict) -> dict:
    """Apply TargetExpandingMeanEncoder."""
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
        bin_idx = int(np.digitize(val, edges_arr[1:-1]))
        
        enc_map = encoding_maps.get(feat_name, {})
        encoded_val = enc_map.get(str(bin_idx), global_mean)
        feats[f"{feat_name}_te"] = float(encoded_val)
    
    return feats


def build_features_batch(csv_path: Path, feature_order: list, bin_edges: dict, te_state: dict = None):
    """Build features in batch (using streaming builder for correctness)."""
    builder = StatefulFeatureBuilder()
    X_rows = []
    
    # Count rows first for progress bar
    print(f"[fast] Counting rows in {csv_path}...")
    total_rows = count_lines(csv_path)
    print(f"[fast] Total rows: {total_rows}")
    
    print(f"[fast] Building features...")
    with csv_path.open("r") as f:
        reader = csv.DictReader(f)
        for row in tqdm(reader, total=total_rows, desc="Features", unit="rows"):
            numeric_row = {k: float(v) for k, v in row.items() if k != "y"}
            feats = builder.build_row(numeric_row)
            
            # Apply binning
            for feat_name, edges in bin_edges.items():
                if feat_name in feats:
                    val = feats[feat_name]
                    edges_arr = np.array(edges)
                    bin_idx = np.digitize(val, edges_arr[1:-1])
                    feats[f"{feat_name}_bin"] = float(bin_idx)
            
            # Apply target encoding
            if te_state:
                feats = apply_target_encoding(feats, te_state)
            
            X_rows.append([feats.get(c, 0.0) for c in feature_order])
    
    return np.array(X_rows, dtype=np.float32)


def clean_probs(probs: np.ndarray, n_classes: int) -> np.ndarray:
    probs = np.asarray(probs, dtype=np.float64).reshape(-1, n_classes)
    probs = np.nan_to_num(probs, nan=1e-15, posinf=1.0, neginf=1e-15)
    probs = np.clip(probs, 1e-15, 1.0)
    row_sums = probs.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1.0, row_sums)
    return probs / row_sums


def predict_batch(X, xgb_model, cat_model, lgb_model, binary_model, stack_model, calib):
    """Batch prediction with full pipeline."""
    n_classes = 5
    
    # Progress bar for stages
    steps = ["XGBoost", "CatBoost", "LightGBM", "Binary", "Stacking", "RareBoost", "Calibration"]
    pbar = tqdm(steps, desc="Inference")
    
    # Base model probs
    pbar.set_description("Inference: XGBoost")
    xgb_probs = clean_probs(xgb_model.predict_proba(X), n_classes)
    pbar.update(1)
    
    pbar.set_description("Inference: CatBoost")
    cat_probs = clean_probs(cat_model.predict_proba(X), n_classes)
    pbar.update(1)
    
    pbar.set_description("Inference: LightGBM")
    if lgb_model is not None:
        lgb_probs = clean_probs(lgb_model.predict(X), n_classes)
    else:
        # Fallback: use XGB + CAT average
        lgb_probs = (xgb_probs + cat_probs) / 2
    pbar.update(1)
    
    # Binary model
    binary_probs = None
    pbar.set_description("Inference: Binary")
    if binary_model is not None:
        binary_probs = binary_model.predict_proba(X)
    pbar.update(1)
    
    # Stacking
    stack_probs = None
    pbar.set_description("Inference: Stacking")
    if stack_model is not None and binary_probs is not None:
        stack_features = np.hstack([xgb_probs, cat_probs, lgb_probs, binary_probs])
        stack_probs = clean_probs(stack_model.predict_proba(stack_features), n_classes)
    pbar.update(1)
    
    # Blend
    xgb_w = calib.get("xgb_weight", 0.3)
    cat_w = calib.get("cat_weight", 0.2)
    lgb_w = calib.get("lgb_weight", 0.2)
    stack_w = calib.get("stack_weight", 0.3)
    
    if stack_probs is not None:
        probs = xgb_w * xgb_probs + cat_w * cat_probs + lgb_w * lgb_probs + stack_w * stack_probs
    else:
        probs = xgb_w * xgb_probs + cat_w * cat_probs + lgb_w * lgb_probs
    
    # RareClassBooster
    pbar.set_description("Inference: RareBoost")
    rare_alpha = calib.get("rare_booster_alpha")
    rare_beta = calib.get("rare_booster_beta")
    if rare_alpha is not None and rare_beta is not None and binary_probs is not None:
        rare_signal = binary_probs[:, 1].reshape(-1, 1)
        rare_mask = np.zeros_like(probs)
        rare_mask[:, 3] = 1
        rare_mask[:, 4] = 1
        boost_factor = 1 + rare_alpha * rare_signal * rare_mask
        probs = probs * boost_factor
        probs[:, 3] *= rare_beta
        probs[:, 4] *= rare_beta
        probs = probs / probs.sum(axis=1, keepdims=True)
    pbar.update(1)
    
    # DiagTemp + OVR
    pbar.set_description("Inference: Calibration")
    diag_temps = np.array(calib.get("diag_temps", [1.0] * 5))
    diag_thresholds = np.array(calib.get("diag_thresholds", [0.0] * 5))
    
    log_probs = np.log(np.clip(probs, 1e-15, 1.0))
    scaled_logits = log_probs / diag_temps
    probs = softmax(scaled_logits, axis=1)
    probs = probs - diag_thresholds
    
    # OVR thresholds - best method based on experiments
    ovr_thresholds = np.array(calib.get("ovr_thresholds", [0.0] * 5))
    adj = probs.copy()
    for c, t in enumerate(ovr_thresholds):
        adj[:, c] = np.where(probs[:, c] >= t, probs[:, c], -np.inf)
    mask = np.all(adj == -np.inf, axis=1)
    adj[mask] = probs[mask]
    
    pbar.update(1)
    
    pbar.set_description("Inference: Done!")
    pbar.close()
    
    return adj.argmax(axis=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", type=Path, default=Path("../data/test.csv"))
    parser.add_argument("--output-path", type=Path, default=Path("submission.csv"))
    args = parser.parse_args()
    
    root = Path(__file__).resolve().parent
    
    # Load configs
    feature_order = json.loads((root / "feature_order.json").read_text())
    calib = json.loads((root / "calibration.json").read_text())
    bin_edges = calib.get("bin_edges", {})
    te_state = calib.get("target_encoding", {})
    
    if bin_edges:
        print(f"[fast] KBinsDiscretizer: {len(bin_edges)} features")
    if te_state and te_state.get("encoding_maps"):
        print(f"[fast] TargetEncoding: {len(te_state.get('encoding_maps', {}))} features")
    
    # Load models
    lgb_available = calib.get("lgb_available", True)  # Default True for backward compatibility
    xgb_model, cat_model, lgb_model, binary_model, stack_model = load_models(root, lgb_available)
    
    # Build features
    X = build_features_batch(args.input_path, feature_order, bin_edges, te_state)
    print(f"[fast] Features shape: {X.shape}")
    
    # Predict
    print("[fast] Running predictions...")
    preds = predict_batch(X, xgb_model, cat_model, lgb_model, binary_model, stack_model, calib)
    
    # Save
    with args.output_path.open("w", newline="") as fw:
        writer = csv.writer(fw)
        writer.writerow(["pred"])
        for p in preds:
            writer.writerow([int(p)])
    
    print(f"[fast] Saved {len(preds)} predictions to {args.output_path}")


if __name__ == "__main__":
    main()
