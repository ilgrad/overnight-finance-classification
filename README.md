# 🏆 Overnight Finance Challenge

> **Multiclass Time Series Classification for Financial Data**  
> Kaggle-style competition solution with **F1-macro = 0.4675**

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-green.svg)](https://xgboost.readthedocs.io)
[![CatBoost](https://img.shields.io/badge/CatBoost-1.2+-yellow.svg)](https://catboost.ai)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.0+-orange.svg)](https://lightgbm.readthedocs.io)
[![GPU](https://img.shields.io/badge/GPU-CUDA%20%7C%20OpenCL-red.svg)](#gpu-acceleration)

---

## 🔗 Resources

- **Dataset**: [Kaggle Dataset](https://www.kaggle.com/datasets/21aec48d66d01ada53f468b9552136a97182a996bae9845a5787c51b0cbf592b)
- **Competition Rules & Problem Description**: [Notion Page](https://www.notion.so/Overnight-Finance-Challenge-ETH-USDC-Predictions-25a2ab88fe6380f996e7c61fc9a9e036)

---

## 📋 Problem Statement

Predict **5-class price movement** from Limit Order Book (LOB) data in a **streaming scenario** — no future data leakage allowed.

| Class | Description | Distribution |
|:-----:|-------------|:------------:|
| 0 | No significant change | 64.09% |
| 1 | Small increase | 16.42% |
| 2 | Small decrease | 16.43% |
| 3 | Large increase | 1.56% |
| 4 | Large decrease | 1.50% |

### Class Distribution Visualization

```
Original Dataset (n = 2,218,127)
────────────────────────────────────────────────────────
Class 0 ████████████████████████████████████████  64.09%  (1,421,692)
Class 1 ██████████                                16.42%  (  364,316)
Class 2 ██████████                                16.43%  (  364,413)
Class 3 █                                          1.56%  (   34,543)
Class 4 █                                          1.50%  (   33,163)
────────────────────────────────────────────────────────

After Resampling (Downsample + SMOTE)
────────────────────────────────────────────────────────
Class 0 █████████████████████                     35.29%  (  484,418)
Class 1 ██████████████                            23.46%  (  321,953)
Class 2 ██████████████                            23.60%  (  323,938)
Class 3 █████                                      8.82%  (  121,104)
Class 4 █████                                      8.82%  (  121,104)
────────────────────────────────────────────────────────
```

### Key Challenge
Classes 3 and 4 represent rare but important price movements (~3% combined). The main challenge is achieving good F1-macro while handling this severe class imbalance.

---

## 🏗️ Solution Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           TRAINING PIPELINE                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────┐    ┌─────────────────┐    ┌────────────────────┐          │
│  │  Raw CSV    │───▶│ StatefulFeature │───▶│  208 LOB Features  │          │
│  │  (2.2M rows)│    │    Builder      │    │   (streaming)      │          │
│  └─────────────┘    └─────────────────┘    └─────────┬──────────┘          │
│                                                       │                      │
│  ┌────────────────────────────────────────────────────▼──────────────────┐  │
│  │                    FEATURE AUGMENTATION                                │  │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌────────────────────┐    │  │
│  │  │ KBinsDiscretizer│  │ Target Encoding │  │ Hybrid Selection   │    │  │
│  │  │   (+23 bins)    │  │  (+9 features)  │  │ (corr + redundancy)│    │  │
│  │  └─────────────────┘  └─────────────────┘  └────────────────────┘    │  │
│  │                         240 Features Total                             │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
│                                          │                                   │
│  ┌───────────────────────────────────────▼───────────────────────────────┐  │
│  │                         RESAMPLING                                     │  │
│  │         RandomUnderSampler (Class 0) + SMOTE (Classes 3,4)            │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
│                                          │                                   │
│  ┌───────────────────────────────────────▼───────────────────────────────┐  │
│  │                      OPTUNA HPO (60 trials)                            │  │
│  │              TPE Sampler | Early Stopping | GPU Acceleration           │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
│                                          │                                   │
│  ┌───────────────────────────────────────▼───────────────────────────────┐  │
│  │                        BASE MODELS                                     │  │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────────────┐  │  │
│  │  │ XGBoost  │  │ CatBoost │  │ LightGBM │  │ Binary Rare Detector │  │  │
│  │  │ (CUDA)   │  │ (CUDA)   │  │ (OpenCL) │  │     (CatBoost)       │  │  │
│  │  │ F1=0.433 │  │ F1=0.440 │  │ F1=0.431 │  │     AUC=0.833        │  │  │
│  │  └────┬─────┘  └────┬─────┘  └────┬─────┘  └──────────┬───────────┘  │  │
│  └───────┼─────────────┼─────────────┼──────────────────┼────────────────┘  │
│          │             │             │                  │                    │
│  ┌───────▼─────────────▼─────────────▼──────────────────▼────────────────┐  │
│  │                      STACKING META-MODEL                               │  │
│  │                   (CatBoost on OOF predictions)                        │  │
│  │                          F1 = 0.440                                    │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
│                                          │                                   │
│  ┌───────────────────────────────────────▼───────────────────────────────┐  │
│  │                    ENSEMBLE BLENDING                                   │  │
│  │          XGB=0.40 | CAT=0.30 | LGB=0.15 | STACK=0.15                  │  │
│  │                       Blend F1 = 0.450                                 │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
│                                          │                                   │
│  ┌───────────────────────────────────────▼───────────────────────────────┐  │
│  │                       CALIBRATION                                      │  │
│  │  ┌────────────────┐  ┌────────────────────┐  ┌──────────────────┐    │  │
│  │  │ RareClassBoost │  │ DiagonalTempScale  │  │ OVR Thresholds   │    │  │
│  │  │    +1.31%      │  │      +0.36%        │  │     +0.12%       │    │  │
│  │  └────────────────┘  └────────────────────┘  └──────────────────┘    │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
│                                          │                                   │
│                              ┌───────────▼───────────┐                       │
│                              │   FINAL F1 = 0.4675   │                       │
│                              │   (+2.75% vs best     │                       │
│                              │    single model)      │                       │
│                              └───────────────────────┘                       │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 📊 Results

### Model Performance Comparison

| Model | F1-macro | Training Time |
|-------|:--------:|:-------------:|
| CatBoost | 0.4400 | 82s |
| Stacking | 0.4404 | 93s |
| XGBoost (Focal Loss) | 0.4331 | 770s |
| LightGBM | 0.4307 | 323s |

### Calibration Impact

```
F1-macro Progression
────────────────────────────────────────────
Baseline Blend      │ 0.4497  │ ████████████████████████
+ RareClassBooster  │ 0.4628  │ █████████████████████████ (+1.31%)
+ DiagTempScale     │ 0.4663  │ █████████████████████████ (+0.36%)
+ OVR Thresholds    │ 0.4675  │ █████████████████████████ (+0.12%)
────────────────────────────────────────────
Total improvement: +1.78% over baseline blend
```

### Final Results

| Metric | Value |
|--------|:-----:|
| **Final F1-macro** | **0.4675** |
| Best single model | 0.4400 |
| Ensemble improvement | +2.75% |
| Total training time | ~126 min |

---

## 🔧 Feature Engineering

### Feature Pipeline

| Stage | Input | Output | Method |
|-------|:-----:|:------:|--------|
| Raw LOB | 41 columns | 208 features | StatefulFeatureBuilder |
| Binning | 23 features | +23 binned | KBinsDiscretizer (quantile, 5 bins) |
| Target Encoding | 9 features | +9 encoded | Expanding Mean (streaming-safe) |
| **Total** | | **240 features** | |

### Key Features (Top-10 by Target Correlation)

| Feature | Correlation | Description |
|---------|:-----------:|-------------|
| `ewma_micro_mad_5` | 0.207 | EWMA of microprice absolute deviation |
| `keltner_position_10` | 0.184 | Position within Keltner channel |
| `microprice_range_5` | 0.182 | Microprice range over 5 periods |
| `microprice_range_20` | 0.175 | Microprice range over 20 periods |
| `keltner_position_20` | 0.170 | Position within wider Keltner channel |
| `mid_price_ratio_3_te` | 0.166 | Target-encoded price ratio |
| `mid_price_1_rolling_std_5` | 0.164 | Rolling volatility |
| `microprice_1_rolling_std_5` | 0.163 | Microprice volatility |
| `keltner_position_10_bin` | 0.159 | Binned Keltner position |
| `spread_abs_mean_te` | 0.159 | Target-encoded spread |

### Streaming-Compliant Design

All features are computed using only **past data** to comply with competition rules:
- Rolling windows (look-back only)
- Expanding mean for target encoding
- KAMA (Kaufman Adaptive Moving Average)
- EWMA (Exponentially Weighted Moving Average)

---

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/overnight-finance-challenge.git
cd overnight-finance-challenge

# Create virtual environment (using uv - recommended)
uv venv .venv --python 3.10
source .venv/bin/activate

# Install dependencies
uv pip install -r requirements.txt
```

### Training

```bash
# Full training with GPU and all optimizations
python train_model.py \
    --train-path data/train.csv \
    --use-gpu \
    --n-estimators 3000 \
    --learning-rate 0.01 \
    --focal-loss \
    --focal-gamma 2.0 \
    --oversample \
    --oversample-ratio 0.25 \
    --downsample \
    --downsample-ratio 0.5 \
    --optuna-trials 60 \
    --val-fraction 0.1
```

### Inference

```bash
# Fast local inference
python fast_inference.py \
    --test-path data/test.csv \
    --output-path predictions.csv

# Competition-format inference (sample-by-sample)
python main.py
```

---

## 🎮 GPU Acceleration

| Framework | Backend | Status |
|-----------|---------|:------:|
| XGBoost | CUDA | ✅ Native support |
| CatBoost | CUDA | ✅ Native support |
| LightGBM | OpenCL | ✅ Works out-of-box |

> **Note**: LightGBM CUDA requires building from source with specific GCC version. OpenCL version is used as stable alternative.

---

## 📁 Project Structure

```
overnight-finance-challenge/
├── final_code/
│   ├── train_model.py      # Main training script
│   ├── main.py             # Competition inference (streaming)
│   ├── fast_inference.py   # Fast batch inference
│   ├── features.py         # StatefulFeatureBuilder
│   ├── requirements.txt    # Dependencies
│   │
│   ├── model_xgb.json      # Trained XGBoost
│   ├── model_cat.cbm       # Trained CatBoost
│   ├── model_lgb.txt       # Trained LightGBM
│   ├── model_stack.cbm     # Stacking meta-model
│   ├── model_binary.cbm    # Rare class detector
│   ├── calibration.json    # Calibration parameters
│   └── feature_order.json  # Feature names
│
├── data/
│   ├── train.csv           # Training data
│   └── test.csv            # Test data
│
└── README.md
```

---

## 🧪 What Didn't Work

| Approach | Issue |
|----------|-------|
| ETNA library | NaN bug with NumPy 2.x |
| F1-EM Optimizer | O(n²) memory, poor results |
| Platt Scaling | Worse than temperature scaling |
| Heavy regularization | Hurt rare class detection |
| Balanced class weights only | Not enough for 1.5% classes |

---

## 💡 Key Insights

1. **Calibration is crucial** — RareClassBooster alone gave +1.31% F1
2. **Downsample majority** before SMOTE — better than SMOTE alone
3. **Calibrate on ORIGINAL data** — never on augmented samples
4. **Stacking adds value** — even when individual models are similar
5. **Focal Loss helps** — focuses on hard-to-classify samples

---


## 📄 License

MIT License — feel free to use this code for learning and reference.

---

<p align="center">
  <b>⭐ If you found this useful, consider giving it a star! ⭐</b>
</p>

