# SSM Sequential Anomaly Detection
**State Space Models (Griffin/Mamba) vs. Transformers for Sequential Anomaly Detection**

This repository contains a comprehensive Proof-of-Concept demonstrating the advantages of State Space Models (SSMs) over standard Transformer architectures for sequential anomaly detection. By leveraging the Griffin architecture (the foundation of Google's RecurrentGemma) and Mamba, we achieve comparable accuracy to Transformers while maintaining O(1) memory complexity and handling significantly longer sequences.

The repository is organized to facilitate academic review and further research into the application of SSMs in financial fraud, ECG anomaly detection, and market surveillance.

---

## Executive Summary

The core hypothesis of this research is that **State Space Models provide a decisive advantage in domains where long-range temporal dependencies are critical**. While Transformers suffer from quadratic O(n²) memory scaling with sequence length, SSMs process sequences with O(1) per-event complexity, enabling theoretically infinite context windows.

### Key Findings

1. **Memory Scaling:** Griffin handles sequences up to **4,096 tokens** on a standard Tesla T4 GPU, whereas a comparable Transformer runs out of memory (OOM) at just 1,024 tokens. This represents an **8x improvement** in context window size on identical hardware.
2. **Accuracy Parity:** On real-world datasets, SSMs achieve parity with Transformers. In the ECG anomaly detection task, Griffin achieved **95.7% accuracy**, compared to the Transformer's 95.8%.
3. **Latency and Throughput:** SSMs offer superior streaming capabilities. In the Bitcoin anomaly detection benchmark, Griffin demonstrated a processing speed of **534 ticks/second** with O(1) memory, allowing for continuous 24/7 streaming without windowing constraints.
4. **Early Warning Capabilities:** The Griffin model successfully predicted **96% of PVC clusters** in ECG data *before* the event occurred, utilizing only the current recurrent hidden state.

---

## Repository Structure

```text
ssm-sequential-anomaly-detection/
├── README.md                 # This document
├── requirements.txt          # Python dependencies
├── notebooks/                
│   └── SSM_Fraud_Detection_POC.ipynb   # Main Colab notebook
├── src/                      # Core model implementations and utilities
│   ├── griffin_model.py      # RG-LRU Griffin implementation (PyTorch)
│   ├── train.py              # Training loops and utilities
│   ├── future_predictor.py   # Next-step prediction architecture
│   ├── data.py               # General data loading utilities
│   ├── btc_*.py              # Bitcoin anomaly detection code
│   └── ecg_*.py              # ECG anomaly detection code
├── results/                  # Raw JSON output from benchmark runs
│   ├── ecg_results.json      # Detailed ECG metrics and memory scaling
│   └── btc_results.json      # Bitcoin anomaly detection metrics
└── figures/                  # Generated visualizations and dashboards
    ├── summary_dashboard.png
    ├── btc_anomaly_dashboard.png
    ├── memory_scaling.png
    ├── memory_complexity.png
    └── latency_comparison.png
```

---

## Experimental Domains

### 1. Financial Fraud Detection (IBM Credit Card Dataset)

The main notebook focuses on the IBM Credit Card Transactions dataset, which provides a highly realistic environment for sequential modeling.

| Property | Value |
|---|---|
| **Transactions** | 24.4 million |
| **Consumers** | 2,000 synthetic users |
| **Avg. transactions per user** | ~12,000 |
| **Fraud rate** | 0.122% |
| **Objective** | SSMs replace complex feature engineering (rolling velocity) by learning long-term patterns from raw sequential data |

### 2. Physiological Monitoring (ECG Anomaly Detection)

Using the MIT-BIH Arrhythmia Database, we compared Griffin and Transformer architectures on continuous time-series data.

| Property | Value |
|---|---|
| **Total beats** | 109,451 across 48 recordings |
| **Griffin F1** | 0.89 |
| **Transformer F1** | 0.89 |
| **Griffin max sequence** | 4,096 tokens |
| **Transformer max sequence** | 512 tokens (OOM at 1,024) |

### 3. Market Anomaly Detection (Bitcoin Price Action)

A demonstration of the streaming capabilities of SSMs on real-time financial data.

| Metric | Griffin (RG-LRU) | Transformer |
|---|---|---|
| Events detected | 67/67 (100%) | 67/67 (100%) |
| Avg. early warning | 114s | 111s |
| Memory per tick | O(1) | O(n²) |
| Inference speed | 534 ticks/s | 459 ticks/s |
| Can stream 24/7 | Yes | Window only |

---

## Model Architectures

### Griffin (RecurrentGemma)

The Griffin architecture combines two key components to achieve efficient long-context modeling:

1. **RG-LRU (Real-Gated Linear Recurrent Unit):** A gated linear recurrence with a diagonal state matrix, providing O(n) memory and compute complexity.
2. **Local Sliding Window Attention:** Attention limited to a fixed window size, capturing local patterns without quadratic scaling.

### Mamba (Selective State Space)

The notebook also evaluates the Mamba architecture, utilizing a Selective State Space model combined with 1D Convolutions to process sequences efficiently while maintaining dynamic state selection based on the input sequence.

---

## Memory Scaling Comparison

| Sequence Length | Griffin (MB) | Transformer (MB) |
|---|---|---|
| 256 | 1.5 | 2.9 |
| 512 | 1.7 | 6.3 |
| 1,024 | 2.1 | 19.3 |
| 2,048 | 2.9 | 70.4 |
| 4,096 | 4.4 | 273.3 |
| 8,192 | 7.6 | 1,081.7 |
| 16,384 | 13.9 | 4,309.3 |
| 32,768 | 26.5 | 17,206.7 |

> Griffin scales linearly O(n). Transformer scales quadratically O(n²) and runs out of memory at 8,192+ tokens on a T4 GPU.

---

## Getting Started

To reproduce the results, we recommend using Google Colab with a T4 GPU.

1. Open `notebooks/SSM_Fraud_Detection_POC.ipynb` in Google Colab.
2. Ensure a GPU runtime is selected (Runtime -> Change runtime type -> T4 GPU).
3. Follow the instructions in the notebook to input Kaggle credentials for dataset downloading.
4. Execute the cells sequentially to train the models and generate the evaluation metrics.

For local execution of the Python scripts, install the required dependencies:
```bash
pip install -r requirements.txt
```

---

## References

- Gu, A., & Dao, T. (2023). *Mamba: Linear-Time Sequence Modeling with Selective State Spaces.* arXiv:2312.00752
- De, S., et al. (2024). *Griffin: Mixing Gated Linear Recurrences with Local Attention for Efficient Language Models.* arXiv:2402.19427
- Moody, G.B., & Mark, R.G. (2001). *The impact of the MIT-BIH Arrhythmia Database.* IEEE Engineering in Medicine and Biology Magazine, 20(3), 45-50.
