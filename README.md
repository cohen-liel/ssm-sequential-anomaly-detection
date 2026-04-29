# SSM Fraud Detection PoC (v9.0)
**State Space Models (Griffin/Mamba) vs. Transformers for Sequential Anomaly Detection**

This repository contains the definitive Proof-of-Concept (v9.0) demonstrating the advantages of State Space Models (SSMs) over standard Transformer architectures for sequential anomaly detection. By leveraging the Griffin architecture (the foundation of Google's RecurrentGemma) and Mamba, we achieve comparable accuracy to Transformers while maintaining O(1) memory complexity and handling significantly longer sequences.

This repository is organized to facilitate academic review and further research into the application of SSMs in financial fraud, ECG anomaly detection, and predictive maintenance.

---

## 📊 Executive Summary

The core hypothesis of this research is that **State Space Models provide a decisive advantage in domains where long-range temporal dependencies are critical**. While Transformers suffer from quadratic O(n²) memory scaling with sequence length, SSMs process sequences with O(1) per-event complexity, enabling theoretically infinite context windows.

### Key Findings (v9.0)

1. **Memory Scaling:** Griffin handles sequences up to **4,096 tokens** on a standard Tesla T4 GPU, whereas a comparable Transformer runs out of memory (OOM) at just 1,024 tokens. This represents an 8x improvement in context window size on identical hardware.
2. **Accuracy Parity:** On real-world datasets, SSMs achieve parity with Transformers. In the ECG anomaly detection task, Griffin achieved **95.7% accuracy**, compared to the Transformer's 95.8%.
3. **Latency and Throughput:** SSMs offer superior streaming capabilities. In the Bitcoin anomaly detection PoC, Griffin demonstrated a processing speed of 534 ticks/second with O(1) memory, allowing for continuous 24/7 streaming without windowing constraints.
4. **Early Warning Capabilities:** The Griffin model successfully predicted 96% of Premature Ventricular Contraction (PVC) clusters in ECG data *before* the event occurred, utilizing only the current recurrent hidden state.

---

## 📁 Repository Structure

```text
ssm-fraud-detection-v9/
├── README.md                 # This document
├── requirements.txt          # Python dependencies
├── notebooks/                
│   └── SSM_Fraud_Detection_POC_v9.ipynb  # The definitive v9.0 Colab notebook
├── src/                      # Core model implementations and utilities
│   ├── griffin_model.py      # RG-LRU Griffin implementation (PyTorch)
│   ├── train.py              # Training loops and utilities
│   ├── future_predictor.py   # Next-step prediction architecture
│   ├── data.py               # General data loading utilities
│   ├── btc_*.py              # Bitcoin anomaly detection specific code
│   └── ecg_*.py              # ECG anomaly detection specific code
├── results/                  # Raw JSON output from benchmark runs
│   ├── ecg_results.json      # Detailed ECG metrics and memory scaling
│   └── btc_results.json      # Bitcoin anomaly detection metrics
└── figures/                  # Generated visualizations and dashboards
    ├── summary_dashboard.png # High-level comparison dashboard
    ├── btc_anomaly_dashboard.png
    ├── memory_scaling.png
    ├── memory_complexity.png
    └── latency_comparison.png
```

---

## 🔬 Experimental Domains

### 1. Financial Fraud Detection (IBM Credit Card Dataset)
The v9.0 notebook focuses on the IBM Credit Card Transactions dataset, which provides a highly realistic environment for sequential modeling.
*   **Scale:** 24.4 million transactions across 2,000 synthetic consumers.
*   **Sequence Length:** Consumers average ~12,000 transactions, requiring models to maintain long-term context.
*   **Imbalance:** A realistic fraud rate of 0.122%.
*   **Objective:** Demonstrate that SSMs can replace complex feature engineering (e.g., rolling velocity features) by natively learning long-term patterns directly from raw sequential data.

### 2. Physiological Monitoring (ECG Anomaly Detection)
Using the MIT-BIH Arrhythmia Database, we compared Griffin and Transformer architectures on continuous time-series data.
*   **Scale:** 109,451 total beats across 48 half-hour recordings.
*   **Result:** Griffin achieved comparable F1 scores (~0.89) while demonstrating linear O(n) memory scaling, proving its viability for real-time edge deployment.

### 3. Market Anomaly Detection (Bitcoin Price Action)
A demonstration of the streaming capabilities of SSMs.
*   **Result:** Griffin successfully detected 100% of defined anomaly events (67/67) with an average early warning time of 114 seconds, operating continuously without the fixed-window limitations of the Transformer baseline.

---

## 🧠 Model Architectures

### Griffin (RecurrentGemma)
The Griffin architecture combines two key components to achieve efficient long-context modeling:
1.  **RG-LRU (Real-Gated Linear Recurrent Unit):** A gated linear recurrence with a diagonal state matrix, providing O(n) memory and compute complexity.
2.  **Local Sliding Window Attention:** Attention limited to a fixed window size, capturing local patterns without quadratic scaling.

### Mamba (Selective State Space)
The v9.0 notebook also evaluates the Mamba architecture, utilizing a Selective State Space model combined with 1D Convolutions to process sequences efficiently while maintaining dynamic state selection based on the input sequence.

---

## 🚀 Getting Started

To reproduce the v9.0 results, we recommend using Google Colab with a T4 GPU.

1.  Open `notebooks/SSM_Fraud_Detection_POC_v9.ipynb` in Google Colab.
2.  Ensure a GPU runtime is selected (`Runtime` -> `Change runtime type` -> `T4 GPU`).
3.  Follow the instructions in the notebook to input Kaggle credentials for dataset downloading.
4.  Execute the cells sequentially to train the models and generate the evaluation metrics.

For local execution of the Python scripts, install the required dependencies:
```bash
pip install -r requirements.txt
```

---
*This repository represents a snapshot of the v9.0 research findings. For inquiries or collaboration regarding the application of SSMs in cybersecurity and fraud detection, please refer to the primary author.*
