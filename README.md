# FedKAN-IDS: A Communication-Efficient Federated Learning Approach for IoT Intrusion Detection

![Status](https://img.shields.io/badge/Status-Research_Prototype-blue)
![Python](https://img.shields.io/badge/Python-3.10-green)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

This repository contains the official implementation of the paper **"FedKAN-IDS: A Communication-Efficient Federated Learning Approach for IoT Intrusion Detection"**.

The project proposes a novel framework integrating **Kolmogorov-Arnold Networks (KAN)** into **Federated Learning (FL)** to secure IoT networks. Our experiments demonstrate that FedKAN achieves state-of-the-art detection accuracy (~99.4%) while reducing communication costs by approximately **50%** compared to traditional MLP-based Federated Learning.

## 🌟 Key Features

*   **Federated KAN Architecture:** Replaces traditional MLPs with efficient KAN layers (B-Splines) for local updates.
*   **Communication Efficiency:** Significantly reduces bandwidth usage suitable for resource-constrained IoT Edge devices.
*   **Non-IID Robustness:** Tested under strict Non-IID data distribution (Dirichlet $\alpha=0.1$) to simulate realistic IoT environments.
*   **Imbalanced Data Handling:** Automatic down-sampling strategy to handle the severe class imbalance in IoT traffic.

## 📂 Project Structure

```text
FedKAN-IDS/
├── data/                   # Dataset storage
│   └── nf_bot_iot_v2/      # NF-BoT-IoT-v2 Parquet file
├── experiments/            # Experiment logs and figures
│   ├── results/            # CSV logs of training
│   └── plots/              # Generated charts (PDF/PNG)
├── src/
│   ├── models/             # Model definitions
│   │   ├── kan.py          # Kolmogorov-Arnold Network implementation
│   │   └── mlp.py          # MLP Baseline
│   └── utils/              # Utilities
│       ├── data_utils.py   # Data loading & Non-IID partitioning
│       └── plot_results.py # Visualization scripts
├── main_fl.py              # Main entry point for FL experiments
├── check_params.py         # Script to compare model parameters
└── requirements.txt        # Python dependencies
```

## 📊 Dataset

We use the **NF-BoT-IoT-v2** dataset, a NetFlow-based dataset for IoT Botnet detection.

*   **Source:** [NF-BoT-IoT-v2 on Kaggle](https://www.kaggle.com/datasets/dhoogla/nfbotiotv2) (derived from the original University of Queensland dataset).
*   **Format:** Parquet (Optimized for speed and storage).
*   **Features:** 43 NetFlow features.
*   **Samples:** ~30M flows (We apply down-sampling for experimental efficiency).

### Setup Instructions:
1.  Download the file `NF-BoT-IoT-V2.parquet`.
2.  Place it in the following directory:
    ```bash
    mkdir -p data/nf_bot_iot_v2/
    # Move the downloaded file here
    ```

## 🚀 Installation

We recommend using `conda` to manage the environment.

```bash
# 1. Create environment
conda create -n fedkan python=3.10 -y
conda activate fedkan

# 2. Install PyTorch (Adjust CUDA version according to your GPU driver)
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia

# 3. Install dependencies
pip install -r requirements.txt
```

## 🏃‍♂️ Usage

### 1. Run FedKAN (Proposed Method)
Run the Federated Learning loop using KAN with Non-IID data ($\alpha=0.1$).

```bash
python main_fl.py --model kan --clients 10 --rounds 50 --alpha 0.1 --gpu 0 --seeds 42 2024 2026
```

### 2. Run FedAvg-MLP (Baseline)
Run the baseline model for comparison.

```bash
python main_fl.py --model mlp --clients 10 --rounds 50 --alpha 0.1 --gpu 0 --seeds 42 2024 2026
```

### 3. Analyze & Plot Results
After running both experiments, generate comparison charts (Communication Efficiency, Convergence).

```bash
python src/utils/plot_results.py
```
*The plots will be saved in `experiments/plots/`.*

## 📈 Experimental Results

Summary of performance on **NF-BoT-IoT-v2** (Non-IID, $\alpha=0.1$, 10 Clients, 50 Rounds):

| Model | Parameters | Final Accuracy | Total Comm. (MB) | Comm. Reduction |
| :--- | :--- | :--- | :--- | :--- |
| **FedAvg-MLP** | 7,106 | 99.48% | 26.62 MB | - |
| **FedKAN (Ours)** | **3,600** | **99.38%** | **13.12 MB** | **~50.7%** |

> **Note:** FedKAN achieves comparable accuracy to the SOTA baseline while requiring only half the communication bandwidth.

## 🤝 Contributing

This project is open for research collaboration. Please open an issue or pull request if you have suggestions.

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
