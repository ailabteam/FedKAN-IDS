import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import glob

# Cấu hình Style cho bài báo Springer
sns.set_theme(style="whitegrid", font_scale=1.2)
plt.rcParams["font.family"] = "serif" # Font giống Times New Roman

RESULTS_DIR = "experiments/results"
PLOTS_DIR = "experiments/plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

def plot_metrics():
    # 1. Load tất cả file CSV kết quả
    all_files = glob.glob(os.path.join(RESULTS_DIR, "*.csv"))
    if not all_files:
        print("❌ No result files found!")
        return
    
    df_list = []
    for f in all_files:
        df_list.append(pd.read_csv(f))
    
    df = pd.concat(df_list, ignore_index=True)
    
    # Map tên model cho đẹp
    df['Model'] = df['Model'].replace({'kan': 'FedKAN (Ours)', 'mlp': 'FedAvg-MLP (Baseline)'})
    
    # 2. Vẽ Accuracy vs Rounds
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x="Round", y="Accuracy", hue="Model", style="Model", markers=True, dashes=False, ci="sd")
    plt.title("Convergence Analysis: FedKAN vs FedMLP")
    plt.ylabel("Global Accuracy")
    plt.xlabel("Communication Rounds")
    plt.ylim(0.8, 1.0) # Zoom vào vùng độ chính xác cao
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/accuracy_convergence.pdf") # PDF cho nét
    plt.savefig(f"{PLOTS_DIR}/accuracy_convergence.png")
    print("✅ Saved Accuracy Plot")

    # 3. Vẽ Communication Efficiency
    # Trục X là dung lượng gửi đi (MB), Trục Y là Accuracy
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x="Comm_MB", y="Accuracy", hue="Model", style="Model", markers=True, dashes=False, ci=None)
    plt.title("Communication Efficiency")
    plt.ylabel("Global Accuracy")
    plt.xlabel("Total Data Transmitted (MB)")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/comm_efficiency.pdf")
    plt.savefig(f"{PLOTS_DIR}/comm_efficiency.png")
    print("✅ Saved Efficiency Plot")

if __name__ == "__main__":
    plot_metrics()
