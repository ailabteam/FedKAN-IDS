import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Cấu hình
RESULTS_DIR = "experiments/results"
PLOTS_DIR = "experiments/plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

# Load dữ liệu
df_kan = pd.read_csv(f"{RESULTS_DIR}/fl_kan_c10_a0.1.csv")
df_mlp = pd.read_csv(f"{RESULTS_DIR}/fl_mlp_c10_a0.1.csv")

def get_comm_at_target(df, target_acc):
    # Tìm round đầu tiên đạt target_acc
    row = df[df['Accuracy'] >= target_acc].head(1)
    if not row.empty:
        return row['Comm_MB'].values[0], row['Round'].values[0]
    return None, None

def print_analysis_table():
    print("\n" + "="*60)
    print(f"{'Metric':<20} | {'FedAvg-MLP':<15} | {'FedKAN (Ours)':<15}")
    print("="*60)
    
    # 1. Final Accuracy
    acc_mlp = df_mlp.iloc[-1]['Accuracy']
    acc_kan = df_kan.iloc[-1]['Accuracy']
    print(f"{'Final Accuracy':<20} | {acc_mlp*100:.2f}%          | {acc_kan*100:.2f}%")
    
    # 2. Total Communication
    comm_mlp = df_mlp.iloc[-1]['Comm_MB']
    comm_kan = df_kan.iloc[-1]['Comm_MB']
    print(f"{'Total Comm (50 Rnds)':<20} | {comm_mlp:.2f} MB        | {comm_kan:.2f} MB")
    
    # 3. Comm to reach 95%
    c95_mlp, r95_mlp = get_comm_at_target(df_mlp, 0.95)
    c95_kan, r95_kan = get_comm_at_target(df_kan, 0.95)
    print(f"{'Comm to reach 95%':<20} | {c95_mlp:.2f} MB (R{r95_mlp})   | {c95_kan:.2f} MB (R{r95_kan})")

    # 4. Comm to reach 99%
    c99_mlp, r99_mlp = get_comm_at_target(df_mlp, 0.99)
    c99_kan, r99_kan = get_comm_at_target(df_kan, 0.99)
    print(f"{'Comm to reach 99%':<20} | {c99_mlp:.2f} MB (R{r99_mlp})  | {c99_kan:.2f} MB (R{r99_kan})")
    print("="*60 + "\n")

def plot_bar_comparison():
    # Tạo Dataframe cho biểu đồ cột
    comm_mlp = df_mlp.iloc[-1]['Comm_MB']
    comm_kan = df_kan.iloc[-1]['Comm_MB']
    
    data = {
        'Model': ['FedAvg-MLP', 'FedKAN (Ours)'],
        'Total Communication (MB)': [comm_mlp, comm_kan]
    }
    df_bar = pd.DataFrame(data)
    
    plt.figure(figsize=(6, 6))
    sns.set_theme(style="whitegrid")
    
    ax = sns.barplot(data=df_bar, x='Model', y='Total Communication (MB)', palette=['#4c72b0', '#dd8452'])
    
    # Thêm số liệu lên đầu cột
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.1f} MB', 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha = 'center', va = 'center', 
                    xytext = (0, 9), 
                    textcoords = 'offset points',
                    fontweight='bold')
        
    plt.title("Total Data Transmitted after 50 Rounds")
    plt.ylabel("Communication Cost (MB)")
    plt.ylim(0, 30)
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/total_comm_bar.png")
    plt.savefig(f"{PLOTS_DIR}/total_comm_bar.pdf")
    print("✅ Saved Bar Chart to experiments/plots/total_comm_bar.png")

if __name__ == "__main__":
    print_analysis_table()
    plot_bar_comparison()
