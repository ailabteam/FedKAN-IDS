import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import argparse
import time
import copy
import os
import gc
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from src.models.kan import KAN
from src.utils.data_utils import load_data, preprocess_data, split_non_iid, get_client_dataloader

# --- CẤU HÌNH ---
DATA_PATH = "data/nf_bot_iot_v2/NF-BoT-IoT-V2.parquet"
RESULTS_DIR = "experiments/results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# --- CLASS MLP BASELINE ---
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), # Thêm 1 lớp ẩn để công bằng với độ sâu của KAN
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, x):
        return self.net(x)

# --- HÀM TRAIN LOCAL ---
def local_train(model, train_loader, epochs, lr, device):
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for _ in range(epochs):
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
    return model.state_dict()

# --- HÀM TEST GLOBAL ---
def test_global(model, test_loader, device):
    model.eval()
    all_preds = []
    all_targets = []
    total_loss = 0
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item()
            
            _, preds = torch.max(output, 1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            
    acc = accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds, average='macro') # Macro cho cân bằng
    precision = precision_score(all_targets, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_targets, all_preds, average='macro', zero_division=0)
    avg_loss = total_loss / len(test_loader)
    
    return acc, f1, precision, recall, avg_loss

# --- HÀM FEDERATED AVERAGING ---
def fed_avg(global_weights, local_weights):
    avg_weights = copy.deepcopy(global_weights)
    for key in avg_weights.keys():
        avg_weights[key] = torch.stack([w[key] for w in local_weights]).mean(0)
    return avg_weights

# --- MAIN FL LOOP ---
def run_experiment(args, seed):
    print(f"\n🚀 STARTING: Model={args.model}, Seed={seed}, Alpha={args.alpha}")
    
    # 1. Load & Balance Data
    # Chiến thuật: Lấy toàn bộ Benign (130k), lấy mẫu Attack (130k) -> Tổng 260k mẫu.
    # Đủ lớn để train, đủ nhỏ để nhanh, và CÂN BẰNG.
    
    df = pd.read_parquet(DATA_PATH, columns=['Label', 'Attack'] + [c for c in pd.read_parquet(DATA_PATH).columns if c not in ['Label', 'Attack']])
    
    df_benign = df[df['Label'] == 0]
    df_attack = df[df['Label'] == 1].sample(n=len(df_benign), random_state=seed) # Down-sample Attack
    
    df_balanced = pd.concat([df_benign, df_attack]).sample(frac=1, random_state=seed).reset_index(drop=True)
    
    print(f"📊 Balanced Data Shape: {df_balanced.shape}")
    print(f"   - Benign: {len(df_benign)}")
    print(f"   - Attack: {len(df_attack)}")
    
    # Prepare X, y
    y = df_balanced['Label'].values
    X = df_balanced.drop(columns=['Label', 'Attack'], errors='ignore').values
    
    # Preprocess
    X, y = preprocess_data(X, y)
    
    # Split Train/Test (80/20)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)
    
    # Create Test Loader (Centralized Test Set for Global Evaluation)
    test_tensor_x = torch.tensor(X_test, dtype=torch.float32)
    test_tensor_y = torch.tensor(y_test, dtype=torch.long)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_tensor_x, test_tensor_y), batch_size=2048, pin_memory=True, num_workers=2)
    
    # Partition Non-IID for Clients
    client_idxs = split_non_iid(X_train, y_train, args.clients, alpha=args.alpha, seed=seed)
    
    # Init Model
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    input_dim = X.shape[1]
    output_dim = 2 # Binary
    
    if args.model == 'kan':
        # KAN: [Input, Hidden, Output]
        # Grid size nhỏ (3-5) để tính toán nhanh
        global_model = KAN([input_dim, 64, output_dim], grid_size=3, spline_order=3).to(device)
    else:
        # MLP: Tương đương về độ phức tạp
        global_model = MLP(input_dim, 128, output_dim).to(device)
        
    global_weights = global_model.state_dict()
    
    # Logs
    results = []
    
    # FL Round Loop
    for round_idx in tqdm(range(args.rounds), desc="FL Rounds"):
        local_weights_list = []
        
        # Client Selection (Nếu muốn scale lớn, chỉ chọn 1 phần)
        # Ở đây chạy full clients (10-20) cho ổn định
        selected_clients = range(args.clients) 
        
        for client_id in selected_clients:
            # Get Local Data
            train_loader = get_client_dataloader(X_train, y_train, client_idxs[client_id], batch_size=args.batch_size)
            
            # Local Train
            local_model = copy.deepcopy(global_model) # Copy weights
            local_model.load_state_dict(global_weights)
            
            w = local_train(local_model, train_loader, args.epochs, args.lr, device)
            local_weights_list.append(w)
            
            del local_model
            
        # Aggregation
        global_weights = fed_avg(global_weights, local_weights_list)
        global_model.load_state_dict(global_weights)
        
        # Evaluation
        if (round_idx + 1) % 1 == 0: # Test every round
            acc, f1, prec, rec, loss = test_global(global_model, test_loader, device)
            
            # Tính Communication Cost (ước lượng sơ bộ qua số params)
            param_size = sum(p.numel() for p in global_model.parameters()) * 4 / (1024*1024) # MB (float32)
            comm_cost = param_size * args.clients * 2 * (round_idx + 1) # Upload + Download
            
            log_entry = {
                "Round": round_idx + 1,
                "Accuracy": acc,
                "F1": f1,
                "Precision": prec,
                "Recall": rec,
                "Loss": loss,
                "Comm_MB": comm_cost,
                "Seed": seed,
                "Model": args.model
            }
            results.append(log_entry)
            
            # Print tiến độ mỗi 5 rounds
            if (round_idx + 1) % 5 == 0:
                print(f"   Round {round_idx+1}: Acc={acc:.4f}, F1={f1:.4f}")
                
        # Dọn dẹp GPU memory
        torch.cuda.empty_cache()

    return pd.DataFrame(results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='kan', choices=['kan', 'mlp'])
    parser.add_argument('--clients', type=int, default=10)
    parser.add_argument('--rounds', type=int, default=20)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--alpha', type=float, default=0.5, help='Dirichlet alpha for Non-IID')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 2024, 2026])
    
    args = parser.parse_args()
    
    all_results = []
    for seed in args.seeds:
        df_res = run_experiment(args, seed)
        all_results.append(df_res)
        
    final_df = pd.concat(all_results)
    
    # Lưu file kết quả tổng hợp
    filename = f"{RESULTS_DIR}/fl_{args.model}_c{args.clients}_a{args.alpha}.csv"
    final_df.to_csv(filename, index=False)
    print(f"\n✅ Experiment Complete! Results saved to {filename}")
