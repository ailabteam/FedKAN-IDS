import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import os

def load_data(file_path, sample_ratio=1.0):
    """
    Load dataset từ CSV hoặc Parquet.
    sample_ratio: Lấy bao nhiêu % dữ liệu (dùng 0.1 để test code cho nhanh, 1.0 khi chạy thật)
    """
    print(f"⏳ Loading data from {file_path}...")
    
    if file_path.endswith('.parquet'):
        df = pd.read_parquet(file_path)
    else:
        # Đọc CSV tối ưu bộ nhớ
        df = pd.read_csv(file_path)

    # Nếu file quá lớn, lấy mẫu ngẫu nhiên
    if sample_ratio < 1.0:
        print(f"⚠️ Sampling {sample_ratio*100}% of data...")
        df = df.sample(frac=sample_ratio, random_state=42)

    # Xử lý Label (Giả sử cột label tên là 'Label' hoặc cột cuối cùng)
    # NF-UQ-NIDS-v2: features, Attack, Label
    # Chúng ta sẽ dùng cột Label (Binary: 0=Benign, 1=Attack) hoặc Attack (Multi-class)
    # Ở đây demo Binary trước
    
    # Tìm cột label (thường là cột cuối hoặc cột có tên 'Label')
    if 'Label' in df.columns:
        y = df['Label'].values
        X = df.drop(columns=['Label', 'Attack', 'IPV4_SRC_ADDR', 'IPV4_DST_ADDR'], errors='ignore') # Bỏ IP để tránh overfitting
    else:
        # Fallback nếu không đúng tên cột
        y = df.iloc[:, -1].values
        X = df.iloc[:, :-1]
        
    # Xử lý NaN/Inf
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Encode Labels nếu là string
    if y.dtype == 'object':
        le = LabelEncoder()
        y = le.fit_transform(y)
        
    print(f"✅ Data loaded. Shape: {X.shape}")
    return X.values, y

def preprocess_data(X, y):
    """
    Chuẩn hóa dữ liệu về [0, 1] cho KAN/Neural Networks
    """
    print("⏳ Normalizing data...")
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y

def split_non_iid(X, y, num_clients, alpha=0.5, seed=42):
    """
    Chia dữ liệu theo phân phối Dirichlet (Non-IID).
    alpha: Mức độ Non-IID. Càng nhỏ (0.1) càng lệch dữ liệu. Càng lớn (100) càng giống IID.
    """
    print(f"⏳ Partitioning data into {num_clients} clients (Non-IID, alpha={alpha})...")
    np.random.seed(seed)
    
    n_classes = len(np.unique(y))
    min_size = 0
    N = len(y)
    
    net_dataidx_map = {}

    while min_size < 10: # Đảm bảo mỗi client có ít nhất 10 mẫu
        idx_batch = [[] for _ in range(num_clients)]
        for k in range(n_classes):
            idx_k = np.where(y == k)[0]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
            
            # Cân bằng lại proportions để không bị lệch quá ở index cuối
            proportions = np.array([p * (len(idx_j) < N / num_clients) for p, idx_j in zip(proportions, idx_batch)])
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
            min_size = min([len(idx_j) for idx_j in idx_batch])

    for j in range(num_clients):
        np.random.shuffle(idx_batch[j])
        net_dataidx_map[j] = idx_batch[j]
        
    print("✅ Partitioning complete.")
    return net_dataidx_map

def get_client_dataloader(X, y, data_idxs, batch_size=1024):
    """
    Tạo DataLoader cho một Client cụ thể
    """
    X_client = torch.tensor(X[data_idxs], dtype=torch.float32)
    y_client = torch.tensor(y[data_idxs], dtype=torch.long)
    dataset = TensorDataset(X_client, y_client)
    
    # Pin_memory=True giúp đẩy lên GPU nhanh hơn
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=2)
