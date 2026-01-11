import pandas as pd
import numpy as np

# Đường dẫn file (Bạn sửa lại nếu tên file tải về khác)
FILE_PATH = "data/nf_bot_iot_v2/NF-BoT-IoT-V2.parquet"

try:
    print(f"⏳ Reading {FILE_PATH}...")
    df = pd.read_parquet(FILE_PATH)
    
    print(f"✅ Loaded successfully. Shape: {df.shape}")
    print("\n--- Columns ---")
    print(df.columns.tolist())
    
    print("\n--- Data Types ---")
    print(df.dtypes)
    
    # Kiểm tra cột Label (Thường là 'Label' hoặc 'Attack')
    target_col = 'Label' if 'Label' in df.columns else df.columns[-1]
    print(f"\n--- Target Column Identified: '{target_col}' ---")
    
    print("Distribution:")
    print(df[target_col].value_counts())
    
    # Kiểm tra xem có cột Attack Category không (cho Multi-class)
    if 'Attack' in df.columns:
        print("\n--- Attack Categories ---")
        print(df['Attack'].value_counts())
        
except Exception as e:
    print(f"❌ Error: {e}")
