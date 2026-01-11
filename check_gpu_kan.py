import torch
import time
from src.models.kan import KAN

# Cấu hình
BATCH_SIZE = 1024
INPUT_DIM = 40   # Giả lập số features của IDS
OUTPUT_DIM = 2   # Binary classification (Attack vs Benign)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(f"Running on: {DEVICE}")

# Tạo data giả lập
x_train = torch.randn(BATCH_SIZE, INPUT_DIM).to(DEVICE)
y_train = torch.randint(0, OUTPUT_DIM, (BATCH_SIZE,)).to(DEVICE)

# Khởi tạo model KAN
# [INPUT_DIM, 64, OUTPUT_DIM] nghĩa là: Input layer -> Hidden 64 -> Output layer
model = KAN([INPUT_DIM, 64, OUTPUT_DIM]).to(DEVICE)

# Test Forward Pass
start_time = time.time()
output = model(x_train)
end_time = time.time()

print("Output shape:", output.shape)
print(f"Forward pass time: {(end_time - start_time)*1000:.2f} ms")

# Test Backward Pass (Training step)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

start_time = time.time()
optimizer.zero_grad()
loss = criterion(output, y_train)
loss.backward()
optimizer.step()
end_time = time.time()

print(f"Backward pass time: {(end_time - start_time)*1000:.2f} ms")
print(f"Initial Loss: {loss.item():.4f}")
print("✅ KAN Model is running correctly on GPU!")
