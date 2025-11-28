import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from data import load_efficiency_data

class EfficiencyDataset(Dataset):
    def __init__(self, x, y):
        self.X = x   # shape (N, 62)
        self.Y = y   # shape (N,) or (N,1)

        if self.Y.ndim == 1:
            self.Y = self.Y[:, None]  # 变成 (N, 1)

        # 转成 torch 张量
        self.X = torch.tensor(self.X, dtype=torch.float32)
        self.Y = torch.tensor(self.Y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


class EfficiencyMLP(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(62, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)     # 输出一个标量效率
        )

    def forward(self, x):
        return self.model(x)
    
# 替换原来的数据加载与 DataLoader 创建为下面内容（添加验证集划分）
# train_x,train_y = load_efficiency_data()
X, y = load_efficiency_data()

# 划分训练集 / 验证集 / 测试集
val_ratio = 0.1
test_ratio = 0.1
rng = np.random.default_rng(42)
idx = rng.permutation(len(X))

n_train = int(len(X) * (1 - val_ratio - test_ratio))
n_val = int(len(X) * val_ratio)

train_idx = idx[:n_train]
val_idx = idx[n_train:n_train + n_val]
test_idx = idx[n_train + n_val:]

train_x, train_y = X[train_idx], y[train_idx]
val_x, val_y = X[val_idx], y[val_idx]
test_x, test_y = X[test_idx], y[test_idx]

train_dataset = EfficiencyDataset(train_x, train_y)
val_dataset = EfficiencyDataset(val_x, val_y)
test_dataset = EfficiencyDataset(test_x, test_y)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EfficiencyMLP().to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

EPOCHS = 200

# 添加用于保存最优模型的变量
best_val = float("inf")

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for x_batch, y_batch in train_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        
        pred = model(x_batch)

        loss = criterion(pred, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(x_batch)

    avg_loss = total_loss / len(train_dataset)

    # 计算验证集损失
    model.eval()
    val_total = 0
    with torch.no_grad():
        for x_val, y_val in val_loader:
            x_val = x_val.to(device)
            y_val = y_val.to(device)
            pred_val = model(x_val)
            loss_val = criterion(pred_val, y_val)
            val_total += loss_val.item() * len(x_val)
    avg_val = val_total / len(val_dataset)

    print(f"Epoch {epoch+1}/{EPOCHS}, Train Loss = {avg_loss:.6f}, Val Loss = {avg_val:.6f}")

    # 保存验证集上最好的模型
    if avg_val < best_val:
        best_val = avg_val
        torch.save(model.state_dict(), "efficiency_mlp_best.pth")
        print(f"验证集损失改善，已保存模型：efficiency_mlp_best.pth")

# 可选：保存训练结束时的最终模型（保持兼容）
torch.save(model.state_dict(), "efficiency_mlp.pth")
print("最终模型已保存为 efficiency_mlp.pth；验证集上最佳模型为 efficiency_mlp_best.pth")

# 使用在验证集上最好的模型在测试集上评估
# 加载最佳模型
best_path = "efficiency_mlp_best.pth"
model.load_state_dict(torch.load(best_path, map_location=device))
model.eval()

test_total = 0.0
mae_total = 0.0
criterion_mae = nn.L1Loss(reduction='sum')

with torch.no_grad():
    for x_test, y_test in test_loader:
        x_test = x_test.to(device)
        y_test = y_test.to(device)
        pred_test = model(x_test)
        loss_test = criterion(pred_test, y_test)  # MSE sum already handled below
        test_total += loss_test.item() * len(x_test) / (len(x_test))  # accumulate per-batch MSE mean
        mae_total += criterion_mae(pred_test, y_test).item()

# 计算平均 MSE（按样本）和 MAE
avg_test_mse = 0.0
# 为了精确按样本平均，重新计算基于 dataset 长度
with torch.no_grad():
    mse_sum = 0.0
    for x_test, y_test in test_loader:
        x_test = x_test.to(device)
        y_test = y_test.to(device)
        pred_test = model(x_test)
        mse_sum += nn.functional.mse_loss(pred_test, y_test, reduction='sum').item()
    avg_test_mse = mse_sum / len(test_dataset)

avg_test_mae = mae_total / len(test_dataset)

print(f"Test MSE = {avg_test_mse:.6f}, Test MAE = {avg_test_mae:.6f}")

# 新增：打印部分测试样本的预测与真值
n_show = min(10, len(test_dataset))
print(f"\n显示 {n_show} 个测试样本的预测 vs 真值（pred -> true）:")
for i in range(n_show):
    x_s, y_s = test_dataset[i]               # 返回 CPU tensor
    x_s = x_s.unsqueeze(0).to(device)
    with torch.no_grad():
        pred_s = model(x_s).squeeze().cpu().item()
    true_s = y_s.squeeze().cpu().item()
    print(f"样本 {i+1}: {pred_s:.6f} -> {true_s:.6f}")