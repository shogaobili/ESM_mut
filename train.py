import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from torch import nn, optim
from model import ESMBackbone  # 假设 model.py 在同一目录下
import argparse

# 加载数据
train_data = np.load("ESM_feat_lable.npy")
labels = torch.tensor(train_data[:, 0], dtype=torch.float32)
features = torch.tensor(train_data[:, 1:], dtype=torch.float32)
ds = TensorDataset(labels, features)
dl = DataLoader(ds, batch_size=16, drop_last=False)

# 获取一个批次的数据并打印
labels, features = next(iter(dl))
print("features = ", features, features.shape)
print("labels = ", labels, labels.shape)

# 定义模型参数
parser = argparse.ArgumentParser()
parser.add_argument('--backbone', type=str, default='esm1b_t33_650M_UR50S')  # 这是一个示例参数
parser.add_argument('--freeze_at', type=int, default=0)
parser.add_argument('--aa_expand', type=str, default='backbone')
args = parser.parse_args([])  # 如果在脚本中运行，传递空列表以避免报错

# 实例化模型
model = ESMBackbone(args)
model.train()  # 切换模型到训练模式

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练循环
num_epochs = 10
for epoch in range(num_epochs):
    for batch in dl:
        batch_labels, batch_features = batch
        optimizer.zero_grad()
        outputs = model(batch_features, batch)
        loss = criterion(outputs['bb_feat'], batch_labels)
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")