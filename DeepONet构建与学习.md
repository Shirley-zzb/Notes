# DeepONet构建与学习

还是宗壮师兄那篇论文，使用了神经网络模型后仍然有着很差的泛化性和外推性，但是使用DeepONet模型居然有着非常好的泛化性和外推性，即使特征的种类/变化很少，非常出乎意料。下面介绍了这个工作过程中对DeepONet的理解与学习。这个过程也增强了对torch部分方法的使用，如数据加载DataLoader的使用、KFold交叉验证的单独使用、数据集划分train_test_split的使用等。

**DeepONet** 是一种基于神经网络的结构，用于学习操作（operators），特别适用于处理输入是函数的情况（例如，管径和坐标点分别作为输入）。在你的问题中，管径 D 和坐标 x,y 是两个不同类型的输入，DeepONet 可以很好地分开处理它们。

### DeepONet 简介：

DeepONet 的核心思想是将输入拆分为两个部分：

1. **Branch Net**（分支网络）：处理输入函数（如管径 D）。
2. **Trunk Net**（主干网络）：处理坐标输入（如 x,y），即函数定义域的输入。

然后，两个网络的输出通过某种方式进行融合（如点积、拼接等），最终得到预测值。在你的情况下，这个预测值是管径 D 和坐标 x,y 对应的温度 T。

### 如何在你的问题中使用 DeepONet：

1. **Branch Net**：用来处理管径 D 的信息。
2. **Trunk Net**：用来处理 x，y 坐标点的空间信息。
3. **融合**：将两个网络的输出融合起来，预测在对应管径和坐标下的温度 T。

### DeepONet 的结构：

你可以设计两个独立的网络，一个专门处理管径 D 的输入，另一个专门处理坐标 x,y，然后将两者的输出通过点积或其他融合方式结合起来。最后，使用全连接层输出温度值 T。

### 实现步骤：

1. **定义 Branch Net**： 这是一个简单的全连接网络，用来处理管径 D 信息。
2. **定义 Trunk Net**： 这是另一个网络，用来处理坐标 x,y。
3. **融合网络**： 将两个网络的输出融合起来，得到最终的预测。

### 详细代码实现：

```python
python复制代码import torch
import torch.nn as nn

# BranchNet: 用于处理管径 D 的信息
class BranchNet(nn.Module):
    def __init__(self):
        super(BranchNet, self).__init__()
        self.fc1 = nn.Linear(1, 64)  # 管径 D 是单一数值输入
        self.fc2 = nn.Linear(64, 64)
    
    def forward(self, D):
        x = torch.relu(self.fc1(D))
        x = torch.relu(self.fc2(x))
        return x

# TrunkNet: 用于处理坐标 x, y 的信息
class TrunkNet(nn.Module):
    def __init__(self):
        super(TrunkNet, self).__init__()
        self.fc1 = nn.Linear(2, 64)  # x, y 是二维输入
        self.fc2 = nn.Linear(64, 64)
    
    def forward(self, xy):
        x = torch.relu(self.fc1(xy))
        x = torch.relu(self.fc2(x))
        return x

# DeepONet: 将 BranchNet 和 TrunkNet 的输出结合起来进行预测
class DeepONet(nn.Module):
    def __init__(self):
        super(DeepONet, self).__init__()
        self.branch_net = BranchNet()  # 管径处理网络
        self.trunk_net = TrunkNet()  # 坐标处理网络
        self.fc = nn.Linear(64, 1)  # 全连接层，输出温度 T
    
    def forward(self, D, xy):
        # 分别通过 Branch Net 和 Trunk Net 处理
        branch_output = self.branch_net(D)  # 管径相关特征
        trunk_output = self.trunk_net(xy)  # 坐标相关特征
        
        # 将 Branch 和 Trunk 的输出进行融合（点积方式）
        combined_output = branch_output * trunk_output  # 逐元素相乘（点积）
        combined_output = torch.sum(combined_output, dim=1, keepdim=True)  # 按维度求和
        
        # 最终通过全连接层输出温度预测
        T = self.fc(combined_output)
        return T

# 测试模型
model = DeepONet()

# 假设我们有 32 个样本，每个样本有一个管径 D 和对应的坐标 x, y
D = torch.randn(32, 1)  # 32 个管径 D
xy = torch.randn(32, 2)  # 对应的 x, y 坐标点

# 预测温度 T
T_pred = model(D, xy)
print(T_pred.shape)  # 输出预测温度的形状
```

### 代码详解：

1. **BranchNet**：用于处理管径 D 的信息，输入是管径的数值，经过两层全连接层提取特征。
   - 输入：管径 D 的数值，形状为 `[batch_size, 1]`。
   - 输出：管径相关的特征，形状为 `[batch_size, 64]`。
2. **TrunkNet**：用于处理坐标 x,y，输入是二维的坐标信息，经过两层全连接层提取空间特征。
   - 输入：坐标 x,y，形状为 `[batch_size, 2]`。
   - 输出：坐标相关的特征，形状为 `[batch_size, 64]`。
3. **DeepONet**：
   - 将 BranchNet 和 TrunkNet 的输出通过点积进行融合。
   - 点积后的结果再通过一个全连接层，最终输出预测的温度 TTT。

### 其他可能的融合方式：

- **拼接（Concatenation）**：你也可以将 BranchNet 和 TrunkNet 的输出拼接起来，然后通过全连接层进一步处理：

  ```python
  python复制代码combined_output = torch.cat((branch_output, trunk_output), dim=1)  # 拼接
  T = self.fc(combined_output)
  ```

- **加法**：除了点积外，还可以通过简单的加法将两个网络的输出融合：

  ```python
  python复制代码combined_output = branch_output + trunk_output  # 加法
  T = self.fc(combined_output)
  ```

### 总结：

- **分离处理特征**：通过 `BranchNet` 和 `TrunkNet` 分别处理管径 D和坐标 x,y 的信息，然后将它们的输出结合起来，模型可以更有效地学习两者之间的独立关系。
- **DeepONet** 是一种灵活的架构，可以根据需要修改不同的融合方式和网络深度。

这个设计可以有效处理管径 D和坐标 x,y 的不同特性，使得模型在预测过程中能更好地学习这两类特征的独立和联合作用。

对于你原来的二维表格数据（即包含管径 \(D\) 和坐标 \(x, y\)），如果想进行 **分离处理**，即分别处理管径 \(D\) 和坐标 \(x, y\) 的特征，并进行预测，可以通过类似 **DeepONet** 的方式进行处理。这种分离处理的方法可以帮助模型学习到不同类型特征的独立作用和相互关系。

### 数据特点：
- **管径 \(D\)**：是一个物理特性，可能与温度 \(T\) 有特定的非线性关系。
- **坐标 \(x, y\)**：表示空间中的位置，温度 \(T\) 可能随空间变化而不同。
- **目标**：预测在给定管径 \(D\) 和坐标 \(x, y\) 下的温度 \(T\)。

### 分离处理思路：
1. **将管径 \(D\) 作为一个单独的特征**：用一个神经网络（如 `BranchNet`）专门处理管径的数值。
2. **将坐标 \(x, y\) 作为空间特征**：用另一个神经网络（如 `TrunkNet`）处理坐标信息。
3. **融合处理后的特征**：将管径相关的特征和坐标相关的特征结合起来（如通过点积或拼接），然后用全连接层输出预测的温度。

### 数据分离处理步骤：
1. **数据划分**：将你的二维表格数据分成两个部分，一个部分是 **管径 \(D\)**，另一个部分是 **坐标 \(x, y\)**。
2. **特征标准化**：通常对于数值数据（如管径、坐标），可以先进行标准化处理，以帮助模型更快收敛。
3. **输入网络**：分别将管径 \(D\) 和坐标 \(x, y\) 输入不同的子网络进行处理，然后将两部分的输出结合起来进行最终的预测。

### 实现步骤：

#### 1. 数据准备：
假设你的原始表格数据形状为 `(n_samples, 3)`，其中 3 列分别是管径 \(D\)、坐标 \(x\) 和坐标 \(y\)，需要先将这些特征分开处理。

```python
import torch
import numpy as np
import pandas as pd

# 假设 data 是你的二维表格数据，包含 D, x, y 三列
# 将 D 和 x, y 分开
D = data[:, 0].reshape(-1, 1)  # 取出第一列作为管径 D
xy = data[:, 1:]  # 取出第二、三列作为坐标 x, y

# 标准化（根据需要）
from sklearn.preprocessing import StandardScaler
scaler_D = StandardScaler()
scaler_xy = StandardScaler()

D = scaler_D.fit_transform(D)  # 对管径 D 进行标准化
xy = scaler_xy.fit_transform(xy)  # 对坐标 x, y 进行标准化

# 转换为 PyTorch 张量
D = torch.tensor(D, dtype=torch.float32)
xy = torch.tensor(xy, dtype=torch.float32)
```

#### 2. 定义分离网络结构：

- **Branch Net**：处理管径 \(D\)。
- **Trunk Net**：处理坐标 \(x, y\)。

```python
import torch.nn as nn

# BranchNet 用于处理管径 D 的信息
class BranchNet(nn.Module):
    def __init__(self):
        super(BranchNet, self).__init__()
        self.fc1 = nn.Linear(1, 64)
        self.fc2 = nn.Linear(64, 64)
    
    def forward(self, D):
        x = torch.relu(self.fc1(D))
        x = torch.relu(self.fc2(x))
        return x

# TrunkNet 用于处理坐标 x, y 的信息
class TrunkNet(nn.Module):
    def __init__(self):
        super(TrunkNet, self).__init__()
        self.fc1 = nn.Linear(2, 64)  # x, y 两个输入
        self.fc2 = nn.Linear(64, 64)
    
    def forward(self, xy):
        x = torch.relu(self.fc1(xy))
        x = torch.relu(self.fc2(x))
        return x
```

#### 3. 融合两个网络的输出并进行预测：
将两个网络的输出融合后通过一个全连接层输出预测结果 \(T\)。

```python
# DeepONet 将 BranchNet 和 TrunkNet 的输出结合起来
class DeepONet(nn.Module):
    def __init__(self):
        super(DeepONet, self).__init__()
        self.branch_net = BranchNet()  # 管径处理网络
        self.trunk_net = TrunkNet()  # 坐标处理网络
        self.fc = nn.Linear(1, 1)  # 最终输出预测温度
    
    def forward(self, D, xy):
        branch_output = self.branch_net(D)  # 管径相关特征
        trunk_output = self.trunk_net(xy)  # 坐标相关特征
        
        # 融合方式：点积
        combined_output = branch_output * trunk_output  # 点积
        combined_output = torch.sum(combined_output, dim=1, keepdim=True)  # 求和
        
        # 最终通过全连接层输出温度预测
        T = self.fc(combined_output)
        return T
```

#### 4. 训练模型：
在训练时，你可以分别输入管径 \(D\) 和坐标 \(x, y\) 来训练模型。

```python
# 模拟数据
n_samples = 49410  # 示例样本数
data = np.random.rand(n_samples, 3)  # 生成随机的 (D, x, y) 数据

# 准备数据
D = data[:, 0].reshape(-1, 1)
xy = data[:, 1:]
D = torch.tensor(D, dtype=torch.float32)
xy = torch.tensor(xy, dtype=torch.float32)

# 假设我们有对应的温度 T 作为标签
T = torch.randn(n_samples, 1)  # 生成随机的温度标签

# 构造数据集
from torch.utils.data import DataLoader, TensorDataset
dataset = TensorDataset(D, xy, T)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# 初始化模型和优化器
model = DeepONet()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# 训练循环
for epoch in range(100):  # 训练 100 个 epoch
    model.train()
    running_loss = 0.0
    for batch_D, batch_xy, batch_T in train_loader:
        optimizer.zero_grad()  # 梯度清零
        outputs = model(batch_D, batch_xy)  # 前向传播
        loss = criterion(outputs, batch_T)  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数
        
        running_loss += loss.item()
    
    print(f"Epoch {epoch+1}, Loss: {running_loss / len(train_loader)}")
```

### 总结：
1. **分离处理**：分别将管径 \(D\) 和坐标 \(x, y\) 分离处理，使用两个不同的子网络处理各自的特征。
2. **融合**：将两个网络的输出通过点积或拼接融合，得到最终的预测值 \(T\)。
3. **训练模型**：使用标准的神经网络训练流程，输入管径和坐标数据，训练模型进行温度预测。

这样的方法可以充分利用管径和坐标的特征独立性，让模型分别学习它们对温度的影响。



后来查阅相关资料，deeponet就是采用两个DNN进行的集成，那如果改成对x，y坐标的处理是cnn会不会有更好的效果？于是进行了尝试，但是发现两个需要注意的问题

1. cnn处理坐标数据后，trunk net的输出与branch net的输出很难对的上，也就是无法进行点积操作了，否则需要很好的进行计算与设计，而且进行拼接、加和等等也会出现维度、大小不一致导致的问题。
2. 添加卷积层后会导致训练极其缓慢

```python
# TrunkNet: 用于处理坐标 x, y 的信息，添加卷积和池化层
class TrunkNet(nn.Module):
    def __init__(self):
        super(TrunkNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)  # 卷积层
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # 池化层
        self.fc1 = nn.Linear(2*4, 64)  # 展平后连接到全连接层
        self.fc2 = nn.Linear(64, 64)
    
    def forward(self, xy):
        # 将输入的坐标 x, y 重塑为 1x1x10x10 形式，以便卷积操作
        # print(xy.size())
        x = xy.view(-1, 1, 4, 8)  # 假设 x, y 是 10x10 的网格
        
        # 通过卷积和池化层
        x = self.pool(torch.relu(self.conv1(x)))
        x = x.view(-1, 2*4)  # 展平
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return x
```

deeponet的常见融合操作：

这个错误信息表明你在进行张量操作时，两个张量在 **非单维度**（即非单一维度，通常是批次或特征维度）的大小不匹配，导致广播（broadcast）失败。

具体来说，错误提示中说：
- **张量 a** 的大小为 1026。
- **张量 b** 的大小为 1024。
- 这两个张量在 **非单维度（非1维度）** 上大小不一致，无法执行广播操作。

### 如何解决：
1. **检查张量的形状**：
   首先，你需要检查两个张量的形状，确保它们在非单维度上大小相同。如果你想要对两个张量进行广播，它们在非单维度上的大小必须相同。

   ```python
   print(tensor_a.shape)
   print(tensor_b.shape)
   ```

   例如：
   - `tensor_a` 的形状可能是 `(1026, ...)`。
   - `tensor_b` 的形状可能是 `(1024, ...)`。

   这里的形状不匹配导致了广播失败。

2. **调整张量的形状**：
   你需要确保在广播操作之前，两个张量在相应的维度上大小相同。可以使用如下方法：

   - **截断或扩展张量**：如果你希望两个张量的大小一致，可以通过截断或填充来调整张量的大小。你可以根据需要手动进行补齐或裁剪：
   
     ```python
     tensor_a = tensor_a[:1024]  # 截断 tensor_a，使其大小为 1024
     ```

   - **手动扩展或缩小张量**：有时候，你可以通过 `torch.cat`、`torch.stack` 等方法手动对张量进行操作，确保它们可以进行后续的广播操作。

3. **检查数据来源**：
   如果这两个张量来自于不同的数据集或不同的操作步骤，确保数据集切分、采样时的大小一致。如果在 `DataLoader` 或其他数据加载过程中出现了尺寸差异，可能需要在数据处理阶段确保所有张量的维度是匹配的。

### 具体场景：
- **数据加载问题**：在数据加载或批次划分中，由于最后一个 batch 的大小不一致，可能会导致某个张量的大小稍大或稍小于预期。此时可以通过设置 `drop_last=True` 确保每个 batch 大小一致，或者通过动态处理批次大小来解决。
  
- **维度不匹配**：有可能是在操作过程中（如矩阵乘法、卷积、拼接等）没有对齐维度，导致广播失败。此时需要确保两个张量在操作前经过了正确的 reshape 或 pad 操作。

### 总结：
- 检查两个张量的形状，找出不匹配的维度。
- 通过截断、填充或重新划分数据来确保张量的维度一致。
- 如果问题出现在数据加载过程中，确保数据集的大小一致，或者在处理时考虑动态调整。



是的，**将 Trunk Net 转换为卷积层后出现的融合操作错误** 和 **训练速度显著下降** 是常见的问题，特别是在以下几种情况下：

### 1. **融合操作报错的原因**：
当你将 Trunk Net 转换为卷积层时，输出的特征维度会发生变化。卷积层输出的张量通常是一个 4D 张量（`[batch_size, channels, height, width]`），而你的 Branch Net 通常是全连接层，输出的是 2D 张量（`[batch_size, features]`）。由于两者的维度不一致，在融合操作（如拼接、点积、加法等）时很容易出现形状不匹配的错误。

**常见问题**：
- **维度不匹配**：卷积层输出的是 4D 特征图，而全连接网络通常输出 2D 张量。直接将它们融合会导致形状不兼容的错误。

**解决方法**：
- **展平卷积层输出**：你可以将卷积层的 4D 输出展平为 2D 向量，然后再与 Branch Net 的输出进行融合。例如：

    ```python
    # 假设卷积层的输出形状为 (batch_size, channels, height, width)
    trunk_output = trunk_output.view(batch_size, -1)  # 展平为 2D 张量 (batch_size, features)
    # 现在 trunk_output 和 branch_output 可以进行拼接或其他融合操作
    ```

- **Global Average Pooling**：可以使用全局平均池化（`Global Average Pooling`），将卷积层的输出转换为 2D 特征向量。这样可以减少特征维度，同时适配后续融合操作。

    ```python
    # 使用全局平均池化，将卷积输出转换为 (batch_size, channels)
    trunk_output = torch.mean(trunk_output, dim=[2, 3])  # 对 height 和 width 维度进行平均
    ```

### 2. **训练速度变慢的原因**：
卷积层通常比全连接层更复杂，尤其是在输入的空间维度较大时，卷积操作会显著增加计算量，导致训练速度变慢。

**常见问题**：
- **高维卷积操作计算复杂度高**：卷积操作需要对每个输入特征图执行多个卷积核操作，随着卷积核的数量（`out_channels`）和输入的空间维度（`height`, `width`）的增加，计算量会大幅上升。
- **I/O 开销增加**：卷积层输出的是高维特征图，数据量增大，可能导致 GPU 内存占用增加，I/O 传输开销增大。

**解决方法**：
- **减少卷积层的深度或输出通道数**：可以通过减少卷积层的深度或减少输出通道数来减小计算量，从而加快训练速度。
- **使用池化层降低分辨率**：在卷积层中引入池化操作（如 `MaxPooling`），可以减少输出特征图的尺寸，进而降低后续层的计算量。
  
    ```python
    self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # 降低空间维度
    ```

- **减少输入数据的分辨率**：如果卷积层处理的是高维输入数据，可以通过下采样减少数据的分辨率，这样可以显著减少卷积层的计算量。
  
- **使用更高效的卷积操作**：可以尝试使用轻量级的卷积操作（如深度可分离卷积 `Depthwise Separable Convolution`），这种方法可以大幅减少计算量。

### 总结：
- **融合报错**：转换为卷积层后，可能导致 Trunk Net 和 Branch Net 输出维度不匹配，通常需要将卷积输出展平或使用全局平均池化。
- **训练速度变慢**：卷积操作的计算复杂度较高，尤其在高维数据上，常见解决方案是减少卷积通道数、引入池化层、下采样输入或使用轻量级卷积操作。

这些问题在将全连接层转换为卷积层时是常见的，通过合理优化，可以改善模型的性能和训练速度。