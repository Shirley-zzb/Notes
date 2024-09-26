# pytorch数据类型

主要是生成tensor变量，具体的tensor形式分类如下：

- 0：scalar 数值
- 1：vector 向量
- 2：matrix 矩阵
- 3：n-dimensional tensor n维张量

## pytorch保存与加载模型

```python
#保存
if loss < best_loss:
        best_loss = loss
        torch.save(nn.state_dict(),PATH)
#加载
nn.load_state_dict(torch.load(PATH))
nn.eval()
```

## pytorch权重损失出现nan

没有约束导致，期望其内积越大，就会越来越大，最后变成无穷大

改变激活函数，变成有约束的

改变目标函数，增加一个阈值，大到一定程度就没用了

正则化，将输出归一化到0-1之间

## pytroch.meshgrid

变成二维坐标

【1，2】 -> [1,1] [2,2]

【3】 -> [3] [3]



【1，2，3】 -> [[1,1,1], [2,2,2], [3,3,3]]

【4，5，6】-> [[4,5,6], [4,5,6], [4,5,6]]

## pytorch构造网络结构的容器函数

1、list

```python
class a(nn.Module):
    def __init__(self):
        super(a.self).__init__()
        self.mods=[nn.Linear(2,3),nn.Linear(3,4)]
    def forward(self,x):
        for i in range(len(self.mods)):
            x=self.mods[i](x)
        return x
model=a()
b=torch.rand(1,2)
model(b)
#该方法的问题在于优化器传入的参数model.parameters()中不包含线性层参数
#反向传播的时候不能自动把线性层参数更新和梯度清零
```

2、Sequential

```python
class a(nn.Module):
    def __init__(self):
        super(a.self).__init__()
        self.mods=nn.Sequential(nn.Linear(2,3),
                               nn.Linear(3,4))
    def forward(self,x):
        return self.mods(x)
model=a()
b=torch.rand(1,2)
model(b)
#要求内部模型按网络结构顺序填写

#更通用的写法
class a(nn.Module):
    def __init__(self):
        super(a.self).__init__()
        self.mods=nn.Sequential()
        for i in range(2,4):
           self.mods.add_module(name:'mymodel{}'.format(i),module=nn.Linear(i,i+1))
    def forward(self,x): #除了self外只能有一个参数
        return self.mods(x)
```

3、ModelList

内部模型是无序的

```python
class a(nn.Module):
    def __init__(self):
        super(a.self).__init__()
        self.mods=[nn.Linear(2,3),nn.Linear(3,4)]
        self.mods=nn.ModuleList(self.mods)
    def forward(self,x):
        for i in range(len(self.mods)):
            x=self.mods[i](x)
        return x
model=a()
b=torch.rand(1,2)
model(b)
```

## pytorch归一化方法

