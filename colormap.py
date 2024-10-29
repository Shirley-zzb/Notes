import matplotlib.pyplot as plt
import numpy as np

# 获取所有可用的colormap名称
cmap_names = plt.colormaps()

# 创建数据
data = np.random.rand(10, 10)

# 计算需要多少个子图
num_cmaps = len(cmap_names)
num_columns = 4  # 每行显示的子图数量
num_rows = (num_cmaps + num_columns - 1) // num_columns  # 计算需要的行数

# 创建图形和子图
fig, axs = plt.subplots(num_rows, num_columns, figsize=(num_columns * 5, num_rows * 3))
axs = axs.flatten()  # 将子图数组展平

# 遍历每个colormap并显示在子图中
for i, cmap_name in enumerate(cmap_names):
    ax = axs[i]
    im = ax.imshow(data, cmap=cmap_name)
    fig.colorbar(im, ax=ax)  # 添加颜色条
    ax.set_title(cmap_name)  # 设置子图标题为colormap名称
    ax.axis('off')  # 关闭坐标轴

# 调整布局
plt.tight_layout()
plt.show()
