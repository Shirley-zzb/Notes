# matplotlib数据类型

```python
import matplotlib
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(8,4)) #设置画布大小，防止保存图片缺失

plt.plot(x,y)
# color='red' 颜色
# linestyle='-.' 线形
# linewidth=2 线宽
# marker=’o' 点形
# markerfacecolor='black' 填充色
# markeredgecolor='red' 轮廓色
# markersize=3 点大小
# markeredgewidth=2 轮廓线宽
# label='Normal' 标签
# zorder=1 图层顺序，越小越贴近背景

plt.legend() #生成图例
# loc='best/lower right' 图例位置
plt.title('normal') #标题
plt.scatter(x,y)

ax1 = plt.gca() #取出坐标轴
ax1.set_title('big title') #设置图标题
# fontname='Arial' 字体
# fontsize=20 字体大小
# weight='bold' 粗体
# style='italic' 斜体
ax1.set_ylabel('T/C')
ax1.set_xlabel('time')
# $^o$ 使用latex

ax1.set_xticks([0,2.5,7,11]) # 刻度包含哪些数
ax1.set_xticklabels('J','a','n','s') # 设置刻度标签
ax1.tick_params(axis='x',direction='in')
# axis='x/y/both' direction='in/out' 刻度方向
# color='blue' 刻度颜色
# length='10' width='2' 刻度长短粗细

fig, ax = plt.subplots(2,2) #画子图
ax[0].plot(x1,y1) #第一个子图
ax[1].plot(x2,y2)
ax[1].set_xlim([0,10])#设置坐标轴范围
ax[1].set_ylim([0,10])
ax.set_yscale('log') #对数坐标轴

ax2 = ax.twinx() #共用x轴
ax3 = ax.twiny() #共用y轴
ax3.plot(x,y)

plt.tight_layout() #紧致布局，防止缺失
plt.savefig('./title.png',dpi=400) #保存图片

plt.rcParams['font.sans-serif'] = ['SimHei'] #显示中文
plt.rcParams['axes.unicode_minus'] = False  #显示负号
import warnings
warnings.filterwarnings('ignore') #去掉warning
```



在matplotlib库中，颜色和符号缩写可以通过以下方式进行设置：

1. 颜色缩写：
   - 'b'：蓝色
   - 'g'：绿色
   - 'r'：红色
   - 'c'：青色
   - 'm'：品红
   - 'y'：黄色
   - 'k'：黑色
   - 'w'：白色
   - '#008000'：RGB颜色（绿色）
   - '0.75'：灰度值（介于0到1之间）
2. 符号缩写：
   - '.'：点标记
   - ','：像素标记
   - 'o'：圆圈标记
   - 'v'：倒三角形标记
   - '^'：正三角形标记
   - '<'：左三角标记
   - '>'：右三角标记
   - '1'：下箭头标记
   - '2'：上箭头标记
   - '3'：左箭头标记
   - '4'：右箭头标记
   - 's'：正方形标记
   - 'p'：五边形标记
   - '*'：星形标记
   - 'h'：六边形标记1
   - 'H'：六边形标记2
   - '+'：加号标记
   - 'x'：乘号标记
   - 'D'：菱形标记
   - 'd'：窄菱形标记
   - '|'：垂直线标记
   - '_'：水平线标记

在matplotlib中，线型可以通过以下几种缩写来设置：

- '-'：实线（默认）
- '--'：虚线
- '-.'：点划线
- ':'：虚点线
- ''：无形线（即不绘制线条）











