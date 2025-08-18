import matplotlib.pyplot as plt
import numpy as np

# 生成数据
x = np.linspace(0, 10, 100)
y = np.sin(x) * 40 + 50

plt.plot(x, y, color="blue")

# 设置 y 轴的范围，包括 [0, 10] 和 [50, 100]
plt.ylim(0, 100)

# 手动添加虚线来表示 [10, 50] 的空隙
plt.hlines(
    10, x.min(), x.max(), colors="gray", linestyles="dashed"
)  # [0, 10] 部分的顶端虚线
plt.hlines(
    50, x.min(), x.max(), colors="gray", linestyles="dashed"
)  # [50, 100] 部分的底端虚线

# 添加提示文本，标注中间的间隙部分
plt.text(5, 30, "... skipped ...", fontsize=12, ha="center")
plt.savefig("tmp.png")

plt.show()
