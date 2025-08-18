import matplotlib.pyplot as plt

# plt.rcParams["font.family"] = "serif"

data1 = []

# 定义数据
data = [
    [0.006700297, 0.015353405, 0.057197427],
    [0.006328435, 0.002561913, 0.00130166],
    [0.001856895, 0.010702471, 0.011737089],
    [0.011671335, 0.035366193, 0.164691943],
]

# data = [
#     [0.59, 0.24, 0.08],
#     [0.15, 0.55, 0.05],
#     [1, 2.54, 5.56]
# ]

data2 = [
    [85.68, 70.41, 30.73],
    [93.47, 74.49, 33.16],
    [86.69, 71.35, 38.22],
    [94.55, 75.74, 41.13],
]

# data2 = [
#     [93.23, 93.68, 61.46],
#     [80.78, 51.39, 4.26],
#     [85.68, 71.82, 33.76],
#     [93.82, 94.12, 62.34],
#     [80.93, 52.24, 5.31],
#     [86.68, 74.36, 39.32],
# ]

data3 = [[91.01, 89.57], [91.89, 93.89]]

# categories = ['Overall', 'Car', 'Pedestrian', 'Cyclist']
# categories = ["Car", "Pedestrian", "Cyclist"]
categories = ["Hard", "Mod", "Easy"]
x_labels = ["0-10", "10-40", "40<"]
markers = ["o", "s", "D", "^"]
colors = ["#5861ac", "#fea040", "#f28080", "#6ab4c1"]
linestyles = [":", "-"]
markersize = 5
plt.figure(dpi=600)
# 为每一行生成一个折线
# for i, row in enumerate(data2):
#     plt.plot(x_labels, row, marker=markers[i], color=colors[i], label=f'{categories[i]}')


for i in range(2):
    plt.plot(
        x_labels,
        data2[i],
        linestyle=linestyles[0],
        marker=markers[i],
        markersize=markersize,
        color=colors[i],
        label=f"{categories[i]}_Raw",
    )
    plt.plot(
        x_labels,
        data2[i + 2],
        linestyle=linestyles[1],
        marker=markers[i],
        markersize=markersize,
        color=colors[i],
        label=f"{categories[i]}_S2D",
    )

i = 2
plt.plot(
    x_labels[:-1],
    data3[0],
    linestyle=linestyles[0],
    marker=markers[i],
    markersize=markersize,
    color=colors[i],
    label=f"{categories[i]}_Raw",
)
plt.plot(
    x_labels[:-1],
    data3[1],
    linestyle=linestyles[1],
    marker=markers[i],
    markersize=markersize,
    color=colors[i],
    label=f"{categories[i]}_S2D",
)

# plt.yscale('log')

# 添加网格线
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
# 添加图例
plt.legend()

# 添加标题和标签
plt.title("Performance w/wo S2D Layer in Different Difficulty (Cyclist)")
# plt.title("Performance w/wo S2D Layer in Different Category (Mod)")
plt.xlabel("Distance (/m)")
plt.ylabel("Detection results in 3D APs (%)")

plt.savefig("S2D_vis_dif.png")

# 显示图表
plt.show()
