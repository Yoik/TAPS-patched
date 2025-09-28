import numpy as np
import matplotlib.pyplot as plt
import os, time

# 项目名和迭代目录
projName = "2A_LSD_80_rc009_10ns_UmSa_"   # 改成你的项目名
iter_dir = "iter000"                       # 如果以后换迭代，可以改这里
base_dir = f"{projName}0/sampling/{iter_dir}"

colvar_file = "COLVAR"
target_col = "p1.sss"   # 也可以改成 "p1.zzz"

def read_colvar(filename, target_col):
    """读取 COLVAR 文件，返回目标列的数值"""
    with open(filename) as f:
        header = None
        for line in f:
            if line.startswith("#! FIELDS"):
                header = line.strip().split()[2:]  # 去掉 "#! FIELDS"
                break
    if header is None:
        return None
    try:
        data = np.loadtxt(filename, comments="#")
    except Exception:
        return None
    if data.size == 0:
        return None
    if data.ndim == 1:
        data = data.reshape(1, -1)
    col_index = header.index(target_col)
    return data[:, col_index]

plt.ion()
fig, ax = plt.subplots(figsize=(8,6))

last_sizes = {}  # 动态记录各节点文件大小

while True:
    ax.clear()
    running, finished, not_started = 0, 0, 0

    # === 动态扫描所有 node 目录（包括 _extra_xxx） ===
    nodes = []
    if os.path.exists(base_dir):
        nodes = sorted([d for d in os.listdir(base_dir) if d.startswith("node")])
    else:
        print(f"[WARN] {base_dir} not found yet, skipping this round.")
        time.sleep(10)
        continue

    # 初始化/更新 last_sizes
    for node in nodes:
        if node not in last_sizes:
            last_sizes[node] = 0

    for node in nodes:
        filename = os.path.join(base_dir, node, colvar_file)

        if not os.path.exists(filename):
            not_started += 1
            continue

        if os.path.getsize(filename) == 0:
            not_started += 1
            continue

        values = read_colvar(filename, target_col)
        if values is None:
            not_started += 1
            continue

        # 判断是否还在更新
        size_now = os.path.getsize(filename)
        if size_now > last_sizes[node]:
            running += 1
        else:
            finished += 1
        last_sizes[node] = size_now

        ax.hist(values, bins=50, density=True, alpha=0.4, label=node)

    ax.set_xlabel(target_col)
    ax.set_ylabel("Probability density")
    ax.set_title(f"{target_col} distributions | Running: {running}, Finished: {finished}, Not started: {not_started}")
    if nodes:  # 避免空 legend 报错
        ax.legend(fontsize=7, ncol=2)
    plt.pause(5.0)  # 每5秒刷新一次
