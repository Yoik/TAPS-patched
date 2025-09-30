import numpy as np
import matplotlib.pyplot as plt
import os, time

# === 基本设置 ===
projName = "2A_LSD_80_rc009_US_raw_"   # 改成你的项目名
iter_dir = "iter000"                   # 如果以后换迭代，可以改这里
base_dir = f"{projName}0/sampling/{iter_dir}"

colvar_file = "COLVAR"
plumed_file = "plumed.dat"
target_col = "p1.sss"   # 也可以改成 "p1.zzz"
kBT = 2.5               # 近似 300K 下的 kBT (kJ/mol)

# === 工具函数 ===
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

def parse_plumed(plumed_path):
    """解析 plumed.dat，提取 AT 和 KAPPA"""
    at, kappa = None, None
    with open(plumed_path) as f:
        for line in f:
            if line.strip().startswith("RESTRAINT"):
                parts = line.split()
                for p in parts:
                    if p.startswith("AT="):
                        at = float(p.split("=")[1])
                    elif p.startswith("KAPPA="):
                        kappa = float(p.split("=")[1])
    return at, kappa

def wham_free_energy(all_values, bias_info, bins=100, kBT=2.5):
    """简化版 WHAM：合并所有窗口数据并重加权"""
    all_data = np.concatenate(list(all_values.values()))
    hist_range = (all_data.min(), all_data.max())
    hist_centers = np.linspace(hist_range[0], hist_range[1], bins)
    P = np.zeros_like(hist_centers)

    for node, values in all_values.items():
        at = bias_info[node]["AT"]
        kappa = bias_info[node]["KAPPA"]
        bias = 0.5 * kappa * (hist_centers - at)**2
        weights = np.exp(-bias/kBT)
        hist, _ = np.histogram(values, bins=bins, range=hist_range, density=True)
        P += hist * weights

    F = -kBT * np.log(P + 1e-12)
    return hist_centers, F - F.min()

# === 绘图初始化 ===
plt.ion()
fig, ax = plt.subplots(figsize=(8,6))
ax2 = ax.twinx()   # 只建一次右轴
last_sizes = {}    # 动态记录各节点文件大小

# === 主循环 ===
while True:
    ax.clear()
    ax2.clear()
    running, finished, not_started = 0, 0, 0
    all_values, bias_info = {}, {}

    # 扫描所有窗口目录
    if os.path.exists(base_dir):
        nodes = sorted([d for d in os.listdir(base_dir) if d.startswith("win")])
    else:
        print(f"[WARN] {base_dir} not found yet, skipping this round.")
        time.sleep(10)
        continue

    for node in nodes:
        if node not in last_sizes:
            last_sizes[node] = 0

        colvar_path = os.path.join(base_dir, node, colvar_file)
        plumed_path = os.path.join(base_dir, node, plumed_file)

        if not os.path.exists(colvar_path) or not os.path.exists(plumed_path):
            not_started += 1
            continue
        if os.path.getsize(colvar_path) == 0:
            not_started += 1
            continue

        values = read_colvar(colvar_path, target_col)
        if values is None:
            not_started += 1
            continue

        # 判断是否还在更新
        size_now = os.path.getsize(colvar_path)
        if size_now > last_sizes[node]:
            running += 1
        else:
            finished += 1
        last_sizes[node] = size_now

        # 保存数据和 bias 信息
        all_values[node] = values
        at, kappa = parse_plumed(plumed_path)
        bias_info[node] = {"AT": at, "KAPPA": kappa}

        # 画每个窗口的分布
        ax.hist(values, bins=50, density=True, alpha=0.4, label=node)

    # 叠加实时 WHAM PMF
    if all_values:
        x, F = wham_free_energy(all_values, bias_info, bins=100, kBT=kBT)
        ax2.plot(x, F, 'k-', lw=2, label="PMF (WHAM)")
        ax2.set_ylabel("Free energy (kJ/mol)")
        ax2.legend(loc="upper right", fontsize=8)

    ax.set_xlabel(target_col)
    ax.set_ylabel("Probability density")
    ax.set_title(f"{target_col} distributions | Running: {running}, Finished: {finished}, Not started: {not_started}")
    if nodes:
        ax.legend(fontsize=7, ncol=2, loc="upper left")
    plt.pause(5.0)  # 每5秒刷新一次
