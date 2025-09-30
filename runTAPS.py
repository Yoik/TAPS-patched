# -*- coding: utf-8 -*-
# 说明：
# 本脚本使用 mpi4py 进行并行控制，驱动 TAPS 工作流（路径采样 + MetaD 采样与分析）。
# 结构上分为：环境与依赖导入 -> MPI 初始化 -> 工具函数 -> 参数与输入文件 -> 主循环(每个 taps 实例) ->
# 数据初始化与广播 -> 迭代循环(准备、采样、分析、更新路径)。
# 每个模块都添加了目的与关键步骤的注释，便于协作与复现。

from __future__ import print_function, division
import warnings
warnings.filterwarnings("ignore", category=UserWarning)  # 安静模式：忽略一般用户级警告，保持日志清爽

# =======================================================================================================================
# 环境与并行：MPI 初始化与基本进程信息
# - 通过 mpi4py 获取通信器、进程总数(size)与当前进程编号(rank)
# - 后续各阶段通过 Barrier 做进程同步，通过 bcast 做对象广播
# =======================================================================================================================
import multiprocessing  # 如需结合多进程本地并行，可扩展使用（当前未直接使用）
from mpi4py import MPI
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
print("###DEBUG### Size:", size, "Rank:", rank, "begining.")

# ======================================================================================================================
# 工具函数：迭代数字格式化
# - 将整数格式化为固定3位，便于路径与文件命名一致性（例如 3 -> "003"）
# ======================================================================================================================
def digits(s1):
    s2 = "%.3d" % s1
    return s2

# ======================================================================================================================
# 依赖导入：TAPS 工作流与常用库
# - TAPS: 核心工作流类，封装 meta_dirs/meta_setup/meta_sample/meta_analyze 等方法
# - Confs: 构象容器或相关数据结构
# - 常用科学计算与文件操作库：mdtraj, numpy, shutil, copy 等
# ======================================================================================================================
from TAPS import *
from Confs import Confs
import time
import errno
import copy
import mdtraj as md
import numpy as np
import shutil
from copy import deepcopy

# =========================================================================================================
# 全局控制参数：taps 实例与迭代设置
# - n_taps: 独立 TAPS 实例的数量（可用于多条初始路径并行推进）
# - n_iter: 每个实例的迭代次数
# - iter_start: 起始迭代编号（用于从某一迭代中断点继续）
# =========================================================================================================
n_start = 0
n_taps = 1

n_iter = 15
iter_start = 0

# =========================================================================================================
# 输入文件与目录约定
# - dirPars/parFile: 参数目录与文件（TAPS 的控制参数）
# - topFile: 拓扑文件（如 GROMACS 的 .gro）
# - p0File: 初始路径（如某一迭代的 .xtc 轨迹）
# - alignFile/rmsFile: 对齐与 RMSD 计算的原子索引文件
# =========================================================================================================
dirPars = '2A_LSD_80/pars'
parFile = 'taps.par'
topFile = 'step7_10.gro'
p0File = 'raw_first50_rc009.xtc'
alignFile = 'align.ndx'
rmsFile = 'rms.ndx'

# =========================================================================================================
# 主循环：遍历多个独立的 taps 实例
# - 每个实例有自己的工作目录 tapsName
# - 仅 rank==0 负责创建目录与重型初始化，其余进程等待并接收广播的上下文
# =========================================================================================================
for i in range(n_start, n_taps + n_start):
    tapsName = '2A_LSD_80_rc009_raw' + str(i)

    # 仅主进程创建工作目录，其他进程同步等待
    if rank == 0 and not os.path.exists(tapsName):
        try:
            os.makedirs(tapsName)
        except OSError as error:
            # 并发创建容错：忽略“已存在”，其他错误抛出
            if error.errno != errno.EEXIST:
                raise
    comm.Barrier()  # 保证所有进程在目录就绪后再继续

    print(tapsName, ":")
    print("  data initialization")

    # -----------------------------------------------------------------------------------------------------
    # 数据与工作流初始化（仅在 rank==0 执行），随后广播给所有进程
    # - 构造 TAPS 对象：加载参数、拓扑与初始路径/索引；记录初始化耗时
    # - 初始化路径列表与演化目录(paths)，导出起始参考路径
    # -----------------------------------------------------------------------------------------------------
    if rank == 0:
        t0 = time.time()
        print("###DEBUG### Size:", size, "Rank:", rank, "running TAPS")
        taps = TAPS(dirPars, parFile, topFile, p0File, alignFile, rmsFile)
        te = time.time()
        print("    time-cost: ", te - t0, ' sec')

        pathList = []
        refPath = copy.copy(taps.refPath)  # 复制参考路径对象，作为迭代起点
        pathList.append(refPath)

        dirEvol = tapsName + '/paths'  # 路径演化记录目录
        if not os.path.exists(dirEvol):
            os.makedirs(dirEvol)

        # 导出初始参考路径到演化目录，命名与 iter_start 对齐
        refPath.pathName = 'iter' + digits(iter_start)
        refPath.exportPath(dirEvol)
    else:
        taps = None
        refPath = None

    # 将 TAPS 上下文与参考路径广播到所有进程，保持状态一致
    taps = comm.bcast(taps, root=0)
    refPath = comm.bcast(refPath, root=0)
    comm.Barrier()

    # =====================================================================================================
    # 迭代循环：每次迭代包含 MetaD 的准备、采样与分析，然后更新路径
    # - 目录结构：sampling/iterXYZ 存放该迭代的运行数据；paths 存放演化的路径快照
    # - rank 角色分工：
    #   * rank==0：生成运行目录、写入输入、触发采样的前置准备、最终导出路径
    #   * 所有 rank：参与采样与分析；通过 Barrier 保持阶段同步
    # =====================================================================================================
    for j in tqdm(range(iter_start, iter_start + n_iter), desc="Iteration", unit="iter"):
        # 迭代标签与目录
        iter = 'iter' + digits(j)
        dirMeta = tapsName + '/sampling/' + iter
        print("  ", iter, ": Preparing MetaD")

        comm.Barrier()
        if rank == 0:
            # 1) 生成每个子运行目录与输入配置（如不同种子或不同窗口）
            dirRUNs = taps.meta_dirs(refPath, dirMeta)

            # 2) 写入/准备 MetaD 所需文件（如 PLUMED 输入、mdp、脚本等）
            t0 = time.time()
            taps.meta_setup(refPath, dirMeta, dirRUNs)
            t1 = time.time()
            print('   timecost: ', t1 - t0, ' sec')
            print("  ", iter, ": Sampling MetaD")
        else:
            dirRUNs = None

        # 广播子运行目录信息，确保所有进程知道该迭代的布局
        dirRUNs = comm.bcast(dirRUNs, root=0)
        comm.Barrier()

        # 为避免资源争抢或文件系统延迟，非主进程可稍作等待（经验性缓冲）
        if rank != 0:
            time.sleep(10)

        # 3) 执行 MetaD 采样（各 rank 协同/分工）
        taps.meta_sample(dirMeta, dirRUNs)
        comm.Barrier()

        # 采样结束到分析开始的耗时记录起点
        t0 = time.time()
        if rank == 0:
            print('   timecost: ', t0 - t1, ' sec')
            print("  ", iter, ": Finding median(z) conformations, update path")

        comm.Barrier()

        # 4) 分析采样结果：提取代表性构象（如中位数 z），构建新的路径对象
        p_meta = taps.meta_analyze(dirMeta, dirRUNs)
        comm.Barrier()
        t1 = time.time()

        if rank == 0:
            # 记录分析耗时，导出更新后的路径快照，并将其作为下一次迭代的参考
            print('   timecost: ', t1 - t0, ' sec')
            p_meta.pathName = iter
            p_meta.exportPath(dirEvol)
            print(' ')
            refPath = deepcopy(p_meta)  # 深拷贝保证后续修改不会影响已导出的快照
        else:
            # 非主进程不做导出，仅参与计算；占位字段置空避免误用
            p_meta.pathName = None
            p_meta.exportPath = None
            refPath = None

        # 广播本次迭代的路径对象与新参考路径，保持所有进程状态同步
        p_meta = comm.bcast(p_meta, root=0)
        refPath = comm.bcast(refPath, root=0)
        comm.Barrier()
