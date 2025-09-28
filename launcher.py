import subprocess, sys

# 统一参数配置
projName = "2A_LSD_80_rc009_10ns_UmSa_"
n_taps = 1
n_iter = 1
iter_start = 0
dirPars = "2A_LSD_80/umbrella"
parFile = "taps.par"
topFile = "step7_10.gro"
p0File = "iter010.xtc"
alignFile = "align.ndx"
rmsFile = "rms.ndx"

# 构造 MPI 命令
mpi_cmd = [
    "mpirun", "-np", "5", sys.executable, "runUmbrella.py",
    "--projName", projName,
    "--n_taps", str(n_taps),
    "--n_iter", str(n_iter),
    "--iter_start", str(iter_start),
    "--dirPars", dirPars,
    "--parFile", parFile,
    "--topFile", topFile,
    "--p0File", p0File,
    "--alignFile", alignFile,
    "--rmsFile", rmsFile
]

# 启动 umbrella
umbrella = subprocess.Popen(mpi_cmd)

# 启动 wham 监控
watch = subprocess.Popen([sys.executable, "wham.py"])
# 等待 umbrella 结束
umbrella.wait()

# umbrella 结束后关闭监控
watch.terminate()
