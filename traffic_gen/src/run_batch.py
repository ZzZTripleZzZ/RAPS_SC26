import os
import subprocess
import shutil
import time

# === 实验配置矩阵 ===
EXPERIMENTS = [
    # 1. 基准测试
    { "name": "lulesh_baseline", "app": "lulesh", "ranks": 8, "args": "-s 15", "scale": 1.0 },
    
    # 2. 模拟 GPU 加速
    { "name": "lulesh_gpu_sim",  "app": "lulesh", "ranks": 8, "args": "-s 15", "scale": 10.0 },
    
    # 3. HPGMG 对比
    { "name": "hpgmg_baseline",  "app": "hpgmg",  "ranks": 8, "args": "5 2",   "scale": 1.0 },
    
    # 4. Replay 测试
    { "name": "replay_lulesh",   "app": "replay", "ranks": 8, "args": "../data/lulesh_baseline/traffic_timeseries.csv", "scale": 1.0 },
]

BASE_DIR = os.getcwd()
DATA_BASE_DIR = os.path.join(BASE_DIR, "../data")
DUMPI_LIB = "/usr/local/sst-dumpi/lib/libdumpi.so"
LOGGER_LIB = os.path.join(BASE_DIR, "liblogger.so")
TRACE_LIB = os.path.join(BASE_DIR, "libmpitrace.so")

BINARIES = {
    "lulesh": "./LULESH/build/lulesh2.0",
    "hpgmg":  "./hpgmg/build/bin/hpgmg-fv-mpi",
    "comd":   "./CoMD/src-mpi/CoMD-mpi",
    "cohmm":  "./CoHMM/cohmm",
    "cosp2":  "./CoSP2/CoSP2-parallel",
    "replay": "./replay_tool"
}

def run_experiment(exp):
    print(f"\n[Manager] >>> Experiment: {exp['name']} ({exp['app']} @ {exp['scale']}x speed)")
    
    output_dir = os.path.join(DATA_BASE_DIR, exp['name'])
    if os.path.exists(output_dir): shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    
    env = os.environ.copy()
    
    if exp['app'] == "replay":
        env["LD_PRELOAD"] = "" 
    else:
        env["LD_PRELOAD"] = f"{LOGGER_LIB}:{TRACE_LIB}:{DUMPI_LIB}"
        env["LOGGER_TIME_SCALE"] = str(exp['scale'])
    
    binary = BINARIES.get(exp['app'])
    if exp['app'] == 'hpgmg' and not os.path.exists(binary): binary = "./hpgmg/build/bin/hpgmg-fv"
    
    if not os.path.exists(binary):
        print(f"❌ Binary missing: {binary}")
        return

    # === 关键修改 ===
    # 添加 --mca btl self,tcp
    # 这告诉 OpenMPI: "只准用 TCP 和 Loopback，绝对不要碰共享内存(vader)"
    # 这样就绕过了 Docker /dev/shm 64MB 的限制
    cmd = [
        "mpirun",
        "--mca", "btl", "self,tcp",  # <--- 新增的救命参数
        "-n", str(exp['ranks']),
        "--oversubscribe",
        "--allow-run-as-root",
        binary
    ] + exp['args'].split()
    
    log_file = os.path.join(output_dir, "runtime.log")
    with open(log_file, "w") as f:
        try:
            subprocess.run(cmd, env=env, stdout=f, stderr=f, check=True)
            print("    ✅ Success.")
        except subprocess.CalledProcessError:
            print(f"    ❌ Failed. See {log_file}")
            return

    if exp['app'] != "replay":
        for file in os.listdir("."):
            if file.startswith("dumpi-") and (file.endswith(".bin") or file.endswith(".meta")):
                shutil.move(file, os.path.join(output_dir, file))
        if os.path.exists("../data/traffic_timeseries.csv"):
            shutil.move("../data/traffic_timeseries.csv", os.path.join(output_dir, "traffic_timeseries.csv"))

if __name__ == "__main__":
    # 简单的检查
    if not os.path.exists("./replay_tool"):
        print("⚠️  Replay tool not found. Did you run build_all.sh?")
        
    for exp in EXPERIMENTS:
        if exp['app'] == "replay":
            src_csv = exp['args']
            if not os.path.exists(src_csv):
                print(f"⏩ Skipping replay {exp['name']}: Source CSV not found.")
                continue
        
        run_experiment(exp)
    
    print("\n🎉 All Done. Run 'python3 analyze_all.py' or 'python3 analyze_advanced.py' to see results.")