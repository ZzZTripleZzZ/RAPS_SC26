#!/bin/bash
# src/run_sim.sh
set -e

# === 配置区域 ===
# 1=LULESH, 2=HPGMG, 3=CoMD, 4=CoHMM, 5=CoSP2
echo "---------------------------------------------------"
echo "Select Benchmark Application:"
echo "  1) LULESH (Shock Hydrodynamics)"
echo "  2) HPGMG  (Geometric Multigrid)"
echo "  3) CoMD   (Molecular Dynamics)"
echo "  4) CoHMM  (Heterogeneous Multiscale)"
echo "  5) CoSP2  (Electronic Structure)"
echo "---------------------------------------------------"
APP_CHOICE=1 # 默认 LULESH

if [ "$APP_CHOICE" -eq 1 ]; then APP_NAME="lulesh"; fi
if [ "$APP_CHOICE" -eq 2 ]; then APP_NAME="hpgmg"; fi
if [ "$APP_CHOICE" -eq 3 ]; then APP_NAME="comd"; fi
if [ "$APP_CHOICE" -eq 4 ]; then APP_NAME="cohmm"; fi
if [ "$APP_CHOICE" -eq 5 ]; then APP_NAME="cosp2"; fi

DATA_DIR="../data/${APP_NAME}"

echo "=========================================================="
echo "🚀 Starting simulation for ${APP_NAME}..."
echo "=========================================================="

# === 1. 环境准备 ===
if [ ! -f /usr/bin/python ]; then ln -s /usr/bin/python3 /usr/bin/python || true; fi

# === 2. 编译工具链 (Toolchain) ===

echo ">>> Building Custom Time-Series Logger..."
mpicc -shared -fPIC -o liblogger.so logger.c

echo ">>> Generating & Building Custom MPI-Trace..."
# 内嵌简单的 Tracer 源码
cat <<EOF > mpitrace.c
#include <mpi.h>
#include <stdio.h>
// 简单的 Tracer：只打印关键信息
int MPI_Send(const void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm) {
    int rank; MPI_Comm_rank(comm, &rank);
    int size; MPI_Type_size(datatype, &size);
    fprintf(stderr, "[TRACE] R%d -> R%d | Send | %d bytes\n", rank, dest, count * size);
    return PMPI_Send(buf, count, datatype, dest, tag, comm);
}
int MPI_Isend(const void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm, MPI_Request *request) {
    int rank; MPI_Comm_rank(comm, &rank);
    int size; MPI_Type_size(datatype, &size);
    fprintf(stderr, "[TRACE] R%d -> R%d | Isend | %d bytes\n", rank, dest, count * size);
    return PMPI_Isend(buf, count, datatype, dest, tag, comm, request);
}
int MPI_Irecv(void *buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm, MPI_Request *request) {
    return PMPI_Irecv(buf, count, datatype, source, tag, comm, request);
}
EOF
mpicc -shared -fPIC -o libmpitrace.so mpitrace.c


# === 3. 准备 Benchmarks ===
if [ "$APP_NAME" == "lulesh" ]; then
    if [ ! -d "LULESH" ]; then git clone https://github.com/LLNL/LULESH.git; fi
    if [ ! -f "LULESH/build/lulesh2.0" ]; then 
        mkdir -p LULESH/build && cd LULESH/build
        cmake .. -DWITH_MPI=ON -DWITH_OPENMP=OFF -DCMAKE_CXX_COMPILER=mpic++ -DCMAKE_BUILD_TYPE=Release > /dev/null
        make -j4 && cd ../..
    fi
    EXE="./LULESH/build/lulesh2.0 -s 15"

elif [ "$APP_NAME" == "hpgmg" ]; then
    if [ ! -d "hpgmg" ]; then git clone https://github.com/hpgmg/hpgmg.git; fi
    if [ ! -f "hpgmg/build/bin/hpgmg-fv" ] && [ ! -f "hpgmg/build/bin/hpgmg-fv-mpi" ]; then
        rm -rf hpgmg/build
        cd hpgmg
        ./configure --CC=mpicc --CFLAGS="-O2" 
        make -C build -j4 
        cd ..
    fi
    if [ -f "./hpgmg/build/bin/hpgmg-fv" ]; then EXE="./hpgmg/build/bin/hpgmg-fv"; else EXE="./hpgmg/build/bin/hpgmg-fv-mpi"; fi
    EXE="$EXE 5 2"

elif [ "$APP_NAME" == "comd" ]; then
    if [ ! -d "CoMD" ]; then git clone https://github.com/ECP-copa/CoMD.git; fi
    if [ ! -f "CoMD/src-mpi/CoMD-mpi" ]; then
        cd CoMD/src-mpi
        make CC=mpicc CFLAGS="-std=c99 -DDOUBLE -DDO_MPI -O2" CoMD-mpi
        cd ../..
    fi
    EXE="./CoMD/src-mpi/CoMD-mpi -x 4 -y 4 -z 4 -N 20"

elif [ "$APP_NAME" == "cohmm" ]; then
    if [ ! -d "CoHMM" ]; then git clone https://github.com/exmatex/CoHMM.git; fi
    if [ ! -f "CoHMM/cohmm" ]; then
        cd CoHMM && make CC=mpicc CXX=mpic++ && cd ..
    fi
    EXE="./CoHMM/cohmm"

elif [ "$APP_NAME" == "cosp2" ]; then
    if [ ! -d "CoSP2" ]; then git clone https://github.com/exmatex/CoSP2.git; fi
    if [ ! -f "CoSP2/CoSP2-parallel" ]; then
        cd CoSP2 && make CC=mpicc PARALLEL=yes && cd ..
    fi
    EXE="./CoSP2/CoSP2-parallel"
fi

# === 4. 运行模拟 ===
export DUMPI_LIB=/usr/local/sst-dumpi/lib/libdumpi.so
export LOGGER_LIB=$(pwd)/liblogger.so
export MPI_TRACE_LIB=$(pwd)/libmpitrace.so

export LD_PRELOAD=$LOGGER_LIB:$MPI_TRACE_LIB:$DUMPI_LIB

echo ">>> Running ${APP_NAME}..."
echo "    (Screen output silenced. Logs redirected to 'debug_trace.log')"

# 关键修改：2> debug_trace.log
# 这会把刷屏的 Trace 信息写到文件里，而普通的程序输出(stdout)可能还会显示一点点
mpirun -n 64 --oversubscribe --allow-run-as-root $EXE 2> debug_trace.log

# === 5. 数据归档 ===
echo ">>> Organizing data into ${DATA_DIR}..."
mkdir -p "$DATA_DIR"

if ls dumpi-*.meta 1> /dev/null 2>&1; then mv dumpi-* "$DATA_DIR/"; fi
if [ -f "../data/traffic_timeseries.csv" ]; then mv "../data/traffic_timeseries.csv" "$DATA_DIR/"; fi

# 把刚才生成的 log 也存起来
mv debug_trace.log "$DATA_DIR/"

echo "✅ Done! Data saved in ${DATA_DIR}"