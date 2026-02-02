#!/bin/bash
# src/build_all.sh
set -e
echo "🛠️  Starting Build Process (Universal Search Version)..."

# ==========================================
# 1. Logger & Tracer
# ==========================================
echo ">> Building Loggers..."
mpicc -shared -fPIC -o liblogger.so logger.c

cat <<EOF > mpitrace.c
#include <mpi.h>
#include <stdio.h>
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

# ==========================================
# 2. Replay Tool
# ==========================================
echo ">> Building Traffic Replayer..."
mpicc -o replay_tool replay.c

# ==========================================
# 3. LULESH
# ==========================================
echo ">> Checking LULESH..."
if [ ! -f "LULESH/build/lulesh2.0" ]; then
    rm -rf LULESH
    git clone https://github.com/LLNL/LULESH.git
    mkdir -p LULESH/build && cd LULESH/build
    cmake .. -DWITH_MPI=ON -DWITH_OPENMP=OFF -DCMAKE_CXX_COMPILER=mpic++ -DCMAKE_BUILD_TYPE=Release > /dev/null
    make -j4
    cd ../..
fi

# ==========================================
# 4. HPGMG
# ==========================================
echo ">> Checking HPGMG..."
if [ ! -f "hpgmg/build/bin/hpgmg-fv-mpi" ] && [ ! -f "hpgmg/build/bin/hpgmg-fv" ]; then
    rm -rf hpgmg
    git clone https://github.com/hpgmg/hpgmg.git
    cd hpgmg
    ./configure --CC=mpicc --CFLAGS="-O2" > /dev/null
    make -C build -j4
    cd ..
fi

# ==========================================
# 5. CoMD
# ==========================================
echo ">> Checking CoMD..."
if [ ! -f "CoMD/src-mpi/CoMD-mpi" ]; then
    rm -rf CoMD
    git clone https://github.com/ECP-copa/CoMD.git
    cd CoMD/src-mpi
    ./generate_info_header CoMD-mpi "mpicc" "-std=c99 -DDOUBLE -DDO_MPI -O2" " -lm "
    mpicc -std=c99 -g -O3 -DDO_MPI -DDOUBLE -o CoMD-mpi *.c -lm
    cd ../..
    echo "   ✅ CoMD build success."
fi

# ==========================================
# 6. CoHMM (Universal Find + Non-blocking)
# ==========================================
echo ">> Checking CoHMM..."
if [ ! -f "CoHMM/cohmm" ]; then
    rm -rf CoHMM
    git clone https://github.com/exmatex/CoHMM.git
    cd CoHMM
    
    echo "   (Searching for source files...)"
    # 使用 find 查找所有 .cc, .cpp, .c 文件
    SRCS=$(find . -name "*.cc" -o -name "*.cpp" -o -name "*.c")
    
    if [ -n "$SRCS" ]; then
        # 尝试编译，如果失败则打印警告但不退出脚本 (|| true)
        echo "   Compiling found sources..."
        mpic++ -O3 -DCOHMM_MPI -o cohmm $SRCS -lm || echo "   ⚠️  CoHMM build failed (Skipping)"
    else
        echo "   ⚠️  No source files found for CoHMM (Skipping)"
    fi
    
    cd ..
fi

# ==========================================
# 7. CoSP2 (Universal Find + Non-blocking)
# ==========================================
echo ">> Checking CoSP2..."
if [ ! -f "CoSP2/CoSP2-parallel" ]; then
    rm -rf CoSP2
    git clone https://github.com/exmatex/CoSP2.git
    cd CoSP2
    
    echo "   (Searching for source files...)"
    # 查找 src-mpi 或当前目录下的 C 文件
    SRCS=$(find . -name "*.c" | grep -v "test") # 排除测试文件
    
    if [ -n "$SRCS" ]; then
        echo "   Compiling found sources..."
        mpicc -O3 -std=c99 -DDO_MPI -o CoSP2-parallel $SRCS -lm || echo "   ⚠️  CoSP2 build failed (Skipping)"
    else
        echo "   ⚠️  No source files found for CoSP2 (Skipping)"
    fi
    
    cd ..
fi

echo "✅ Build script finished. (Check warnings above if any App failed)"