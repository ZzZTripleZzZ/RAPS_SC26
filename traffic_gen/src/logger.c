// src/logger.c (v4.0 - Virtual Time Dilation)
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define MAX_RANKS 512

static FILE *log_fp = NULL;
static int my_rank = -1;
static int comm_size = -1;

// === 时间控制变量 ===
static double time_scale = 1.0;       // 缩放因子 (默认 1.0)
static double virtual_clock = 0.0;    // 当前的虚拟时间
static double last_wall_clock = 0.0;  // 上一次离开 MPI 的物理时间

void init_logger() {
    if (my_rank == -1) {
        MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
        MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
        
        // 读取环境变量
        char *env_scale = getenv("LOGGER_TIME_SCALE");
        if (env_scale) {
            time_scale = atof(env_scale);
            if (time_scale <= 0) time_scale = 1.0;
        }
        if (my_rank == 0 && time_scale != 1.0) {
            printf(">>> [Logger] Virtual Time Dilation Enabled: Scale = %.2fX (Simulating faster compute)\n", time_scale);
        }

        // 初始化时钟
        last_wall_clock = MPI_Wtime();
        virtual_clock = 0.0;

        char fname[64];
        sprintf(fname, "/tmp/traffic_rank_%d.csv", my_rank);
        log_fp = fopen(fname, "w");
    }
}

// 更新虚拟时钟 (在进入 MPI 调用前执行)
// 逻辑: 计算上一段代码跑了多久 (Compute Time)，将其除以 Scale，加到虚拟时钟上
void update_clock_pre() {
    if (my_rank == -1) init_logger();
    
    double now = MPI_Wtime();
    double compute_duration = now - last_wall_clock;
    
    // 关键: 压缩计算时间
    virtual_clock += compute_duration / time_scale;
}

// 更新物理时钟 (在离开 MPI 调用后执行)
// 逻辑: 重置基准点，确保 MPI 函数内部的耗时不会被缩放 (网络是物理设备，不能缩放)
void update_clock_post() {
    // 这里我们假设 MPI 调用本身的耗时也计入虚拟时间，但不缩放
    // 或者更简单的: 重新对齐 last_wall_clock
    double now = MPI_Wtime();
    
    // 把 MPI 内部消耗的时间原样加到虚拟时钟上 (模拟真实网络延迟)
    // virtual_clock += (now - (last_wall_clock + compute_duration)); 
    // 上面的公式化简后，其实就是我们需要把 virtual_clock 向前推进 "MPI Duration"
    // 但为了简化实现，我们暂不记录 MPI 内部耗时到 virtual_clock (假设那是瞬间完成注入)，
    // 主要是为了让 Trace 里的 timestamps 紧凑。
    
    last_wall_clock = now;
}

void record_traffic(int dest, int count, MPI_Datatype datatype) {
    if (dest == MPI_PROC_NULL || !log_fp) return;
    
    int type_size;
    MPI_Type_size(datatype, &type_size);
    long long bytes = (long long)count * type_size;
    
    if (dest < MAX_RANKS) {
        // 记录: VirtualTime, Source, Target, Bytes
        fprintf(log_fp, "%.6f,%d,%d,%lld\n", virtual_clock, my_rank, dest, bytes);
    }
}

// === 拦截器 ===

int MPI_Send(const void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm) {
    update_clock_pre();
    record_traffic(dest, count, datatype);
    int ret = PMPI_Send(buf, count, datatype, dest, tag, comm);
    update_clock_post();
    return ret;
}

int MPI_Isend(const void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm, MPI_Request *request) {
    update_clock_pre();
    record_traffic(dest, count, datatype);
    int ret = PMPI_Isend(buf, count, datatype, dest, tag, comm, request);
    update_clock_post();
    return ret;
}

int MPI_Irecv(void *buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm, MPI_Request *request) {
    update_clock_pre();
    // Irecv 不产生流量，但它是计算阶段的边界，所以也要更新时钟
    int ret = PMPI_Irecv(buf, count, datatype, source, tag, comm, request);
    update_clock_post();
    return ret;
}

int MPI_Wait(MPI_Request *request, MPI_Status *status) {
    update_clock_pre();
    int ret = PMPI_Wait(request, status);
    update_clock_post();
    return ret;
}

int MPI_Waitall(int count, MPI_Request array_of_requests[], MPI_Status array_of_statuses[]) {
    update_clock_pre();
    int ret = PMPI_Waitall(count, array_of_requests, array_of_statuses);
    update_clock_post();
    return ret;
}

// 拦截其他常用集合通信 (Collectives) 以保证时钟同步精度
int MPI_Barrier(MPI_Comm comm) {
    update_clock_pre();
    int ret = PMPI_Barrier(comm);
    update_clock_post();
    return ret;
}

int MPI_Allreduce(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm) {
    update_clock_pre();
    // Reduce 也会产生流量，简单起见暂不记录到 CSV (RAPS 主要看 P2P)
    int ret = PMPI_Allreduce(sendbuf, recvbuf, count, datatype, op, comm);
    update_clock_post();
    return ret;
}

int MPI_Finalize() {
    if (log_fp) fclose(log_fp);
    PMPI_Barrier(MPI_COMM_WORLD); // 确保大家写完
    
    if (my_rank == 0) {
        printf(">>> [Logger] Merging time-series data...\n");
        FILE *out = fopen("../data/traffic_timeseries.csv", "w");
        if (out) {
            fprintf(out, "Time,Source,Target,Bytes\n");
            char line[256];
            for (int i = 0; i < comm_size; i++) {
                char fname[64];
                sprintf(fname, "/tmp/traffic_rank_%d.csv", i);
                FILE *in = fopen(fname, "r");
                if (in) {
                    while (fgets(line, sizeof(line), in)) fputs(line, out);
                    fclose(in);
                    remove(fname);
                }
            }
            fclose(out);
            printf(">>> [Logger] Saved to ../data/traffic_timeseries.csv\n");
        }
    }
    return PMPI_Finalize();
}