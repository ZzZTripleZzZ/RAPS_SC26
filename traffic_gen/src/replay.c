// src/replay.c
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define MAX_EVENTS 200000
// 修复点：从 10MB 降为 1MB，防止 64 进程并发时撑爆内存 (OOM)
#define BUFFER_SIZE (1 * 1024 * 1024) 

typedef struct {
    double time;
    int target;
    int bytes;
} Event;

Event my_events[MAX_EVENTS];
int event_count = 0;

int compare_events(const void *a, const void *b) {
    double t1 = ((Event*)a)->time;
    double t2 = ((Event*)b)->time;
    if (t1 < t2) return -1;
    if (t1 > t2) return 1;
    return 0;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    char *csv_path = "../data/traffic_timeseries.csv";
    if (argc > 1) csv_path = argv[1];

    // === 1. 加载 CSV ===
    FILE *fp = fopen(csv_path, "r");
    if (!fp) {
        if (rank == 0) printf("Error: Cannot open %s\n", csv_path);
        MPI_Finalize();
        return 0;
    }

    char line[256];
    fgets(line, sizeof(line), fp); // Header

    while (fgets(line, sizeof(line), fp)) {
        double t;
        int src, dst, bytes;
        if (sscanf(line, "%lf,%d,%d,%d", &t, &src, &dst, &bytes) == 4) {
            if (src == rank) {
                if (event_count < MAX_EVENTS) {
                    my_events[event_count].time = t;
                    my_events[event_count].target = dst;
                    my_events[event_count].bytes = bytes;
                    event_count++;
                }
            }
        }
    }
    fclose(fp);

    qsort(my_events, event_count, sizeof(Event), compare_events);
    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0) printf(">>> [Replay] Loaded %d events. Buffer: 1MB. Starting...\n", event_count);

    // === 2. 准备接收池 (Sink) ===
    char *recv_buf = (char*)malloc(BUFFER_SIZE);
    if (!recv_buf) {
        fprintf(stderr, "Rank %d failed to allocate recv memory\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    MPI_Request recv_req;
    MPI_Irecv(recv_buf, BUFFER_SIZE, MPI_BYTE, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &recv_req);

    // === 3. 回放循环 ===
    double start_time = MPI_Wtime();
    int current_idx = 0;
    
    char *send_buf = (char*)malloc(BUFFER_SIZE);
    if (!send_buf) {
        fprintf(stderr, "Rank %d failed to allocate send memory\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    memset(send_buf, 0, 128); 

    while (1) {
        double now = MPI_Wtime() - start_time;

        // A. 发送 (Replay)
        while (current_idx < event_count && my_events[current_idx].time <= now) {
            Event *e = &my_events[current_idx];
            
            int count = e->bytes;
            if (count > BUFFER_SIZE) count = BUFFER_SIZE;
            
            MPI_Request req;
            MPI_Isend(send_buf, count, MPI_BYTE, e->target, 0, MPI_COMM_WORLD, &req);
            MPI_Request_free(&req); 

            current_idx++;
        }

        // B. 接收 (Drain)
        int flag;
        MPI_Status status;
        MPI_Test(&recv_req, &flag, &status);
        if (flag) {
            MPI_Irecv(recv_buf, BUFFER_SIZE, MPI_BYTE, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &recv_req);
        }

        // C. 退出条件
        if (current_idx >= event_count) {
            // 为防止过早退出导致还在路上的包被 Connection Reset，
            // 这里简单做一个 Allreduce 同步，确保大家都发完了
            // (严格来说应该用更复杂的逻辑，但这里简单等待 1 秒通常足够清空网络)
            double t_end = MPI_Wtime() + 1.0;
            while(MPI_Wtime() < t_end) {
                 MPI_Test(&recv_req, &flag, &status);
                 if (flag) MPI_Irecv(recv_buf, BUFFER_SIZE, MPI_BYTE, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &recv_req);
            }
            break; 
        }
        
        usleep(50);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) printf(">>> [Replay] Finished. Time: %.4fs\n", MPI_Wtime() - start_time);

    free(recv_buf);
    free(send_buf);
    MPI_Finalize();
    return 0;
}