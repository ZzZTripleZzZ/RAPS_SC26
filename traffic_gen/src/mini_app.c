#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// 定义最大 Rank 数 (为了简化矩阵打印)
#define MAX_RANKS 64

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // 1. 初始化流量计数器 (Traffic Matrix Row)
    // my_traffic[target_rank] = bytes_sent
    long long my_traffic[MAX_RANKS];
    memset(my_traffic, 0, sizeof(my_traffic));

    // 模拟 Frontier 常见的 "Halo Exchange"
    int left = (rank - 1 + size) % size;
    int right = (rank + 1) % size;
    
    int data_size = 1000; // int count
    int bytes = data_size * sizeof(int);
    int send_buf[1000];
    int recv_buf[1000];

    // --- 第一轮通信 (向右) ---
    MPI_Sendrecv(send_buf, data_size, MPI_INT, right, 0,
                 recv_buf, data_size, MPI_INT, left, 0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    // 记录流量
    my_traffic[right] += bytes;

    // --- 第二轮通信 (向左) ---
    MPI_Sendrecv(send_buf, data_size, MPI_INT, left, 1,
                 recv_buf, data_size, MPI_INT, right, 1,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    // 记录流量
    my_traffic[left] += bytes;

    // 2. 收集全局流量矩阵 (Gather Traffic Matrix)
    // 只有 Rank 0 需要这张全图
    long long global_matrix[MAX_RANKS * MAX_RANKS];
    MPI_Gather(my_traffic, MAX_RANKS, MPI_LONG_LONG,
               global_matrix, MAX_RANKS, MPI_LONG_LONG,
               0, MPI_COMM_WORLD);

    // 3. Rank 0 将矩阵写入 CSV 文件
    if (rank == 0) {
        FILE *fp = fopen("../data/traffic_matrix.csv", "w");
        if (fp) {
            // 写入 CSV Header
            fprintf(fp, "Source,Target,Bytes\n");
            for (int src = 0; src < size; src++) {
                for (int dst = 0; dst < size; dst++) {
                    long long volume = global_matrix[src * MAX_RANKS + dst];
                    // 只记录有流量的边 (稀疏矩阵)
                    if (volume > 0) {
                        fprintf(fp, "%d,%d,%lld\n", src, dst, volume);
                    }
                }
            }
            fclose(fp);
            printf("Success: Real traffic matrix exported to data/traffic_matrix.csv\n");
        } else {
            printf("Error: Could not open file for writing.\n");
        }
    }

    MPI_Finalize();
    return 0;
}