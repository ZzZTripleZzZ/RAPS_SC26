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
