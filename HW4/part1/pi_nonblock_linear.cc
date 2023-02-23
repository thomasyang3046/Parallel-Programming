#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/types.h>
#include <unistd.h>

int main(int argc, char** argv)
{
    // --- DON'T TOUCH ---
    MPI_Init(&argc, &argv);
    double start_time = MPI_Wtime();
    double pi_result;
    long long int tosses = atoi(argv[1]);
    int world_rank, world_size;
    // ---

    // TODO: MPI init
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Status status;
    long long int n = tosses / world_size;
    unsigned int seed = time(NULL) + world_rank;
    long long int total = 0;
    long long int local_count = 0;

    for (long long int i = 0; i < n; i++)
    {
        double x = (double)rand_r(&seed) / (RAND_MAX);
        double y = (double)rand_r(&seed) / (RAND_MAX);
        if (x * x + y * y <= 1)
            local_count++;
    }
    
    if (world_rank > 0)
    {
        // TODO: MPI workers
        MPI_Request request;
        MPI_Isend(&local_count, 1, MPI_LONG_LONG_INT, 0, world_rank, MPI_COMM_WORLD, &request);
    }
    else if (world_rank == 0)
    {
        // TODO: non-blocking MPI communication.
        // Use MPI_Irecv, MPI_Wait or MPI_Waitall.
        total = local_count;
        MPI_Status status[world_size - 1];
        MPI_Request requests[world_size - 1];
        long long int result[world_size - 1];
        for (int node = 1; node < world_size; node++)
            MPI_Irecv(&result[node - 1], 1, MPI_LONG_LONG_INT, node, node, MPI_COMM_WORLD, &requests[node - 1]);

        MPI_Waitall(world_size - 1, requests, status);
        for (int i = 0; i < world_size - 1; i++)
            total += result[i];
    }

    if (world_rank == 0)
    {
        // TODO: PI result
        pi_result = 4.0 * (double)total / tosses;
        // --- DON'T TOUCH ---
        double end_time = MPI_Wtime();
        printf("%lf\n", pi_result);
        printf("MPI running time: %lf Seconds\n", end_time - start_time);
        // ---
    }

    MPI_Finalize();
    return 0;
}
