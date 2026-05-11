#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>

#define N 2000000
#define G 400

void kmean_mpi(int local_n, long local_V[], long R[], int A[], int rank)
{
    int i, j, min, iter = 0;
    long dif, t;

    long local_S[G];
    int local_A[G];

    long global_S[G];
    int global_A[G];

    do
    {
        // reset locales
        for (i = 0; i < G; i++)
        {
            local_S[i] = 0;
            local_A[i] = 0;
        }

        // asignación local
        for (i = 0; i < local_n; i++)
        {
            min = 0;
            long dif_min = labs(local_V[i] - R[0]);

            for (j = 1; j < G; j++)
            {
                long d = labs(local_V[i] - R[j]);
                if (d < dif_min)
                {
                    min = j;
                    dif_min = d;
                }
            }

            local_S[min] += local_V[i];
            local_A[min] += 1;
        }

        // reducción global
        MPI_Allreduce(local_S, global_S, G, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(local_A, global_A, G, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

        // actualizar centroides
        dif = 0;
        for (i = 0; i < G; i++)
        {
            t = R[i];

            if (global_A[i] > 0)
                R[i] = global_S[i] / global_A[i];

            dif += labs(t - R[i]);
            A[i] = global_A[i];
        }

        // convergencia global
        MPI_Allreduce(MPI_IN_PLACE, &dif, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);

        iter++;

    } while (dif != 0);

    if (rank == 0)
        printf("iter %d\n", iter);
}

void qs(int ii, int fi, long fV[], int fA[])
{
    int i, f;
    long pi, pa, vtmp, vta, vfi, vfa;

    pi = fV[ii];
    pa = fA[ii];

    i = ii + 1;
    f = fi;

    vtmp = fV[i];
    vta = fA[i];

    while (i <= f)
    {
        if (vtmp < pi)
        {
            fV[i - 1] = vtmp;
            fA[i - 1] = vta;

            i++;

            vtmp = fV[i];
            vta = fA[i];
        }
        else
        {
            vfi = fV[f];
            vfa = fA[f];

            fV[f] = vtmp;
            fA[f] = vta;

            f--;

            vtmp = vfi;
            vta = vfa;
        }
    }

    fV[i - 1] = pi;
    fA[i - 1] = pa;

    if (ii < f)
        qs(ii, f, fV, fA);

    if (i < fi)
        qs(i, fi, fV, fA);
}

int main(int argc, char *argv[])
{
    int rank, size, i;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    long *V = NULL;
    long R[G];
    int A[G];

    int *counts = malloc(size * sizeof(int));
    int *displs = malloc(size * sizeof(int));

    int base = N / size;
    int extra = N % size;

    // reparto irregular
    for (i = 0; i < size; i++)
    {
        counts[i] = base + (i < extra ? 1 : 0);
        displs[i] = (i == 0) ? 0 : displs[i - 1] + counts[i - 1];
    }

    int local_n = counts[rank];
    long *local_V = malloc(local_n * sizeof(long));

    // solo root genera datos
    if (rank == 0)
    {
        V = malloc(N * sizeof(long));

        for (i = 0; i < N; i++)
            V[i] = rand() % 1000000;

        for (i = 0; i < G; i++)
            R[i] = V[i];
    }

    // repartir datos
    MPI_Scatterv(V, counts, displs, MPI_LONG,
                 local_V, local_n, MPI_LONG,
                 0, MPI_COMM_WORLD);

    // compartir centroides iniciales
    MPI_Bcast(R, G, MPI_LONG, 0, MPI_COMM_WORLD);

    // ejecutar kmeans
    kmean_mpi(local_n, local_V, R, A, rank);

    // imprimir resultados
    if (rank == 0)
    {
        qs(0, G - 1, R, A);
        for (i = 0; i < G; i++)
            printf("R[%d] : %ld tiene %d agrupados\n", i, R[i], A[i]);
    }

    MPI_Finalize();
    return 0;
}