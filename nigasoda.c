#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>

#define N 2000000   // Total de Valores
#define G 400       // Total de Grupos

// Usamos punteros para manejar memoria local en cada proceso
long *valores_locales;
long centros[G];
int  volumen_global[G];

void kmean_mpi(int n_total, int n_grupos, long centros[], int volumen_global[], int rank, int size) {
    int i, j, min, iter = 0;
    long dif_local, dif_global, t;

    // Cuántos elementos le tocan a este proceso
    int n_locales = n_total / size;
    long sumas_locales[G];
    int volumen_local[G];
    int *grupos_locales = malloc(n_locales * sizeof(int));

    do {
        // 1. Fase de Clasificación: Cada proceso clasifica sus propios puntos
        for (i = 0; i < n_locales; i++) {
            min = 0;
            long dist_min = abs(valores_locales[i] - centros[0]);

            for (j = 1; j < n_grupos; j++) {
                long d = abs(valores_locales[i] - centros[j]);
                if (d < dist_min) {
                    min = j;
                    dist_min = d;
                }
            }
            grupos_locales[i] = min;
        }

        // 2. Reiniciar acumuladores locales
        for (i = 0; i < n_grupos; i++) {
            sumas_locales[i] = 0;
            volumen_local[i] = 0;
        }

        // 3. Acumular localmente
        for (i = 0; i < n_locales; i++) {
            sumas_locales[grupos_locales[i]] += valores_locales[i];
            volumen_local[grupos_locales[i]] += 1;
        }

        // 4. Comunicación Global: Reducción de sumas y volúmenes
        long sumas_globales[G];
        MPI_Allreduce(sumas_locales, sumas_globales, n_grupos, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(volumen_local, volumen_global, n_grupos, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

        // 5. Actualizar centros y calcular convergencia
        dif_local = 0;
        for (i = 0; i < n_grupos; i++) {
            t = centros[i];
            if (volumen_global[i] > 0)
                centros[i] = sumas_globales[i] / volumen_global[i];

            dif_local += abs(t - centros[i]);
        }

        // Todos deben estar de acuerdo en si "dif" es 0 para parar
        MPI_Allreduce(&dif_local, &dif_global, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);

        iter++;
    } while (dif_global > 0);

    if (rank == 0) printf("Iteraciones totales: %d\n", iter);
    free(grupos_locales);
}

// Quicksort se mantiene igual (solo se ejecuta en el rank 0)
void qs(int ii, int fi, long fV[], int fA[]) {
    /* ... (tu código original de qs) ... */
    int i, f; long pi, pa, vtmp, vta, vfi, vfa;
    pi = fV[ii]; pa = fA[ii]; i = ii + 1; f = fi;
    vtmp = fV[i]; vta = fA[i];
    while (i <= f) {
        if (vtmp < pi) { fV[i - 1] = vtmp; fA[i - 1] = vta; i++; vtmp = fV[i]; vta = fA[i]; }
        else { vfi = fV[f]; vfa = fA[f]; fV[f] = vtmp; fA[f] = vta; f--; vtmp = vfi; vta = vfa; }
    }
    fV[i - 1] = pi; fA[i - 1] = pa;
    if (ii < f) qs(ii, f, fV, fA);
    if (i < fi) qs(i, fi, fV, fA);
}

int main(int argc, char** argv) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int n_locales = N / size;
    valores_locales = malloc(n_locales * sizeof(long));

    // Inicialización de datos
    // Para que todos tengan datos coherentes, usamos la misma semilla o el rank 0 reparte

    long *temp_full = NULL;
    if (rank == 0) {
        temp_full = malloc(N * sizeof(long));
        for (int i = 0; i < N; i++) temp_full[i] = (rand() % rand()) / N;; // Simplificado para el ejemplo
        for (int i = 0; i < G; i++) centros[i] = temp_full[i];
    }

    // Distribuir los datos y los centros iniciales
    MPI_Scatter(temp_full, n_locales, MPI_LONG, valores_locales, n_locales, MPI_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(centros, G, MPI_LONG, 0, MPI_COMM_WORLD);

    // Ejecutar K-Means paralelo
    kmean_mpi(N, G, centros, volumen_global, rank, size);

    // Finalización
    if (rank == 0) {
        qs(0, G - 1, centros, volumen_global);
        for (int i = 0; i < G; i++)
            printf("R[%d] : %ld tiene %d agrupados\n", i, centros[i], volumen_global[i]);
        free(temp_full);
    }

    free(valores_locales);
    MPI_Finalize();
    return 0;
}
