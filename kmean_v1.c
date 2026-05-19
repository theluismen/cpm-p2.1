#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>

#define N 2000000   // Total de Valores: 2 Millones
#define G 400       // Total de Grupos

long *valores_g;    // Vector de los N valores ( rank=0 )
long *valores_l;    // Vector de los N/nodos valores locales
long centros[G];    // Vector para los centroides de cada grupo
int  volumen_g[G];  // Vector para la cantidad de valores en cada grupo

void kmean_mpi ( int n_valores, int n_grupos, long valores[], long centros[], int volumen_g[], int rank )
{
    int  i, j, min, iter = 0;   // Variables de ayuda
    long dif_l, dif_g, t;       // Variables dif local, dif global y temporal
    long sumas_l[n_grupos];     // Suma de valores locales
    long sumas_g[n_grupos];     // Suma de valores globales
    int  volumen_l[n_grupos];   // Cantidades de grupos locales
    int  *grupos_l;             // grupos[i] = g -> grupos[i] pertenece al grupo g

    grupos_l = malloc(n_valores * sizeof(int));

    do
    {
        /* Agrupación de Elementos */
        for ( i = 0; i < n_valores; i++ )
        {
            min = 0;
            dif_l = abs(valores[i] - centros[0]);

            for ( j = 1; j < n_grupos; j++ )
            {
                if ( abs(valores[i] - centros[j]) < dif_l )
                {
                    min = j;
                    dif_l = abs(valores[i] - centros[j]);
                }
            }

            grupos_l[i] = min;
        }

        for ( i = 0; i < n_grupos; i++ )
        {
            sumas_l[i]   = 0;
            volumen_l[i] = 0;
        }

        /* Preparar Sumas y Totales */
        for ( i = 0; i < n_valores; i++ )
        {
            sumas_l[grupos_l[i]]   = sumas_l[grupos_l[i]] + valores[i];
            volumen_l[grupos_l[i]] = volumen_l[grupos_l[i]] + 1;
        }

        /* Reducir las sumas de datos locales a sumas de datos globales */
        MPI_Allreduce(sumas_l, sumas_g, n_grupos, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
        /* Reducir las volumenes de datos locales a volumenes de datos globales */
        MPI_Allreduce(volumen_l, volumen_g, n_grupos, MPI_INT, MPI_SUM, MPI_COMM_WORLD);


        /* Actualizar Centroides */
        dif_l = 0;

        for ( i = 0; i < n_grupos; i++ )
        {
            t = centros[i];

            if ( volumen_g[i] )
                centros[i] = sumas_g[i] / volumen_g[i];

            dif_l = dif_l + abs(t - centros[i]);
        }

        /* Reducir las dif locales a una dif global */
        MPI_Allreduce(&dif_l, &dif_g, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);

        iter++;

    } while ( dif_g );

    if ( rank == 0 ) {
        printf("iter %d\n", iter);
    }
    free(grupos_l);
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

int main ( int argc, char** argv )
{
    /* Variables */
    int i;
    int rank, nodos;
    int n_valores_l;    // Número de Valores Locales (N/nodos)

    /* Inicializar entorno MPI */
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);   // rank  : Identificador de nodo
    MPI_Comm_size(MPI_COMM_WORLD, &nodos);  // nodes : Cantidad de nodos

    /* Calcular Porciones de Datos*/
    n_valores_l = N / nodos;
    /* Asignar memoria a vector local de Valores */
    valores_l   = malloc( n_valores_l * sizeof(long) );

    /* Proceso ROOT */
    if ( rank == 0 ) {
        /* Asignar memoria a vector global de Valores */
        valores_g = malloc( N * sizeof(long) );

        /* Inicializar N Valores */
        for ( i = 0; i < N; i++ )
            valores_g[i] = ( rand() % rand() ) / N;

        /* Inicializar G centroides */
        for ( i = 0; i < G; i++ )
            centros[i] = valores_g[i];
    }

    /* PARTICIONAR N Valores entre nodes nodos */
    MPI_Scatter(valores_g, n_valores_l, MPI_LONG, valores_l, n_valores_l, MPI_LONG, 0, MPI_COMM_WORLD);
    /* REPLICAR G Centroides a los nodes nodos */
    MPI_Bcast(centros, G, MPI_LONG, 0, MPI_COMM_WORLD);

    /* Calcular G Grupos */
    kmean_mpi(n_valores_l, G, valores_l, centros, volumen_g, rank);

    /* Proceso ROOT */
    if ( rank == 0 ) {
        /* Ordenar Centroides y Cantidades */
        qs(0, G - 1, centros, volumen_g);

        /* Mostrar Resultados */
        for ( i = 0; i < G; i++ )
            printf("R[%d] : %ld tiene %d agrupados\n", i, centros[i], volumen_g[i]);

        /* Liberar Memoria Dinámica */
        free(valores_g);
    }

    /* Liberar Memoria Dinámica */
    free(valores_l);
    /* Terminar entorno de MPI */
    MPI_Finalize();

    return 0;
}
