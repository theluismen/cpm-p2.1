#include <stdlib.h>
#include <stdio.h>

#define N 2000000   // Total de Valores: 2 Millones
#define G 400       // Total de Grupos

long valores[N];    // Vector de los N valores
long centros[G];    // Vector para los centroides de cada grupo
int  volumen[G];    // Vector para la cantidad de valores en cada grupo

void kmean(int n_valores, int n_grupos, long valores[], long centros[], int volumen[])
{
    int     i, j, min, iter = 0;
    long    dif, t;
    long    sumas[G];   // Suma de valores de cada grupo
    int     grupos[N];  // grupos[i] = g -> grupos[i] pertenece al grupo g

    do
    {
        for (i = 0; i < n_valores; i++)
        {
            min = 0;
            dif = abs(valores[i] - centros[0]);

            for (j = 1; j < n_grupos; j++)
            {
                if (abs(valores[i] - centros[j]) < dif)
                {
                    min = j;
                    dif = abs(valores[i] - centros[j]);
                }
            }

            grupos[i] = min;
        }

        for (i = 0; i < n_grupos; i++)
        {
            sumas[i] = 0;
            volumen[i] = 0;
        }

        for (i = 0; i < n_valores; i++)
        {
            sumas[grupos[i]] = sumas[grupos[i]] + valores[i];
            volumen[grupos[i]] = volumen[grupos[i]] + 1;
        }

        dif = 0;

        for (i = 0; i < n_grupos; i++)
        {
            t = centros[i];

            if (volumen[i])
                centros[i] = sumas[i] / volumen[i];

            dif = dif + abs(t - centros[i]);
        }

        iter++;

    } while (dif);

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

int main()
{
    int i;

    for (i = 0; i < N; i++)
        valores[i] = (rand() % rand()) / N;

    // primeros candidatos
    for (i = 0; i < G; i++)
        centros[i] = valores[i];

    // calcular los G más representativos
    kmean(N, G, valores, centros, volumen);

    qs(0, G - 1, centros, volumen);

    for (i = 0; i < G; i++)
        printf("R[%d] : %ld tiene %d agrupados\n", i, centros[i], volumen[i]);

    return 0;
}
