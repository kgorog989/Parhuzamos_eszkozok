#include "aco.h"

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void init_distance_matrix(const char *distances, int num_cities, double (*city_distances)[num_cities])
{
    FILE *fp;
    if ((fp = fopen(distances, "r")) == NULL)
    {
        printf("File opening error");
        exit(-1);
    }

    int i, j;

    for (i = 0; i < num_cities; i++)
    {
        for (j = 0; j < num_cities; j++)
        {
            fscanf(fp, "%lf", city_distances[i] + j);
        }
    }

    fclose(fp);
}

void init_pheromones(int num_cities, double (*pheromones)[num_cities])
{
    for (int i = 0; i < num_cities; i++)
    {
        for (int j = 0; j < num_cities; j++)
        {
            pheromones[i][j] = 100;
        }
    }
}

void init_ants(int num_ants, int num_iterations, int num_cities, int (*ant_tours)[num_cities], double (*ant_lengths)[num_iterations])
{
    for (int i = 0; i < num_ants; i++)
    {
        for (int k = 0; k < num_iterations; k++)
        {
            ant_tours[i][k][0] = rand() % num_cities;
            ant_lengths[i][k] = 0;
        }
    }
}

void init_ant_randoms(int num_ants, int num_iterations, int num_cities, double (*ant_randoms)[num_cities])
{
    for (int i = 0; i < num_ants; i++)
    {
        for (int j = 0; j < num_iterations; j++)
        {
            for (int k = 0; k < num_cities; k++)
            {
                ant_randoms[i][j][k] = (double)rand() / RAND_MAX;
            }
        }
    }
}

void init_visited_cities(int num_ants, int num_iterations, int num_cities, int (*ant_tours)[num_cities], int (*visited_cities)[num_cities])
{
    for (int i = 0; i < num_ants; i++)
    {
        for (int j = 0; j < num_iterations; j++)
        {
            for (int k = 0; k < num_cities; k++)
            {
                visited_cities[i][j][k] = 0;
            }
            visited_cities[i][j][0] = 1;
        }
    }
}

void find_best_tour(int num_cities, int num_iterations, int num_ants, int *ant_tours, double (*ant_lengths)[num_iterations], int *best_tour, double *best_length)
{
    for (int i = 0; i < num_ants; i++)
    {
        for (int j = 0; j < num_iterations; j++)
        {
            if (ant_lengths[i][j] < *best_length)
            {
                *best_length = ant_lengths[i][j];
                for (int k = 0; k < num_cities; k++)
                {
                    best_tour[k] = ant_tours[i * num_iterations * num_cities + j * num_cities + k];
                }
            }
        }
    }
}