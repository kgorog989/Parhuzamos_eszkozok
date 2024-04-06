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

void init_ants(int num_ants, int num_cities, int (*ant_tours)[num_cities], double *ant_lengths)
{
    for (int i = 0; i < num_ants; i++)
    {
        ant_tours[i][0] = rand() % num_cities;
        for (int j = 1; j < num_cities; j++)
        {
            ant_tours[i][j] = -1;
        }
        ant_lengths[i] = 0;
    }
}

void init_ant_randoms(int num_ants, int num_cities, double (*ant_randoms)[num_cities])
{
    for (int i = 0; i < num_ants; i++)
    {
        for (int j = 0; j < num_cities; j++)
        {
            ant_randoms[i][j] = (double)rand() / RAND_MAX;
        }
    }
}

void init_visited_cities(int num_ants, int num_cities, int (*ant_tours)[num_cities], int (*visited_cities)[num_cities])
{
    for (int i = 0; i < num_ants; i++)
    {
        for (int j = 0; j < num_cities; j++)
        {
            visited_cities[i][j] = 0;
        }
        visited_cities[i][ant_tours[i][0]] = 1;
    }
}

void update_pheromones(int num_cities, int num_ants, double (*pheromones)[num_cities], int (*ant_tours)[num_cities], double *ant_lengths)
{
    int city1, city2;
    // Evaporating pheromones
    for (int i = 0; i < num_cities; i++)
    {
        for (int j = 0; j < num_cities; j++)
        {
            pheromones[i][j] *= (1.0 - RHO);
        }
    }
    // Depositing pheromones
    for (int i = 0; i < num_ants; i++)
    {
        for (int j = 0; j < num_cities - 1; j++)
        {
            city1 = ant_tours[i][j];
            city2 = ant_tours[i][j + 1];
            pheromones[city1][city2] += Q / ant_lengths[i];
            pheromones[city2][city1] += Q / ant_lengths[i];
        }
        city1 = ant_tours[i][num_cities - 1];
        city2 = ant_tours[i][0];
        pheromones[city1][city2] += Q / ant_lengths[i];
        pheromones[city2][city1] += Q / ant_lengths[i];
    }
}

void find_best_tour(int num_cities, int num_ants, int (*ant_tours)[num_cities], double *ant_lengths, int *best_tour, double *best_length)
{
    for (int i = 0; i < num_ants; i++)
    {
        if (ant_lengths[i] < *best_length)
        {
            *best_length = ant_lengths[i];
            for (int j = 0; j < num_cities; j++)
            {
                best_tour[j] = ant_tours[i][j];
            }
        }
    }
}