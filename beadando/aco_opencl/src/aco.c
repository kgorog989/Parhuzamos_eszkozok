#include "aco.h"

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void init_distance_matrix(const char *distances, int num_cities, double city_distances[num_cities][num_cities])
{
    FILE *fp;
    if ((fp = fopen(distances, "r")) == NULL)
    {
        printf("File opening error");
        exit(-1);
    }

    for (int i = 0; i < num_cities; i++)
    {
        for (int j = 0; j < num_cities; j++)
        {
            fscanf(fp, "%lf", &city_distances[i][j]);
        }
    }

    fclose(fp);
}

void init_pheromones(int num_cities, double pheromones[num_cities][num_cities])
{
    for (int i = 0; i < num_cities; i++)
    {
        for (int j = 0; j < num_cities; j++)
        {
            pheromones[i][j] = 100;
        }
    }
}

void init_ants(int num_ants, int num_iterations, int num_cities, int (*ant_tours)[num_iterations][num_cities], double (*ant_lengths)[num_iterations]) {
    for (int i = 0; i < num_ants; i++) {
        for (int k = 0; k < num_iterations; k++) {
            (*ant_tours)[k][0] = rand() % num_cities;
            (*ant_lengths)[k] = 0;
        }
        ant_tours++;
        ant_lengths++;
    }
}

void init_ant_randoms(int num_ants, int num_iterations, int num_cities, double (*ant_randoms)[num_iterations][num_cities]) {
    for (int i = 0; i < num_ants; i++) {
        for (int j = 0; j < num_iterations; j++) {
            for (int k = 0; k < num_cities; k++) {
                (*ant_randoms)[j][k] = (double)rand() / RAND_MAX;
            }
        }
        ant_randoms++;
    }
}

void init_visited_cities(int num_ants, int num_iterations, int num_cities, int (*ant_tours)[num_iterations][num_cities], int (*visited_cities)[num_iterations][num_cities]) {
    for (int i = 0; i < num_ants; i++) {
        for (int j = 0; j < num_iterations; j++) {
            for (int k = 0; k < num_cities; k++) {
                (*visited_cities)[j][k] = 0;
            }
            (*visited_cities)[j][(*ant_tours)[j][0]] = 1;
        }
        ant_tours++;
        visited_cities++;
    }
}

void find_best_tour(int num_cities, int num_iterations, int num_ants, int (*ant_tours)[num_iterations][num_cities], double (*ant_lengths)[num_iterations], int *best_tour, double *best_length) {
    *best_length = INFINITY;
    for (int i = 0; i < num_ants; i++) {
        for (int j = 0; j < num_iterations; j++) {
            if ((*ant_lengths)[j] < *best_length) {
                *best_length = (*ant_lengths)[j];
                for (int k = 0; k < num_cities; k++) {
                    best_tour[k] = (*ant_tours)[j][k];
                }
            }
        }
        ant_tours++;
        ant_lengths++;
    }
}