#include "aco.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <limits.h>

int main(int argc, char *argv[])
{
    srand(time(NULL));

    double total_time;
    clock_t start, end;

    int num_iterations = 100;
    int num_ants;
    int max_ants = 10;
    int num_cities = 312;
    double city_distances[num_cities][num_cities];
    double pheromones[num_cities][num_cities];
    int best_tour[num_cities];
    double best_length;

    FILE *file;
    if ((file = fopen("data/times_usca312.txt", "w")) == NULL)
    {
        printf("File opening error");
        exit(-1);
    }

    for (num_ants = 2; num_ants <= max_ants; num_ants++)
    {

        start = clock();

        int ant_tours[num_ants][num_cities];
        double ant_lengths[num_ants];
        best_length = INFINITY;
        init_distance_matrix("data/usca312.txt", num_cities, city_distances);
        init_pheromones(num_cities, pheromones);

        for (int i = 0; i < num_iterations; i++)
        {

            init_ants(num_ants, num_cities, ant_tours, ant_lengths);
            generate_solutions(num_ants, num_cities, city_distances, pheromones, ant_tours, ant_lengths);
            update_pheromones(num_cities, num_ants, pheromones, ant_tours, ant_lengths);
            find_best_tour(num_cities, num_ants, ant_tours, ant_lengths, best_tour, &best_length);
        }

        printf("\nBest tour: ");
        for (int i = 0; i < num_cities; i++)
        {
            printf("%d ", best_tour[i]);
        }
        printf("\nBest tour length: %lf\n", best_length);
        end = clock();
        total_time = ((double)(end - start)) / CLK_TCK;

        fprintf(file, "%d %lf %lf\n", num_ants, total_time, best_length);
    }

    fclose(file);

    return 0;
}
