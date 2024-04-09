#ifndef ACO_H
#define ACO_H

#define ALPHA 1.0
#define BETA 2.0
#define RHO 0.5
#define Q 100.0

void init_distance_matrix(const char *distances, int num_cities, double (*city_distances)[num_cities]);
void init_pheromones(int num_cities, double (*pheromones)[num_cities]);
void init_ants(int num_ants, int num_iterations, int num_cities, int (*ant_tours)[num_cities], double (*ant_lengths)[num_iterations]);
void init_ant_randoms(int num_ants, int num_iterations, int num_cities, double (*ant_randoms)[num_cities]);
void init_visited_cities(int num_ants, int num_iterations, int num_cities, int (*ant_tours)[num_cities], int (*visited_cities)[num_cities]);
void find_best_tour(int num_cities, int num_iterations, int num_ants, int (*ant_tours)[num_cities], double (*ant_lengths)[num_iterations], int *best_tour, double *best_length);

#endif
