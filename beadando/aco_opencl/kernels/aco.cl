#define ALPHA 1.0
#define BETA 2.0
#define RHO 0.5
#define Q 100.0

__kernel void generate_solutions(const int num_ants, const int num_cities,
                                 __global double *city_distances,
                                 __global double *pheromones,
                                 __global int *ant_tours,
                                 __global double *ant_lengths,
                                 const __global double *ant_randoms,
                                 __global int *visited_cities) {
  int i = get_global_id(0);

  if (i < num_ants) {
    printf("generate id : %d", i);
    double total_probabilities, random_num, probability_sum,
        probability_to_move;
    int current_city, next_city, next_city_found;

    // Set the current city
    current_city = ant_tours[i * num_cities];

    for (int k = 1; k < num_cities; k++) {
      total_probabilities = 0;
      random_num = ant_randoms[i * num_cities + k];
      probability_sum = 0;

      for (int l = 0; l < num_cities; l++) {
        if (visited_cities[i * num_cities + l] == 0) {
          total_probabilities +=
              pow(pheromones[current_city * num_cities + l], ALPHA) *
              pow(1.0 / city_distances[current_city * num_cities + l], BETA);
        }
      }

      next_city_found = 0;
      for (int m = 0; m < num_cities; m++) {
        if (visited_cities[i * num_cities + m] == 0) {
          probability_sum +=
              pow(pheromones[current_city * num_cities + m], ALPHA) *
              pow(1.0 / city_distances[current_city * num_cities + m], BETA);
          probability_to_move = probability_sum / total_probabilities;
          if (probability_to_move >= random_num) {
            next_city_found = 1;
            next_city = m;
            break;
          }
        }
      }

      if (next_city_found == 0) {
        for (int n = 0; n < num_cities; n++) {
          if (visited_cities[i * num_cities + n] == 0) {
            next_city = n;
            break;
          }
        }
      }

      ant_tours[i * num_cities + k] = next_city;
      ant_lengths[i] += city_distances[current_city * num_cities + next_city];
      visited_cities[i * num_cities + next_city] = 1;
      current_city = next_city;
    }

    // Adding the distance between the starting and ending city to the length of
    // the tour
    ant_lengths[i] +=
        city_distances[ant_tours[i * num_cities + num_cities - 1] * num_cities +
                       ant_tours[i * num_cities]];
  }
}

__kernel void iterate(const int num_ants, const int num_cities,
                      __global double *city_distances,
                      __global double *pheromones, __global int *ant_tours,
                      __global double *ant_lengths,
                      const __global double *ant_randoms,
                      __global int *visited_cities, const int num_iterations) {
  int id = get_global_id(0);
  printf("\n inside id: %d", id);
  for (int i = 0; i < num_iterations; i++) {

    generate_solutions(
        num_ants, num_cities, city_distances, pheromones,
        &ant_tours[id * num_iterations * num_cities + i * num_cities],
        &ant_lengths[id * num_iterations],
        &ant_randoms[id * num_iterations * num_cities + i * num_cities],
        &visited_cities[id * num_iterations * num_cities + i * num_cities]);

    // Updating pheromones
    int city1, city2;
    // Evaporating pheromones
    for (int k = 0; k < num_cities; k++)
    {
        for (int j = 0; j < num_cities; j++)
        {
            pheromones[k * num_cities + j] *= (1.0 - RHO);
        }
    }
    // Depositing pheromones
    for (int k = 0; k < num_ants; k++)
    {
        for (int j = 0; j < num_cities - 1; j++)
        {
            city1 = ant_tours[id * num_iterations * num_cities + i * num_cities + j];
            city2 = ant_tours[id * num_iterations * num_cities + i * num_cities + j + 1];
            pheromones[city2 * num_cities + city1] += Q / ant_lengths[id * num_iterations + i];
            pheromones[city1 * num_cities + city2] += Q / ant_lengths[id * num_iterations + i];
        }
        city1 = ant_tours[id * num_iterations * num_cities + i * num_cities + num_cities - 1];
        city2 = ant_tours[id * num_iterations * num_cities + i * num_cities];
        pheromones[city2 * num_cities + city1] += Q / ant_lengths[id * num_iterations + i];
        pheromones[city1 * num_cities + city2] += Q / ant_lengths[id * num_iterations + i];
    }
  }
}
