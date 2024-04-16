#define ALPHA 1.0
#define BETA 2.0
#define RHO 0.5
#define Q 100.0

__kernel void iterate(const int num_ants, const int num_cities,
                      __global double *city_distances,
                      __global double *pheromones, __global int *ant_tours,
                      __global double *ant_lengths,
                      const __global double *ant_randoms,
                      __global int *visited_cities, const int num_iterations) {
  int ant_id = get_global_id(0);
  if (ant_id > num_ants) {
    return;
  }
  //printf("start ant id: %d\n", ant_id);

  for (int i = 0; i < num_iterations; i++) {
    double total_probabilities, random_num, probability_sum,
        probability_to_move;
    int current_city, next_city, next_city_found;

    // Set the current city
    current_city =
        ant_tours[ant_id * num_iterations * num_cities + i * num_cities];

    for (int k = 1; k < num_cities; k++) {
      total_probabilities = 0;
      random_num = ant_randoms[ant_id * num_iterations * num_cities +
                               i * num_cities + k];
      probability_sum = 0;

      for (int l = 0; l < num_cities; l++) {
        if (visited_cities[ant_id * num_iterations * num_cities +
                           i * num_cities + l] == 0) {
          total_probabilities +=
              pow(pheromones[current_city * num_cities + l], ALPHA) *
              pow(1.0 / city_distances[current_city * num_cities + l], BETA);
        }
      }

      next_city_found = 0;
      for (int m = 0; m < num_cities; m++) {
        if (visited_cities[ant_id * num_iterations * num_cities +
                           i * num_cities + m] == 0) {
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
          if (visited_cities[ant_id * num_iterations * num_cities +
                             i * num_cities + n] == 0) {
            next_city = n;
            break;
          }
        }
      }

      ant_tours[ant_id * num_iterations * num_cities + i * num_cities + k] =
          next_city;
      ant_lengths[ant_id * num_iterations + i] +=
          city_distances[current_city * num_cities + next_city];
      visited_cities[ant_id * num_iterations * num_cities + i * num_cities +
                     next_city] = 1;
      current_city = next_city;
    }

    // Adding the distance between the starting and ending city to the length of
    // the tour
    ant_lengths[ant_id * num_iterations + i] +=
        city_distances[ant_tours[ant_id * num_iterations * num_cities +
                                 i * num_cities + num_cities - 1] *
                           num_cities +
                       ant_tours[ant_id * num_iterations * num_cities +
                                 i * num_cities]];

    barrier(CLK_GLOBAL_MEM_FENCE);

    if (ant_id == 1) {
      // Updating pheromones
      int city1, city2;
      // Evaporating pheromones
      for (int k = 0; k < num_cities; k++) {
        for (int j = 0; j < num_cities; j++) {
          pheromones[k * num_cities + j] *= (1.0 - RHO);
        }
      }
      // Depositing pheromones
      for (int k = 0; k < num_ants; k++) {
        for (int l = 0; l < num_cities - 1; l++) {
          int tour_index = k * num_iterations * num_cities + i * num_cities + l;
          city1 = ant_tours[tour_index];
          city2 = ant_tours[tour_index + 1];
          pheromones[city2 * num_cities + city1] +=
              Q / ant_lengths[k * num_iterations + i];
          pheromones[city1 * num_cities + city2] +=
              Q / ant_lengths[k * num_iterations + i];
        }
        // Depositing pheromones to the first and last tour
        int tour_index = k * num_iterations * num_cities + i * num_cities;
        city1 = ant_tours[tour_index - 1];
        city2 = ant_tours[tour_index];
        pheromones[city2 * num_cities + city1] +=
            Q / ant_lengths[k * num_iterations + i];
        pheromones[city1 * num_cities + city2] +=
            Q / ant_lengths[k * num_iterations + i];
      }
    }
    barrier(CLK_GLOBAL_MEM_FENCE);
  }
}
