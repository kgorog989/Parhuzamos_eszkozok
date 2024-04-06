#define ALPHA 1.0
#define BETA 2.0

__kernel void generate_solutions(const int num_ants, 
                                 const int num_cities, 
                                 __global double* city_distances,
                                 __global double* pheromones,
                                 __global int* ant_tours,
                                 __global double* ant_lengths,
                                 const __global double* ant_randoms,
                                 __global int* visited_cities) {
    int i = get_global_id(0);

    if (i < num_ants) {
        double total_probabilities, random_num, probability_sum, probability_to_move;
        int current_city, next_city, next_city_found;

        // Set the current city
        current_city = ant_tours[i * num_cities];

        for (int k = 1; k < num_cities; k++) {
            total_probabilities = 0;
            random_num = ant_randoms[i * num_cities + k];
            probability_sum = 0;
            
            for (int l = 0; l < num_cities; l++) {
                if (visited_cities[i * num_cities + l] == 0) {
                    total_probabilities += pow(pheromones[current_city * num_cities + l], ALPHA) * pow(1.0 / city_distances[current_city * num_cities + l], BETA);
                }
            }

            next_city_found = 0;
            for (int m = 0; m < num_cities; m++) {
                if (visited_cities[i * num_cities + m] == 0) {
                    probability_sum += pow(pheromones[current_city * num_cities + m], ALPHA) * pow(1.0 / city_distances[current_city * num_cities + m], BETA);
                    probability_to_move = probability_sum / total_probabilities;
                    if (probability_to_move >= random_num) {
                        next_city_found = 1;
                        next_city = m;
                        break;
                    }
                }
            }

            if (next_city_found == 0) {
                for (int n = num_cities - 1; n >= 0; n--) {
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

        // Adding the distance between the starting and ending city to the length of the tour
        ant_lengths[i] += city_distances[ant_tours[i * num_cities + num_cities - 1] * num_cities + ant_tours[i * num_cities]];
    }
}
