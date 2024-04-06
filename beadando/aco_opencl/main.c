#include "aco.h"

#define CL_TARGET_OPENCL_VERSION 220
#include <CL/cl.h>

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

    cl_int err;
    int error_code;

    int num_iterations = 100;
    int num_ants;
    int max_ants = 3;
    int num_cities = 312;
    double city_distances[num_cities][num_cities];
    double pheromones[num_cities][num_cities];
    int best_tour[num_cities];
    double best_length;

    // Get platform
    cl_uint n_platforms;
    cl_platform_id platform_id;
    err = clGetPlatformIDs(1, &platform_id, &n_platforms);
    if (err != CL_SUCCESS)
    {
        printf("[ERROR] Error calling clGetPlatformIDs. Error code: %d\n", err);
        return 0;
    }

    // Get device
    cl_device_id device_id;
    cl_uint n_devices;
    err = clGetDeviceIDs(
        platform_id,
        CL_DEVICE_TYPE_GPU,
        1,
        &device_id,
        &n_devices);
    if (err != CL_SUCCESS)
    {
        printf("[ERROR] Error calling clGetDeviceIDs. Error code: %d\n", err);
        return 0;
    }

    // Create OpenCL context
    cl_context context = clCreateContext(NULL, n_devices, &device_id, NULL, NULL, NULL);

    // Build the program
    const char *kernel_code = load_kernel_source("kernels/aco.cl", &error_code);
    if (error_code != 0)
    {
        printf("Source code loading error!\n");
        return 0;
    }
    cl_program program = clCreateProgramWithSource(context, 1, &kernel_code, NULL, NULL);
    const char options[] = "-D SET_ME=1234";
    err = clBuildProgram(
        program,
        1,
        &device_id,
        options,
        NULL,
        NULL);
    if (err != CL_SUCCESS)
    {
        printf("Build error! Code: %d\n", err);
        size_t real_size;
        err = clGetProgramBuildInfo(
            program,
            device_id,
            CL_PROGRAM_BUILD_LOG,
            0,
            NULL,
            &real_size);
        char *build_log = (char *)malloc(sizeof(char) * (real_size + 1));
        err = clGetProgramBuildInfo(
            program,
            device_id,
            CL_PROGRAM_BUILD_LOG,
            real_size + 1,
            build_log,
            &real_size);
        // build_log[real_size] = 0;
        printf("Real size : %d\n", real_size);
        printf("Build log : %s\n", build_log);
        free(build_log);
        return 0;
    }
    size_t sizes_param[10];
    size_t real_size;
    err = clGetProgramInfo(
        program,
        CL_PROGRAM_BINARY_SIZES,
        10,
        sizes_param,
        &real_size);
    printf("Real size   : %d\n", real_size);
    printf("Binary size : %d\n", sizes_param[0]);

    // Create the command queue
    cl_command_queue command_queue = clCreateCommandQueue(context, device_id, NULL, NULL);

    // Fill the distance matrix
    FILE *file;
    if ((file = fopen("data/times_usca312.txt", "w")) == NULL)
    {
        printf("File opening error");
        exit(-1);
    }
    init_distance_matrix("data/usca312.txt", num_cities, city_distances);

    // Prepare OpenCL memory object for the distance matrix and copy it to the GPU
    cl_mem buf_city_distances = clCreateBuffer(context, CL_MEM_READ_ONLY, num_cities * num_cities * sizeof(double), NULL, NULL);
    clEnqueueWriteBuffer(command_queue, buf_city_distances, CL_TRUE, 0, num_cities * num_cities * sizeof(double), city_distances, 0, NULL, NULL);

    printf("\ncreating Kernel...\n");
    cl_kernel kernel = clCreateKernel(program, "generate_solutions", &err);
    if (err != CL_SUCCESS)
    {
        printf("%d\n", err);
        return 0;
    }
    printf("\nKernel created\n");

    // Set kernel arguments
    err = clSetKernelArg(kernel, 1, sizeof(int), (void *)&num_cities);
    if (err != CL_SUCCESS)
    {
        printf("%d\n", err);
        return 0;
    }
    


    err = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&buf_city_distances);
    if (err != CL_SUCCESS)
    {
        printf("%d\n", err);
        return 0;
    }

    for (num_ants = 2; num_ants <= max_ants; num_ants++)
    {

        start = clock();

        int ant_tours[num_ants][num_cities];
        double ant_lengths[num_ants];
        double ant_randoms[num_ants][num_cities];
        int visited_cities[num_ants][num_cities];
        best_length = INFINITY;

        init_pheromones(num_cities, pheromones);

        cl_mem buf_pheromones = clCreateBuffer(context, CL_MEM_READ_WRITE, num_cities * num_cities * sizeof(double), NULL, NULL);
        cl_mem buf_ant_tours = clCreateBuffer(context, CL_MEM_READ_WRITE, num_ants * num_cities * sizeof(int), NULL, NULL);
        cl_mem buf_ant_lengths = clCreateBuffer(context, CL_MEM_READ_WRITE, num_ants * sizeof(double), NULL, NULL);
        cl_mem buf_ant_randoms = clCreateBuffer(context, CL_MEM_READ_WRITE, num_ants * num_cities * sizeof(int), NULL, NULL);
        cl_mem buf_visited_cities = clCreateBuffer(context, CL_MEM_READ_WRITE, num_ants * num_cities * sizeof(int), NULL, NULL);

        clEnqueueWriteBuffer(command_queue, buf_pheromones, CL_TRUE, 0, num_cities * num_cities * sizeof(double), pheromones, 0, NULL, NULL);

        for (int i = 0; i < num_iterations; i++)
        {
            init_ants(num_ants, num_cities, ant_tours, ant_lengths);
            init_ant_randoms(num_ants, num_cities, ant_randoms);
            init_visited_cities(num_ants, num_cities, ant_tours, visited_cities);


            clEnqueueWriteBuffer(command_queue, buf_ant_tours, CL_TRUE, 0, num_ants * num_cities * sizeof(int), ant_tours, 0, NULL, NULL);
            clEnqueueWriteBuffer(command_queue, buf_ant_lengths, CL_TRUE, 0, num_ants * sizeof(double), ant_lengths, 0, NULL, NULL);
            clEnqueueWriteBuffer(command_queue, buf_ant_randoms, CL_TRUE, 0, num_ants * num_cities * sizeof(int), ant_randoms, 0, NULL, NULL);
            clEnqueueWriteBuffer(command_queue, buf_visited_cities, CL_TRUE, 0, num_ants * num_cities * sizeof(int), visited_cities, 0, NULL, NULL);

            // Set kernel arguments
            err = clSetKernelArg(kernel, 0, sizeof(int), (void *)&num_ants);
            if (err != CL_SUCCESS)
            {
                printf("%d\n", err);
                return 0;
            }

            err = clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *)&buf_pheromones);
            if (err != CL_SUCCESS)
            {
                printf("%d\n", err);
                return 0;
            }

            err = clSetKernelArg(kernel, 4, sizeof(cl_mem), (void *)&buf_ant_tours);
            if (err != CL_SUCCESS)
            {
                printf("%d\n", err);
                return 0;
            }

            err = clSetKernelArg(kernel, 5, sizeof(cl_mem), (void *)&buf_ant_lengths);
            if (err != CL_SUCCESS)
            {
                printf("%d\n", err);
                return 0;
            }

            err = clSetKernelArg(kernel, 6, sizeof(cl_mem), (void *)&buf_ant_randoms);
            if (err != CL_SUCCESS)
            {
                printf("%d\n", err);
                return 0;
            }

            err = clSetKernelArg(kernel, 7, sizeof(cl_mem), (void *)&buf_visited_cities);
            if (err != CL_SUCCESS)
            {
                printf("%d\n", err);
                return 0;
            }

            // Size specification
            size_t local_work_size = 256;
            size_t n_work_groups = (num_cities + local_work_size + 1) / local_work_size;
            size_t global_work_size = n_work_groups * local_work_size;

            // Apply the kernel on the range
            clEnqueueNDRangeKernel(
                command_queue,
                kernel,
                1,
                NULL,
                &global_work_size,
                &local_work_size,
                0,
                NULL,
                NULL);

            clFinish(command_queue);

            // Copy the output matrices back to the CPU memory
            err = clEnqueueReadBuffer(command_queue, buf_pheromones, CL_TRUE, 0, num_cities * num_cities * sizeof(double), pheromones, 0, NULL, NULL);
            if (err != CL_SUCCESS)
            {
                printf("%d\n", err);
                return 0;
            }

            clFinish(command_queue);

            err = clEnqueueReadBuffer(command_queue, buf_ant_tours, CL_TRUE, 0, num_ants * num_cities * sizeof(int), ant_tours, 0, NULL, NULL);
            if (err != CL_SUCCESS)
            {
                printf("%d\n", err);
                return 0;
            }

            clFinish(command_queue);

            err = clEnqueueReadBuffer(command_queue, buf_ant_lengths, CL_TRUE, 0, num_ants * sizeof(double), ant_lengths, 0, NULL, NULL);
            if (err != CL_SUCCESS)
            {
                printf("%d\n", err);
                return 0;
            }

            clFinish(command_queue);

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

        // Release the mem objects
        clReleaseMemObject(buf_pheromones);
        clReleaseMemObject(buf_ant_tours);
        clReleaseMemObject(buf_ant_lengths);
        clReleaseMemObject(buf_ant_randoms);
        clReleaseMemObject(buf_visited_cities);
    }

    fclose(file);

    // Release the distance matrix mem object
    clReleaseMemObject(buf_city_distances);

    // Release the resources
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseContext(context);
    clReleaseDevice(device_id);

    return 0;
}
