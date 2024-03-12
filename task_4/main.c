#include "kernel_loader.h"

#define CL_TARGET_OPENCL_VERSION 220
#include <CL/cl.h>

#include <stdio.h>
#include <stdlib.h>

void print_matrix(int lines, int columns, float *matrix);

// Threadblock sizes (e.g. for kernels myGEMM1 or myGEMM2)
#define TS 32

int main(void)
{
    int i;
    cl_int err;
    int error_code;

    // Set the sizes
    int K = 5;
    int M = 5;
    int N = 5;

    // Create the matrices and initialize them with random values
    float *A = (float *)malloc(M * K * sizeof(float *));
    float *B = (float *)malloc(K * N * sizeof(float *));
    float *C = (float *)malloc(M * N * sizeof(float *));
    for (int i = 0; i < M * K; i++)
    {
        A[i] = 3.9 * i + i * i + 3.7;
    }
    for (int i = 0; i < K * N; i++)
    {
        B[i] = 1.1 * i + 0.51 * i * i + 16.9;
    }
    for (int i = 0; i < M * N; i++)
    {
        C[i] = 0.0;
    }

    // Print matrices
    printf("\nA matrix: %f", A[1*M+1]);
    print_matrix(M, K, A);
    print_matrix(K, N, B);
    print_matrix(M, N, C);

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
    const char *kernel_code = load_kernel_source("kernels/sample.cl", &error_code);
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

    // Prepare OpenCL memory objects
    cl_mem bufA = clCreateBuffer(context, CL_MEM_READ_ONLY, M * K * sizeof(float), NULL, NULL);
    cl_mem bufB = clCreateBuffer(context, CL_MEM_READ_ONLY, K * N * sizeof(float), NULL, NULL);
    cl_mem bufC = clCreateBuffer(context, CL_MEM_READ_WRITE, M * N * sizeof(float), NULL, NULL);

    // Copy matrices to the GPU
    clEnqueueWriteBuffer(command_queue, bufA, CL_TRUE, 0, M * K * sizeof(float), A, 0, NULL, NULL);
    clEnqueueWriteBuffer(command_queue, bufB, CL_TRUE, 0, K * N * sizeof(float), B, 0, NULL, NULL);
    clEnqueueWriteBuffer(command_queue, bufC, CL_TRUE, 0, M * N * sizeof(float), C, 0, NULL, NULL);

    printf("\ncreating Kernel...\n");
    cl_kernel kernel = clCreateKernel(program, "multiply_matrices", &err);
    if (err != CL_SUCCESS)
    {
        printf("%d\n", err);
        return 0;
    }
    printf("\nKernel created\n");

    // Set kernel arguments
    err = clSetKernelArg(kernel, 0, sizeof(int), (void *)&M);
    if (err != CL_SUCCESS)
    {
        printf("%d\n", err);
        return 0;
    }

    err = clSetKernelArg(kernel, 1, sizeof(int), (void *)&N);
    if (err != CL_SUCCESS)
    {
        printf("%d\n", err);
        return 0;
    }

    err = clSetKernelArg(kernel, 2, sizeof(int), (void *)&K);
    if (err != CL_SUCCESS)
    {
        printf("%d\n", err);
        return 0;
    }

    err = clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *)&bufA);
    if (err != CL_SUCCESS)
    {
        printf("%d\n", err);
        return 0;
    }

    err = clSetKernelArg(kernel, 4, sizeof(cl_mem), (void *)&bufB);
    if (err != CL_SUCCESS)
    {
        printf("%d\n", err);
        return 0;
    }

    err = clSetKernelArg(kernel, 5, sizeof(cl_mem), (void *)&bufC);
    if (err != CL_SUCCESS)
    {
        printf("%d\n", err);
        return 0;
    }

    printf("\narguments set\n");

    // Size specification
    const size_t local[2] = {TS, TS};
    const size_t global[2] = {M, N};

    // Apply the kernel on the range
    cl_event event = NULL;
    clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, global, local, 0, NULL, &event);

    // Wait for calculations to be finished
    clWaitForEvents(1, &event);

    // Copy the output matrix C back to the CPU memory
    err = clEnqueueReadBuffer(command_queue, bufC, CL_TRUE, 0, M * N * sizeof(float), C, 0, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        printf("%d\n", err);
        return 0;
    }

    clFinish(command_queue);

    // Print the result, matrix C
    print_matrix(M, N, C);

    // Free the OpenCL memory objects
    clReleaseMemObject(bufA);
    clReleaseMemObject(bufB);
    clReleaseMemObject(bufC);

    // Release the resources
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseContext(context);
    clReleaseDevice(device_id);

    // Free the host memory objects
    free(A);
    free(B);
    free(C);
}

void print_matrix(int lines, int columns, float *matrix)
{
    int row, col;
    printf("\n/////////////////\n");
    for (row = 0; row < lines; row++)
    {
        for (col = 0; col < columns; col++)
        {
            printf("%f     ", matrix[row * columns + col]);
        }
        printf("\n");
    }
    printf("\n/////////////////\n");
}
