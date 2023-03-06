#include <cstdio>
#include <iomanip>
#include <iostream>
#include "assert.h"
#include <time.h>
#include <driver_types.h>
#include <device_launch_parameters.h>
#include <crt/device_functions.h>
using namespace std;
__global__ void piCalc(int* r, float* x, float* y, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
        r[i] = pow(x[i], 2) + pow(y[i], 2) > 1 ? 0 : 1;
}
__global__ void reduce(int* inData, int* outData)
{
    __shared__ int data[1024];
    int tid = threadIdx.x;
    int i = 2 * blockIdx.x * blockDim.x + threadIdx.x;

    data[tid] = inData[i] + inData[i + blockDim.x];      // Загружаем в общую память

    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
            data[tid] += data[tid + s];

        __syncthreads();
    }

    if (tid == 0)                                       // write result of block reduction
        outData[blockIdx.x] = data[0];
}


long clp2(long x) {
    long p2 = 1;
    while (1) {
        if (p2 >= x)
            return p2;
        p2 <<= 1;
    }
    return 0;
}
const int BLOCK_SIZE = 1024;
const int N = 1024;
int main()
{
    int n = clp2(N);

    float* x = (float*)calloc(N * sizeof(float), sizeof(float));
    float* y = (float*)calloc(N * sizeof(float), sizeof(float));
    int* r_CPU = (int*)calloc(N * sizeof(int), sizeof(int));
    int* r_GPU = (int*)calloc(n * sizeof(int), sizeof(int));
    int* sum_GPU = (int*)calloc(n * sizeof(int), sizeof(int));
    // заполнение данных
    for (int i = 0; i < N; i++)
    {
        x[i] = (float)(rand() % 1000000) / 1000000;
        y[i] = (float)(rand() % 1000000) / 1000000;
        sum_GPU[i] = 0;
    }
    // подсчет на CPU
    double cpuTime;

    clock_t start_cpu, end_cpu;
    start_cpu = clock();

    for (int i = 0; i < N; i++)
    {
        r_CPU[i] = pow(x[i], 2) + pow(y[i], 2) > 1 ? 0 : 1;
    }
    int sum = 0;
    for (int i = 0; i < N; i++)
    {
        if (r_CPU[i] == 1)
            sum++;
    }
    end_cpu = clock();
    cpuTime = (1.0f * (end_cpu - start_cpu) / CLOCKS_PER_SEC) * 1000;
    printf("Time spent executing cpu: %lf ms\n", cpuTime);
    //добьем заранее нулями то, где не будет данных
    //в элементы до N мы че нибудь положив в функции ядра
    for (int i = N; i < n; i++) {
        r_GPU[i] = 0;
    }


    cudaEvent_t start, stop;
    float gpuTime = 0.0f;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);


    //выделяем память для GPU
    float* xdev = NULL;
    cudaError_t cuerr = cudaMalloc((void**)&xdev, N * sizeof(float));
    if (cuerr != cudaSuccess) {
        fprintf(stderr, "Cannot allocate device array for x: %s\n",
            cudaGetErrorString(cuerr));
        return 0;
    }
    float* ydev = NULL;
    cuerr = cudaMalloc((void**)&ydev, N * sizeof(float));
    if (cuerr != cudaSuccess) {
        fprintf(stderr, "Cannot allocate device array for y: %s\n",
            cudaGetErrorString(cuerr));
        return 0;
    }
    // туда будем писать 0 или 1
    int* rdev = NULL;
    cuerr = cudaMalloc((void**)&rdev, n * sizeof(int));
    if (cuerr != cudaSuccess) {
        fprintf(stderr, "Cannot allocate device array for r: %s\n",
            cudaGetErrorString(cuerr));
        return 0;
    }
    // вспомогательный массив для суммирования
    int* sum_gpu = NULL;
    cuerr = cudaMalloc((void**)&sum_gpu, n * sizeof(int));
    if (cuerr != cudaSuccess) {
        fprintf(stderr, "Cannot allocate device array for r: %s\n",
            cudaGetErrorString(cuerr));
        return 0;
    }



    cuerr = cudaMemcpy(xdev, x, N * sizeof(float), cudaMemcpyHostToDevice);
    if (cuerr != cudaSuccess) {
        fprintf(stderr, "Cannot copy a array from host to device: %s\n",
            cudaGetErrorString(cuerr));
        return 0;
    }
    cuerr = cudaMemcpy(ydev, y, N * sizeof(float), cudaMemcpyHostToDevice);
    if (cuerr != cudaSuccess) {
        fprintf(stderr, "Cannot copy b array from host to device: %s\n",
            cudaGetErrorString(cuerr));
        return 0;
    }
    cuerr = cudaMemcpy(rdev, r_GPU, n * sizeof(float), cudaMemcpyHostToDevice);
    if (cuerr != cudaSuccess) {
        fprintf(stderr, "Cannot copy b array from host to device: %s\n",
            cudaGetErrorString(cuerr));
        return 0;
    }


    dim3 dimBlock(N > BLOCK_SIZE ? BLOCK_SIZE : N, 1, 1);
    dim3 dimGrid(N / BLOCK_SIZE + 1, 1, 1);
    piCalc << <dimGrid, dimBlock >> > (rdev, xdev, ydev, N);

    int grid_size = n / (2 * BLOCK_SIZE);

    if (grid_size == 0) {
        grid_size = 1;
    }

    dim3 dimBlockSum(BLOCK_SIZE, 1, 1);
    dim3 dimGridSum(grid_size, 1, 1);

    reduce << <dimGridSum, dimBlockSum >> > (rdev, sum_gpu);

    cudaDeviceSynchronize();
    cudaMemcpy(sum_GPU, sum_gpu, N * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 1; i < grid_size; i++) {
        sum_GPU[0] += sum_GPU[i];
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpuTime, start, stop);
    printf("Time spent executing by the GPU: %.4f ms\n", gpuTime);
    printf("SpeedUp: %.4f \n", cpuTime / gpuTime);
    printf("pi CPU: %.6lf \n", (double)sum * 4 / N);
    printf("pi GPU: %.6lf \n", (double)sum_GPU[0] * 4 / N);
    return 0;
}
