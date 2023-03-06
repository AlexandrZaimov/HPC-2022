#include "/studies/hpc/EasyBMP.h"
#include "/studies/hpc/EasyBMP_BMP.h"
#include "/studies/hpc/EasyBMP_DataStructures.h"
#include "/studies/hpc/EasyBMP_VariousBMPutilities.h"
#include "/studies/hpc/EasyBMP.cpp"
#include <iostream>
#include <vector>
#include <algorithm> 
#include <string>
#include <iomanip>
#include <stdio.h>
#include <chrono>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void addKernel(float* c, float* a, float* b, unsigned int m, unsigned int n, unsigned int k)
{
    //int count_threads = gridDim.x * blockDim.x;
    int Row = blockIdx.x * blockDim.x + threadIdx.x;
    int Col = blockIdx.y * blockDim.y + threadIdx.y;
    if ((Row < m) && (Col < k))
    {
        for (int i = 0; i < n; i++)
        {
            c[Row * k + Col] += a[Row * n + i] * b[i * k + Col];
        }
    }
}
int main()
{
    
    int m = 512;
    int n = 512;
    int k = 512;

    printf("m = %d\n", m);
    printf("n = %d\n", n);
    printf("k = %d\n", k);

    // Выделение памяти на CPU
    float* a = (float*)calloc(m * n * sizeof(int), sizeof(float));
    float* b = (float*)calloc(n * k * sizeof(int), sizeof(float));
    float* c = (float*)calloc(m * k * sizeof(int), sizeof(float));
    float* c_linear = (float*)calloc(m * k * sizeof(int), sizeof(float));

    // Заполнение массива матрицы a,b, 2 матрицы с? GPu  CPu
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
            a[i * n + j] = rand() % 10;
    }
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < k; j++)
            b[i * k + j] = rand() % 10;
    }

    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < k; j++)
        {
            c[i * k + j] = 0;
            c_linear[i * k + j] = 0;
        }
    }
    printf("\n");


// Выделяем память на GPU
    float* adev = NULL;
    cudaError_t cuerr = cudaMalloc((void**)&adev, m * n * sizeof(float));
    if (cuerr != cudaSuccess) {
        fprintf(stderr, "Cannot allocate device array for a: %s\n",
            cudaGetErrorString(cuerr));
        return 0;
    }
    float* bdev = NULL;
    cuerr = cudaMalloc((void**)&bdev, n * k * sizeof(float));
    if (cuerr != cudaSuccess) {
        fprintf(stderr, "Cannot allocate device array for b: %s\n",
            cudaGetErrorString(cuerr));
        return 0;
    }
    float* cdev = NULL;
    cuerr = cudaMalloc((void**)&cdev, m * k * sizeof(float));
    if (cuerr != cudaSuccess) {
        fprintf(stderr, "Cannot allocate device array for c: %s\n",
            cudaGetErrorString(cuerr));
        return 0;
    }

    // Замер времени 
    cudaEvent_t start, stop;
    float gpuTime = 0.0f;
    cuerr = cudaEventCreate(&start);
    if (cuerr != cudaSuccess) {
        fprintf(stderr, "Cannot create CUDA start event: %s\n",
            cudaGetErrorString(cuerr));
        return 0;
    }
    cuerr = cudaEventCreate(&stop);
    if (cuerr != cudaSuccess) {
        fprintf(stderr, "Cannot create CUDA end event: %s\n",
            cudaGetErrorString(cuerr));
        return 0;
    }
    // Старт времени GPU
    cuerr = cudaEventRecord(start, 0);
    if (cuerr != cudaSuccess) {
        fprintf(stderr, "Cannot record CUDA event: %s\n",
            cudaGetErrorString(cuerr));
        return 0;
    }
    // Копирование с CPU на GPU
    cuerr = cudaMemcpy(adev, a, m * n * sizeof(float), cudaMemcpyHostToDevice);
    if (cuerr != cudaSuccess) {
        fprintf(stderr, "Cannot copy a array from host to device: %s\n",
            cudaGetErrorString(cuerr));
        return 0;
    }
    cuerr = cudaMemcpy(bdev, b, n * k * sizeof(float), cudaMemcpyHostToDevice);
    if (cuerr != cudaSuccess) {
        fprintf(stderr, "Cannot copy b array from host to device: %s\n",
            cudaGetErrorString(cuerr));
        return 0;
    }
    cuerr = cudaMemcpy(cdev, c, m * k * sizeof(float), cudaMemcpyHostToDevice);
    if (cuerr != cudaSuccess) {
        fprintf(stderr, "Cannot copy b array from host to device: %s\n",
            cudaGetErrorString(cuerr));
        return 0;
    }


   
    //Запуск ядра
    //GRID_SIZE, BLOCK_SIZE
    dim3 DimGrid(m / 32 + 1, k / 32 + 1, 1);
    dim3 DimBlock(32, 32, 1);
    addKernel <<< DimGrid, DimBlock >> > (cdev, adev, bdev, m, n, k);

    cuerr = cudaGetLastError();
    if (cuerr != cudaSuccess)
    {
        fprintf(stderr, "Cannot launch CUDA kernel: %s\n",
            cudaGetErrorString(cuerr));
        return 0;
    }
    // Синхронизация потоков
    cuerr = cudaDeviceSynchronize();
    if (cuerr != cudaSuccess)
    {
        fprintf(stderr, "Cannot synchronize CUDA kernel: %s\n",
            cudaGetErrorString(cuerr));
        return 0;
    }

    // Стоп времени GPU
    cuerr = cudaEventRecord(stop, 0);
    if (cuerr != cudaSuccess)
    {
        fprintf(stderr, "Cannot copy c array from device to host: %s\n",
            cudaGetErrorString(cuerr));
        return 0;
    }

    // Копируем результат c GPU на CPU
    cuerr = cudaMemcpy(c, cdev, m * k * sizeof(float), cudaMemcpyDeviceToHost);
    if (cuerr != cudaSuccess)
    {
        fprintf(stderr, "Cannot copy c array from device to host: %s\n",
            cudaGetErrorString(cuerr));
        return 0;
    }

    // Время работы
    cuerr = cudaEventElapsedTime(&gpuTime, start, stop);
   
    printf("Time gpu: %.9f ms\n", gpuTime);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(adev);
    cudaFree(bdev);
    cudaFree(cdev);


    //
    //Начало и конец времени 
    //
    clock_t start_linear = clock();
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < k; j++)
        {
            for (int q = 0; q < n; q++)
            {
                c_linear[i * k + j] += a[i * n + q] * b[q * k + j];
            }
        }
    }
    clock_t end_linear = clock();
    double linear_time = ((double)(end_linear - start_linear) / CLOCKS_PER_SEC) * 1000;
    printf("Time linear: %.9f ms\n", linear_time);

    float error = 0;
    for (int i = 0; i < m * k; i++) {
        c[i] -= c_linear[i];
        if (abs(c[i]) > error) error = abs(c[i]);
    }

    printf("Speedup: %.9f\n", linear_time / gpuTime);
    printf("Error: %.9f\n", error);

    free(a);
    free(b);
    free(c);
    free(c_linear);
    return 0;
}
