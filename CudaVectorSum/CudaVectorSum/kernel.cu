#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <chrono>
#include <iostream>

#define SIZE 256
#define SHMEM_SIZE 256 * 4

using namespace std;

void verify_result(int* h_v, int N, int GpuRes) {
	chrono::system_clock::time_point startCpu = chrono::system_clock::now();
	for (int i = 1; i < N; i++) {
		h_v[0] += h_v[i];
	}
	chrono::system_clock::time_point endCpu = chrono::system_clock::now();
	auto timeCpu = chrono::duration_cast<chrono::nanoseconds>(endCpu - startCpu).count();
	cout << "CPU TIME : " << timeCpu << "ns\n";
	printf("CPU res is %d \n", h_v[0]);
}

__global__ void sum_reduction(int* v, int* v_r) {
	// разделяем память 
	__shared__ int partial_sum[SHMEM_SIZE];

	// Находим ИД потока
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	// Загружаем в память элементы
	partial_sum[threadIdx.x] = v[tid];


	//Делим блоки на 2, 
	for (int s = blockDim.x / 2; s > 0; s >>= 1) {
		if (threadIdx.x < s) {
			partial_sum[threadIdx.x] += partial_sum[threadIdx.x + s];
		}

	}



	if (threadIdx.x == 0) {
		v_r[blockIdx.x] = partial_sum[0];
	}
}

void initialize_vector(int* v, int n) {
	for (int i = 0; i < n; i++) {
		v[i] = rand() % 10;
	}
}

int main() {
	// Размер вектора
	int n = 1 << 8;
	size_t bytes = n * sizeof(int);

	int* h_v, * h_v_r;
	int* d_v, * d_v_r;

	// Выделяем память

	h_v = (int*)malloc(bytes);
	h_v_r = (int*)malloc(bytes);
	cudaMalloc(&d_v, bytes);
	cudaMalloc(&d_v_r, bytes);

	initialize_vector(h_v, n);

	// Копируем с CPU на GPU
	cudaMemcpy(d_v, h_v, bytes, cudaMemcpyHostToDevice);


	int TB_SIZE = SIZE;

	int GRID_SIZE = n / TB_SIZE;

	chrono::system_clock::time_point start = chrono::system_clock::now();

	// Вызываем ядро
	sum_reduction << <GRID_SIZE, TB_SIZE >> > (d_v, d_v_r);

	sum_reduction << <1, TB_SIZE >> > (d_v_r, d_v_r);

	chrono::system_clock::time_point end = chrono::system_clock::now();
	auto time = chrono::duration_cast<chrono::nanoseconds>(end - start).count();

	// Копируем с GPU на CPU
	cudaMemcpy(h_v_r, d_v_r, bytes, cudaMemcpyDeviceToHost);
	printf("%d",h_v_r[0]);
	
	cout << "GPU TIME : " << time << "ns\n";
	printf("GPU result is %d \n", h_v_r[0]);
	verify_result(h_v, n, h_v_r[0]);

	return 0;
}