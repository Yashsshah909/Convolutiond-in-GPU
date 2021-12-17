#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <iostream>
#include "common.h"
const int BLOCK_SIZE = 32;

__global__ void ConvolutionSharedMem(int* A, int* B, int* C, int n, int kernelDim)
{
	int col = blockIdx.x * (BLOCK_SIZE - kernelDim + 1) + threadIdx.x;
	int row = blockIdx.y * (BLOCK_SIZE - kernelDim + 1) + threadIdx.y;
	int row_i = row - kernelDim + 1;
	int col_i = col - kernelDim + 1;

	float tmp = 0;

	__shared__ float shm[BLOCK_SIZE][BLOCK_SIZE];

	if (row_i < n && row_i >= 0 && col_i < n && col_i >= 0)
	{
		//printf("\n A[col_i * n + row_i] : %d", A[col_i * n + row_i]);
		shm[threadIdx.y][threadIdx.x] = A[col_i * n + row_i];
	}
	else
	{
		shm[threadIdx.y][threadIdx.x] = 0;
	}

	__syncthreads();

	if (threadIdx.y < (BLOCK_SIZE - kernelDim + 1) && threadIdx.x < (BLOCK_SIZE - kernelDim + 1) && row < (n - kernelDim + 1) && col < (n - kernelDim + 1))
	{
		for (int i = 0; i < kernelDim; i++)
			for (int j = 0; j < kernelDim; j++)
				tmp += shm[threadIdx.y + i][threadIdx.x + j] * C[j*kernelDim + i];
		B[col*n + row] = tmp;
		
	}
}



bool convolutionGPUShared(int* matrix, int* result, int* kernel , int size, int kernelSize)
{

	cudaError_t status;

	size_t bytes_m = kernelSize * kernelSize * sizeof(int);
	size_t bytes_n = size * size * sizeof(int);

	int* d_A;
	int* d_B;
	int* d_C;

	status = cudaMalloc((void**)&d_A, bytes_n);
	if (status != cudaSuccess)
	{
		printf("Error: %s  in cudaMalloc for A\n", cudaGetErrorString(status));
		return EXIT_FAILURE;
	}

	status = cudaMalloc((void**)&d_B, bytes_n);
	if (status != cudaSuccess)
	{
		printf("Error: %s  in cudaMalloc for B\n", cudaGetErrorString(status));
		return EXIT_FAILURE;
	}
	status = cudaMalloc((void**)&d_C, bytes_m);
	if (status != cudaSuccess)
	{
		printf("Error: %s  in cudaMalloc for C\n", cudaGetErrorString(status));
		return EXIT_FAILURE;
	}
	


	status = cudaMemcpy(d_A, matrix, bytes_n, cudaMemcpyHostToDevice);
	if (status != cudaSuccess)
	{
		printf("Error: %s  in cudaMemcpy for A\n", cudaGetErrorString(status));
		return EXIT_FAILURE;
	}

	status = cudaMemcpy(d_C, kernel, bytes_m, cudaMemcpyHostToDevice);
	if (status != cudaSuccess)
	{
		printf("Error:  %s  in cudaMemcpy for C\n", cudaGetErrorString(status));
		return EXIT_FAILURE;
	}
	// Calculate grid dimensions
	int blockSize = BLOCK_SIZE;
	int gridSize = (size + blockSize - 1) / blockSize;

	// Dimension launch arguments
	dim3 block_dim(blockSize, blockSize);
	dim3 grid_dim(gridSize, gridSize);

	ConvolutionSharedMem << < grid_dim, block_dim >> > (d_A, d_B, d_C, size, kernelSize);
	status = cudaGetLastError();
	if (status != cudaSuccess)
	{
		printf( "Kernel Error %s  in launching kernel\n", cudaGetErrorString(status));
		cudaFree(d_A);
		cudaFree(d_B);
		cudaFree(d_C);
		return false;
	}

	status = cudaMemcpy(result, d_B, bytes_n, cudaMemcpyDeviceToHost);
	if (status != cudaSuccess)
	{
		printf("Memcpy error %s \n", cudaGetErrorString(status));
		cudaFree(d_A);
		cudaFree(d_B);
		cudaFree(d_C);
		return false;
	}

	// Copy the result back to the CPU

	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);

	return true;
}
