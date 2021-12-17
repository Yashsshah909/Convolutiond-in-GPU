#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <iostream>
#include "common.h"
__constant__ int mask[7 * 7];

__global__ void convolution_2d(int *matrix, int *result, int N, int MASK_DIM, int MASK_OFFSET) {
	// Calculate the global thread positions
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	// Starting index for calculation
	int start_r = row - MASK_OFFSET;
	int start_c = col - MASK_OFFSET;

	// Temp value for accumulating the result
	int temp = 0;

	// Iterate over all the rows
	for (int i = 0; i < MASK_DIM; i++) {
		// Go over each column
		for (int j = 0; j < MASK_DIM; j++) {
			// Range check for rows
			if ((start_r + i) >= 0 && (start_r + i) < N) {
				// Range check for columns
				if ((start_c + j) >= 0 && (start_c + j) < N) {
					// Accumulate result
					temp += matrix[(start_r + i) * N + (start_c + j)] *		mask[i * MASK_DIM + j];
				}
			}
		}
	}

	// Write back the result
	result[row * N + col] = temp;
}


bool convolutionGPU(int *matrix, int *h_mask, int *result, int size, int kernelSize, int MASK_OFFSET)
{
	// Error return value
	cudaError_t status;
	// Number of bytes in the matrix.
		// Size of the mask in bytes
	size_t bytes_m = kernelSize * kernelSize * sizeof(int);
	size_t bytes_n = size * size * sizeof(int);

	// Allocate device memory
	int *d_matrix;
	int *d_result;
	cudaMalloc(&d_matrix, bytes_n);
	status = cudaGetLastError();
	if (status != cudaSuccess) {
		std::cout << "Memory Allocation Failed " << cudaGetErrorString(status) << std::endl;
		return false;
	}
	cudaMalloc(&d_result, bytes_n);
	if (status != cudaSuccess) {
		std::cout << "Memory Allocation Failed " << cudaGetErrorString(status) << std::endl;
		return false;
	}
	// Copy data to the device
	cudaMemcpy(d_matrix, matrix, bytes_n, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(mask, h_mask, bytes_m);

	// Calculate grid dimensions
	int THREADS = 32;
	int BLOCKS = (size + THREADS - 1) / THREADS;

	// Dimension launch arguments
	dim3 block_dim(THREADS, THREADS);
	dim3 grid_dim(BLOCKS, BLOCKS);

	// Perform 2D Convolution
	convolution_2d << <grid_dim, block_dim >> > (d_matrix, d_result, size, kernelSize,  MASK_OFFSET);
	// Check for errors
	status = cudaGetLastError();
	if (status != cudaSuccess) {
		std::cout << "Kernel failed: " << cudaGetErrorString(status) << std::endl;
		cudaFree(d_matrix);
		cudaFree(d_result);
		return false;
	}
	// Copy the result back to the CPU
	cudaMemcpy(result, d_result, bytes_n, cudaMemcpyDeviceToHost);
	cudaFree(d_matrix);
	cudaFree(d_result);
	return true;
}
