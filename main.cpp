#include <cstdlib> // malloc(), free()
#include <ctime> // time(), clock()
#include <cmath> // sqrt()
#include <iostream>
#include "common.h"

const int  kernelDim = 2;
const int ITERS =100;

int N = 32;


void initializeMatrix(int *m, int n) {
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			m[n * i + j] = rand() % 100;
		}
	}
}
void displayResults(int *matrix)
{
	for (int row = 0; row < N; row++) {
		for (int col = 0; col < N; col++)
			std::cout << matrix[row * N + col] << " ";
		
		std::cout << std::endl;
	}
	std::cout << std::endl;
}
int main() {
	
	float tcpu, tgpu, tgpuShared;
	clock_t start, end;
	
	while(N <= 65536) {

		std::cout << "For Size N : " << N << std::endl;
		// Size of the matrix (in bytes)
		size_t bytes_n = N * N * sizeof(int);
		int kernelOf = kernelDim / 2;
		// Allocate the matrix and initialize it
		int *matrix = new int[N * N];
		int *result = new int[N * N];
		int *resultShared = new int[N * N];
		int *cpuOutput = new int[N* N];
		initializeMatrix(matrix, N);

		// Size of the mask in bytes
		size_t bytes_m = kernelDim * kernelDim * sizeof(int);

		// Allocate the mask and initialize it
		int *kernelMatrix = new int[kernelDim * kernelDim];
		initializeMatrix(kernelMatrix, kernelDim);


		//********************CPU

		start = clock();
		for (int i = 0; i < ITERS; i++) {
			convolutionCPU(matrix, kernelMatrix, N, cpuOutput, kernelDim, kernelOf);
		}

		end = clock();
		tcpu = (float)(end - start) * 1000 / (float)CLOCKS_PER_SEC / ITERS;
		// Display the results
		std::cout << "Host Computation took " << tcpu << " ms:" << std::endl;

		//display matrix
		//std::cout << " \n CPU result \n ";
		//displayResults(cpuOutput);

		//**************** CPU ends

		//****************GPU const mem
		bool gpuStatus = convolutionGPU(matrix, kernelMatrix, result, N, kernelDim, kernelOf);
		if (!gpuStatus) {
			std::cout << "\n * Device error! * \n" << std::endl;
			return 1;
		}
		start = clock();
		for (int i = 0; i < ITERS; i++) {
			convolutionGPU(matrix, kernelMatrix, result, N, kernelDim, kernelOf);
		}
		end = clock();
		tgpu = (float)(end - start) * 1000 / (float)CLOCKS_PER_SEC / ITERS;
		// Display the results
		std::cout << "Device Computation took for Const Mem : " << tgpu << " ms:" << std::endl;

		//display matrix
		//std::cout << " \n GPU result \n ";
		//displayResults( result);

		//****************GPU ends
		float sum = 0, delta = 0;
		for (int i = 0; i < N*N; i++) {
			delta += (cpuOutput[i] - result[i]) * (cpuOutput[i] - result[i]);
			sum += (cpuOutput[i] * result[i]);
		}
		float L2norm = sqrt(delta / sum);
		//std::cout << "Relative error: " << L2norm << "\n" <<((L2norm < 1e-6) ? "Passed" : "Failed") << std::endl;


		//****************const GPU ends 


		//****************shared GPU ends 
		bool gpuStatusShared = convolutionGPUShared(matrix, resultShared, kernelMatrix, N, kernelDim);
		if (!gpuStatusShared) {
			std::cout << "\n * Device error! for shared * \n" << std::endl;
			return 1;
		}
		start = clock();
		for (int i = 0; i < ITERS; i++) {
			convolutionGPUShared(matrix, resultShared, kernelMatrix, N, kernelDim);
		}
		end = clock();
		tgpuShared = (float)(end - start) * 1000 / (float)CLOCKS_PER_SEC / ITERS;
		// Display the results
		std::cout << "Device Computation took for Shared Mem : " << tgpuShared << " ms:" << std::endl;


		//display matrix
		//std::cout << " \n GPU result  Shared Mem\n ";
		//displayResults( resultShared);



		sum = 0, delta = 0;
		for (int i = 0; i < N*N; i++) {
			delta += (cpuOutput[i] - resultShared[i]) * (cpuOutput[i] - resultShared[i]);
			sum += (cpuOutput[i] * resultShared[i]);
		}
		long double L2normShared = sqrt(delta / sum);
		//std::cout << "Relative error with CPU and Shared: " << L2normShared << "\n" <<	((L2normShared < 1e-6) ? "Passed" : "Failed") << std::endl;

		//****************shared GPU ends

		// Free the memory 
		delete[] matrix;
		delete[] result;
		delete[] kernelMatrix;

		N = N * 2;
		std::cout << " \n ********************************** \n ";
	}

	return 0;
}
