#include "common.h"


// Verifies the 2D convolution result on the CPU
// Takes:
//  m:      Original matrix
//  mask:   Convolutional mask
//  result: Result from the GPU
//  N:      Dimensions of the matrix
void convolutionCPU(int *m, int *mask, int N, int *cpuOutput, int MASK_DIM, int MASK_OFFSET)
{
	// Temp value for accumulating results
	int temp;

	// Intermediate value for more readable code
	int offset_r;
	int offset_c;

	// Go over each row
	for (int i = 0; i < N; i++) {
		// Go over each column
		for (int j = 0; j < N; j++) {
			// Reset the temp variable
			temp = 0;

			// Go over each mask row
			for (int k = 0; k < MASK_DIM; k++) {
				// Update offset value for row
				offset_r = i - MASK_OFFSET + k;

				// Go over each mask column
				for (int l = 0; l < MASK_DIM; l++) {
					// Update offset value for column
					offset_c = j - MASK_OFFSET + l;

					// Range checks if we are hanging off the matrix
					if (offset_r >= 0 && offset_r < N) {
						if (offset_c >= 0 && offset_c < N) {
							// Accumulate partial results
							temp += m[offset_r * N + offset_c] * mask[k * MASK_DIM + l];
						}
					}
				}
			}
			cpuOutput[i*N + j] = temp;
		}
	}
}
