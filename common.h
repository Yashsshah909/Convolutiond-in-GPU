#ifndef COMMON_H
#define COMMON_H
// CPU Implementation
void convolutionCPU(int *, int *,  int, int *, int , int);


// CUDA Implementation (returns false on failure)
bool convolutionGPU(int *, int *, int *, int N,int ,int);

bool convolutionGPUShared(int *, int *,int* ,int,int);
#endif