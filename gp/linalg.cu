#include <curand_mtgp32_kernel.h>
#include <device_launch_parameters.h>
#include "utils_cuda.h"

#define BlockSize 256

__global__ void power_cu(float * d_a,int n,float pow){
    // Element-wise power
    int i = threadIdx.x+blockIdx.x+blockDim.x;
    if(i<n)
        d_a[i]=powf(d_a[i],pow);
}

void power_gpu(float *d_a,int n, float pow){
    // call the cuda implementation
    dim3 blocksize(BlockSize);
    dim3 gridsize=div_ceil(n,BlockSize);
    power_cu<<<gridsize,blocksize>>>(d_a,n,pow);
}

__global__ void matmul_cu(float *d_a,float *d_b,int n){
    // Element-wise multiplication
    int i =threadIdx.x+blockIdx.x*blockDim.x;
    if(i<n)
        d_a[i]=d_a[i]*d_b[i];
}

void matmul_gpu(float *d_a,float *d_b,int n){
    // call the cuda implementation
    dim3 blocksize(BlockSize);
    dim3 gridsize=div_ceil(n,BlockSize);
    matmul_cu<<<gridsize,blocksize>>>(d_a,d_b,n);
}

__global__ void diag_cu(float* d_a,int n,float *d_diag){
    int i=threadIdx.x+blockIdx.x*blockDim.x;
    if(i<n)
        d_diag[i]=d_a[i*n+i];
}

float* diag_gpu(float *d_a,int n){
    float *d_diag;
    checkCudaErrors(cudaMalloc((void **)&d_diag,n*sizeof(float)));
    dim3 blocksize(BlockSize);
    dim3 gridsize=div_ceil(n,BlockSize);
    diag_cu<<<gridsize,blocksize>>>(d_a,n,d_diag);
    return d_diag;
}

__global__ void diagAdd_cu(float *d_a,int n,float alpha){
    // add a number (alpha) to all the diagonal elements
    int i=threadIdx.x+blockIdx.x*blockDim.x;
    if(i<n)
        d_a[i*n+i]+=alpha;
}

void diagAdd_gpu(float *d_a,int n,float alpha){
    dim3 blocksize(BlockSize);
    dim3 gridsize=div_ceil(n,BlockSize);
    diagAdd_cu<<<gridsize,blocksize>>>(d_a,n,alpha);
}



