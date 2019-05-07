//
// Created by ali on 07.05.19.
//

#ifndef GP_GPU_KERNELS_H
#define GP_GPU_KERNELS_H

#include <cublas_v2.h>
#include <cusolverDn.h>

typedef float (*kernelfunc)(float*, float*, int, float, float,float);
typedef enum {
    cudagpSquaredExponentialKernel,
    cudagpExponentialKernel,
    cudagpRationalQuadraticKernel
} kernelstring_enum;

__device__ __host__ float sq_e_k(float *x,float *y,int d,float lsc,float sig);
__device__ __host__ float e_k(float *x,float *y,int d,float lsc,float sig);
__device__ __host__ float rqk(float *x,float *y,int d,float lsc,float a,float sig);
__device__ __host__ kernelfunc get_kernel(kernelstring_enum kernel_id);

#endif //GP_GPU_KERNELS_H
