

//Differences between __device__ and __global__ functions are:
//
//__device__ functions can be called only from the device, and it is executed only in the device.
//
//__global__ functions can be called from the host, and it is executed in the device.
// In CUDA function type qualifiers __device__ and __host__ can be used together in which case
// the function is compiled for both the host and the device.
// This allows to eliminate copy-paste

#include "kernels.h"

#include <math.h>
#include <stdio.h>

// L2 : ||x-y||_2
__device__ __host__ float  L2(float *x,float *y,int d){
    float sum=0.0;
    for(int i=0;i<d;++i)
        sum+=(x[i]-y[i])*(x[i]-y[i]);
    return sum;
}

// squared exponentional kernel A.K.A. the Radial Basis Function kernel
__device__ __host__ float sq_e_k(float *x,float *y,int d,float lsc,float sig,float a){
    // lsc: lengthscale ; sig : sigma
    float r=L2(x,y,d);
    return sig *expf(-r/(lsc*lsc*2));
}

//exponentional kernel
__device__ __host__ float e_k(float *x,float *y,int d,float lsc,float sig,float a){
    // lsc: lengthscale ; sig : sigma
    float r=L2(x,y,d);
    return sig*expf(-sqrtf(r)/lsc);
}

// rational quadratic kernel (infinite sum of RBF kernels)
__device__ __host__ float rqk(float *x,float *y,int d,float lsc,float sig,float a){
    // lsc: lengthscale ; sig : sigma
    float r=L2(x,y,d);
    return sig*powf(1+r/(2*a*lsc*lsc),-a);
}

// decide which kernel to use
__device__ __host__ kernelfunc get_kernel(kernelstring_enum kernel_id){
    switch(kernel_id){
        case cudagpSquaredExponentialKernel:
            return sq_e_k;
        case cudagpExponentialKernel:
            return e_k;
        case cudagpRationalQuadraticKernel:
            return rqk;
        default:
            printf("Invalid Kernel, default is squared exponential kernel");
            return sq_e_k;
    }
}

