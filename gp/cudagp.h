//
// Created by ali on 09.05.19.
//

#ifndef GP_GPU_CUDAGP_H
#define GP_GPU_CUDAGP_H

#include <stdio.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <cstdlib>

struct dataset{
    float *X,*Y;
    int n,d;
};

typedef enum {
    cudagpSquaredExponentialKernel,
    cudagpExponentialKernel,
    cudagpRationalQuadraticKernel
} kernelstring_enum;

struct parameters{
    kernelstring_enum kernel;
    float **values;
    int numparam;
};

struct cudagphandle_t{
    int num;
    dataset *d_dataset;
    parameters d_parameters;
    cusolverDnHandle_t cusolver;
    cublasHandle_t cublas;
};

struct prediction{
    float *mean,*var;
    int t;
};

static int numParams(kernelstring_enum kernel) {
    switch(kernel) {
        case cudagpSquaredExponentialKernel:
            return 2;
        case cudagpExponentialKernel:
            return 1;
        case cudagpRationalQuadraticKernel:
            return 2;
        default:
            fprintf(stderr, "ERROR: Invalid kernel method.");
            exit(EXIT_FAILURE);
            return -1;
    }
}


#endif //GP_GPU_CUDAGP_H
