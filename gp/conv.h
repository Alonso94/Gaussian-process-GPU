//
// Created by ali on 07.05.19.
//

#ifndef GP_GPU_CONV_H
#define GP_GPU_CONV_H

#include <cusolverDn.h>
#include "kernels.h"

float* comp_cov_gpu(float *d_x, int n , int d, kernelstring_enum kernel, float lsc, float sig, float a);
float* comp_cross_cov_gpu(float *d_x,int n ,int d,float *d_xtest,int t , kernelstring_enum kernel,float lsc,float sig,float a);
float* conditionalMean(float *d_y,int n, float *d_cov, float *d_xtest,int t, float *d_covy,cusolverDnHandle_t cusolver,cublasHandle_t cublash);
float* conditionalCov(int n, int d,float *d_cov, float *d_xtest,int t, float *d_covy,kernelstring_enum kern,float lsc,float sig,float a,cusolverDnHandle_t cusolver,cublasHandle_t cublash);
void chol(float d_cov,int n,cusolverDnHandle_t cusolver);
float* eye(int n);


#endif //GP_GPU_CONV_H
