//
// Created by ali on 07.05.19.
//

#ifndef GP_GPU_LINALOG_H
#define GP_GPU_LINALOG_H

void power_gpu(float *d_a,int n, float pow);

void matmul_gpu(float *d_a,float *d_b,int n);

float *diag_gpu(float *d_a,int n);

void diagAdd_gpu(float *d_a,int n,float alpha);

#endif //GP_GPU_LINALOG_H
