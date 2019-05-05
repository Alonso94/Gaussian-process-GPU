//
// Created by Palash on 19-02-2018.
//
#include <stdio.h>
#include <ctime>
#include <malloc.h>
#include "cudaMain.h"
#include "cudaHeaders.h"

using namespace std;

//define thread to block number
#define BLOCK_SIZE 64

double rnd(double fmin=0,double fmax=10){
    double f = (double)rand() / RAND_MAX;
    return fmin + f * (fmax - fmin);
}

void cpu_matmul(double *a,double *b,double *c,int m,int n,int k){
    //size a : m*k
    //size b : k*n
    //size c : m*n
    for(int i=0;i<m;i++)
        for(int j=0;j<n;j++){
            double tmp=0.0;
            for(int w=0;w<k;w++)
                tmp+=a[i*k+w]*b[w*n+j];
            c[i*n+j]=tmp;
        }
}

__global__ void gpu_matmul(double *a,double *b,double *c,int m,int n,int k){
    int i=blockIdx.y * blockDim.y + threadIdx.y;
    int j=blockIdx.x * blockDim.x + threadIdx.x;
    double sum=0;
    if (i<m && j<n){
        for (int w=0;w<k;w++)
            sum+=a[i*k+w]*b[w*n+j];
        c[i*n+j]=sum;
    }
}

__global__ void transpose(double *in,double *out,int m, int n){
    int i=blockIdx.y * blockDim.y + threadIdx.y;
    int j=blockIdx.x * blockDim.x + threadIdx.x;
    int ind,new_ind;
    if(j <n && i <m){
        ind= j*n+i;
        new_ind= i*m + j;
        out[new_ind]=in[ind];
    }
}

int cudaMain(int argc, char **argv) {
    int m=1024,n=1024,k=1020;
    dim3 dimGrid((n + BLOCK_SIZE - 1) / BLOCK_SIZE, (m + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    cout<<"begin"<<endl;
    srand(1024);
    double *a,*b,*c,*cc;
    //memory allocation in the host (CPU)
    cudaMallocHost((void**)&a,sizeof(double)*m*k);
    cudaMallocHost((void**)&b,sizeof(double)*k*n);
    cudaMallocHost((void**)&c,sizeof(double)*m*n);
    cudaMallocHost((void**)&cc,sizeof(double)*m*n);
    //generate random arrays
    for(int i=0;i<m;i++)
        for(int j=0;j<k;++j)
            a[i * k + j] = rnd();
    for(int i=0;i<k;i++)
        for(int j=0;j<n;++j)
            a[i*n+j]=rnd();

    cout<<"memory allocation done"<<endl;

    //to record the execution time
    float t_gpu,t_cpu;
    cudaEvent_t start,stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);


    ////////////////////////////////////
    // GPU test
    ////////////////////////////////////
    // start recording the time
    cudaEventRecord(start,0);
    //memory allocation on device (GPU)
    double *cu_a,*cu_b,*cu_c;
    cudaMalloc((void**)&cu_a, sizeof(double)*m*k);
    cudaMalloc((void**)&cu_b, sizeof(double)*k*n);
    cudaMalloc((void**)&cu_c, sizeof(double)*m*n);
    //copy matrix from host (CPU) to device (GPU)
    cudaMemcpy(cu_a,a, sizeof(double)*m*k, cudaMemcpyHostToDevice);
    cudaMemcpy(cu_b,b,sizeof(double)*k*n, cudaMemcpyHostToDevice);
    // call the gpu function
    gpu_matmul<<<dimGrid,dimBlock>>>(cu_a,cu_b,cu_c,m,n,k);
    //copy the result
    cudaMemcpy(c,cu_a, sizeof(double)*m*n,cudaMemcpyDeviceToHost);
    cudaThreadSynchronize();
    //stop the timer
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    //calculate the time
    cudaEventElapsedTime(&t_gpu, start, stop);
    printf("t_gpu = %5f ms\n",t_gpu);

    ////////////////////////////////////
    // CPU test
    ////////////////////////////////////
    // start recording the time
    cudaEventRecord(start,0);
    //call the dpu function
    cpu_matmul(a,b,cc,m,n,k);
    //stop the timer
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    //calculate the time
    cudaEventElapsedTime(&t_cpu, start, stop);
    printf("t_cpu = %5f ms \n",t_cpu);

    //////////////////////////////////////
    // compare the results
    /////////////////////////////////////
    double diff=0.0;
    for(int i=0;i<m;++i)
        for (int j=0;j<n;++j){
                diff+=c[i*n+j]!=cc[i*n+j];
        }
    printf("Sum of differences between elements of result matrix = %f",diff);

    cudaFree(a);
    cudaFree(b);
    cudaFree(c);
    cudaFree(cc);
    cudaFree(cu_a);
    cudaFree(cu_b);
    cudaFree(cu_c);

    return 0;
}
