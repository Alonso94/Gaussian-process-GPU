#include <stdio.h>
#include <cusolverDn.h>
#include <cublas_v2.h>

#include "cudagp.h"
#include "conv.h"
#include "utils_cuda.h"

// intitalize sparseGP
// 1. compute the full covariance matrix.
// 2. data points are clustered.
// 3. each cluster is assigned to an expert (seperate experts)

dataset data2device(const float *h_x,const float *h_y,const int n,const int d){
    float *d_x,*d_y;
    checkCudaErrors(cudaMalloc((void**)&d_x, sizeof(float)*n*d));
    checkCudaErrors(cudaMalloc((void**)&d_y, sizeof(float)*n));

    checkCudaErrors(cudaMemcpyAsync(d_x,h_x, sizeof(float)*n*d,cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpyAsync(d_y,h_y, sizeof(float)*n,cudaMemcpyHostToDevice));

    dataset d_data;
    d_data.X=d_x;
    d_data.Y=d_y;
    d_data.d=d;
    d_data.n=n;

    return d_data;
}

parameters initDeviceParams(kernelstring_enum kernel,float *h_defaultParams,bool useDefaultPdrams){
    int np=numParams(kernel);
    float *h_params;
    if(!useDefaultPdrams){
        h_params=(float*)malloc(np*sizeof(float1));
        for(int i=0;i<np;i++)
            h_params[i]=rand()/RAND_MAX;
    } else
        h_params=h_defaultParams;
    float *d_params_val;
    checkCudaErrors(cudaMalloc((void**)&d_params_val, sizeof(float)*np));
    checkCudaErrors(cudaMemcpy(d_params_val,h_params, sizeof(float)*np,cudaMemcpyHostToDevice));

    parameters d_params;
    d_params.kernel=kernel;
    d_params.values=d_params_val;
    d_params.numparam=np;
    return d_params;
}

cublasHandle_t initCublas(){
    cublasHandle_t cublashandle;
    cublasCreate_v2(&cublashandle);
    return cublashandle;
}

cusolverDnHandle_t initCusolver(){
    cusolverDnHandle_t  cusolverhandle;
    cusolverDnCreate(&cusolverhandle);
    return cusolverhandle;
}

//split dataset into clusters
int* splitDataset(int n,int numCluster){
    int cluster_size=div_ceil(n,numCluster);
    int *start_ind=(int*)malloc(numCluster* sizeof(int));
    start_ind[0]=0;
    for(int i=0;i<numCluster,i++)
        start_ind[i]=(start_ind[i-1]+cluster_size<n) ? (start_ind[i-1]+cluster_size): (n);
    return start_ind;
}


cudagphandle_t initializeCudaGP(float *h_x,float *h_y,int n, int d, kernelstring_enum kernel,float* defaultparams){
    cudagphandle_t cudagphandle;
    cudagphandle.num=1;//num clusters
    cudagphandle.d_dataset=(dataset*)malloc(sizeof(dataset));
    cudagphandle.d_dataset[0]=data2device(h_x,h_y,n,d);
    cudagphandle.d_parameters = initDeviceParams(kernel, defaultparams, true);
    cudagphandle.cusolver=initCusolver();
    cudagphandle.cublas=initCublas();
    return cudagphandle;
}

cudagphandle_t initializeCudaDGP(float *h_x,float *h_y,int n, int d, kernelstring_enum kernel, int numClusters){
    cudagphandle_t cudagphandle;
    cudagphandle.num=numClusters;
    cudagphandle.d_dataset=(dataset*)malloc(sizeof(dataset)*numClusters);
    int *start=splitDataset(n,numClusters);
    for(int i=0;i<numClusters;i++) {
        int size = (i == numClusters - ) ? (n - start[i]) : (start[i + 1] - start[i]);
        cudagphandle.d_dataset[i] = data2device(&h_x[start[i]], &h_y[start[i]], size, d);
    }
    cudagphandle.d_parameters = initDeviceParams(kernel, 0, false);
    cudagphandle.cusolver=initCusolver();
    cudagphandle.cublas=initCublas();
    return cudagphandle;
}

cudagphandle_t initializeCudaDGP(float *h_x,float *h_y,int n, int d, kernelstring_enum kernel, int numClusters,float * defaultParams){
    cudagphandle_t cudagphandle;
    cudagphandle.num=numClusters;
    cudagphandle.d_dataset=(dataset*)malloc(sizeof(dataset)*numClusters);
    int *start=splitDataset(n,numClusters);
    for(int i=0;i<numClusters;i++) {
        int size = (i == numClusters - ) ? (n - start[i]) : (start[i + 1] - start[i]);
        cudagphandle.d_dataset[i] = data2device(&h_x[start[i]], &h_y[start[i]], size, d);
    }
    cudagphandle.d_parameters = initDeviceParams(kernel, defaultParams, true);
    cudagphandle.cusolver=initCusolver();
    cudagphandle.cublas=initCublas();
    return cudagphandle;
}

void freeCudaGP(cudagphandle_t a){
    for(int i=0;i<a.num;++i){
        checkCudaErrors(cudaFree(a.d_dataset[i].X));
        checkCudaErrors(cudaFree(a.d_dataset[i].Y));
    }
    checkCudaErrors(cudaFree(a.d_parameters.values));
    checkCusolverErrors(cusolverDnDestroy(a.cusolver));
    cublasDestroy_v2(a.cublas);
}

