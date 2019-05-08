#include <stdio.h>
#include <device_launch_parameters.h>

#include "kernels.h"
#include "utils_cuda.h"

#define Blocksize 256
#define Blocksize2D 32


// compute the full covariance matrix; K(X,X) -> outputs: d_conv
__global__ void  compute_cov(float *d_x,int n ,int d, kernelstring_enum kernel,float lsc,float sig,float a, float *d_cov){
    int i=threadIdx.x+blockIdx.x*blockDim.x;
    int j=threadIdx.y+blockIdx.y*blockDim.y;
    if(i<n && j<n){
        kernelfunc kern=get_kernel(kernel);
        // all possible combinatorial
        float *vec_x=&d_x[i*d];
        float *vec_y=&d_x[j*d];
        d_cov[i*n+j]=kern(vec_x,vec_y,d,lsc,sig,a);
    }
}

float* comp_cov_gpu(float *d_x,int n ,int d, kernelstring_enum kernel,float lsc,float sig,float a){
    // call the cuda function
    dim3 blocksize(Blocksize2D,Blocksize2D);
    dim3 gridsize=div_ceil(dim3(n,n),blocksize);
    float *d_cov;
    checkCudaErrors(cudaMalloc((void**)&d_cov, sizeof(float)*n*n));
    compute_cov<<<gridsize,blocksize>>>(d_x,n,d,kernel,lsc,sig,a,d_cov);
    return d_cov;
}

// compute rectangular covariance matrix K(X,Xtest)
// compute the full covariance matrix; K(X,X) -> outputs: d_conv
__global__ void  compute_cross_cov(float *d_x,int n ,float *d_xtest,int t ,int d, kernelstring_enum kernel,float lsc,float sig,float a, float *d_cov){
    int i=threadIdx.x+blockIdx.x*blockDim.x;
    int j=threadIdx.y+blockIdx.y*blockDim.y;
    if(i<t && j<n){
        kernelfunc kern=get_kernel(kernel);
        // all possible combinatorial
        float *vec_x=&d_xtest[i*d];
        float *vec_y=&d_x[j*d];
        d_cov[i*n+j]=kern(vec_x,vec_y,d,lsc,sig,a);
    }
}

float* comp_cross_cov_gpu(float *d_x,int n ,int d,float *d_xtest,int t , kernelstring_enum kernel,float lsc,float sig,float a) {
    // call the cuda function
    dim3 blocksize(Blocksize2D, Blocksize2D);
    dim3 gridsize = div_ceil(dim3(t, n), blocksize);
    float *d_cov;
    checkCudaErrors(cudaMalloc((void **) &d_cov, sizeof(float) * t * n));
    compute_cross_cov << < gridsize, blocksize >> > (d_x, n, d, d_xtest, t, kernel, lsc, sig, a, d_cov);
    return d_cov;
}

// conditional mean of the test data given the prior gp
// kfy*kfyy^-1
float* conditionalMean(float *d_y,int n, float *d_cov, float *d_xtest,int t, float *d_covy,cusolverDnHandle_t cusolver,cublasHandle_t cublash){
    //container for the intermediate value
    float *d_interm;
    checkCudaErrors(cudaMalloc((void**)&d_interm, sizeof(float)*n));
    checkCudaErrors(cudaMemcpy(d_interm,d_y, sizeof(float)*n,cudaMemcpyDeviceToDevice));
    // kyy^-1*y
    int *d_devinfo;
    checkCudaErrors(cudaMalloc((void**)&d_devinfo, sizeof(int)));
    checkCudaErrors(cusolverDnSpotrs(cusolver,CHOLESKY_FILL_MODE,n,1,d_cov,n,d_interm,n,d_devinfo));
    int h_devinfo=0;
    checkCudaErrors(cudaMemcpy(&h_devinfo,d_devinfo, sizeof(int),cudaMemcpyDeviceToHost));
    if(h_devinfo!=0){
        printf("Error in kyf^-1*y");
        exit(EXIT_FAILURE);
    }
    float *d_mean;
    checkCudaErrors(cudaMalloc((void**)&d_mean, sizeof(float)*t));
    float alpha=1.0f,beta=0.0f;
    checkCublasErrors(cublasSgemv_v2(cublash,CUBLAS_OP_T,n,t,&alpha,d_covy,n,d_interm,1,&beta,d_mean,1));
    cudaFree(d_interm);
    cudaFree(d_devinfo);
    return d_mean;
}

// conditional covariance of the test data given the prior gp
// kff-kfy*kyy^-1*kyf
float* conditionalCov(int n, int d,float *d_cov, float *d_xtest,int t, float *d_covy,kernelstring_enum kern,float lsc,float sig,float a,cusolverDnHandle_t cusolver,cublasHandle_t cublash){
    //container for the intermediate value
    float *d_interm;
    checkCudaErrors(cudaMalloc((void**)&d_interm, sizeof(float)*n));
    checkCudaErrors(cudaMemcpy(d_interm,d_covy, sizeof(float)*n,cudaMemcpyDeviceToDevice));
    // kyy^-1*kyf
    int *d_devinfo;
    checkCudaErrors(cudaMalloc((void**)&d_devinfo, sizeof(int)));
    checkCudaErrors(cusolverDnSpotrs(cusolver,CHOLESKY_FILL_MODE,n,t,d_cov,n,d_interm,n,d_devinfo));
    int h_devinfo=0;
    checkCudaErrors(cudaMemcpy(&h_devinfo,d_devinfo, sizeof(int),cudaMemcpyDeviceToHost));
    if(h_devinfo!=0){
        printf("Error in kyf^-1*kyf");
        exit(EXIT_FAILURE);
    }
    //calculate kff
    float *d_covff=comp_cov_gpu(d_xtest,t,d,kern,lsc,sig,a);
    //calculate kff-kfy*kyy^-1*kyf
    float alpha=-1.0f,beta=1.0f;
    checkCublasErrors(cublasSgemm_v2(cublash,CUBLAS_OP_T,CUBLAS_OP_N,t,t,n,&alpha,d_covy,n,d_interm,n,&beta,d_covff,t));
    cudaFree(d_interm);
    cudaFree(d_devinfo);
    return d_covff;
}

// compute Cholesky decomposition
void chol(float d_cov,int n,cusolverDnHandle_t cusolver){
    int Lwork=0;float* d_workspace;
    checkCusolverErrors(cusolverDnSgeqrf_bufferSize(cusolver,CHOLESKY_FILL_MODE,n,d_cov,n,&Lwork));
    checkCudaErrors(cudaMalloc((void**)&d_workspace, sizeof(float)*Lwork));
    int *d_devinfo;
    checkCudaErrors(cudaMalloc((void**)&d_devinfo, sizeof(int)));
    checkCudaErrors(cusolverDnSpotrs(cusolver,CHOLESKY_FILL_MODE,n,d_cov,n,d_workspace,Lwork,d_devinfo));
    int h_devinfo=0;
    checkCudaErrors(cudaMemcpy(&h_devinfo,d_devinfo, sizeof(int),cudaMemcpyDeviceToHost));
    if(h_devinfo!=0){
        printf("Error in Cholesky Decomposition");
        exit(EXIT_FAILURE);
    }
}

__global__ void eye_k(int n,float* d_eye){
    int i=threadIdx.x+blockIdx.x*blockDim.x;
    int j=threadIdx.y+blockIdx.y*blockDim.y;
    if (i<n && j<n){
        d_eye[i*n+j]=(i==j);
    }
}

float* eye(int n){
    float* d_eye;
    checkCudaErrors(cudaMalloc((void**)&d_eye, sizeof(float)*n*n));
    dim3 blocksize(Blocksize2D,Blocksize2D);
    dim3 gridsize=div_ceil(dim3(n,n),blocksize);
    eye_k<<<gridsize,blocksize>>>(n,d_eye);
    return d_eye;
}