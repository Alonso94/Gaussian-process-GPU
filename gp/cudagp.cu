#include "cudagp.h"
#include <stdio.h>
#include <math.h>
#include "kernels.h"
#include "utils_cuda.h"
#include <time.h>
#include <math_functions.h>

void randMatrixf(float* h_matrix,int n){
    for(int i=0;i<n;++i)
        h_matrix[i]=rand()/(float)RAND_MAX*100;
}

void printMatrix(float* a,int rows,int cols){
    printf("Matrix %d x %d:\n",rows,cols);
    for(int i=0;i<rows;++i){
        for(int j=0;j<cols;++j)
            printf("%.5f ",a[i*cols+j]);
        printf("\n");
    }
}

void printDiag(float* a,int rows,int cols){
    printf("Diagonal of matrix:\n");
    for(int i=0;i<min(rows,cols);++i)
        printf("%.5f ",a[i*cols+i]);
    printf("\n");
}

int readData(float *x,float *y,int n){
    FILE *in=fopen("test_data","r");
    if(!in)
        printf("Failed to read");
    int i=0;
    char line[100];
    double a,b;
    while(i<n && fgets(line, sizeof(line),in)!=nullptr){
        scanf(line,"%f\t%f[^\n]",&a,&b);
        x[i]=a;
        y[i]=b;
        i++;
    }
    return i;
}

float* cov_matrix(float *x,int n,int d,kernelstring_enum kernel,float lsc,float sig,float a){
    float* h_cov=(float*)malloc(sizeof(float)*n*n);
    kernelfunc kern=get_kernel(kernel);
    for(int i=0;i<n;++i)
        for(int j=0;j<n;++j)
            h_cov[i*n+j]=kern(&x[i*d],&x[j*d],d,lsc,sig,a);
    return h_cov;
}

float* linespace(int min,int max,int len){
    float *x=(float*) malloc(len* sizeof(float));
    for(int i=0;i<len;++i)
        x[i]=min+(max-min)*(i/(float)(len-1));
    return x;
}

float* uniform(float min,float max,int len){
    float *x=(float*) malloc(len* sizeof(float));
    for(int i=0;i<len;++i)
        x[i]=min+(max-min+1)*((float)rand()/(float)RAND_MAX);
    return x;
}

float* func(float *x,int n){
    float *y=(float*) malloc(n* sizeof(float));
    for(int i=0;i<n;++i)
        y[i]=sin(x[i]+((float)rand()/(float)RAND_MAX)*0.4);
    return y;
}

float get_mean(float *x,int n){
    float sum=0.0;
    for(int i=0;i<n;++i)
        sum+=x[i];
    return sum/n;
}

float get_std(float *x,int n){
    float var=0.0;
    float mean=get_mean(x,n);
    for(int i=0;i<n;++i)
        var+=(x[i]-mean)*(x[i]-mean);
    return sqrt(var/(n-1));
}

//int main(int argc,const char** argv){
//    srand(0);
//    int n=300,d=1;
//    float *x=uniform(-400,400,n);
//    float *y=func(x,n);
//    int t=201;
//    float *x_test=linespace(-400,400,t);
//    float lsc=1.8,sig=1.15,a=0;
//    cudagphandle_t cudagphandle=initializaeCudaGP(x,y,n,d,cudagpSquaredExponentialKernel,lsc,sig,a);
//    prediction pred=predict(cudagphandle,x_test,t);
//
//    free(x_test);
//    free(x);
//    free(y);
//    freeCudaGP(cudagphandle);
//    free(pred.mean);
//    free(pred.var);
//}

