#include <sys/time.h>
#include <stdio.h>

//TODO for writing to file, will be deleted
#include <stdlib.h>
//TODO: could include later
//#include <device_launch_parameters.h>
#include <cuda_runtime.h>
//#include "../inc/helper_cuda.h"

// time stamp function in seconds
double getTimeStamp() {
 struct timeval tv ;
 gettimeofday( &tv, NULL ) ;
 return (double) tv.tv_usec/1000000 + tv.tv_sec ;
}
void initData(float* data, int n){ 
 int i,j,k;
 for(i = 0; i < n; i++){
  for(j = 0; j < n; j++){
   for(k = 0; k < n; k++){
    data[i*n*n + j*n + k] = (float) (i+j+k)*1.1;
   }
  }
 }
}
void debugPrint(float* data, int n){
 int i,j,k;
 for(i = 0; i < n; i++){
  printf("--------layer %d--------\n",i);
  for(j = 0; j < n; j++){
   for(k = 0; k < n; k++){
    printf("%lf ",data[i*n*n + j*n + k]);
   }
   printf("\n");
  }
  printf("\n");
 }
 printf("\n");
}
// host side matrix addition
void h_stencil(float *a, float *b, int n){
 int i,j,k;
 for(i = 1; i < n-1; i++){
  for(j = 1; j < n-1; j++){
   for(k = 1; k < n-1; k++){
    a[i*n*n + j*n + k] = 0.8*(b[(i-1)*n*n+j*n+k]+b[(i+1)*n*n+j*n+k]+b[i*n*n+(j-1)*n+k]+b[i*n*n+(j+1)*n+k]+b[i*n*n+j*n+(k-1)]+b[i*n*n+j*n+(k+1)]);
   }
  }
 }
}
// host side matrix addition
float h_sum(float *data, int n){
 int i,j,k;
 float ret=0;
 for(i = 0; i < n; i++){
  for(j = 0; j < n; j++){
   for(k = 0; k < n; k++){
    ret += data[i*n*n + j*n + k]*((i+j+k)?1:-1);
   }
  }
 }
 return ret;
}
// device-side matrix addition
//__global__ void f_addmat( float *A, float *B, float *C, int nx, int ny ){
// // kernel code might look something like this
// // but you may want to pad the matrices and index into them accordingly
// int ix = threadIdx.x + blockIdx.x*blockDim.x ;
// int iy = threadIdx.y + blockIdx.y*blockDim.y ;
// int idx = iy*nx + ix ;
// if( (ix<nx) && (iy<ny) )
// C[idx] = A[idx] + B[idx] ;
//}
__global__ void f_addmat( float *A, float *B, float *C, int nx, int ny ){
 // kernel code might look something like this
 // but you may want to pad the matrices and index into them accordingly
 int ix = threadIdx.x;
 int iy = threadIdx.y*blockDim.x + blockIdx.x*blockDim.x*blockDim.y;
 int idx = iy + ix ;
 if(idx<nx*ny)
 C[idx] = A[idx] + B[idx] ;
}
int main( int argc, char *argv[] ) {
 // get program arguments
 if( argc != 2) {
 printf("Error: wrong number of args\n") ;
 exit(1) ;
 }
 int n = atoi( argv[1] ) ; // should check validity
 int noElems = n*n*n ;
 int bytes = noElems * sizeof(float) ;
 // but you may want to pad the matrices…

 // alloc memory host-side
 float *h_A = (float *) malloc( bytes ) ;
 float *h_B = (float *) malloc( bytes ) ;

 // init matrices with random data
 //initData( h_A, noElems ) ; initData( h_B, noElems ) ;
 initData(h_B, n);
 memset(h_A, 0, bytes);
 h_stencil(h_A,h_B,n);
 debugPrint(h_A,n);
 float result = h_sum(h_A,n);
 printf("%lf ",result);
}
