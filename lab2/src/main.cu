#include <sys/time.h>
#include <stdio.h>
#include<math.h>
//TODO for writing to file, will be deleted
#include <stdlib.h>
//TODO: could include later
//#include <device_launch_parameters.h>
#include <cuda_runtime.h>
//#include "../inc/helper_cuda.h"

// time stamp function in ms
double getTimeStamp() {
 struct timeval tv ;
 gettimeofday( &tv, NULL ) ;
 return (double) tv.tv_usec/1000 + tv.tv_sec*1000 ;
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
 for(i = 0; i < 3; i++){
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
// host side validation 
bool val(float *a, float *b, int n){
 int i,j,k;
 bool match = true;
 for(i = 0; i < n; i++){
  for(j = 0; j < n; j++){
   for(k = 0; k < n; k++){
    if(match && (round(a[i*n*n + j*n + k]*100)/100 != round(b[i*n*n+j*n+k]*100)/100)){
     //printf("%d,%d,%d expect %lf, actual %lf\n",i,j,k,h_A[i*n*n + j*n + k],h_dA[i*n*n+j*n+k]);
     match = false;
     //break;
    }
   }
  }
 }
 return match;
}
float h_sum(float *data, int n){
 int i,j,k;
 float ret=0;
 for(i = 0; i < n; i++){
  for(j = 0; j < n; j++){
   for(k = 0; k < n; k++){
    ret += data[i*n*n + j*n + k]*(((i+j+k)%2)?1:-1);
   }
  }
 }
 return ret;
}
__device__ void globalToShared(float *sm, float *b, int l, int n, int smx, int smy, int ix, int iy){
  sm[smx+smy*(blockDim.x+2)] = b[ix + iy*n + l*n*n];
  if(smx==1){
   sm[0+smy*(blockDim.x+2)] = b[ix-1 + iy*n + l*n*n];
  }
  if(smx==blockDim.x || ix==n-2){
   sm[smx+1+smy*(blockDim.x+2)] = b[ix+1 + iy*n + l*n*n];
  }
  if(smy==1){
   sm[smx] = b[ix + (iy-1)*n + l*n*n];
  }
  if(smy==blockDim.y || iy==n-2){
   sm[smx+(smy+1)*(blockDim.x+2)] = b[ix + (iy+1)*n + l*n*n];
  }
}
__global__ void kernal( float *a, float *b, int n){
 extern __shared__ float sm[];
 int ix = threadIdx.x + 1;
 int iy = threadIdx.y + 1;
 int gx = threadIdx.x + 1 + blockIdx.x*blockDim.x;
 int gy = threadIdx.y + 1 + blockIdx.y*blockDim.y;
 float down,up,self;
 float l1;
 if(gx<n-1&&gy<n-1){
  globalToShared(sm, b, 0, n, ix, iy, gx, gy);
  __syncthreads();
  down = sm[ix + iy*(blockDim.x+2)];
  globalToShared(sm, b, 1, n, ix, iy, gx, gy);
  __syncthreads();
  self = sm[ix + iy*(blockDim.x+2)];
  l1 = sm[ix-1 + iy*(blockDim.x+2)] + sm[ix+1 + iy*(blockDim.x+2)] + sm[ix + (iy-1)*(blockDim.x+2)] + sm[ix + (iy+1)*(blockDim.x+2)];
  __syncthreads();
  int layer;
  #pragma unroll
  for(layer = 2; layer < n; layer++){
   globalToShared(sm, b, layer, n, ix, iy, gx, gy);
   __syncthreads();
   up = sm[ix + iy*(blockDim.x+2)];
   a[gx + gy*n + (layer-1)*n*n] = 0.8*(down+up+l1);
   down = self;
   self = up;
   l1 = sm[ix-1 + iy*(blockDim.x+2)] + sm[ix+1 + iy*(blockDim.x+2)] + sm[ix + (iy-1)*(blockDim.x+2)] + sm[ix + (iy+1)*(blockDim.x+2)];
   __syncthreads();
  }
 }
}
int main( int argc, char *argv[] ) {
 // get program arguments
 if( argc != 2) {
 printf("Error: wrong number of args\n") ;
 exit(1) ;
 }
 int n = atoi( argv[1] );
 int noElems = n*n*n ;
 int bytes = noElems * sizeof(float) ;

 // alloc memory host-side
 float *h_A = (float *) malloc( bytes ) ;
 float *h_B = (float *) malloc( bytes ) ;
 float *h_dA = (float *) malloc( bytes ) ;

 // init matrices with random data
 initData(h_B, n);
 memset(h_A, 0, bytes);
 
 // alloc memory dev-side
 float *d_A, *d_B ;
 cudaMalloc( (void **) &d_A, bytes ) ;
 cudaMalloc( (void **) &d_B, bytes ) ;

 double timeStampA = getTimeStamp() ;
 //transfer data to dev
 cudaMemcpy( d_B, h_B, bytes, cudaMemcpyHostToDevice ) ;
 double timeStampB = getTimeStamp() ;

 //debugPrint(h_B, n);
 // invoke Kernel
 dim3 block(32, 32);
 dim3 grid((n-2+block.x-1)/block.x,(n-2+block.y-1)/block.y);
 kernal<<<grid,block,(1024+33*4)*sizeof(float)>>>(d_A,d_B,n);
 cudaDeviceSynchronize() ;
 //cudaDeviceProp GPUprop;
 //cudaGetDeviceProperties(&GPUprop,0);
 //printf("maxgridsize x is %d\n",GPUprop.maxGridSize[0]);

 double timeStampC = getTimeStamp() ;
 //copy data back
 cudaMemcpy( h_dA, d_A, bytes, cudaMemcpyDeviceToHost ) ;
 double timeStampD = getTimeStamp() ;

 h_stencil(h_A,h_B,n);
 //float h_Result = h_sum(h_A,n);
 float h_dResult = h_sum(h_dA,n);
 
 // print out results
 //if(!memcmp(h_A,h_dA,n*n*n*sizeof(float))){
 if(val(h_A,h_dA,n)){
  //debugPrint(h_hC, nx, ny);
  //debugPrint(h_dC, nx, ny);
  FILE* fptr;
  fptr = fopen("time.log","a");
  fprintf(fptr,"%d: %lf, %.6f %.6f %.6f %.6f\n", n, h_dResult, timeStampD-timeStampA, timeStampB-timeStampA, timeStampC-timeStampB, timeStampD-timeStampC);
  fclose(fptr);
  printf("%lf %d\n", h_dResult, (int)round(timeStampD-timeStampA));
 }else{
  //debugPrint(h_A, n);
  //debugPrint(h_dA, n);
  printf("Error: function failed.\n");
 }
 
 // free GPU resources
 cudaFree(d_A);
 cudaFree(d_B);
 cudaDeviceReset();
}
