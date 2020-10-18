#include <sys/time.h>
#include <stdio.h>
#include<math.h>
//TODO for writing to file, will be deleted
#include <stdlib.h>
//TODO: could include later
//#include <device_launch_parameters.h>
#include <cuda_runtime.h>
//#include "../inc/helper_cuda.h"
#define NUM_STREAMS 4 
texture<float, 1, cudaReadModeElementType> texRef;
// time stamp function in ms
double getTimeStamp() {
 struct timeval tv ;
 gettimeofday( &tv, NULL ) ;
 return (double) tv.tv_usec/1000 + tv.tv_sec*1000 ;
}
void initData(float* data, int n, int pad_n){
 memset(data, 0, pad_n*pad_n*n*sizeof(float));
 int i,j,k;
 for(i = 0; i < n; i++){
  for(j = 0; j < n; j++){
   for(k = 0; k < n; k++){
    data[i*pad_n*pad_n + j*pad_n + k] = (float) (i+j+k)*1.1;
   }
  }
 }
}
void debugPaddingPrint(float* data, int n, int height){
 int i,j,k;
 for(i = 0; i < height; i++){
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
void debugPrint(float* data, int n, int pad_n){
 int i,j,k;
 for(i = 0; i < n; i++){
  printf("--------layer %d--------\n",i);
  for(j = 0; j < n; j++){
   for(k = 0; k < n; k++){
    printf("%lf ",data[i*pad_n*pad_n + j*pad_n + k]);
   }
   printf("\n");
  }
  printf("\n");
 }
 printf("\n");
}
// host side matrix addition
void h_stencil(float *a, float *b, int n, int pad_n){
 int i,j,k;
 for(i = 1; i < n-1; i++){
  for(j = 1; j < n-1; j++){
   for(k = 1; k < n-1; k++){
    a[i*pad_n*pad_n + j*pad_n + k] = 0.8*(b[(i-1)*pad_n*pad_n+j*pad_n+k]+b[(i+1)*pad_n*pad_n+j*pad_n+k]+b[i*pad_n*pad_n+(j-1)*pad_n+k]+b[i*pad_n*pad_n+(j+1)*pad_n+k]+b[i*pad_n*pad_n+j*pad_n+(k-1)]+b[i*pad_n*pad_n+j*pad_n+(k+1)]);
   }
  }
 }
}
// host side validation 
bool val(float *a, float *b, int n, int pad_n){
 int i,j,k;
 bool match = true;
 for(i = 1; i < n-1; i++){
  for(j = 1; j < n-1; j++){
   for(k = 1; k < n-1; k++){
    if(match && (round(a[i*pad_n*pad_n + j*pad_n + k]*100)/100 != round(b[i*pad_n*pad_n+j*pad_n+k]*100)/100)){
     //printf("%d,%d,%d expect %lf, actual %lf\n",i,j,k,h_A[i*n*n + j*n + k],h_dA[i*n*n+j*n+k]);
     match = false;
     //break;
    }
   }
  }
 }
 return match;
}
float h_sum(float *data, int n, int pad_n){
 int i,j,k;
 float ret=0;
 for(i = 1; i < n-1; i++){
  for(j = 1; j < n-1; j++){
   for(k = 1; k < n-1; k++){
    ret += data[i*pad_n*pad_n + j*pad_n + k]*(((i+j+k)%2)?1:-1);
   }
  }
 }
 return ret;
}
__const__ __device__ int indexX[32][32] = //indexX[threadY][threadX]
{
 { 0, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
 {-1,-2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,1},
 
 {-1, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1},
 {-1, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1},
 {-1, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1},
 {-1, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1},
 {-1, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1},
 {-1, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1},
 {-1, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1},
 {-1, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1},
 {-1, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1},
 {-1, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1},
 {-1, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1},
 {-1, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1},
 {-1, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1},
 {-1, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1},
 {-1, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1},
 {-1, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1},
 {-1, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1},
 {-1, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1},
 {-1, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1},
 {-1, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1},
 {-1, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1},
 {-1, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1},
 {-1, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1},
 {-1, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1},
 {-1, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1},
 {-1, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1},
 {-1, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1},
 {-1, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1},
 
 {-1,-2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,1},
 { 0, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
};
__const__ __device__ int indexY[32][32] = //indexY[threadY][threadX]
{
 {-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
 { 0,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-1, 0},
 
 { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
 { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
 { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
 { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
 { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
 { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
 { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
 { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
 { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
 { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
 { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
 { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
 { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
 { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
 { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
 { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
 { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
 { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
 { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
 { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
 { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
 { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
 { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
 { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
 { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
 { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
 { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
 { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
 
 { 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0},
 { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
};

__device__ void globalToShared(float *sm, float *b, int l, int n, int smx, int smy, int ix, int iy){
  sm[smx+smy*(blockDim.x+2)] = tex1Dfetch(texRef,ix + iy*n + l*n*n);
  int marginOffsetX = indexX[smy-1][smx-1];
  int marginOffsetY = indexY[smy-1][smx-1];
  sm[smx+marginOffsetX+(smy+marginOffsetY)*(blockDim.x+2)] = tex1Dfetch(texRef,ix+marginOffsetX + (iy+marginOffsetY)*n + l*n*n);
  //sm[smx+smy*(blockDim.x+2)] = b[ix + iy*n + l*n*n];
  //int marginOffsetX = indexX[smy-1][smx-1];
  //int marginOffsetY = indexY[smy-1][smx-1];
  //sm[smx+marginOffsetX+(smy+marginOffsetY)*(blockDim.x+2)] = b[ix+marginOffsetX + (iy+marginOffsetY)*n + l*n*n];
  //if(smx==1){
  // sm[0+smy*(blockDim.x+2)] = b[ix-1 + iy*n + l*n*n];
  //}
  //if(smx==blockDim.x){
  // sm[smx+1+smy*(blockDim.x+2)] = b[ix+1 + iy*n + l*n*n];
  //}
  //if(smy==1){
  // sm[smx] = b[ix + (iy-1)*n + l*n*n];
  //}
  //if(smy==blockDim.y){
  // sm[smx+(smy+1)*(blockDim.x+2)] = b[ix + (iy+1)*n + l*n*n];
  //}
}
__global__ void kernal( float *a, float *b, int n, int height){
 extern __shared__ float sm[];
 int ix = threadIdx.x + 1;
 int iy = threadIdx.y + 1;
 int gx = threadIdx.x + 1 + blockIdx.x*blockDim.x;
 int gy = threadIdx.y + 1 + blockIdx.y*blockDim.y;
 float down,up,self;
 float l1;
 down =tex1Dfetch(texRef,gx+gy*n);
 globalToShared(sm, b, 1, n, ix, iy, gx, gy);
 __syncthreads();
 self = sm[ix + iy*(blockDim.x+2)];
 l1 = sm[ix-1 + iy*(blockDim.x+2)] + sm[ix+1 + iy*(blockDim.x+2)] + sm[ix + (iy-1)*(blockDim.x+2)] + sm[ix + (iy+1)*(blockDim.x+2)];
 __syncthreads();
 int layer;
 for(layer = 2; layer < height; layer++){
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
int main( int argc, char *argv[] ) {
 // get program arguments
 if( argc != 2) {
 printf("Error: wrong number of args\n") ;
 exit(1) ;
 }
 int n = atoi( argv[1] );
 int pad_n = n + 32 - (n-2)%32;
 //int pad_offset = pad_n-n;
 //printf("padding is %d\n",pad_n);
 int noElems = pad_n*pad_n*n ;
 int bytes = noElems * sizeof(float) ;

 // alloc memory host-side
 float *h_A = (float *) malloc( bytes ) ;
 //float *h_B = (float *) malloc( bytes ) ;
 //float *h_dA = (float *) malloc( bytes ) ;
 float *h_B;
 float *h_dA;
 cudaMallocHost((void**)&h_B,bytes);
 cudaMallocHost((void**)&h_dA,bytes);
 // init matrices with random data
 initData(h_B, n, pad_n);
 memset(h_A, 0, bytes);
 
 //debugPaddingPrint(h_B,pad_n,n);
 // alloc memory dev-side
 float *d_A, *d_B ;
 cudaMalloc( (void **) &d_A, bytes ) ;
 cudaMalloc( (void **) &d_B, bytes ) ;
 
 cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
 double timeStampA = getTimeStamp() ;
 //stream creation
 int batch_size = (noElems+NUM_STREAMS-1)/NUM_STREAMS;
 int last_batch = noElems-(NUM_STREAMS-1)*batch_size;
 cudaStream_t stream[NUM_STREAMS+1];
 for (int i = 1; i < NUM_STREAMS; i++){
  cudaStreamCreate(&(stream[i]));
  cudaMemcpyAsync(&d_B[(i-1)*batch_size],&h_B[(i-1)*batch_size],batch_size*sizeof(float),cudaMemcpyHostToDevice,stream[i]);
 }
 cudaStreamCreate(&(stream[NUM_STREAMS]));
 cudaMemcpyAsync(&d_B[(NUM_STREAMS-1)*batch_size],&h_B[(NUM_STREAMS-1)*batch_size],last_batch*sizeof(float),cudaMemcpyHostToDevice,stream[NUM_STREAMS]);

 //sync all streams and done
 for(int i = 1; i < NUM_STREAMS+1; i++){
  cudaStreamSynchronize(stream[i]);
 }
 //transfer data to dev
 //cudaMemcpy( d_B, h_B, bytes, cudaMemcpyHostToDevice ) ;
 cudaBindTexture(NULL,texRef,d_B,bytes);
 double timeStampB = getTimeStamp() ;

 //debugPrint(h_B, n);
 // invoke Kernel
 dim3 block(32, 32);
 dim3 grid((pad_n-2+block.x-1)/block.x,(pad_n-2+block.y-1)/block.y);
 //printf("grid x %d, grid y %d\n",grid.x,grid.y);
 kernal<<<grid,block,(1024+33*4)*sizeof(float)>>>(d_A,d_B,pad_n,n);
 cudaDeviceSynchronize() ;
 //cudaDeviceProp GPUprop;
 //cudaGetDeviceProperties(&GPUprop,0);
 //printf("maxgridsize x is %d\n",GPUprop.maxGridSize[0]);

 double timeStampC = getTimeStamp() ;
 //copy data back
 cudaMemcpy( h_dA, d_A, bytes, cudaMemcpyDeviceToHost ) ;
 double timeStampD = getTimeStamp() ;

 h_stencil(h_A,h_B,n,pad_n);
 float h_dResult = h_sum(h_dA,n,pad_n);
 
 // print out results
 //if(!memcmp(h_A,h_dA,n*n*n*sizeof(float))){
 if(val(h_A,h_dA,n,pad_n)){
  //debugPrint(h_hC, nx, ny);
  //debugPrint(h_dC, nx, ny);
  FILE* fptr;
  fptr = fopen("time.log","a");
  fprintf(fptr,"%d: %lf, %.6f %.6f %.6f %.6f\n", n, h_dResult, timeStampD-timeStampA, timeStampB-timeStampA, timeStampC-timeStampB, timeStampD-timeStampC);
  fclose(fptr);
  printf("%lf %d\n", h_dResult, (int)round(timeStampD-timeStampA));
 }else{
  debugPrint(h_A, n, pad_n);
  debugPrint(h_dA, n, pad_n);
  printf("Error: function failed.\n");
 }
 
 // free GPU resources
 //cudaFreeHost(h_B);
 //cudaFreeHost(h_dA);
 cudaDeviceReset();
}
