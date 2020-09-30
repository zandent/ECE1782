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
void initDataA(float* data, int nx, int ny){ 
 int i,j;
 for(i = 0; i < nx; i++){
  for(j = 0; j < ny; j++){
   data[i*ny + j] = (float) (i+j)/3.0;
  }
 }
}
void initDataB(float* data, int nx, int ny){ 
 int i,j;
 for(i = 0; i < nx; i++){
  for(j = 0; j < ny; j++){
   data[i*ny + j] = (float)3.14*(i+j);
  }
 }
}
void debugPrint(float* data, int nx, int ny){
 int i,j;
 for(i = 0; i < nx; i++){
  for(j = 0; j < ny; j++){
   printf("%f ",data[i*ny + j]);
  }
  printf("\n");
 }
 printf("\n");
}
// host side matrix addition
void h_addmat(float *A, float *B, float *C, int nx, int ny){
 int i;
 for(i = 0; i < nx*ny; i++){
  C[i] = A[i] + B[i];
 }
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
 int ix = threadIdx.x*blockDim.y+blockIdx.x*blockDim.x*blockDim.y;
 int iy = threadIdx.y;
 int idx = iy + ix ;
 if(idx<nx*ny)
 C[idx] = A[idx] + B[idx] ;
}
int main( int argc, char *argv[] ) {
 // get program arguments
 if( argc != 3) {
 printf("Error: wrong number of args\n") ;
 exit(1) ;
 }
 int nx = atoi( argv[1] ) ; // should check validity
 int ny = atoi( argv[2] ) ; // should check validity
 int noElems = nx*ny ;
 int bytes = noElems * sizeof(float) ;
 // but you may want to pad the matrices…

 // alloc memory host-side
 float *h_A = (float *) malloc( bytes ) ;
 float *h_B = (float *) malloc( bytes ) ;
 float *h_hC = (float *) malloc( bytes ) ; // host result
 float *h_dC = (float *) malloc( bytes ) ; // gpu result

 // init matrices with random data
 //initData( h_A, noElems ) ; initData( h_B, noElems ) ;
 initDataA(h_A, nx, ny);
 initDataB(h_B, nx, ny);
 // alloc memory dev-side
 float *d_A, *d_B, *d_C ;
 cudaMalloc( (void **) &d_A, bytes ) ;
 cudaMalloc( (void **) &d_B, bytes ) ;
 cudaMalloc( (void **) &d_C, bytes ) ;

 double timeStampA = getTimeStamp() ;
 //transfer data to dev
 cudaMemcpy( d_A, h_A, bytes, cudaMemcpyHostToDevice ) ;
 cudaMemcpy( d_B, h_B, bytes, cudaMemcpyHostToDevice ) ;
 // note that the transfers would be twice as fast if h_A and h_B
 // matrices are pinned
 double timeStampB = getTimeStamp() ;

 // invoke Kernel
 dim3 block( 8, 8 ) ; // you will want to configure this
 //int block = 64;
 //int grid = (noElems + block-1)/block;
 int grid = (noElems + block.x*block.y-1)/(block.x*block.y);
 //dim3 grid( (nx + block.x-1)/block.x, (ny + block.y-1)/block.y ) ;
 cudaDeviceProp GPUprop;
 cudaGetDeviceProperties(&GPUprop,0);
 printf("maxgridsize x is %d\n",GPUprop.maxGridSize[0]);
 printf("noelems is %d\n",noElems);
 printf("gridx is %d\n",grid);
 //printf("gridx is %d and grid y is %d\n",grid.x,grid.y);

 f_addmat<<<grid, block>>>( d_A, d_B, d_C, nx, ny ) ;
 cudaDeviceSynchronize() ;

 double timeStampC = getTimeStamp() ;
 //copy data back
 cudaMemcpy( h_dC, d_C, bytes, cudaMemcpyDeviceToHost ) ;
 double timeStampD = getTimeStamp() ;

 // free GPU resources
 cudaFree( d_A ) ; cudaFree( d_B ) ; cudaFree( d_C ) ;
 cudaDeviceReset() ;

 // check result
 h_addmat( h_A, h_B, h_hC, nx, ny ) ;
 
 // print out results
 if(!memcmp(h_hC,h_dC,nx*ny)){
  //debugPrint(h_hC, nx, ny);
  //debugPrint(h_dC, nx, ny);
  FILE* fptr;
  fptr = fopen("time.log","a");
  fprintf(fptr,"%.6f %.6f %.6f %.6f\n", timeStampD-timeStampA, timeStampB-timeStampA, timeStampC-timeStampB, timeStampD-timeStampC);
  fclose(fptr);
  printf("%.6f %.6f %.6f %.6f\n", timeStampD-timeStampA, timeStampB-timeStampA, timeStampC-timeStampB, timeStampD-timeStampC);
 }else{
  printf("Error: function failed.\n");
 }
}
