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
__global__ void f_addmat( float *A, float *B, float *C, int nx, int ny ){
 // kernel code might look something like this
 // but you may want to pad the matrices and index into them accordingly
 //__shared__ float sA[32][32];
 //__shared__ float sB[32][32];
 //__shared__ float sC[32][32];

 int ix = threadIdx.x;
 int iy = threadIdx.y*blockDim.x + blockIdx.x*blockDim.x*blockDim.y;
 int idx = (iy + ix)*4 ;
 if(idx<nx*ny){
  //int sidx = threadIdx.y*blockDim.x + threadIdx.x;
  int size = ((nx*ny-idx)<4) ? (nx*ny-idx) : 4;
  //printf("sidx is %d, idx is %d, size is %d\n", sidx, idx, size);
  for(int i = idx; i < idx + size; i++){
   //sA[threadIdx.x][threadIdx.y] = A[i];
   //sB[threadIdx.x][threadIdx.y] = B[i];
   //__syncthreads();
   //sC[threadIdx.x][threadIdx.y] = sA[threadIdx.x][threadIdx.y] + sB[threadIdx.x][threadIdx.y];
   //__syncthreads();
   //C[i] = sC[threadIdx.x][threadIdx.y];
   C[i] = A[i] + B[i];
  }
 }
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
 float *h_hA = (float *) malloc( bytes ) ;
 float *h_hB = (float *) malloc( bytes ) ;
 float *h_hC = (float *) malloc( bytes ) ; // host result
 //float *h_dC = (float *) malloc( bytes ) ; // gpu result
 float *h_A, *h_B, *h_dC;
 float *d_A, *d_B, *d_C ;
 cudaHostAlloc((void**)&h_A,bytes,cudaHostAllocWriteCombined|cudaHostAllocMapped);
 cudaHostAlloc((void**)&h_B,bytes,cudaHostAllocWriteCombined|cudaHostAllocMapped);
 cudaHostAlloc((void**)&h_dC,bytes,cudaHostAllocWriteCombined|cudaHostAllocMapped);
 cudaHostGetDevicePointer( &d_A, h_A, 0 );
 cudaHostGetDevicePointer( &d_B, h_B, 0 );
 cudaHostGetDevicePointer( &d_C, h_dC, 0 );
 // init matrices with random data
 //initData( h_A, noElems ) ; initData( h_B, noElems ) ;
 initDataA(h_A, nx, ny);
 initDataB(h_B, nx, ny);
 initDataA(h_hA, nx, ny);
 initDataB(h_hB, nx, ny);
 // alloc memory dev-side
 //cudaMalloc( (void **) &d_A, bytes ) ;
 //cudaMalloc( (void **) &d_B, bytes ) ;
 //cudaMalloc( (void **) &d_C, bytes ) ;

 double timeStampA = getTimeStamp() ;
 //transfer data to dev
 //cudaMemcpy( d_A, h_A, bytes, cudaMemcpyHostToDevice ) ;
 //cudaMemcpy( d_B, h_B, bytes, cudaMemcpyHostToDevice ) ;
 // note that the transfers would be twice as fast if h_A and h_B
 // matrices are pinned
 double timeStampB = getTimeStamp() ;

 // invoke Kernel
 dim3 block( 32, 32 ) ; // you will want to configure this
 //int block = 64;
 //int grid = (noElems + block-1)/block;
 //int grid = (noElems + block.x*block.y-1)/(block.x*block.y);
 int grid = ((noElems+3)/4 + block.x*block.y-1)/(block.x*block.y);
 //dim3 grid( (nx + block.x-1)/block.x, (ny + block.y-1)/block.y ) ;
 //cudaDeviceProp GPUprop;
 //cudaGetDeviceProperties(&GPUprop,0);
 //printf("sharedmemperblk is %d\n",GPUprop.sharedMemPerBlock);
 //printf("maxgridsize x is %d\n",GPUprop.maxGridSize[0]);
 //printf("noelems is %d\n",noElems);
 //printf("gridx is %d\n",grid);
 //printf("gridx is %d and grid y is %d\n",grid.x,grid.y);

 f_addmat<<<grid, block>>>( d_A, d_B, d_C, nx, ny ) ;
 cudaDeviceSynchronize() ;

 double timeStampC = getTimeStamp() ;
 //copy data back
 //cudaMemcpy( h_dC, d_C, bytes, cudaMemcpyDeviceToHost ) ;
 double timeStampD = getTimeStamp() ;

 // free GPU resources
 cudaFree( d_A ) ; cudaFree( d_B ) ; cudaFree( d_C ) ;
 cudaDeviceReset() ;

 // check result
 h_addmat( h_hA, h_hB, h_hC, nx, ny ) ;
 
 // print out results
 bool match = true;
 //for(int i = 0; i < nx*ny; i++){
 // if(h_hC[i]!=h_dC[i]){match=false;break;}
 //}
 if(match){
  //debugPrint(h_hC, nx, ny);
  //debugPrint(h_dC, nx, ny);
  FILE* fptr;
  fptr = fopen("time.log","a");
  fprintf(fptr,"%dX%d %.6f %.6f %.6f %.6f\n", nx, ny, timeStampD-timeStampA, timeStampB-timeStampA, timeStampC-timeStampB, timeStampD-timeStampC);
  fclose(fptr);
  printf("%.6f %.6f %.6f %.6f\n", timeStampD-timeStampA, timeStampB-timeStampA, timeStampC-timeStampB, timeStampD-timeStampC);
 }else{
  debugPrint(h_hC, nx, ny);
  debugPrint(h_dC, nx, ny);
  printf("Error: function failed.\n");
 }
}
