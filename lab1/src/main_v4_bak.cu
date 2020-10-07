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
__global__ void f_addmat( float *A, float *B, int nx, int ny/*, int padrow*/){
 // kernel code might look something like this
 // but you may want to pad the matrices and index into them accordingly
 //__shared__ float sA[32][32];
 //__shared__ float sB[32][32];
 //__shared__ float sC[32][32];

 int ix = threadIdx.x;
 int iy = threadIdx.y*blockDim.x + blockIdx.x*blockDim.x*blockDim.y;
 int idx = (iy + ix)*4 ;
 //int col = idx-padrow*(int)(idx/padrow);
 //if(idx<nx*padrow && col<ny){
 if(idx<nx*ny){
  //int sidx = threadIdx.y*blockDim.x + threadIdx.x;
  int size = ((nx*ny-idx)<4) ? (nx*ny-idx) : 4;
  //int size=4;
  //if((ny-col)<4){
  // size = ny-col;
  //}
  //if(col<4){
  // size += col;
  //}

  //float tmpA[4];
  //float tmpB[4];
  //memcpy(tmpA,&A[idx],size);
  //memcpy(tmpB,&B[idx],size);
  //for(int j = 0; j < size; j++){
  // tmpB[j] += tmpA[j];
  //}
  //memcpy(&B[idx],tmpB,size);
  //printf("sidx is %d, idx is %d, size is %d\n", sidx, idx, size);
  #pragma unroll
  for(int i = idx; i < idx + size; i++){
   //sA[threadIdx.x][threadIdx.y] = A[i];
   //sB[threadIdx.x][threadIdx.y] = B[i];
   //__syncthreads();
   //sC[threadIdx.x][threadIdx.y] = sA[threadIdx.x][threadIdx.y] + sB[threadIdx.x][threadIdx.y];
   //__syncthreads();
   //C[i] = sC[threadIdx.x][threadIdx.y];
   //printf("index %d\n",i);
   B[i] += A[i];
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
 //size_t pitchA,pitchB, pitchC;
 //cudaMallocPitch(&d_A,&pitchA,ny*sizeof(float),nx);
 //cudaMallocPitch(&d_B,&pitchB,ny*sizeof(float),nx);
 //cudaMallocPitch(&d_C,&pitchC,ny*sizeof(float),nx);
 //float *h_ddC = (float *) malloc(nx*pitchC);
 cudaMalloc( (void **) &d_C, bytes ) ;
 cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
 double timeStampA = getTimeStamp() ;
 //transfer data to dev
 cudaMemcpy( d_A, h_A, bytes, cudaMemcpyHostToDevice ) ;
 cudaMemcpy( d_B, h_B, bytes, cudaMemcpyHostToDevice ) ;
 //printf("pA is %d and pB is %d\n",pitchA,pitchB);
 //cudaMemcpy2D( d_A, pitchA, h_A,ny*sizeof(float),ny*sizeof(float),nx,cudaMemcpyHostToDevice ) ;
 //cudaMemcpy2D( d_B, pitchB, h_B,ny*sizeof(float),ny*sizeof(float),nx,cudaMemcpyHostToDevice ) ;
 // note that the transfers would be twice as fast if h_A and h_B
 // matrices are pinned
 double timeStampB = getTimeStamp() ;

 // invoke Kernel
 dim3 block( 32, 32 ) ; // you will want to configure this
 //int block = 64;
 //int grid = (noElems + block-1)/block;
 int grid = ((noElems+3)/4 + block.x*block.y-1)/(block.x*block.y);
 //int grid = (((pitchA/4*nx*sizeof(float))+3)/4 + block.x*block.y-1)/(block.x*block.y);
 //dim3 grid( (nx + block.x-1)/block.x, (ny + block.y-1)/block.y ) ;
 //cudaDeviceProp GPUprop;
 //cudaGetDeviceProperties(&GPUprop,0);
 //printf("sharedmemperblk is %d\n",GPUprop.sharedMemPerBlock);
 //printf("maxgridsize x is %d\n",GPUprop.maxGridSize[0]);
 //printf("noelems is %d\n",noElems);
 //printf("gridx is %d\n",grid);
 //printf("gridx is %d and grid y is %d\n",grid.x,grid.y);

 f_addmat<<<grid, block>>>( d_A, d_B, nx, ny/*, pitchA/(sizeof(float))*/ ) ;
 cudaDeviceSynchronize() ;

 double timeStampC = getTimeStamp() ;
 //copy data back
 cudaMemcpy( h_dC, d_B, bytes, cudaMemcpyDeviceToHost ) ;
 //cudaMemcpy2D( h_ddC, pitchB, d_B,ny*sizeof(float),ny*sizeof(float),nx,cudaMemcpyDeviceToHost ) ;
 double timeStampD = getTimeStamp() ;

 // free GPU resources
 cudaFree( d_A ) ; cudaFree( d_B ) ; cudaFree( d_C ) ;
 cudaDeviceReset() ;

 // check result
 h_addmat( h_A, h_B, h_hC, nx, ny ) ;

 //for(int i = 0; i < nx; i++){
 // for(int j = 0; j < pitchC/4; j++){
 //  if(j<ny){
 //   h_dC[i*ny+j] = h_ddC[i*pitchC/4+j];
 //  }
 // }
 //} 
 // print out results
 if(!memcmp(h_hC,h_dC,nx*ny*sizeof(float))){
  //debugPrint(h_hC, nx, ny);
  //debugPrint(h_dC, nx, ny);
  FILE* fptr;
  fptr = fopen("time.log","a");
  fprintf(fptr,"%dX%d %.6f %.6f %.6f %.6f\n", nx, ny, timeStampD-timeStampA, timeStampB-timeStampA, timeStampC-timeStampB, timeStampD-timeStampC);
  fclose(fptr);
  printf("%.6f %.6f %.6f %.6f\n", timeStampD-timeStampA, timeStampB-timeStampA, timeStampC-timeStampB, timeStampD-timeStampC);
 }else{
  //debugPrint(h_hC, nx, ny);
  //debugPrint(h_dC, nx, ny);
  printf("Error: function failed.\n");
 }
}
