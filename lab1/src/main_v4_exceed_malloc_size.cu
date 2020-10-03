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
void initData(float* data0, float* data1, int nx, int ny){ 
 int i,j;
 int idx;
 for(i = 0; i < nx; i++){
  for(j = 0; j < ny; j++){
   //printf("i is %d and j is %d\n",i,j);
   idx = 2*(i*ny+j);
   if(idx<nx*ny){
    data0[idx] = (float)(i+j)/3.0;
    data0[idx+1] = (float)3.14*(i+j);
   }else{
    if((nx*ny)%2){
     data1[idx-1-nx*ny] = (float)(i+j)/3.0;
     data1[idx-nx*ny] = (float)3.14*(i+j);
    }else{
     data1[idx-nx*ny] = (float)(i+j)/3.0;
     data1[idx+1-nx*ny] = (float)3.14*(i+j);
    }
   } 
  }
 }
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
void h_addmat(float *A, float* B, float *C, int nx, int ny){
 int i;
 for(i = 0; i < 2*nx*ny; i+=2){
  if(i<nx*ny){
   C[i/2] = A[i] + A[i+1];
  //}else if(i==nx*ny-1){
   //C[i/2] = A[i] + B[i+1-nx*ny];
  }else{
   if((nx*ny)%2){
    C[i/2] = B[i-nx*ny] + B[i-1-nx*ny];
   }else{
    C[i/2] = B[i-nx*ny] + B[i+1-nx*ny];
   }
  }
 }
}
// device-side matrix addition
__global__ void f_addmat( float *A, float* midA, float *C, int nx, int ny ){
 // kernel code might look something like this
 // but you may want to pad the matrices and index into them accordingly
 //__shared__ float sA[32][32];
 //__shared__ float sB[32][32];
 //__shared__ float sC[32][32];

 int ix = threadIdx.x;
 int iy = threadIdx.y*blockDim.x + blockIdx.x*blockDim.x*blockDim.y;
 int idx = (iy + ix)*4 ;
 float* tmp;
 int size;
 int i;
 if((2*idx)<(nx*ny)&&((nx*ny-2*idx)>=8)){
  tmp=A;
  if((nx*ny+((nx*ny)%2)-2*idx)<16){
   size = (nx*ny+((nx*ny)%2)-2*idx)/2;
  }else{
   size = 4;
  }
  i = 2*idx;
 }else{
  tmp=midA;
  if( (2*idx) < (nx*ny+((nx*ny)%2)) ){
   idx = (nx*ny+((nx*ny)%2))/2;
  }
  if((nx*ny*2-2*idx)<=2*((iy + ix)*4+4-idx)){
   size = (nx*ny*2-2*idx)/2;
  }else{
   size = (iy + ix)*4+4-idx;
  }
  i = 2*idx-nx*ny-(nx*ny)%2;
 }
 if(idx<nx*ny){
  int j;
  for(j = 0; j < size; j++){
   C[idx+j]   = tmp[i+2*j]   + tmp[i+2*j+1];
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
 float *h_A = (float *) malloc( bytes + (noElems%2)*sizeof(float)) ;
 float *h_B = (float *) malloc( bytes - (noElems%2)*sizeof(float)) ;
 float *h_hC = (float *) malloc( bytes ) ; // host result
 float *h_dC = (float *) malloc( bytes ) ; // gpu result

 // init matrices with random data
 initData( h_A, h_B, nx, ny) ;
 // alloc memory dev-side
 float *d_A, *d_midA, *d_C ;
 cudaMalloc( (void **) &d_A, bytes + (noElems%2)*sizeof(float)) ;
 //printf("int is %d, size is %d\n",bytes + (noElems%2)*sizeof(float),sizeof(d_A));
 cudaMalloc( (void **) &d_midA, bytes - (noElems%2)*sizeof(float)) ;
 cudaMalloc( (void **) &d_C, bytes ) ;

 double timeStampA = getTimeStamp() ;
 //transfer data to dev
 cudaMemcpy( d_A, h_A, bytes+(noElems%2)*sizeof(float), cudaMemcpyHostToDevice ) ;
 cudaMemcpy( d_midA, h_B, bytes-(noElems%2)*sizeof(float), cudaMemcpyHostToDevice ) ;
 // note that the transfers would be twice as fast if h_A and h_B
 // matrices are pinned
 double timeStampB = getTimeStamp() ;

 // invoke Kernel
 dim3 block( 32, 32 ) ; // you will want to configure this
 //int block = 64;
 //int grid = (noElems + block-1)/block;
 //int grid = (noElems + block.x*block.y-1)/(block.x*block.y);
 int grid = ((noElems+3)/4 + 1 + block.x*block.y-1)/(block.x*block.y);
 //dim3 grid( (nx + block.x-1)/block.x, (ny + block.y-1)/block.y ) ;
 //cudaDeviceProp GPUprop;
 //cudaGetDeviceProperties(&GPUprop,0);
 //printf("sharedmemperblk is %d\n",GPUprop.sharedMemPerBlock);
 //printf("maxgridsize x is %d\n",GPUprop.maxGridSize[0]);
 //printf("noelems is %d\n",noElems);
 //printf("gridx is %d\n",grid);
 //printf("gridx is %d and grid y is %d\n",grid.x,grid.y);

 f_addmat<<<grid, block>>>( d_A, d_midA, d_C, nx, ny ) ;
 cudaDeviceSynchronize() ;

 double timeStampC = getTimeStamp() ;
 //copy data back
 cudaMemcpy( h_dC, d_C, bytes, cudaMemcpyDeviceToHost ) ;
 double timeStampD = getTimeStamp() ;

 // free GPU resources
 cudaFree( d_A ) ; cudaFree( d_midA ) ; cudaFree( d_C ) ;
 cudaDeviceReset() ;

 // check result
 h_addmat( h_A, h_B, h_hC, nx, ny ) ;
 
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
  debugPrint(h_hC, nx, ny);
  debugPrint(h_dC, nx, ny);
  printf("Error: function failed.\n");
 }
}
