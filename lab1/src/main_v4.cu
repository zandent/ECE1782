#include <sys/time.h>
#include <stdio.h>
//TODO for writing to file, will be deleted
#include <stdlib.h>
//TODO: could include later
//#include <device_launch_parameters.h>
#include <cuda_runtime.h>
//#include "../inc/helper_cuda.h"
#define NUM_STREAMS 8 
double timeStampB;
double timeStampC;
double timeStampD;
// time stamp function in seconds
double getTimeStamp() {
 struct timeval tv ;
 gettimeofday( &tv, NULL ) ;
 return (double) tv.tv_usec/1000000 + tv.tv_sec ;
}
void myCallBackB(cudaStream_t stream,cudaError_t status, void*  userData ){
 timeStampB=getTimeStamp();
}
void myCallBackC(cudaStream_t stream,cudaError_t status, void*  userData ){
 timeStampC=getTimeStamp();
}
void myCallBackD(cudaStream_t stream,cudaError_t status, void*  userData ){
 timeStampD=getTimeStamp();
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
__global__ void f_addmat( float *A, float *B, int len/*, int padrow*/){
 // kernel code might look something like this
 // but you may want to pad the matrices and index into them accordingly
 //__shared__ float sA[32][32];
 //__shared__ float sB[32][32];
 //__shared__ float sC[32][32];

 int ix = threadIdx.x;
 int iy = threadIdx.y*blockDim.x + blockIdx.x*blockDim.x*blockDim.y;
 int idx = iy + ix ;
 //int col = idx-padrow*(int)(idx/padrow);
 //if(idx<nx*padrow && col<ny){
 //if(idx<gridDim.x/4*blockDim.x*blockDim.y){
  //int sidx = threadIdx.y*blockDim.x + threadIdx.x;
  //int size = ((nx*ny-idx)<4) ? (nx*ny-idx) : 4;
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
  for(int i = idx; i < len; i+=gridDim.x*blockDim.x*blockDim.y){
   //sA[threadIdx.x][threadIdx.y] = A[i];
   //sB[threadIdx.x][threadIdx.y] = B[i];
   //__syncthreads();
   //sC[threadIdx.x][threadIdx.y] = sA[threadIdx.x][threadIdx.y] + sB[threadIdx.x][threadIdx.y];
   //__syncthreads();
   //C[i] = sC[threadIdx.x][threadIdx.y];
   //printf("index %d\n",i);
   B[i] += A[i];
  }
 //}
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
 float *d_A, *d_B ;
 cudaHostAlloc((void**)&h_A,bytes,cudaHostAllocWriteCombined|cudaHostAllocMapped);
 cudaHostAlloc((void**)&h_B,bytes,cudaHostAllocWriteCombined|cudaHostAllocMapped);
 cudaHostAlloc((void**)&h_dC,bytes,cudaHostAllocWriteCombined);
 //cudaHostGetDevicePointer( &d_A, h_A, 0 );
 //cudaHostGetDevicePointer( &d_B, h_B, 0 );
 //cudaHostGetDevicePointer( &d_C, h_dC, 0 );
 // init matrices with random data
 //initData( h_A, noElems ) ; initData( h_B, noElems ) ;
 initDataA(h_A, nx, ny);
 initDataB(h_B, nx, ny);
 initDataA(h_hA, nx, ny);
 initDataB(h_hB, nx, ny);
 // alloc memory dev-side
 cudaMalloc( (void **) &d_A, bytes ) ;
 cudaMalloc( (void **) &d_B, bytes ) ;
 cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
 double timeStampA = getTimeStamp() ;
 //transfer data to dev
 //cudaMemcpy( d_A, h_A, bytes, cudaMemcpyHostToDevice ) ;
 //cudaMemcpy( d_B, h_B, bytes, cudaMemcpyHostToDevice ) ;
 //printf("pA is %d and pB is %d\n",pitchA,pitchB);
 //cudaMemcpy2D( d_A, pitchA, h_A,ny*sizeof(float),ny*sizeof(float),nx,cudaMemcpyHostToDevice ) ;
 //cudaMemcpy2D( d_B, pitchB, h_B,ny*sizeof(float),ny*sizeof(float),nx,cudaMemcpyHostToDevice ) ;
 // note that the transfers would be twice as fast if h_A and h_B
 // matrices are pinned
 //double timeStampB = getTimeStamp() ;
 //double timeStampC = getTimeStamp() ;
 //double timeStampD = getTimeStamp() ;

 // invoke Kernel
 dim3 block( 32, 32 ) ; // you will want to configure this
 //int block = 64;
 //int grid = (noElems + block-1)/block;
 int grid = ((noElems+3)/4/NUM_STREAMS + block.x*block.y-1)/(block.x*block.y);
 //int grid = ((noElems/NUM_STREAMS+3)/4 + block.x*block.y-1)/(block.x*block.y);
 //int grid = (((pitchA/4*nx*sizeof(float))+3)/4 + block.x*block.y-1)/(block.x*block.y);
 //dim3 grid( (nx + block.x-1)/block.x, (ny + block.y-1)/block.y ) ;
 //cudaDeviceProp GPUprop;
 //cudaGetDeviceProperties(&GPUprop,0);
 //printf("sharedmemperblk is %d\n",GPUprop.sharedMemPerBlock);
 //printf("maxgridsize x is %d\n",GPUprop.maxGridSize[0]);
 //printf("noelems is %d\n",noElems);
 //printf("prev num is %d\n",noElems/NUM_STREAMS);
 //printf("align num is %d\n",noElems/NUM_STREAMS-(noElems/NUM_STREAMS)%8);
 int align_idx = noElems/NUM_STREAMS-(noElems/NUM_STREAMS)%8;
 //printf("grid is %d\n",grid);
 //printf("gridx is %d and grid y is %d\n",grid.x,grid.y);

 //f_addmat<<<grid, block>>>( d_A, d_B, nx, ny/*, pitchA/(sizeof(float))*/ ) ;
 //cudaDeviceSynchronize() ;

 cudaStream_t stream[NUM_STREAMS+1];
 for (int i = 1; i < NUM_STREAMS+1; i++){
  cudaStreamCreate(&(stream[i]));
 }
 int i;
 for(i = 1; i < NUM_STREAMS; i++){
  cudaMemcpyAsync(&d_A[(i-1)*align_idx],&h_A[(i-1)*align_idx],align_idx*sizeof(float),cudaMemcpyHostToDevice,stream[i]);
  cudaMemcpyAsync(&d_B[(i-1)*align_idx],&h_B[(i-1)*align_idx],align_idx*sizeof(float),cudaMemcpyHostToDevice,stream[i]);
  cudaStreamAddCallback(stream[i],myCallBackB,(void*)&i,0);
  //printf("index is %d, num is %d\n",(i-1)*nx*ny/NUM_STREAMS,nx*ny/NUM_STREAMS );
  f_addmat<<<grid, block, 0, stream[i]>>>( d_A+(i-1)*align_idx, d_B+(i-1)*align_idx,align_idx) ;
  cudaStreamAddCallback(stream[i],myCallBackC,(void*)&i,0);
  cudaMemcpyAsync(&h_dC[(i-1)*align_idx],&d_B[(i-1)*align_idx],align_idx*sizeof(float),cudaMemcpyDeviceToHost,stream[i]);
  cudaStreamAddCallback(stream[i],myCallBackD,(void*)&i,0);
 }
 grid =((noElems-(NUM_STREAMS-1)*align_idx+3)/4+ block.x*block.y-1)/(block.x*block.y);
 //printf("grid final is %d\n",grid);
 //printf("index is %d, num is %d\n",(NUM_STREAMS-1)*nx*ny/NUM_STREAMS,nx*ny-(NUM_STREAMS-1)*nx*ny/NUM_STREAMS);
 cudaMemcpyAsync(&d_A[(NUM_STREAMS-1)*align_idx],&h_A[(NUM_STREAMS-1)*align_idx],(noElems-(NUM_STREAMS-1)*align_idx)*sizeof(float),cudaMemcpyHostToDevice,stream[NUM_STREAMS]);
 cudaMemcpyAsync(&d_B[(NUM_STREAMS-1)*align_idx],&h_B[(NUM_STREAMS-1)*align_idx],(noElems-(NUM_STREAMS-1)*align_idx)*sizeof(float),cudaMemcpyHostToDevice,stream[NUM_STREAMS]);
 cudaStreamAddCallback(stream[i],myCallBackB,(void*)&i,0);
 f_addmat<<<grid, block, 0, stream[NUM_STREAMS]>>>( d_A+(NUM_STREAMS-1)*align_idx, d_B+(NUM_STREAMS-1)*align_idx,noElems-(NUM_STREAMS-1)*align_idx) ;
 cudaStreamAddCallback(stream[i],myCallBackC,(void*)&i,0);
 cudaMemcpyAsync(&h_dC[(NUM_STREAMS-1)*align_idx],&d_B[(NUM_STREAMS-1)*align_idx],(noElems-(NUM_STREAMS-1)*align_idx)*sizeof(float),cudaMemcpyDeviceToHost,stream[NUM_STREAMS]);
 cudaStreamAddCallback(stream[i],myCallBackD,(void*)&i,0);
 for(int i = 1; i < NUM_STREAMS+1; i++){
  cudaStreamSynchronize(stream[i]);
 }
 //f_addmat<<<grid, block>>>( d_A, d_B, nx*ny/*, pitchA/(sizeof(float))*/ ) ;
 //cudaDeviceSynchronize() ;
 //copy data back
 //cudaMemcpy( h_dC, d_B, bytes, cudaMemcpyDeviceToHost ) ;
 //cudaMemcpy2D( h_ddC, pitchB, d_B,ny*sizeof(float),ny*sizeof(float),nx,cudaMemcpyDeviceToHost ) ;

 // check result
 h_addmat( h_hA, h_hB, h_hC, nx, ny ) ;

 //for(int i = 0; i < nx; i++){
 // for(int j = 0; j < pitchC/4; j++){
 //  if(j<ny){
 //   h_dC[i*ny+j] = h_ddC[i*pitchC/4+j];
 //  }
 // }
 //} 
 // print out results
 if(!memcmp(h_hC,h_dC,nx*ny*sizeof(float))){
 //if(1){
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
  printf("Error: Results not matched.\n");
 }
 // free GPU resources
 cudaFreeHost( h_A ) ; cudaFreeHost( h_B ) ; cudaFreeHost( h_dC ) ;
 cudaDeviceReset() ;
}
