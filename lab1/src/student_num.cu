#include <sys/time.h>
#include <stdio.h>
#include <cuda_runtime.h>
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
 int ix = threadIdx.x;
 int iy = threadIdx.y*blockDim.x + blockIdx.x*blockDim.x*blockDim.y;
 int idx = iy + ix ;
 #pragma unroll
 for(int i = idx; i < len; i+=gridDim.x*blockDim.x*blockDim.y){
  B[i] += A[i];
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
 float *d_A, *d_B ;
 cudaHostAlloc((void**)&h_A,bytes,cudaHostAllocWriteCombined|cudaHostAllocMapped);
 cudaHostAlloc((void**)&h_B,bytes,cudaHostAllocWriteCombined|cudaHostAllocMapped);
 cudaHostAlloc((void**)&h_dC,bytes,cudaHostAllocWriteCombined);
 // init matrices with random data
 initDataA(h_A, nx, ny);
 initDataB(h_B, nx, ny);
 initDataA(h_hA, nx, ny);
 initDataB(h_hB, nx, ny);
 // alloc memory dev-side
 cudaMalloc( (void **) &d_A, bytes ) ;
 cudaMalloc( (void **) &d_B, bytes ) ;
 cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
 double timeStampA = getTimeStamp() ;

 // invoke Kernel
 dim3 block( 32, 32 ) ; // you will want to configure this
 int grid = ((noElems+3)/4/NUM_STREAMS + block.x*block.y-1)/(block.x*block.y);
 int align_idx = noElems/NUM_STREAMS-(noElems/NUM_STREAMS)%8;

 cudaStream_t stream[NUM_STREAMS+1];
 for (int i = 1; i < NUM_STREAMS+1; i++){
  cudaStreamCreate(&(stream[i]));
 }
 int i;
 for(i = 1; i < NUM_STREAMS; i++){
  cudaMemcpyAsync(&d_A[(i-1)*align_idx],&h_A[(i-1)*align_idx],align_idx*sizeof(float),cudaMemcpyHostToDevice,stream[i]);
  cudaMemcpyAsync(&d_B[(i-1)*align_idx],&h_B[(i-1)*align_idx],align_idx*sizeof(float),cudaMemcpyHostToDevice,stream[i]);
  cudaStreamAddCallback(stream[i],myCallBackB,(void*)&i,0);
  f_addmat<<<grid, block, 0, stream[i]>>>( d_A+(i-1)*align_idx, d_B+(i-1)*align_idx,align_idx) ;
  cudaStreamAddCallback(stream[i],myCallBackC,(void*)&i,0);
  cudaMemcpyAsync(&h_dC[(i-1)*align_idx],&d_B[(i-1)*align_idx],align_idx*sizeof(float),cudaMemcpyDeviceToHost,stream[i]);
  cudaStreamAddCallback(stream[i],myCallBackD,(void*)&i,0);
 }
 grid =((noElems-(NUM_STREAMS-1)*align_idx+3)/4+ block.x*block.y-1)/(block.x*block.y);
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

 // check result
 h_addmat( h_hA, h_hB, h_hC, nx, ny ) ;

 // print out results
 if(!memcmp(h_hC,h_dC,nx*ny*sizeof(float))){
  //debugPrint(h_hC, nx, ny);
  //debugPrint(h_dC, nx, ny);
  //FILE* fptr;
  //fptr = fopen("time.log","a");
  //fprintf(fptr,"%dX%d %.6f %.6f %.6f %.6f\n", nx, ny, timeStampD-timeStampA, timeStampB-timeStampA, timeStampC-timeStampB, timeStampD-timeStampC);
  //fclose(fptr);
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
