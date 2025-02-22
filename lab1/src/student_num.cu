#include <sys/time.h>
#include <stdio.h>
#include <cuda_runtime.h>
#define NUM_STREAMS 4 
//For time log by callback function
double timeStampB=0;
double timeStampC=0;
double timeStampD=0;
double timeKernal=0;
// time stamp function in seconds
double getTimeStamp() {
 struct timeval tv ;
 gettimeofday( &tv, NULL ) ;
 return (double) tv.tv_usec/1000000 + tv.tv_sec ;
}
//The three callback functions are used to called when finishing memcpyasyc and kernal completion
void myCallBackB(cudaStream_t stream,cudaError_t status, void*  userData ){
 timeStampB = getTimeStamp();
}
void myCallBackC(cudaStream_t stream,cudaError_t status, void*  userData ){
 timeStampC = getTimeStamp();
 //Through looking nvvp graph, the kernal executation is non-overlap and totally spread out. So I got each timeStampC-timeStampB for each kernal and take a sum
 timeKernal += timeStampC-timeStampB;
}
void myCallBackD(cudaStream_t stream,cudaError_t status, void*  userData ){
 timeStampD=getTimeStamp();
}
//initDataA function to init matrix A with selected values
void initDataA(float* data, int nx, int ny){ 
 int i,j;
 for(i = 0; i < nx; i++){
  for(j = 0; j < ny; j++){
   data[i*ny + j] = (float) (i+j)/3.0;
  }
 }
}
//initDataB function to init matrix B with selected values
void initDataB(float* data, int nx, int ny){ 
 int i,j;
 for(i = 0; i < nx; i++){
  for(j = 0; j < ny; j++){
   data[i*ny + j] = (float)3.14*(i+j);
  }
 }
}
//for debug: print each element in matrix
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
__global__ void f_addmat( float *A, float *B, int len){
 int ix = threadIdx.x;
 int iy = threadIdx.y*blockDim.x + blockIdx.x*blockDim.x*blockDim.y;
 int idx = iy + ix ;
 //for loop will be unrolled
 //stride loop access 4 elements, the stride is one grid size since the kernal's grid is 1/4 of total size per stream
 //assign B = B + A then I dont need access the third matrix C to save time.
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
 //the following allocate array pinned on memory to boost memcpy
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
 dim3 block( 32, 32 ) ;
 //I use 4 streams to pipeline memcpy h2d, kernal exec and memcpy d2h
 //grid determines the number of blocks. "/4" means shrink into 1/4 of total size for kernal executing 4 elements addition
 //"/NUM_STREAMS" means assign blocks to 4 streams uniformly
 int grid = ((noElems+3)/4/NUM_STREAMS + block.x*block.y-1)/(block.x*block.y);
 //align_idx is to align 4 data in 32-byte data access for each kernal executing
 int align_idx = noElems/NUM_STREAMS-(noElems/NUM_STREAMS)%8;

 //stream creation
 cudaStream_t stream[NUM_STREAMS+1];
 for (int i = 1; i < NUM_STREAMS+1; i++){
  cudaStreamCreate(&(stream[i]));
 }
 int i;
 for(i = 1; i < NUM_STREAMS; i++){
  //async memcpy for A and B
  cudaMemcpyAsync(&d_A[(i-1)*align_idx],&h_A[(i-1)*align_idx],align_idx*sizeof(float),cudaMemcpyHostToDevice,stream[i]);
  cudaMemcpyAsync(&d_B[(i-1)*align_idx],&h_B[(i-1)*align_idx],align_idx*sizeof(float),cudaMemcpyHostToDevice,stream[i]);
  //add callback to get timestamp update when each of memcpy per stream. I can get the last completion of data copying
  cudaStreamAddCallback(stream[i],myCallBackB,(void*)&i,0);
  //kernal invoked
  f_addmat<<<grid, block, 0, stream[i]>>>( d_A+(i-1)*align_idx, d_B+(i-1)*align_idx,align_idx) ;
  //add callback to get lastest stamp when finishing kernal per stream
  cudaStreamAddCallback(stream[i],myCallBackC,(void*)&i,0);
  //async memcpy back to host
  cudaMemcpyAsync(&h_dC[(i-1)*align_idx],&d_B[(i-1)*align_idx],align_idx*sizeof(float),cudaMemcpyDeviceToHost,stream[i]);
  //add callback to get lastest timestamp when finishing data copying
  cudaStreamAddCallback(stream[i],myCallBackD,(void*)&i,0);
 }
 //Here is to run the last stream. It is out of loop since the size of remaing data is different from aligned data size
 grid =((noElems-(NUM_STREAMS-1)*align_idx+3)/4+ block.x*block.y-1)/(block.x*block.y);
 cudaMemcpyAsync(&d_A[(NUM_STREAMS-1)*align_idx],&h_A[(NUM_STREAMS-1)*align_idx],(noElems-(NUM_STREAMS-1)*align_idx)*sizeof(float),cudaMemcpyHostToDevice,stream[NUM_STREAMS]);
 cudaMemcpyAsync(&d_B[(NUM_STREAMS-1)*align_idx],&h_B[(NUM_STREAMS-1)*align_idx],(noElems-(NUM_STREAMS-1)*align_idx)*sizeof(float),cudaMemcpyHostToDevice,stream[NUM_STREAMS]);
 cudaStreamAddCallback(stream[i],myCallBackB,(void*)&i,0);
 f_addmat<<<grid, block, 0, stream[NUM_STREAMS]>>>( d_A+(NUM_STREAMS-1)*align_idx, d_B+(NUM_STREAMS-1)*align_idx,noElems-(NUM_STREAMS-1)*align_idx) ;
 cudaStreamAddCallback(stream[i],myCallBackC,(void*)&i,0);
 cudaMemcpyAsync(&h_dC[(NUM_STREAMS-1)*align_idx],&d_B[(NUM_STREAMS-1)*align_idx],(noElems-(NUM_STREAMS-1)*align_idx)*sizeof(float),cudaMemcpyDeviceToHost,stream[NUM_STREAMS]);
 cudaStreamAddCallback(stream[i],myCallBackD,(void*)&i,0);

 //sync all streams and done
 for(int i = 1; i < NUM_STREAMS+1; i++){
  cudaStreamSynchronize(stream[i]);
 }

 // check result
 h_addmat( h_hA, h_hB, h_hC, nx, ny ) ;

 // print out results
 if(!memcmp(h_hC,h_dC,nx*ny*sizeof(float))){//results compare
  printf("%.6f %.6f %.6f %.6f\n", timeStampD-timeStampA, timeStampB-timeStampA, timeKernal, timeStampD-timeStampC);
 }else{
  //for debug print
  //debugPrint(h_hC, nx, ny);
  //debugPrint(h_dC, nx, ny);
  printf("Error: Results not matched.\n");
 }
 // free GPU resources
 cudaFreeHost( h_A ) ; cudaFreeHost( h_B ) ; cudaFreeHost( h_dC ) ;
 cudaDeviceReset() ;
}
