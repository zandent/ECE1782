#include <sys/time.h>
#include <stdio.h>
#include<math.h>
//TODO for writing to file, will be deleted
//#include <stdlib.h>
#include <cuda_runtime.h>
#define NUM_STREAMS 16 
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
 for(i = 0; i < n; i++){
  for(j = 0; j < n; j++){
   for(k = 0; k < n; k++){
    if(a[i*n*n + j*n + k] != b[i*n*n+j*n+k]){
     //printf("%d,%d,%d expect %lf, actual %lf\n",i,j,k,a[i*n*n + j*n + k],b[i*n*n+j*n+k]);
     return false;
    }
   }
  }
 }
 return true;
}
double h_sum(float *data, int n){
 int i,j,k;
 double ret=0;
 for(i = 1; i < n-1; i++){
  for(j = 1; j < n-1; j++){
   for(k = 1; k < n-1; k++){
    ret += data[i*n*n + j*n + k]*(((i+j+k)%2)?1:-1);
   }
  }
 }
 return ret;
}
double h_rsum(float *data, int n){
 int i,j,k;
 double ret=0;
 for(i = 1; i < n-1; i++){
  for(j = 1; j < n-1; j++){
   for(k = 1; k < n-1; k++){
    ret += roundf(data[i*n*n + j*n + k]*100)/100*(((i+j+k)%2)?1:-1);
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
__global__ void kernal( float *a, float *b, int n, int height){
 extern __shared__ float sm[];
 int ix = threadIdx.x + 1;
 int iy = threadIdx.y + 1;
 int gx = threadIdx.x + 1 + blockIdx.x*blockDim.x;
 int gy = threadIdx.y + 1 + blockIdx.y*blockDim.y;
 float down,up,self;
 float l1,l2,l3,l4;
 if(gx<n-1&&gy<n-1){
  down = b[gx+gy*n];
  globalToShared(sm, b, 1, n, ix, iy, gx, gy);
  __syncthreads();
  self = sm[ix + iy*(blockDim.x+2)];
  l1 = sm[ix-1 + iy*(blockDim.x+2)];
  l2 = sm[ix+1 + iy*(blockDim.x+2)];
  l3 = sm[ix + (iy-1)*(blockDim.x+2)];
  l4 = sm[ix + (iy+1)*(blockDim.x+2)];
  __syncthreads();
  int layer;
  #pragma unroll
  for(layer = 2; layer < height; layer++){
   globalToShared(sm, b, layer, n, ix, iy, gx, gy);
   __syncthreads();
   up = sm[ix + iy*(blockDim.x+2)];
   a[gx + gy*n + (layer-1)*n*n] = 0.8*(down+up+l1+l2+l3+l4);
   down = self;
   self = up;
   l1 = sm[ix-1 + iy*(blockDim.x+2)];
   l2 = sm[ix+1 + iy*(blockDim.x+2)];
   l3 = sm[ix + (iy-1)*(blockDim.x+2)];
   l4 = sm[ix + (iy+1)*(blockDim.x+2)];
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
 //int pad_n = n + 32 - (n-2)%32;
 int noElems = n*n*n ;
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
 initData(h_B, n);
 memset(h_A, 0.0, bytes);
 memset(h_dA, 0.0, bytes);
 
 // alloc memory dev-side
 float *d_A, *d_B ;
 cudaMalloc( (void **) &d_A, bytes ) ;
 cudaMalloc( (void **) &d_B, bytes ) ;
 
 //debugPrint(h_B, n);
 // invoke Kernel
 dim3 block(32, 32);
 dim3 grid((n-2+block.x-1)/block.x,(n-2+block.y-1)/block.y);
 double timeStampA = getTimeStamp() ;
 double timeStampD = getTimeStamp() ;
 if(n>=250){
 //transfer data to dev
 //stream creation
 int batch_h = (n+NUM_STREAMS-1)/NUM_STREAMS;
 int batch_size = n*n*batch_h;
 int last_batch = noElems-(NUM_STREAMS-1)*batch_size;
 int b_size[NUM_STREAMS];
 b_size[0] = batch_h;
 b_size[NUM_STREAMS-1] = n-(NUM_STREAMS-1)*batch_h + 2;
 for(int k = 1; k < NUM_STREAMS-1; k++){
  b_size[k] = batch_h+2;
 }
 int offset[NUM_STREAMS];
 offset[0] = 0;
 for(int k = 1; k < NUM_STREAMS; k++){
  offset[k] = k*batch_h-2;
 }
 //for(int k = 0; k < NUM_STREAMS; k++){
 // printf("b_size %d is %d\n",k,b_size[k]);
 // printf("off %d is %d\n",k,offset[k]);
 //}

 timeStampA = getTimeStamp() ;
 cudaStream_t stream[NUM_STREAMS+1];
 for (int i = 1; i < NUM_STREAMS; i++){
  cudaStreamCreate(&(stream[i]));
  cudaMemcpyAsync(&d_B[(i-1)*batch_size],&h_B[(i-1)*batch_size],batch_size*sizeof(float),cudaMemcpyHostToDevice,stream[i]);
  kernal<<<grid,block,(1024+33*4)*sizeof(float)>>>(d_A+n*n*offset[i-1],d_B+n*n*offset[i-1],n,b_size[i-1]);
  cudaMemcpyAsync(&h_dA[n*n*(1+offset[i-1])],&d_A[n*n*(1+offset[i-1])],(b_size[i-1]-2)*n*n*sizeof(float),cudaMemcpyDeviceToHost,stream[i]);
 }
 cudaStreamCreate(&(stream[NUM_STREAMS]));
 cudaMemcpyAsync(&d_B[(NUM_STREAMS-1)*batch_size],&h_B[(NUM_STREAMS-1)*batch_size],last_batch*sizeof(float),cudaMemcpyHostToDevice,stream[NUM_STREAMS]);
 kernal<<<grid,block,(1024+33*4)*sizeof(float)>>>(d_A+n*n*offset[NUM_STREAMS-1],d_B+n*n*offset[NUM_STREAMS-1],n,b_size[NUM_STREAMS-1]);
 cudaMemcpyAsync(&h_dA[n*n*(1+offset[NUM_STREAMS-1])],&d_A[n*n*(1+offset[NUM_STREAMS-1])],(b_size[NUM_STREAMS-1]-2)*n*n*sizeof(float),cudaMemcpyDeviceToHost,stream[NUM_STREAMS]);

 //sync all streams and done
 for(int i = 1; i < NUM_STREAMS+1; i++){
  cudaStreamSynchronize(stream[i]);
 }
 timeStampD = getTimeStamp() ;
 }else{
 timeStampA = getTimeStamp() ;
 //transfer data to dev
 cudaMemcpy( d_B, h_B, bytes, cudaMemcpyHostToDevice ) ;

 //debugPrint(h_B, n);
 // invoke Kernel
 dim3 block(32, 32);
 dim3 grid((n-2+block.x-1)/block.x,(n-2+block.y-1)/block.y);
 kernal<<<grid,block,(1024+33*4)*sizeof(float)>>>(d_A,d_B,n,n);
 cudaDeviceSynchronize() ;
 //cudaDeviceProp GPUprop;
 //cudaGetDeviceProperties(&GPUprop,0);
 //printf("maxgridsize x is %d\n",GPUprop.maxGridSize[0]);

 //copy data back
 cudaMemcpy( h_dA, d_A, bytes, cudaMemcpyDeviceToHost ) ;
 timeStampD = getTimeStamp() ;
 }
 h_stencil(h_A,h_B,n);
 //h_dA = h_A;
 bool match = val(h_A,h_dA,n);
 //float h_Result = h_rsum(h_A,n);
 float h_dResult = h_sum(h_dA,n);
 
 // print out results
 //if(!memcmp(h_A,h_dA,n*n*n*sizeof(float))){
 if(match){
  //debugPrint(h_A, n);
  //debugPrint(h_dC, nx, ny);
  //FILE* fptr;
  //fptr = fopen("time.log","a");
  //fprintf(fptr,"%d: %lf %.6f\n", n, h_dResult, timeStampD-timeStampA); 
  //fclose(fptr);
  //printf("%lf %lf %d\n", h_dResult, h_Result, (int)round(timeStampD-timeStampA));
  printf("%lf %d\n", h_dResult, (int)round(timeStampD-timeStampA));
 }else{
  //debugPrint(h_A, n);
  //debugPrint(h_dA, n);
  //FILE* fptr;
  //fptr = fopen("time.log","a");
  //fprintf(fptr,"%d Error: function failed.\n", n);
  //fclose(fptr);
  printf("Error: function failed.\n");
 }
 
 // free GPU resources
 cudaFree(d_A);
 cudaFree(d_B);
 cudaDeviceReset();
}

