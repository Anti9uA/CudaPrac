#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <cuda.h>

using namespace std;

int* vec1, * vec2;  // 호스트의 벡터
int* gpuVec, * cpuVec;  // GPU와 CPU의 결과값을 담을 벡터

__global__ void vecAddGPU(int* VEC1, int* VEC2, int* RESULT) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    RESULT[i] = VEC1[i] + VEC2[i];
}

void vecAddCPU(int* VEC1, int* VEC2, int* RESULT, int N) {
    for (int i = 0; i < N; i++) {
        RESULT[i] = VEC1[i] * VEC2[i];
    }
}

int main(int argc, char** argv)
{
    cout << "Vector Addition Start!!" << endl;
    int n = 100000000;
    int nBytes = n * sizeof(int);
    int block_size, block_no;
    vec1 = (int*)malloc(nBytes);
    vec2 = (int*)malloc(nBytes);
    gpuVec = (int*)malloc(nBytes);
    cpuVec = (int*)malloc(nBytes);

    int* vec1_Cuda;
    int* vec2_Cuda, 
    int* gpuVec_Cuda;
    block_size = 4;
    block_no = n / block_size;
    dim3 dimBlock(block_size, 1, 1);
    dim3 dimGrid(block_no, 1, 1);

    // 인덱스 값만큼 벡터에 값 할당
    for (int i = 0; i < n; i++) {
        vec1[i] = i;
        vec2[i] = i;
    }

    // cudaMalloc으로 글로벌 메모리 할당 
    cout << "Allocating to global memory..." << endl;
    cudaMalloc((void**)&vec1_Cuda, n * sizeof(int));
    cudaMalloc((void**)&vec2_Cuda, n * sizeof(int));
    cudaMalloc((void**)&gpuVec_Cuda, n * sizeof(int));

    // cudaMemcpy()로 CPU(vec1,vec2)에서 GPU(vec1_Cuda,vec2_Cuda)로 값 복사
    cout << "Copying to Device..." << endl;
    cudaMemcpy(vec1_Cuda, vec1, n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(vec2_Cuda, vec2, n * sizeof(int), cudaMemcpyHostToDevice);
    
    // GPU 작업 시작!
    cout << "GPU Start!!" << endl;
    clock_t start_gpu = clock();
    cout << "GPU running..." << endl;
    vecAddGPU <<<block_no, block_size >>> (vec1_Cuda, vec2_Cuda, gpuVec_Cuda);
    cudaThreadSynchronize();
    clock_t end_gpu = clock();
    double time_gpu = (double)(end_gpu - start_gpu) / CLOCKS_PER_SEC;

    // 디바이스(GPU)의 결과값들을 다시 호스트(CPU)로 복사
    cudaMemcpy(gpuVec, gpuVec_Cuda, n * sizeof(int), cudaMemcpyDeviceToHost);
    cout << "GPU time >> " << time_gpu << endl;

    // CPU 작업 시작!
    cout << "\nCPU Start!!" << endl;
    clock_t start_cpu = clock();
    cout << "CPU running..." << endl;
    vecAddCPU(vec1, vec2, cpuVec, n);
    clock_t end_cpu = clock();
    double time_cpu = (double)(end_cpu - start_cpu) / CLOCKS_PER_SEC;
    cout << "CPU time >> "<< time_cpu << endl;

    // 메모리 해제
    cudaFree(vec1_Cuda);
    cudaFree(vec2_Cuda);
    cudaFree(gpuVec_Cuda);
    return 0;
}