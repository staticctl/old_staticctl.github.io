
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>


/*

__global__ void hello_world(void)
{
    printf("GPU: Hello world!\n");
}
int main(int argc, char** argv)
{
    printf("CPU: Hello world!\n");
    hello_world <<<1, 10 >>> ();
    cudaDeviceReset();//if no this line ,it can not output hello world from gpu
    return 0;
}
*/
/*
* https://github.com/Tony-Tan/CUDA_Freshman
* 3_sum_arrays
*/
#include <cuda_runtime.h>
#include <stdio.h>
//#include "freshman.h"

/*
void sumArrays(float* a, float* b, float* res, const int size)
{
    for (int i = 0; i < size; i += 4)
    {
        res[i] = a[i] + b[i];
        res[i + 1] = a[i + 1] + b[i + 1];
        res[i + 2] = a[i + 2] + b[i + 2];
        res[i + 3] = a[i + 3] + b[i + 3];
    }

__global__ void sumArraysGPU(float* a, float* b, float* res)
{
    int i = threadIdx.x;
    res[i] = a[i] + b[i];
}
*/
/*
int main(int argc, char** argv)
{
    int dev = 0;
    cudaSetDevice(dev);

    int nElem = 32;
    printf("Vector size:%d\n", nElem);
    int nByte = sizeof(float) * nElem;
    float* a_h = (float*)malloc(nByte);
    float* b_h = (float*)malloc(nByte);
    float* res_h = (float*)malloc(nByte);
    float* res_from_gpu_h = (float*)malloc(nByte);
    memset(res_h, 0, nByte);
    memset(res_from_gpu_h, 0, nByte);

    float* a_d, * b_d, * res_d;
    CHECK(cudaMalloc((float**)&a_d, nByte));
    CHECK(cudaMalloc((float**)&b_d, nByte));
    CHECK(cudaMalloc((float**)&res_d, nByte));

    // 产生浮点型随机数 initialData
    initialData(a_h, nElem);
    initialData(b_h, nElem);

    CHECK(cudaMemcpy(a_d, a_h, nByte, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(b_d, b_h, nByte, cudaMemcpyHostToDevice));

    dim3 block(nElem);
    dim3 grid(nElem / block.x);
    sumArraysGPU << <grid, block >> > (a_d, b_d, res_d);
    printf("Execution configuration<<<%d,%d>>>\n", block.x, grid.x);

    CHECK(cudaMemcpy(res_from_gpu_h, res_d, nByte, cudaMemcpyDeviceToHost));
    sumArrays(a_h, b_h, res_h, nElem);

    checkResult(res_h, res_from_gpu_h, nElem);
    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(res_d);

    free(a_h);
    free(b_h);
    free(res_h);
    free(res_from_gpu_h);

    return 0;
}
*/

/*
*1_check_dimension
*/
/*
#include <cuda_runtime.h>
#include <stdio.h>
__global__ void checkIndex(void)
{
    printf("threadIdx:(%d,%d,%d) blockIdx:(%d,%d,%d) blockDim:(%d,%d,%d)\
  gridDim(%d,%d,%d)\n", threadIdx.x, threadIdx.y, threadIdx.z,
        blockIdx.x, blockIdx.y, blockIdx.z, blockDim.x, blockDim.y, blockDim.z,
        gridDim.x, gridDim.y, gridDim.z);
}
int main(int argc, char** argv)
{
    int nElem = 6;
    dim3 block(3);
    dim3 grid((nElem + block.x - 1) / block.x);
    printf("grid.x %d grid.y %d grid.z %d\n", grid.x, grid.y, grid.z);
    printf("block.x %d block.y %d block.z %d\n", block.x, block.y, block.z);
    checkIndex << <grid, block >> > ();
    cudaDeviceReset();
    return 0;
}
*/


/*
*2_grid_block
*/
#include <cuda_runtime.h>
#include <stdio.h>
int main(int argc, char** argv)
{
    int nElem = 1024;
    dim3 block(1024);
    dim3 grid((nElem - 1) / block.x + 1);
    printf("grid.x %d block.x %d\n", grid.x, block.x);

    block.x = 512;
    grid.x = (nElem - 1) / block.x + 1;
    printf("grid.x %d block.x %d\n", grid.x, block.x);

    block.x = 256;
    grid.x = (nElem - 1) / block.x + 1;
    printf("grid.x %d block.x %d\n", grid.x, block.x);

    block.x = 128;
    grid.x = (nElem - 1) / block.x + 1;
    printf("grid.x %d block.x %d\n", grid.x, block.x);

    cudaDeviceReset();
    return 0;
}