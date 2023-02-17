#ifndef FRESHMAN_H
#define FRESHMAN_H

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <math.h>
#include <time.h>

#ifdef _WIN32
#	include <windows.h>
#else
#	include <sys/time.h>
#endif
#ifdef _WIN32

//宏定义
#define CHECK(call)\
{\
  const cudaError_t error=call;\
  if(error!=cudaSuccess)\
  {\
      printf("ERROR: %s:%d,",__FILE__,__LINE__);\
      printf("code:%d,reason:%s\n",error,cudaGetErrorString(error));\
      exit(1);\
  }\
}

int gettimeofday(struct timeval* tp, void* tzp)
{
    time_t clock;
    struct tm tm;
    SYSTEMTIME wtm;
    GetLocalTime(&wtm);
    tm.tm_year = wtm.wYear - 1900;
    tm.tm_mon = wtm.wMonth - 1;
    tm.tm_mday = wtm.wDay;
    tm.tm_hour = wtm.wHour;
    tm.tm_min = wtm.wMinute;
    tm.tm_sec = wtm.wSecond;
    tm.tm_isdst = -1;
    clock = mktime(&tm);
    tp->tv_sec = clock;
    tp->tv_usec = wtm.wMilliseconds * 1000;
    return (0);
}
#endif

//计时函数
double cpuSecond()
{
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return((double)tp.tv_sec + (double)tp.tv_usec * 1e-6);
}

//产生浮点型随机数
void initialData(float* ip, int size)
{
    //使用srand和rand产生随机数
    time_t t;
    srand((unsigned)time(&t));  //https://www.runoob.com/w3cnote/cpp-rand-srand.html
    for (int i = 0; i < size; i++)
    {
        ip[i] = (float)(rand() & 0xffff) / 1000.0f;
    }
}

//产生整型随机数
void initialData_int(int* ip, int size)
{
    time_t t;
    srand((unsigned)time(&t));
    for (int i = 0; i < size; i++)
    {
        ip[i] = int(rand() & 0xff);
    }
}

//
void printMatrix(float* C, const int nx, const int ny)
{
    float* ic = C;
    printf("Matrix<%d,%d>:", ny, nx);
    for (int i = 0; i < ny; i++)
    {
        for (int j = 0; j < nx; j++)
        {
            printf("%6f ", C[j]);
        }
        ic += nx;
        printf("\n");
    }
}

//初始化设备
void initDevice(int devNum)
{
    int dev = devNum;

    //打印设备信息
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("Using device %d: %s\n", dev, deviceProp.name);
    printf("Maximun number of thread per block: %d\n", deviceProp.maxThreadsPerBlock);
    printf("Maximun size of each dimension of a block: %d x %d x %d\n", deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
    printf("Maximun size of each dimension of a grid: %d x %d x %d\n", deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
    printf("Maximu memory pitch %lu bytes\n", deviceProp.memPitch);
    printf("Total amount of global memory: %.2f GBytes (%llu bytes)\n", (float)deviceProp.totalGlobalMem / pow(1024.0, 3), deviceProp.totalGlobalMem);

    //设置设备ID
    CHECK(cudaSetDevice(dev));

}

//检查计算结果
void checkResult(float* hostRef, float* gpuRef, const int N)
{
    long epsilon = 1.0E-8;
    for (int i = 0; i < N; i++)
    {
        if (abs(long(hostRef[i] - gpuRef[i])) > epsilon)
        {
            printf("Results don\'t match!\n");
            printf("%f(hostRef[%d] )!= %f(gpuRef[%d])\n", hostRef[i], i, gpuRef[i], i);
            return;
        }
    }
    printf("Check result success!\n");
}
#endif//FRESHMAN_H
