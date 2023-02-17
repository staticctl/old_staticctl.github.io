#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define PI 3.14159265358979323846


#include <stdio.h>

#include <math.h>
#include <omp.h>
#include <ctime>


__global__ void dft2d_kernel(float* in_array, cuComplex* out_array, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        cuComplex sum = make_cuComplex(0.0f, 0.0f);
        for (int i = 0; i < width; i++) {
            for (int j = 0; j < height; j++) {
                float angle = 2.0f * M_PI * (float)(i * x + j * y) / (float)(width * height);
                cuComplex factor = make_cuComplex(cosf(angle), -sinf(angle));
                cuComplex input_value = make_cuComplex(in_array[j * width + i], 0.0f);
                cuComplex product = cuCmulf(input_value, factor);
                sum = cuCaddf(sum, product);
            }
        }
        out_array[y * width + x] = sum;
    }
}
