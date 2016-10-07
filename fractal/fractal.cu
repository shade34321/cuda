/*
* Shade Alabsa
* Fractal example
* CS 7172
* Edward Jung
* HW 3
* Most is taken from the example in the book with little changes.
*/
#include <stdio.h>
#include "cpu_bitmap.h"

#define DIM 1000

struct cuComplex {
    float r;
    float i;
    __device__ cuComplex(float a, float b): r(a), i(b) {}

    __device__ float magnitude2(void) {
        return r * r + i * i;
    }

    __device__ cuComplex operator+(const cuComplex& a) {
        return cuComplex(r+a.r, i+a.i);
    }
};

/*
 These error functions is ripped from there example. I didn't want to include
 their entire file nor did I want to just ignore all the errors. 
*/
static void HandleError( cudaError_t err, const char *file, int line) {
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
        exit(EXIT_FAILURE);
    }
}

#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__))

__device__ int julia(int x, int y) {
    const float scale = 1.65;
    float jx = scale * (float)(DIM/2 - x)/(DIM/2);
    float jy = scale * (float)(DIM/2 -y)/(DIM/2);

    cuComplex c(-0.8, 0.156);
    cuComplex a(jx, jy);

    int i = 0;

    for (i = 0; i<200;i++) {
        a = a*a+c;
        if (a.magnitude2() > 1000)
            return 0;
    }

    return 1;
}

__global__ void kernel(unsigned char *ptr) {
    int x = blockIdx.x;
    int y = blockIdx.y;
    int offset = x + y * gridDim.x;

    int juliaValue = julia(x, y);
    ptr[offset*4 + 0] = 255 * juliaValue;
    ptr[offset*4 + 1] = 0;
    ptr[offset*4 + 2] =  0;
    ptr[offset*4 + 3] = 255;
}

int main(void) {
    CPUBitmap bitmap(DIM, DIM);
    unsigned char *dev_bitmap;

    HANDLE_ERROR(cudaMalloc((void**)&dev_bitmap, bitmap.image_size()));

    dim3 grid(DIM, DIM);
    kernel<<<grid,1>>>(dev_bitmap);

    HANDLE_ERROR(cudaMemcpy(bitmap.get_ptr(), dev_bitmap, bitmap.image_size(),
    cudaMemcpyDeviceToHost));

    bitmap.display_and_exit();

    HANDLE_ERROR(cudaFree(dev_bitmap));

    return 0;
}
