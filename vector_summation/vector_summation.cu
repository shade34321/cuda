/*
* Shade Alabsa
* Vector Summation Example from Cuda By Example
* CS 7172
* Edward Jung
* HW 3
* Most is taken from the example in the book with little changes.
*/
#include <stdio.h>

#define N (32 * 1024)

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

/*
    Performs the summation of the vectors on the GPU
*/

__global__ void add(float *a, float *b, float *c) {
    int tid = blockIdx.x;
    while(tid < N) {
        c[tid] = a[tid] + b[tid];
        tid += gridDim.x;
    }
}

void printArray(float *c) {
    for (int i = 0; i < N; i++) {
        printf("%f ", c[i]);

        if (i % 10 == 0) {
            printf("\n");
        }
    }

    printf("\n");
}

int main(void) {
    float *a, *b, *c;
    float *dev_a, *dev_b, *dev_c;

    a = (float*)malloc( N * sizeof(float) );
    b = (float*)malloc( N * sizeof(float) );
    c = (float*)malloc( N * sizeof(float) );

    
    HANDLE_ERROR( cudaMalloc((void**)&dev_a, N * sizeof(float) ));
    HANDLE_ERROR( cudaMalloc((void**)&dev_b, N * sizeof(float) ));
    HANDLE_ERROR( cudaMalloc((void**)&dev_c, N * sizeof(float) ));

    for (int i = 0; i<N; i++) {
        a[i] = i * 3.14;
        b[i] = 2 * i;
    }

    HANDLE_ERROR( cudaMemcpy(dev_a, a, N * sizeof(float),
                                cudaMemcpyHostToDevice));
    HANDLE_ERROR( cudaMemcpy(dev_b, b, N * sizeof(float),
                                cudaMemcpyHostToDevice));
    
    add<<<512,1>>>(dev_a, dev_b, dev_c);

    HANDLE_ERROR(cudaMemcpy(c, dev_c, N * sizeof(float),
                                cudaMemcpyDeviceToHost));

    bool success = true;
    for (int i = 0; i < N; i++) {
        if (a[i] + b[i] != c[i]) {
            printf( "Error %f + %f != %f\n", a[i], b[i], c[i]);
            success = false;
        }
    }

    if (success) {
        printf("We did it!\n");
        printArray(c);
    }

    HANDLE_ERROR( cudaFree(dev_a));
    HANDLE_ERROR( cudaFree(dev_b));
    HANDLE_ERROR( cudaFree(dev_c));

    free(a);
    free(b);
    free(c);

    return 0;
}
