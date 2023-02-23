#include <stdio.h>
#include <stdlib.h>
#include "hostFE.h"
#include "helper.h"
#include <cuda.h>
__global__ void conv(float *inputImage, float *outputImage, float *filter,
                     const int imageHeight, const int imageWidth, const int filterWidth)
{
    int halffilterSize = filterWidth / 2;
    int current_y = blockIdx.y * blockDim.y + threadIdx.y;
    int current_x = blockIdx.x * blockDim.x + threadIdx.x;
    int k, l;
    float sum = 0;

    for (k = -halffilterSize;k <= halffilterSize;k++) 
    {
        if (current_y + k >= 0 && current_y + k < imageHeight)
            for (l = -halffilterSize; l <= halffilterSize; l++)
                if (current_x + l >= 0 && current_x + l < imageWidth)
                    sum += inputImage[(current_y + k) * imageWidth + current_x + l] *filter[(k + halffilterSize) * filterWidth + l + halffilterSize];
    }
    outputImage[current_y * imageWidth + current_x] = sum;
}

void hostFE(int filterWidth, float *filter, int imageHeight, int imageWidth,
            float *inputImage, float *outputImage, cl_device_id *device,
            cl_context *context, cl_program *program)
{
    int filterSize = filterWidth * filterWidth * sizeof(float);
    int imageSize = imageHeight * imageWidth * sizeof(float);

    float *inputbuffer;
    float *outputbuffer;
    float *filterbuffer;
    cudaMalloc(&inputbuffer, imageSize);
    cudaMalloc(&outputbuffer, imageSize);
    cudaMalloc(&filterbuffer, filterSize);

    cudaMemcpy(filterbuffer, filter, filterSize, cudaMemcpyHostToDevice);
    cudaMemcpy(inputbuffer, inputImage, imageSize, cudaMemcpyHostToDevice);

    dim3 block(20, 20);
    dim3 grid(imageWidth / 20, imageHeight / 20);
    conv<<<grid, block>>>(inputbuffer, outputbuffer, filterbuffer, imageHeight, imageWidth, filterWidth);

    cudaMemcpy(outputImage, outputbuffer, imageSize, cudaMemcpyDeviceToHost);

    cudaFree(filterbuffer);
    cudaFree(inputbuffer);
    cudaFree(outputbuffer);
}