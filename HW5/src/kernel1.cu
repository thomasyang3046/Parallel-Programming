#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
__global__ void mandelKernel(float lowerX, float lowerY, float stepX, float stepY, int *gpu_data, int resX, int resY, int maxIterations) {
    // To avoid error caused by the floating number, use the following pseudo code
    //
    // float x = lowerX + thisX * stepX;
    // float y = lowerY + thisY * stepY;
    int current_x = blockIdx.x * blockDim.x + threadIdx.x;
	int current_y = blockIdx.y * blockDim.y + threadIdx.y;
    float c_re=lowerX + current_x * stepX;
    float c_im=lowerY + current_y * stepY;
    float z_re = c_re;
    float z_im = c_im;
    int i;
    for (i = 0; i < maxIterations; ++i)
	{
		if (z_re * z_re + z_im * z_im > 4.f)
		break;

		float new_re = z_re * z_re - z_im * z_im;
		float new_im = 2.f * z_re * z_im;
		z_re = c_re + new_re;
		z_im = c_im + new_im;
	}
    int index=current_y*resX+current_x;
    gpu_data[index]=i;
}

// Host front-end function that allocates the memory and launches the GPU kernel
void hostFE (float upperX, float upperY, float lowerX, float lowerY, int* img, int resX, int resY, int maxIterations)
{
    float stepX = (upperX - lowerX) / resX;
    float stepY = (upperY - lowerY) / resY;

	dim3 block(16, 16);
	dim3 grid(resX/16.0, resY/16.0);

	int *cpu_data = (int*)malloc(resX*resY*sizeof(int));
    int *gpu_data;
    cudaMalloc(&gpu_data, resX*resY*sizeof(int));
	
	mandelKernel <<< grid, block >>> (lowerX, lowerY, stepX, stepY, gpu_data, resX, resY, maxIterations);
	
	cudaMemcpy(cpu_data, gpu_data, resX*resY*sizeof(int), cudaMemcpyDeviceToHost);
	memcpy(img, cpu_data, resX*resY*sizeof(int));

	free(cpu_data);
	cudaFree(gpu_data);
}
