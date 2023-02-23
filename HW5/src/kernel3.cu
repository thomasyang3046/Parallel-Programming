#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
__global__ void mandelKernel(float lowerX, float lowerY, float stepX, float stepY, int *gpu_data, int resX, int resY, int maxIterations,int scale) {
    // To avoid error caused by the floating number, use the following pseudo code
    //
    // float x = lowerX + thisX * stepX;
    // float y = lowerY + thisY * stepY;
    int current_x = (blockIdx.x * blockDim.x + threadIdx.x)*scale;
	int current_y = (blockIdx.y * blockDim.y + threadIdx.y)*scale;
    for(int j=current_y;j<current_y+scale;j++)
        for(int i=current_x;i<current_x+scale;i++)
        {
            float c_re=lowerX + i * stepX;
            float c_im=lowerY + j * stepY;
            float z_re = c_re;
            float z_im = c_im;
            int k;
            for (k = 0; k < maxIterations; ++k)
            {
                if (z_re * z_re + z_im * z_im > 4.f)
                break;

                float new_re = z_re * z_re - z_im * z_im;
                float new_im = 2.f * z_re * z_im;
                z_re = c_re + new_re;
                z_im = c_im + new_im;
            }
            int index=j*resX+i;
            gpu_data[index]=k;
        }
}

// Host front-end function that allocates the memory and launches the GPU kernel
void hostFE (float upperX, float upperY, float lowerX, float lowerY, int* img, int resX, int resY, int maxIterations)
{
    float stepX = (upperX - lowerX) / resX;
    float stepY = (upperY - lowerY) / resY;
    int scale=2;
	dim3 block(16/scale, 16/scale);
	dim3 grid(resX/16.0, resY/16.0);
    size_t pitch;

	int *cpu_data;
    int *gpu_data;
    cudaHostAlloc(&cpu_data, resX*resY*sizeof(int), cudaHostAllocMapped);
    cudaMallocPitch(&gpu_data, &pitch, resX * sizeof(int), resY);
	
	mandelKernel <<< grid, block >>> (lowerX, lowerY, stepX, stepY, gpu_data, resX, resY, maxIterations,scale);
	cudaDeviceSynchronize();
    
	cudaMemcpy(cpu_data, gpu_data, resX*resY*sizeof(int), cudaMemcpyDeviceToHost);
	memcpy(img, cpu_data, resX*resY*sizeof(int));

	cudaFreeHost(cpu_data);
	cudaFree(gpu_data);
}
