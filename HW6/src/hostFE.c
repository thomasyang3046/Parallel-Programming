#include <stdio.h>
#include <stdlib.h>
#include "hostFE.h"
#include "helper.h"

void hostFE(int filterWidth, float *filter, int imageHeight, int imageWidth,
            float *inputImage, float *outputImage, cl_device_id *device,
            cl_context *context, cl_program *program)
{
    cl_int status;
    int filterSize = filterWidth * filterWidth*sizeof(float);
    int imageSize = imageHeight * imageWidth * sizeof(float);

    cl_command_queue queue=clCreateCommandQueue(*context, *device, 0, NULL);

    cl_mem filterbuffer = clCreateBuffer(*context, CL_MEM_USE_HOST_PTR, filterSize,filter,NULL);
    cl_mem inputbuffer = clCreateBuffer(*context, CL_MEM_USE_HOST_PTR, imageSize, inputImage, NULL);
    cl_mem outputbuffer = clCreateBuffer(*context, CL_MEM_WRITE_ONLY, imageSize, NULL, NULL);

    cl_kernel mykernel = clCreateKernel(*program, "convolution", NULL);
    // set kernel arg
    clSetKernelArg(mykernel, 0, sizeof(cl_mem), (void *)&inputbuffer);
    clSetKernelArg(mykernel, 1, sizeof(cl_mem), (void *)&outputbuffer);
    clSetKernelArg(mykernel, 2, sizeof(cl_mem), (void *)&filterbuffer);
    clSetKernelArg(mykernel, 3, sizeof(cl_int), (void *)&imageWidth);
    clSetKernelArg(mykernel, 4, sizeof(cl_int), (void *)&imageHeight);
    clSetKernelArg(mykernel, 5, sizeof(cl_int), (void *)&filterWidth);

    //Set local and global workgroup sizes 
    size_t localws[2] = {20,20} ;
    size_t globalws[2] = {imageWidth, imageHeight};//divisible by 20
    clEnqueueNDRangeKernel(queue, mykernel, 2, 0, globalws, localws, 0, NULL, NULL);
    clEnqueueReadBuffer(queue, outputbuffer, CL_TRUE, 0, imageSize, outputImage, 0, NULL, NULL);
    // release mem
    /*clReleaseCommandQueue(queue);
    clReleaseMemObject(filterbuffer);
    clReleaseMemObject(inputbuffer);
    clReleaseMemObject(outputbuffer);
    clReleaseKernel(mykernel);*/
}