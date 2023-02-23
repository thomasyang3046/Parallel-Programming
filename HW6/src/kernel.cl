__kernel void convolution(__global float *inputImage, __global float * outputImage, __constant float *filter,const  int imageWidth,const int imageHeight, const int filterWidth) 
{
    int halffilterSize = filterWidth / 2;
    float sum;
    int k, l,i = get_global_id(0),j = get_global_id(1);
    sum = 0;
    for (k = -halffilterSize; k <= halffilterSize; k++)
    {
        if (j + k >= 0 && j + k < imageHeight)
            for (l = -halffilterSize; l <= halffilterSize; l++)
            {
                if (i + l >= 0 && i + l < imageWidth)
                    sum += inputImage[(j + k) * imageWidth + i + l] * filter[(k + halffilterSize) * filterWidth + l + halffilterSize];
            }
    }
    outputImage[j * imageWidth + i] = sum;
}
