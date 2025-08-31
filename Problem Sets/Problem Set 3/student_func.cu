/* Udacity Homework 3
   HDR Tone-mapping

  Background HDR
  ==============

  A High Dynamic Range (HDR) image contains a wider variation of intensity
  and color than is allowed by the RGB format with 1 byte per channel that we
  have used in the previous assignment.  

  To store this extra information we use single precision floating point for
  each channel.  This allows for an extremely wide range of intensity values.

  In the image for this assignment, the inside of church with light coming in
  through stained glass windows, the raw input floating point values for the
  channels range from 0 to 275.  But the mean is .41 and 98% of the values are
  less than 3!  This means that certain areas (the windows) are extremely bright
  compared to everywhere else.  If we linearly map this [0-275] range into the
  [0-255] range that we have been using then most values will be mapped to zero!
  The only thing we will be able to see are the very brightest areas - the
  windows - everything else will appear pitch black.

  The problem is that although we have cameras capable of recording the wide
  range of intensity that exists in the real world our monitors are not capable
  of displaying them.  Our eyes are also quite capable of observing a much wider
  range of intensities than our image formats / monitors are capable of
  displaying.

  Tone-mapping is a process that transforms the intensities in the image so that
  the brightest values aren't nearly so far away from the mean.  That way when
  we transform the values into [0-255] we can actually see the entire image.
  There are many ways to perform this process and it is as much an art as a
  science - there is no single "right" answer.  In this homework we will
  implement one possible technique.

  Background Chrominance-Luminance
  ================================

  The RGB space that we have been using to represent images can be thought of as
  one possible set of axes spanning a three dimensional space of color.  We
  sometimes choose other axes to represent this space because they make certain
  operations more convenient.

  Another possible way of representing a color image is to separate the color
  information (chromaticity) from the brightness information.  There are
  multiple different methods for doing this - a common one during the analog
  television days was known as Chrominance-Luminance or YUV.

  We choose to represent the image in this way so that we can remap only the
  intensity channel and then recombine the new intensity values with the color
  information to form the final image.

  Old TV signals used to be transmitted in this way so that black & white
  televisions could display the luminance channel while color televisions would
  display all three of the channels.
  

  Tone-mapping
  ============

  In this assignment we are going to transform the luminance channel (actually
  the log of the luminance, but this is unimportant for the parts of the
  algorithm that you will be implementing) by compressing its range to [0, 1].
  To do this we need the cumulative distribution of the luminance values.

  Example
  -------

  input : [2 4 3 3 1 7 4 5 7 0 9 4 3 2]
  min / max / range: 0 / 9 / 9

  histo with 3 bins: [4 7 3]

  cdf : [4 11 14]


  Your task is to calculate this cumulative distribution by following these
  steps.

*/

#include "utils.h"
#define NUM_MIN_MAX_ROW 32
#define NUM_PREFIX_ROW 32

__global__ void calculate_min_max(int numCols, int numRows, int numGroupX, const float* const d_logLuminance,
    float* min_logLumGroups, float* max_logLumGroups)
{
    // ȫ���߳�ID
    int PosX = blockDim.x * blockIdx.x + threadIdx.x;
    int PosY = blockDim.y * blockIdx.y + threadIdx.y;

    if (PosX >= numCols || PosY >= numRows)
    {
        return;
    }
    __shared__ float SharedMinLogLum[NUM_MIN_MAX_ROW];
    __shared__ float SharedMaxLogLum[NUM_MIN_MAX_ROW];
    //if (blockIdx.x == 0 && blockIdx.y == 0)
    //{
    //    printf("threadIdx: %d, value: %f\n", threadIdx.x, d_logLuminance[PosX + PosY * numCols]);
    //}
	SharedMinLogLum[threadIdx.x] = d_logLuminance[PosX + PosY * numCols];
    SharedMaxLogLum[threadIdx.x] = d_logLuminance[PosX + PosY * numCols];

    __syncthreads();

    for (int Step = 1; Step < NUM_MIN_MAX_ROW; Step <<= 1)
    {
        int PrevThreadIdx = max(0, (int)threadIdx.x - Step);
        {
			SharedMinLogLum[threadIdx.x] = min(SharedMinLogLum[threadIdx.x], SharedMinLogLum[PrevThreadIdx]);
			SharedMaxLogLum[threadIdx.x] = max(SharedMaxLogLum[threadIdx.x], SharedMaxLogLum[PrevThreadIdx]);
        }
		__syncthreads();
    }

    //if (blockIdx.x == 0 && blockIdx.y == 0)
    //{
    //    printf("Shared threadIdx: %d, value: %f\n", threadIdx.x, SharedMinLogLum[threadIdx.x]);
    //}
	if (threadIdx.x == NUM_MIN_MAX_ROW - 1)
	{
        min_logLumGroups[blockIdx.x + blockIdx.y * numGroupX] = SharedMinLogLum[threadIdx.x];
        max_logLumGroups[blockIdx.x + blockIdx.y * numGroupX] = SharedMaxLogLum[threadIdx.x];
	}
}

__global__ void calc_bin(int numCols, int numRows, const float* const d_logLuminance,
    unsigned int* const d_cdf,
    float min_logLumGroups, float lumRange, const size_t numBins)
{
    // ȫ���߳�ID
    int PosX = blockDim.x * blockIdx.x + threadIdx.x;
    int PosY = blockDim.y * blockIdx.y + threadIdx.y;

    if (PosX >= numCols || PosY >= numRows)
    {
        return;
    }
    __shared__ float SharedLogLuminances[NUM_MIN_MAX_ROW];
    __shared__ unsigned int SharedBin[NUM_MIN_MAX_ROW];
	SharedLogLuminances[threadIdx.x] = d_logLuminance[PosX + PosY * numCols];
    SharedBin[threadIdx.x] = -1;

    __syncthreads();

    unsigned int bin = (SharedLogLuminances[threadIdx.x] - min_logLumGroups) / lumRange * numBins;
    SharedBin[threadIdx.x] = bin;
    __syncthreads();
 
   
    if (threadIdx.x == NUM_MIN_MAX_ROW - 1)
    {
        for (int i = 0; i < NUM_MIN_MAX_ROW; i++)
        {
            if (SharedBin[threadIdx.x] != -1)
            {
                atomicAdd(d_cdf + SharedBin[i], 1);
            }
        }
    }
}

__global__ void calc_prefix(unsigned int* const d_cdf, unsigned int* const d_GridSum, const size_t numBins)
{
    // ȫ���߳�ID
    int PosX = blockDim.x * blockIdx.x + threadIdx.x;

    if (PosX >= numBins)
    {
        return;
    }
    __shared__ float SharedCDF[NUM_PREFIX_ROW];

    SharedCDF[threadIdx.x] = d_cdf[PosX];

    __syncthreads();

    for (int Step = 1; Step < NUM_PREFIX_ROW; Step <<= 1)
    {
        int PrevThreadIdx = (int)threadIdx.x - Step;
        if (PrevThreadIdx >= 0)
        {
            SharedCDF[threadIdx.x] = SharedCDF[threadIdx.x] + SharedCDF[PrevThreadIdx];
        }
        __syncthreads();
    }
    __syncthreads();


    if (threadIdx.x == 0)
    {
        d_GridSum[blockIdx.x] = SharedCDF[NUM_PREFIX_ROW - 1];
        for (int i = 0; i < NUM_PREFIX_ROW; i++)
        {
            d_cdf[PosX + i] = SharedCDF[i];
        }
    }
}

__global__ void calc_grid_prefix(unsigned int* const d_GridSum, const size_t numBins)
{
    // ȫ���߳�ID
    int PosX = blockDim.x * blockIdx.x + threadIdx.x;

    if (PosX >= numBins)
    {
        return;
    }
    __shared__ float SharedGridSum[NUM_PREFIX_ROW];

    SharedGridSum[threadIdx.x] = d_GridSum[PosX];

    __syncthreads();

    for (int Step = 1; Step < NUM_PREFIX_ROW; Step <<= 1)
    {
        int PrevThreadIdx = (int)threadIdx.x - Step;
        if (PrevThreadIdx >= 0)
        {
            SharedGridSum[threadIdx.x] = SharedGridSum[threadIdx.x] + SharedGridSum[PrevThreadIdx];
        }

        __syncthreads();
    }
    __syncthreads();


    if (threadIdx.x == 0)
    {
        for (int i = 0; i < NUM_PREFIX_ROW; i++)
        {
            d_GridSum[PosX + i] = SharedGridSum[i];
        }
    }
}

__global__ void calc_final_result(unsigned int* const d_GridSum, unsigned int* d_bin, unsigned int* d_cdf, const size_t numBins)
{
    // ȫ���߳�ID
    int PosX = blockDim.x * blockIdx.x + threadIdx.x;

    if (PosX >= numBins)
    {
        return;
    }
    unsigned int PrevBlockSum = blockIdx.x > 0 ? d_GridSum[blockIdx.x - 1] : 0;

	d_cdf[PosX] = d_bin[PosX] - d_cdf[PosX] + PrevBlockSum;
    //d_cdf[PosX] = blockIdx.x;
}

void your_histogram_and_prefixsum(const float* const d_logLuminance,
                                  unsigned int* const d_cdf,
                                  float &min_logLum,
                                  float &max_logLum,
                                  const size_t numRows,
                                  const size_t numCols,
                                  const size_t numBins)
{
    //TODO
    /*Here are the steps you need to implement
    1) find the minimum and maximum value in the input logLuminance channel
        store in min_logLum and max_logLum
    2) subtract them to find the range
    3) generate a histogram of all the values in the logLuminance channel using
        the formula: bin = (lum[i] - lumMin) / lumRange * numBins
    4) Perform an exclusive scan (prefix sum) on the histogram to get
        the cumulative distribution of luminance values (this should go in the
        incoming d_cdf pointer which already has been allocated for you)       */

    {
        size_t gridX = std::ceil((float)numCols / NUM_MIN_MAX_ROW);
        size_t gridY = numRows;
        const dim3 blockSize(NUM_MIN_MAX_ROW, 1, 1);
        const dim3 gridSize(gridX, gridY, 1);
        float* min_logLumGroups, *max_logLumGroups;
        float* h_min_logLumGroups = new float[gridX * gridY];
        float* h_max_logLumGroups = new float[gridX * gridY];
        checkCudaErrors(cudaMalloc(&min_logLumGroups, sizeof(float) * gridX * gridY));
        checkCudaErrors(cudaMalloc(&max_logLumGroups, sizeof(float) * gridX * gridY));
        calculate_min_max << <gridSize, blockSize >> > (numCols, numRows, gridX, d_logLuminance, min_logLumGroups, max_logLumGroups);
        cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaMemcpy(h_min_logLumGroups, min_logLumGroups, sizeof(float) * gridX * gridY, cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(h_max_logLumGroups, max_logLumGroups, sizeof(float) * gridX * gridY, cudaMemcpyDeviceToHost));

        min_logLum = std::numeric_limits<float>::max();
        for (int i = 0; i < gridX * gridY; i++)
        {
			min_logLum = std::min(min_logLum, h_min_logLumGroups[i]);
        }
        max_logLum = std::numeric_limits<float>::lowest();
        //std::cout << "group0: min:" << h_min_logLumGroups[0] << " max:" << h_max_logLumGroups[0] << std::endl;
        //std::cout << "min_logLum: ";
        for (int i = 0; i < gridX * gridY; i++)
        {
			//std::cout << h_max_logLumGroups[i] << " ";
            max_logLum = std::max(max_logLum, h_max_logLumGroups[i]);
        }
        //std::cout << std::endl;
        //std::cout << "min_logLum: " << min_logLum << ", max_logLum: " << max_logLum << std::endl;
        checkCudaErrors(cudaFree(min_logLumGroups));
        checkCudaErrors(cudaFree(max_logLumGroups));
		delete[] h_min_logLumGroups;
        delete[] h_max_logLumGroups;
    }

    // ����bin�����ݣ��Ȳ�ֱ�Ӵ浽d_cdf��
    unsigned int* d_bin;
    {
        checkCudaErrors(cudaMalloc(&d_bin, sizeof(unsigned int) * numBins));
		checkCudaErrors(cudaMemset(d_bin, 0, sizeof(unsigned int) * numBins));
    }

    {
        float lumRange = max_logLum - min_logLum;
        size_t gridX = std::ceil((float)numCols / NUM_MIN_MAX_ROW);
        size_t gridY = numRows;
        const dim3 blockSize(NUM_MIN_MAX_ROW, 1, 1);
        const dim3 gridSize(gridX, gridY, 1);
		//unsigned int* h_debug_cdf = new unsigned int[numBins];
        calc_bin << <gridSize, blockSize >> > (numCols, numRows, d_logLuminance, d_bin, min_logLum, lumRange, numBins);
        cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
        //checkCudaErrors(cudaMemcpy(h_debug_cdf, d_bin, sizeof(unsigned int) * numBins, cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(d_cdf, d_bin, sizeof(unsigned int) * numBins, cudaMemcpyDeviceToDevice));
        //std::cout << "bin: ";
        //for (int i = 0; i < NUM_MIN_MAX_ROW; i++)
        //{
        //    std::cout << h_debug_cdf[i] << " ";
        //}
        //std::cout << std::endl;
        //delete h_debug_cdf;
    }

    
    {
        size_t gridX = std::ceil((float)numBins / NUM_PREFIX_ROW);
        const dim3 blockSize(NUM_PREFIX_ROW, 1, 1);
        const dim3 gridSize(gridX, 1, 1);

        unsigned int* d_GridSum;
        checkCudaErrors(cudaMalloc(&d_GridSum, sizeof(unsigned int) * gridX));
        checkCudaErrors(cudaMemset(d_GridSum, 0, sizeof(unsigned int) * gridX));
        //unsigned int* h_debug_prefix_bin = new unsigned int[numBins];
        //unsigned int* h_debug_prefix_bin_sum = new unsigned int[gridX];
        calc_prefix << <gridSize, blockSize >> > (d_bin, d_GridSum, numBins);
        cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
  //      checkCudaErrors(cudaMemcpy(h_debug_prefix_bin, d_bin, sizeof(unsigned int) * numBins, cudaMemcpyDeviceToHost));
  //      checkCudaErrors(cudaMemcpy(h_debug_prefix_bin_sum, d_GridSum, sizeof(unsigned int) * gridX, cudaMemcpyDeviceToHost));
		//std::cout << "prefix bin: ";
  //      for (int i = 0; i < numBins; i++)
  //      {
  //          std::cout << h_debug_prefix_bin[i] << " ";
  //      }
  //        delete h_debug_prefix_bin;
  //      std::cout << "\nbin Sum: ";
  //      for (int i = 0; i < gridX; i++)
  //      {
  //          std::cout << h_debug_prefix_bin_sum[i] << " ";
  //      }
  //      std::cout << std::endl;
        {
            // ����grid sum��ǰ׺��
            dim3 blockSumSize(gridX, 1, 1);
            dim3 gridSumSize(1, 1, 1);
            calc_grid_prefix << <gridSumSize, blockSumSize >> > (d_GridSum, gridX);
            cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
            //checkCudaErrors(cudaMemcpy(h_debug_prefix_bin_sum, d_GridSum, sizeof(unsigned int) * gridX, cudaMemcpyDeviceToHost));
            //std::cout << "\nbin Sum[P]: ";
            //for (int i = 0; i < gridX; i++)
            //{
            //    std::cout << h_debug_prefix_bin_sum[i] << " ";
            //}
            //std::cout << std::endl;
        }


        {
            //unsigned int* h_debug_cdf = new unsigned int[numBins];
            calc_final_result << <gridSize, blockSize >> > (d_GridSum, d_bin, d_cdf, numBins);
            cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
            //checkCudaErrors(cudaMemcpy(h_debug_cdf, d_cdf, sizeof(unsigned int)* numBins, cudaMemcpyDeviceToHost));
            //std::cout << "prefix cdf: ";
            //for (int i = 0; i < numBins; i++)
            //{
            //    std::cout << h_debug_cdf[i] << " ";
            //}
            //std::cout << std::endl;
        }

        
    }

    checkCudaErrors(cudaFree(d_bin));
}
