//Udacity HW 4
//Radix Sorting

#include "utils.h"
#include <thrust/host_vector.h>

/* Red Eye Removal
   ===============
   
   For this assignment we are implementing red eye removal.  This is
   accomplished by first creating a score for every pixel that tells us how
   likely it is to be a red eye pixel.  We have already done this for you - you
   are receiving the scores and need to sort them in ascending order so that we
   know which pixels to alter to remove the red eye.

   Note: ascending order == smallest to largest

   Each score is associated with a position, when you sort the scores, you must
   also move the positions accordingly.

   Implementing Parallel Radix Sort with CUDA
   ==========================================

   The basic idea is to construct a histogram on each pass of how many of each
   "digit" there are.   Then we scan this histogram so that we know where to put
   the output of each digit.  For example, the first 1 must come after all the
   0s so we have to know how many 0s there are to be able to start moving 1s
   into the correct position.

   1) Histogram of the number of occurrences of each digit
   2) Exclusive Prefix Sum of Histogram
   3) Determine relative offset of each digit
        For example [0 0 1 1 0 0 1]
                ->  [0 1 0 1 2 3 2]
   4) Combine the results of steps 2 & 3 to determine the final
      output location for each element and move it there

   LSB Radix sort is an out-of-place sort and you will need to ping-pong values
   between the input and output buffers we have provided.  Make sure the final
   sorted results end up in the output buffer!  Hint: You may need to do a copy
   at the end.

 */

__global__ void Scan(
    unsigned int* d_inputVals,
    unsigned int* d_output,
    unsigned int  expectValue,
    int           n
)
{
    int global_index_1d = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (global_index_1d < n)
    {
        unsigned int output_value = d_inputVals[global_index_1d] & 1;

        d_output[global_index_1d] = output_value;
    }
}

#define NUM_GROUP_THREAD 256
__global__ void PrefixExclusiveScan(unsigned int* d_input, unsigned int* d_groupSum, const size_t n)
{
	int globalIndex = blockDim.x * blockIdx.x + threadIdx.x;
    
    __shared__ float SharedInput[NUM_GROUP_THREAD];
    SharedInput[threadIdx.x] = globalIndex < n ? d_input[globalIndex] : 0;

    __syncthreads();

    for (int Step = 1; Step < NUM_GROUP_THREAD; Step <<= 1)
    {
        int PrevThreadIdx = (int)threadIdx.x - Step;
        if (PrevThreadIdx >= 0)
        {
            SharedInput[threadIdx.x] = SharedInput[threadIdx.x] + SharedInput[PrevThreadIdx];
        }
        __syncthreads();
    }
    __syncthreads();

    if (threadIdx.x == 0)
    {
		d_groupSum[blockIdx.x] = SharedInput[NUM_GROUP_THREAD - 1];
        for (int i = 0; i < NUM_GROUP_THREAD; i++)
        {
            const int finalIndex = globalIndex + i;
            if (finalIndex < n)
            {
				d_input[finalIndex] = SharedInput[i];
            }
        }
    }
}

__global__ void PrefixGroupSum(unsigned int* const d_groupSum, const size_t numGroup)
{
    int globalIndex = blockDim.x * blockIdx.x + threadIdx.x;

    __shared__ float SharedGroupSum[NUM_GROUP_THREAD];

    SharedGroupSum[threadIdx.x] = globalIndex < numGroup ? d_groupSum[globalIndex] : 0;

    __syncthreads();

    for (int Step = 1; Step < NUM_GROUP_THREAD; Step <<= 1)
    {
        int PrevThreadIdx = (int)threadIdx.x - Step;
        if (PrevThreadIdx >= 0)
        {
            SharedGroupSum[threadIdx.x] = SharedGroupSum[threadIdx.x] + SharedGroupSum[PrevThreadIdx];
        }

        __syncthreads();
    }
    __syncthreads();


    if (threadIdx.x == 0)
    {
        for (int i = 0; i < NUM_GROUP_THREAD; i++)
        {
            const int finalIndex = globalIndex + i;
            if (finalIndex < numGroup)
            {
                d_groupSum[finalIndex] = SharedGroupSum[i];
            }
        }
    }
}

__global__ void CompositeFinalPrefixSum(unsigned int* const d_prefixGroupSum, unsigned int* d_prefixSum, unsigned int* d_input, const size_t numValues)
{
    int globalIndex = blockDim.x * blockIdx.x + threadIdx.x;

    if (globalIndex >= numValues)
    {
        return;
    }
    unsigned int PrevBlockSum = blockIdx.x > 0 ? d_prefixGroupSum[blockIdx.x - 1] : 0;

    d_input[globalIndex] = d_prefixSum[globalIndex] - d_input[globalIndex] + PrevBlockSum;
}

void PrefixSum(unsigned int* d_inputVals,
	unsigned int* d_output,
	const size_t numElems)
{
	size_t blockSizeX = NUM_GROUP_THREAD;
	size_t gridX = (numElems + blockSizeX - 1) / blockSizeX;

    unsigned int* d_groupSum;
    checkCudaErrors(cudaMalloc(&d_groupSum, sizeof(unsigned int) * gridX));
    checkCudaErrors(cudaMemset(d_groupSum, 0, sizeof(unsigned int) * gridX));

    unsigned int* d_prefixSumResult;
	checkCudaErrors(cudaMalloc(&d_prefixSumResult, sizeof(unsigned int) * gridX));
    checkCudaErrors(cudaMemcpy(d_prefixSumResult, d_inputVals, sizeof(unsigned int) * numElems, cudaMemcpyDeviceToDevice));

    checkCudaErrors(cudaMemcpy(d_output, d_inputVals, sizeof(unsigned int) * numElems, cudaMemcpyDeviceToDevice));
    {
        const dim3 blockSize(NUM_GROUP_THREAD, 1, 1);
        const dim3 gridSize(gridX, 1, 1);
        PrefixExclusiveScan << <gridSize, blockSize >> > (d_prefixSumResult, d_groupSum, numElems);
    }
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
    {
        size_t groupSumGridX = (gridX + blockSizeX - 1) / blockSizeX;
        const dim3 blockSize(NUM_GROUP_THREAD, 1, 1);
        const dim3 gridSize(groupSumGridX, 1, 1);
        PrefixGroupSum<<<gridSize, blockSize>>> (d_groupSum, gridX);
	}
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
	{
		const dim3 blockSize(NUM_GROUP_THREAD, 1, 1);
		const dim3 gridSize(gridX, 1, 1);
        CompositeFinalPrefixSum << <gridSize, blockSize >> > (d_groupSum, d_prefixSumResult, d_output, numElems);
	}
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    checkCudaErrors(cudaFree(d_groupSum));
}

void your_sort(unsigned int* const d_inputVals,
               unsigned int* const d_inputPos,
               unsigned int* const d_outputVals,
               unsigned int* const d_outputPos,
               const size_t numElems)
{ 
    //TODO
    //PUT YOUR SORT HERE
    for (unsigned int i = 0; i < 32; i++)
    {
		unsigned int* d_scanResult;

    }
}
