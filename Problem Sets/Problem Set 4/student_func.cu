//Udacity HW 4
//Radix Sorting

#include "utils.h"
#include <thrust/host_vector.h>
#include <iostream>
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

template<typename Type>
void PrintDeviceMemory(size_t numElems, Type* deviceVals, const std::string& name, size_t offset = 0)
{
    std::vector<Type> h_inputVals;
    h_inputVals.resize(numElems);
    checkCudaErrors(cudaMemcpy(h_inputVals.data(), deviceVals, sizeof(Type) * numElems, cudaMemcpyDeviceToHost));
    std::cout << name << ":";
    for (int i = offset; i < numElems; i++)
    {
        std::cout << h_inputVals[i] << ", ";
    }
    std::cout << std::endl;
}

__global__ void Scan(
    unsigned int* d_inputVals,
    unsigned int* d_inputPos,
    unsigned int* d_result0,
    unsigned int* d_result1,
    unsigned int  mask,
    unsigned int  shift,
    int           n
)
{
    int global_index_1d = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (global_index_1d < n)
    {
        if (d_inputVals[d_inputPos[global_index_1d]] & mask)
        {
            atomicAdd(&d_result1[global_index_1d], 1);
        }
        else
        {
            atomicAdd(&d_result0[global_index_1d], 1);
        }
    }
}

#define NUM_GROUP_THREAD 512
__global__ void PrefixExclusiveScan(unsigned int* d_input, unsigned int* d_groupSum, const size_t n)
{
	int globalIndex = blockDim.x * blockIdx.x + threadIdx.x;

    __shared__ float SharedInput[NUM_GROUP_THREAD];
    SharedInput[threadIdx.x] = globalIndex < n ? d_input[globalIndex] : 0;

    __syncthreads();

    for (int Step = 1; Step < NUM_GROUP_THREAD; Step <<= 1)
    {
        unsigned int temp = SharedInput[threadIdx.x];
        __syncthreads();

        int PrevThreadIdx = (int)threadIdx.x - Step;
        if (PrevThreadIdx >= 0)
        {
            temp += SharedInput[PrevThreadIdx];
        }
        __syncthreads();

        SharedInput[threadIdx.x] = temp;
    }
    __syncthreads();

    if (threadIdx.x == 0)
    {
        atomicAdd(&d_groupSum[blockIdx.x], SharedInput[NUM_GROUP_THREAD - 1]);
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
        unsigned int temp = SharedGroupSum[threadIdx.x];
        __syncthreads();

        int PrevThreadIdx = (int)threadIdx.x - Step;
        if (PrevThreadIdx >= 0)
        {
            temp += SharedGroupSum[PrevThreadIdx];
        }
        __syncthreads();

        SharedGroupSum[threadIdx.x] = temp;
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

__global__ void CompositeFinalPrefixSum(unsigned int* const d_prefixGroupSum, unsigned int* d_prefixSum, unsigned int* d_input, unsigned int* d_sum, const size_t numValues)
{
    int globalIndex = blockDim.x * blockIdx.x + threadIdx.x;

    if (globalIndex >= numValues)
    {
        return;
    }
    unsigned int PrevBlockSum = blockIdx.x > 0 ? d_prefixGroupSum[blockIdx.x - 1] : 0;

    d_input[globalIndex] = d_prefixSum[globalIndex] - d_input[globalIndex] + PrevBlockSum;

    if (globalIndex == numValues - 1)
    {
        *d_sum = d_prefixSum[globalIndex] + PrevBlockSum;
    }
}

void PrefixSum(unsigned int* d_inputVals,
	unsigned int* d_output,
    unsigned int* d_sum,
	const size_t numElems)
{
	size_t blockSizeX = NUM_GROUP_THREAD;
	size_t gridX = (numElems + blockSizeX - 1) / blockSizeX;

    unsigned int* d_groupSum;
    checkCudaErrors(cudaMalloc(&d_groupSum, sizeof(unsigned int) * gridX));
    checkCudaErrors(cudaMemset(d_groupSum, 0, sizeof(unsigned int) * gridX));

    unsigned int* d_prefixSumResult;
	checkCudaErrors(cudaMalloc(&d_prefixSumResult, sizeof(unsigned int) * numElems));
    checkCudaErrors(cudaMemcpy(d_prefixSumResult, d_inputVals, sizeof(unsigned int) * numElems, cudaMemcpyDeviceToDevice));

    checkCudaErrors(cudaMemcpy(d_output, d_inputVals, sizeof(unsigned int) * numElems, cudaMemcpyDeviceToDevice));
    {
        const dim3 blockSize(blockSizeX, 1, 1);
        const dim3 gridSize(gridX, 1, 1);
        PrefixExclusiveScan<<<gridSize, blockSize>>>(d_prefixSumResult, d_groupSum, numElems);
        //cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
    }

    {
        size_t groupSumGridX = (gridX + blockSizeX - 1) / blockSizeX;
        const dim3 blockSize(blockSizeX, 1, 1);
        const dim3 gridSize(groupSumGridX, 1, 1);
        PrefixGroupSum<<<gridSize, blockSize>>> (d_groupSum, gridX);
        //cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
	}
    
	{
		const dim3 blockSize(blockSizeX, 1, 1);
		const dim3 gridSize(gridX, 1, 1);
        CompositeFinalPrefixSum << <gridSize, blockSize >> > (d_groupSum, d_prefixSumResult, d_output, d_sum, numElems);
        //cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
	}

    checkCudaErrors(cudaFree(d_groupSum));
    checkCudaErrors(cudaFree(d_prefixSumResult));
}

__global__ void Sort(
    unsigned int* d_inputVals,
    unsigned int* d_inputPos, 
    unsigned int* d_prefixSum0,
    unsigned int* d_prefixSum1,
    unsigned int* d_outputPos,
    unsigned int* sum0,
    unsigned int  mask,
    unsigned int  shift,
    const size_t  numValues)
{
    int global_index_1d = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (global_index_1d < numValues)
    {
        unsigned int inputPos = d_inputPos[global_index_1d];
        if (d_inputVals[inputPos] & mask)
        {
            d_outputPos[*sum0 + d_prefixSum1[global_index_1d]] = inputPos;
        }
        else
        {
            d_outputPos[d_prefixSum0[global_index_1d]] = inputPos;
        }
    }
}

__global__ void FillValues(
    unsigned int* d_inputVals,
    unsigned int* d_inputPos,
    unsigned int* d_outputVals,
    const size_t  numValues)
{
    int global_index_1d = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (global_index_1d < numValues)
    {
        d_outputVals[global_index_1d] = d_inputVals[d_inputPos[global_index_1d]];
    }
}

void your_sort(unsigned int* const d_inputVals,
               unsigned int* const d_inputPos,
               unsigned int* const d_outputVals,
               unsigned int* const d_outputPos,
               size_t numElems)
{ 
    //TODO
    //PUT YOUR SORT HERE

    size_t blockSizeX = NUM_GROUP_THREAD;
    size_t gridX = (numElems + blockSizeX - 1) / blockSizeX;
    const dim3 blockSize(blockSizeX, 1, 1);
    const dim3 gridSize(gridX, 1, 1);

    unsigned int* d_scanResult0, *d_scanResult1;
    checkCudaErrors(cudaMalloc(&d_scanResult0, sizeof(unsigned int) * numElems));
    checkCudaErrors(cudaMalloc(&d_scanResult1, sizeof(unsigned int) * numElems));

    unsigned int* d_prefixSumScanResult0, *d_prefixSumScanResult1;
    checkCudaErrors(cudaMalloc(&d_prefixSumScanResult0, sizeof(unsigned int) * numElems));
    checkCudaErrors(cudaMalloc(&d_prefixSumScanResult1, sizeof(unsigned int) * numElems));

    unsigned int* d_inputIndex;
    checkCudaErrors(cudaMalloc(&d_inputIndex, sizeof(unsigned int) * numElems));
    checkCudaErrors(cudaMemcpy(d_inputIndex, d_inputPos, sizeof(unsigned int) * numElems, cudaMemcpyDeviceToDevice));

    unsigned int* d_outputIndex;
    checkCudaErrors(cudaMalloc(&d_outputIndex, sizeof(unsigned int) * numElems));

    unsigned int* d_sum0;
    checkCudaErrors(cudaMalloc(&d_sum0, sizeof(unsigned int)));

    unsigned int* d_sum1;
    checkCudaErrors(cudaMalloc(&d_sum1, sizeof(unsigned int)));

    for (unsigned int i = 0; i < sizeof(unsigned int) * 8; i++)
    {
        unsigned int shift = i;
        unsigned int mask = 1 << shift;
        checkCudaErrors(cudaMemset(d_scanResult0, 0, sizeof(unsigned int) * numElems));
        checkCudaErrors(cudaMemset(d_scanResult1, 0, sizeof(unsigned int) * numElems));
        Scan<<<gridSize, blockSize>>>(d_inputVals, d_inputIndex, d_scanResult0, d_scanResult1, mask, shift, numElems);
        //cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
        PrefixSum(d_scanResult0, d_prefixSumScanResult0, d_sum0, numElems);
        PrefixSum(d_scanResult1, d_prefixSumScanResult1, d_sum1, numElems);
        Sort<<<gridSize, blockSize>>>(d_inputVals, d_inputIndex, d_prefixSumScanResult0, d_prefixSumScanResult1, d_outputIndex, d_sum0, mask, shift, numElems);
        //cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
        std::swap(d_inputIndex, d_outputIndex);
    }

    FillValues<<<gridSize, blockSize>>>(d_inputVals, d_inputIndex, d_outputVals, numElems);
    checkCudaErrors(cudaMemcpy(d_outputPos, d_inputIndex, sizeof(unsigned int) * numElems, cudaMemcpyDeviceToDevice));
}
