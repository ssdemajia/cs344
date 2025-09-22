/* Udacity HW5
   Histogramming for Speed

   The goal of this assignment is compute a histogram
   as fast as possible.  We have simplified the problem as much as
   possible to allow you to focus solely on the histogramming algorithm.

   The input values that you need to histogram are already the exact
   bins that need to be updated.  This is unlike in HW3 where you needed
   to compute the range of the data and then do:
   bin = (val - valMin) / valRange to determine the bin.

   Here the bin is just:
   bin = val

   so the serial histogram calculation looks like:
   for (i = 0; i < numElems; ++i)
     histo[val[i]]++;

   That's it!  Your job is to make it run as fast as possible!

   The values are normally distributed - you may take
   advantage of this fact in your implementation.

*/


#include "utils.h"
#include <vector>
#define NUM_BLOCK_X 256
#define NUM_BINS 1024
#define NUM_PARTS 1024


// 使用Atomic操作的直方图计算
__global__ void AtomicHistogram(const unsigned int* const vals, //INPUT
	unsigned int* const histo,      //OUPUT
	int numVals)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < numVals)
	{
		atomicAdd(&histo[vals[i]], 1);
	}
}

__global__ void LocalHistogram(const unsigned int* const vals, //INPUT
	unsigned int* const histo,      //OUPUT
	int numVals)
{
	__shared__ unsigned int localHisto[NUM_BINS];

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int tid = threadIdx.x;

	// Initialize shared memory histogram
	if (tid < NUM_BINS)
		localHisto[tid] = 0;
	__syncthreads();

	// Build local histogram
	if (i < numVals)
	{
		atomicAdd(&localHisto[vals[i]], 1);
	}

	__syncthreads();

	// Write back to global memory
	if (tid < NUM_BINS)
	{
		atomicAdd(&histo[blockIdx.x * NUM_BINS + tid], localHisto[tid]);
	}
}

__global__ void MergeHistogram(const unsigned int* const intermediateHisto, //INPUT
	unsigned int* const histo,      //OUPUT
	int numParts)
{
	int bin = blockIdx.x * blockDim.x + threadIdx.x;

	if (bin < NUM_BINS)
	{
		//printf("bin: %d,", bin);
		unsigned int sum = 0;
		for (int i = 0; i < numParts; ++i)
		{
			int pos = i * NUM_BINS + bin;
			sum += intermediateHisto[pos];
		}
		histo[bin] = sum;
	}
}

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

void computeHistogram(const unsigned int* const d_vals, //INPUT
                      unsigned int* const d_histo,      //OUTPUT
                      const unsigned int numBins,
                      const unsigned int numElems)
{
	size_t blockX = NUM_BLOCK_X;
	size_t gridX = (numElems + blockX - 1) / blockX;
	const dim3 blockSize(blockX, 1, 1);
	const dim3 gridSize(gridX, 1, 1);
	AtomicHistogram << <gridSize, blockSize >> > (d_vals, d_histo, numElems);
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
	//PrintDeviceMemory(NUM_BINS, d_histo, "Result0:");
}


void computeHistogram2(const unsigned int* const d_vals, //INPUT
	unsigned int* d_intermediate_histo,
	unsigned int* const d_histo,      //OUTPUT
	const unsigned int numBins,
	const unsigned int numElems)
{
	size_t gridX = (numElems + NUM_PARTS - 1) / NUM_PARTS;

	{
		const dim3 blockSize(NUM_PARTS, 1, 1);
		const dim3 gridSize(gridX, 1, 1);
		LocalHistogram << <gridSize, blockSize >> > (d_vals, d_intermediate_histo, numElems);
		cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
		//PrintDeviceMemory(NumIntermediate, d_intermediate_histo, "Intermediate:", (gridX - 1) * NUM_BINS);
	}

	{
		gridX = 32;
		const dim3 blockSize(gridX, 1, 1);
		const dim3 gridSize((numBins + gridX - 1) / gridX, 1, 1);
		MergeHistogram << <gridSize, blockSize >> > (d_intermediate_histo, d_histo, gridX);
		cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
	}
	//PrintDeviceMemory(NUM_BINS, d_histo, "Result1:");
	
}