// Homework 2
// Image Blurring
//
// In this homework we are blurring an image. To do this, imagine that we have
// a square array of weight values. For each pixel in the image, imagine that we
// overlay this square array of weights on top of the image such that the center
// of the weight array is aligned with the current pixel. To compute a blurred
// pixel value, we multiply each pair of numbers that line up. In other words, we
// multiply each weight with the pixel underneath it. Finally, we add up all of the
// multiplied numbers and assign that value to our output for the current pixel.
// We repeat this process for all the pixels in the image.

// To help get you started, we have included some useful notes here.

//****************************************************************************

// For a color image that has multiple channels, we suggest separating
// the different color channels so that each color is stored contiguously
// instead of being interleaved. This will simplify your code.

// That is instead of RGBARGBARGBARGBA... we suggest transforming to three
// arrays (as in the previous homework we ignore the alpha channel again):
//  1) RRRRRRRR...
//  2) GGGGGGGG...
//  3) BBBBBBBB...
//
// The original layout is known an Array of Structures (AoS) whereas the
// format we are converting to is known as a Structure of Arrays (SoA).

// As a warm-up, we will ask you to write the kernel that performs this
// separation. You should then write the "meat" of the assignment,
// which is the kernel that performs the actual blur. We provide code that
// re-combines your blurred results for each color channel.

//****************************************************************************

// You must fill in the gaussian_blur kernel to perform the blurring of the
// inputChannel, using the array of weights, and put the result in the outputChannel.

// Here is an example of computing a blur, using a weighted average, for a single
// pixel in a small image.
//
// Array of weights:
//
//  0.0  0.2  0.0
//  0.2  0.2  0.2
//  0.0  0.2  0.0
//
// Image (note that we align the array of weights to the center of the box):
//
//    1  2  5  2  0  3
//       -------
//    3 |2  5  1| 6  0       0.0*2 + 0.2*5 + 0.0*1 +
//      |       |
//    4 |3  6  2| 1  4   ->  0.2*3 + 0.2*6 + 0.2*2 +   ->  3.2
//      |       |
//    0 |4  0  3| 4  2       0.0*4 + 0.2*0 + 0.0*3
//       -------
//    9  6  5  0  3  9
//
//         (1)                         (2)                 (3)
//
// A good starting place is to map each thread to a pixel as you have before.
// Then every thread can perform steps 2 and 3 in the diagram above
// completely independently of one another.

// Note that the array of weights is square, so its height is the same as its width.
// We refer to the array of weights as a filter, and we refer to its width with the
// variable filterWidth.

//****************************************************************************

// Your homework submission will be evaluated based on correctness and speed.
// We test each pixel against a reference solution. If any pixel differs by
// more than some small threshold value, the system will tell you that your
// solution is incorrect, and it will let you try again.

// Once you have gotten that working correctly, then you can think about using
// shared memory and having the threads cooperate to achieve better performance.

//****************************************************************************

// Also note that we've supplied a helpful debugging function called checkCudaErrors.
// You should wrap your allocation and copying statements like we've done in the
// code we're supplying you. Here is an example of the unsafe way to allocate
// memory on the GPU:
//
// cudaMalloc(&d_red, sizeof(unsigned char) * numRows * numCols);
//
// Here is an example of the safe way to do the same thing:
//
// checkCudaErrors(cudaMalloc(&d_red, sizeof(unsigned char) * numRows * numCols));
//
// Writing code the safe way requires slightly more typing, but is very helpful for
// catching mistakes. If you write code the unsafe way and you make a mistake, then
// any subsequent kernels won't compute anything, and it will be hard to figure out
// why. Writing code the safe way will inform you as soon as you make a mistake.

// Finally, remember to free the memory you allocate at the end of the function.

//****************************************************************************

#include "utils.h"

__global__
void gaussian_blur(const unsigned char* const inputChannel,
                   unsigned char* const outputChannel,
                   int numRows, int numCols,
                   const float* const filter, const int filterWidth)
{
  // TODO
  
  // NOTE: Be sure to compute any intermediate results in floating point
  // before storing the final result as unsigned char.

  // NOTE: Be careful not to try to access memory that is outside the bounds of
  // the image. You'll want code that performs the following check before accessing
  // GPU memory:
  //
  // if ( absolute_image_position_x >= numCols ||
  //      absolute_image_position_y >= numRows )
  // {
  //     return;
  // }
  
  // NOTE: If a thread's absolute position 2D position is within the image, but some of
  // its neighbors are outside the image, then you will need to be extra careful. Instead
  // of trying to read such a neighbor value from GPU memory (which won't work because
  // the value is out of bounds), you should explicitly clamp the neighbor values you read
  // to be within the bounds of the image. If this is not clear to you, then please refer
  // to sequential reference solution for the exact clamping semantics you should follow.

    // 全局线程ID
    int PosX = blockDim.x * blockIdx.x + threadIdx.x;
    int PosY = blockDim.y * blockIdx.y + threadIdx.y;

    if (PosX >= numCols || PosY >= numRows)
    {
        return;
    }

    float FinalValue = 0.0f;
    for (int i = 0; i < filterWidth; i++)
    {
        for (int j = 0; j < filterWidth; j++)
        {
            // Calculate the absolute position of the pixel in the image
            int absPosX = PosX + i - filterWidth / 2;
            int absPosY = PosY + j - filterWidth / 2;
            absPosX = min(max(absPosX, 0), numCols-1);
            absPosY = min(max(absPosY, 0), numRows-1);
            unsigned char inputValue = inputChannel[absPosX + absPosY * numCols];
            float weight = filter[i + j * filterWidth];
            FinalValue += weight * (float)inputValue;
        }
    }
	outputChannel[PosX + PosY * numCols] = static_cast<unsigned char>(FinalValue);
}

__global__
void gaussian_blur_shared(const unsigned char* const inputChannel,
    unsigned char* const outputChannel,
    int numRows, int numCols,
    const float* const filter, const int filterWidth)
{
    int PosX = blockDim.x * blockIdx.x + threadIdx.x;
    int PosY = blockDim.y * blockIdx.y + threadIdx.y;

    if (PosX >= numCols || PosY >= numRows)
    {
        return;
    }
    __shared__ float sh_arr[81];
    
    sh_arr[threadIdx.x + 9 * threadIdx.y] = filter[threadIdx.x + 9 * threadIdx.y];
    
    __syncthreads();

    float FinalValue = 0.0f;
    for (int i = 0; i < filterWidth; i++)
    {
        for (int j = 0; j < filterWidth; j++)
        {
            // Calculate the absolute position of the pixel in the image
            int absPosX = PosX + i - filterWidth / 2;
            int absPosY = PosY + j - filterWidth / 2;
            absPosX = min(max(absPosX, 0), numCols - 1);
            absPosY = min(max(absPosY, 0), numRows - 1);
            unsigned char inputValue = inputChannel[absPosX + absPosY * numCols];
            float weight = sh_arr[i + j * filterWidth];
            FinalValue += weight * (float)inputValue;
        }
    }
    outputChannel[PosX + PosY * numCols] = static_cast<unsigned char>(FinalValue);
}

__global__
void gaussian_blur_shared2(const unsigned char* const inputChannel,
    unsigned char* const outputChannel,
    int numRows, int numCols,
    const float* const filter, const int filterWidth)
{
    int PosX = blockIdx.x;
    int PosY = blockIdx.y;

    //if (PosX >= numCols || PosY >= numRows)
    //{
    //    return;
    //}
    __shared__ float sh_arr[9][9];


    int absPosX = PosX + threadIdx.x - filterWidth / 2;
    int absPosY = PosY + threadIdx.y - filterWidth / 2;
    absPosX = min(max(absPosX, 0), numCols - 1);
    absPosY = min(max(absPosY, 0), numRows - 1);
    unsigned char inputValue = inputChannel[absPosX + absPosY * numCols];
    float weight = filter[threadIdx.x + threadIdx.y * filterWidth];

    sh_arr[threadIdx.x][threadIdx.y] = weight * (float)inputValue;

    __syncthreads();

    float FinalValue = 0.0f;
    for (int i = 0; i < filterWidth; i++)
    {
        for (int j = 0; j < filterWidth; j++)
        {
            FinalValue += sh_arr[i][j];
        }
    }
    outputChannel[blockIdx.x + blockIdx.y * numCols] = static_cast<unsigned char>(FinalValue);
}


// 分别使用横竖两个方向分别进行卷积
__global__
void gaussian_blur_horizental(const unsigned char* const inputChannel,
    float* const outputChannel,
    int numRows, int numCols,
    const float* const filter, const int filterWidth)
{
    // 全局线程ID
    int PosX = blockDim.x * blockIdx.x + threadIdx.x;
    int PosY = blockDim.y * blockIdx.y + threadIdx.y;

    if (PosX >= numCols || PosY >= numRows)
    {
        return;
    }

    float FinalValue = 0.0f;
    for (int i = 0; i < filterWidth; i++)
    {
        int absPosX = PosX + i - filterWidth / 2;
        absPosX = min(max(absPosX, 0), numCols - 1);
        unsigned char inputValue = inputChannel[absPosX + PosY * numCols];
        float weight = filter[i];
        FinalValue += weight * (float)inputValue;
    }

    outputChannel[PosX + PosY * numCols] = FinalValue;
}

// 分别使用横竖两个方向分别进行卷积
__global__
void gaussian_blur_vertical(const float* const inputChannel,
    unsigned char* const outputChannel,
    int numRows, int numCols,
    const float* const filter, const int filterWidth)
{
    // 全局线程ID
    int PosX = blockDim.x * blockIdx.x + threadIdx.x;
    int PosY = blockDim.y * blockIdx.y + threadIdx.y;

    if (PosX >= numCols || PosY >= numRows)
    {
        return;
    }

    float FinalValue = 0.0f;
    for (int j = 0; j < filterWidth; j++)
    {
        int absPosY = PosY + j - filterWidth / 2;
        absPosY = min(max(absPosY, 0), numRows - 1);
        float inputValue = inputChannel[PosX + absPosY * numCols];
        float weight = filter[j];
        FinalValue += weight * inputValue;
    }
    outputChannel[PosX + PosY * numCols] = static_cast<unsigned char>(FinalValue);
}

//This kernel takes in an image represented as a uchar4 and splits
//it into three images consisting of only one color channel each
__global__
void separateChannels(const uchar4* const inputImageRGBA,
                      int numRows,
                      int numCols,
                      unsigned char* const redChannel,
                      unsigned char* const greenChannel,
                      unsigned char* const blueChannel)
{
    int PosX = blockDim.x * blockIdx.x + threadIdx.x;
    int PosY = blockDim.y * blockIdx.y + threadIdx.y;
    int BlockSize = blockDim.x * blockDim.y * blockDim.z;

    if (PosX >= numCols || PosY >= numRows )
    {
        return;
    }

    int Index = PosX + PosY * numCols;
    uchar4 Value = inputImageRGBA[Index];
	redChannel[Index] = Value.x;
	greenChannel[Index] = Value.y;
	blueChannel[Index] = Value.z;
}

//This kernel takes in three color channels and recombines them
//into one image.  The alpha channel is set to 255 to represent
//that this image has no transparency.
__global__
void recombineChannels(const unsigned char* const redChannel,
                       const unsigned char* const greenChannel,
                       const unsigned char* const blueChannel,
                       uchar4* const outputImageRGBA,
                       int numRows,
                       int numCols)
{
  const int2 thread_2D_pos = make_int2( blockIdx.x * blockDim.x + threadIdx.x,
                                        blockIdx.y * blockDim.y + threadIdx.y);

  const int thread_1D_pos = thread_2D_pos.y * numCols + thread_2D_pos.x;

  //make sure we don't try and access memory outside the image
  //by having any threads mapped there return early
  if (thread_2D_pos.x >= numCols || thread_2D_pos.y >= numRows)
    return;

  unsigned char red   = redChannel[thread_1D_pos];
  unsigned char green = greenChannel[thread_1D_pos];
  unsigned char blue  = blueChannel[thread_1D_pos];

  //Alpha should be 255 for no transparency
  uchar4 outputPixel = make_uchar4(red, green, blue, 255);

  outputImageRGBA[thread_1D_pos] = outputPixel;
}

unsigned char *d_red, *d_green, *d_blue;
float         *d_filter, *d_filter_vector;

void allocateMemoryAndCopyToGPU(const size_t numRowsImage, const size_t numColsImage,
                                const float* const h_filter, const float* const h_filter_vector, const size_t filterWidth)
{

  //allocate memory for the three different channels
  //original
  checkCudaErrors(cudaMalloc(&d_red,   sizeof(unsigned char) * numRowsImage * numColsImage));
  checkCudaErrors(cudaMalloc(&d_green, sizeof(unsigned char) * numRowsImage * numColsImage));
  checkCudaErrors(cudaMalloc(&d_blue,  sizeof(unsigned char) * numRowsImage * numColsImage));

  //TODO:
  //Allocate memory for the filter on the GPU
  //Use the pointer d_filter that we have already declared for you
  //You need to allocate memory for the filter with cudaMalloc
  //be sure to use checkCudaErrors like the above examples to
  //be able to tell if anything goes wrong
  //IMPORTANT: Notice that we pass a pointer to a pointer to cudaMalloc
  checkCudaErrors(cudaMalloc(&d_filter, sizeof(float) * filterWidth * filterWidth));
  checkCudaErrors(cudaMemcpy(d_filter, h_filter, sizeof(float) * filterWidth * filterWidth, cudaMemcpyHostToDevice));

  checkCudaErrors(cudaMalloc(&d_filter_vector, sizeof(float) * filterWidth));
  checkCudaErrors(cudaMemcpy(d_filter_vector, h_filter_vector, sizeof(float) * filterWidth, cudaMemcpyHostToDevice));
}

void your_gaussian_blur(const uchar4 * const h_inputImageRGBA, uchar4 * const d_inputImageRGBA,
                        uchar4* const d_outputImageRGBA, const size_t numRows, const size_t numCols,
                        unsigned char *d_redBlurred, 
                        unsigned char *d_greenBlurred, 
                        unsigned char *d_blueBlurred,
    float* d_redBlurred_float,
    float* d_greenBlurred_float,
    float* d_blueBlurred_float,
                        const int filterWidth)
{
  size_t threadInBlock = 9;
  size_t gridX = std::ceil((float)numCols / threadInBlock);
  size_t gridY = std::ceil((float)numRows / threadInBlock);
  const dim3 blockSize(threadInBlock, threadInBlock, 1);
  const dim3 gridSize(gridX, gridY, 1);

  //TODO: Launch a kernel for separating the RGBA image into different color channels
  separateChannels<<<gridSize, blockSize>>>(d_inputImageRGBA, numRows, numCols, d_red, d_green, d_blue);
  // Call cudaDeviceSynchronize(), then call checkCudaErrors() immediately after
  // launching your kernel to make sure that you didn't make any mistakes.
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

  //TODO: Call your convolution kernel here 3 times, once for each color channel.
  //gaussian_blur_shared <<<gridSize, blockSize>>>(d_red, d_redBlurred, numRows, numCols, d_filter, filterWidth);
  //gaussian_blur_shared <<<gridSize, blockSize>>>(d_green, d_greenBlurred, numRows, numCols, d_filter, filterWidth);
  //gaussian_blur_shared <<<gridSize, blockSize>>>(d_blue, d_blueBlurred, numRows, numCols, d_filter, filterWidth);

  //const dim3 gridSize2(numCols, numRows, 1);
  //const dim3 blockSize2(filterWidth, filterWidth, 1);
  //gaussian_blur_shared2 << <gridSize2, blockSize2 >> > (d_red, d_redBlurred, numRows, numCols, d_filter, filterWidth);
  //gaussian_blur_shared2 << <gridSize2, blockSize2 >> > (d_green, d_greenBlurred, numRows, numCols, d_filter, filterWidth);
  //gaussian_blur_shared2 << <gridSize2, blockSize2 >> > (d_blue, d_blueBlurred, numRows, numCols, d_filter, filterWidth);
   

    {
        gaussian_blur_horizental << <gridSize, blockSize >> > (d_red, d_redBlurred_float, numRows, numCols, d_filter_vector, filterWidth);
        gaussian_blur_vertical << <gridSize, blockSize >> > (d_redBlurred_float, d_redBlurred, numRows, numCols, d_filter_vector, filterWidth);

        gaussian_blur_horizental << <gridSize, blockSize >> > (d_green, d_greenBlurred_float, numRows, numCols, d_filter_vector, filterWidth);
        gaussian_blur_vertical << <gridSize, blockSize >> > (d_greenBlurred_float, d_greenBlurred, numRows, numCols, d_filter_vector, filterWidth);

        gaussian_blur_horizental << <gridSize, blockSize >> > (d_blue, d_blueBlurred_float, numRows, numCols, d_filter_vector, filterWidth);
        gaussian_blur_vertical << <gridSize, blockSize >> > (d_blueBlurred_float, d_blueBlurred, numRows, numCols, d_filter_vector, filterWidth);
    }
  
  // Again, call cudaDeviceSynchronize(), then call checkCudaErrors() immediately after
  // launching your kernel to make sure that you didn't make any mistakes.
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

  // Now we recombine your results. We take care of launching this kernel for you.
  //
  // NOTE: This kernel launch depends on the gridSize and blockSize variables,
  // which you must set yourself.
  recombineChannels<<<gridSize, blockSize>>>(d_redBlurred,
                                             d_greenBlurred,
                                             d_blueBlurred,
                                             d_outputImageRGBA,
                                             numRows,
                                             numCols);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

}


//Free all the memory that we allocated
//TODO: make sure you free any arrays that you allocated
void cleanup() {
  checkCudaErrors(cudaFree(d_red));
  checkCudaErrors(cudaFree(d_green));
  checkCudaErrors(cudaFree(d_blue));
}
