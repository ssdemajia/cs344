//Udacity HW 6
//Poisson Blending

/* Background
   ==========

   The goal for this assignment is to take one image (the source) and
   paste it into another image (the destination) attempting to match the
   two images so that the pasting is non-obvious. This is
   known as a "seamless clone".

   The basic ideas are as follows:

   1) Figure out the interior and border of the source image
   2) Use the values of the border pixels in the destination image 
      as boundary conditions for solving a Poisson equation that tells
      us how to blend the images.
   
      No pixels from the destination except pixels on the border
      are used to compute the match.

   Solving the Poisson Equation
   ============================

   There are multiple ways to solve this equation - we choose an iterative
   method - specifically the Jacobi method. Iterative methods start with
   a guess of the solution and then iterate to try and improve the guess
   until it stops changing.  If the problem was well-suited for the method
   then it will stop and where it stops will be the solution.

   The Jacobi method is the simplest iterative method and converges slowly - 
   that is we need a lot of iterations to get to the answer, but it is the
   easiest method to write.

   Jacobi Iterations
   =================

   Our initial guess is going to be the source image itself.  This is a pretty
   good guess for what the blended image will look like and it means that
   we won't have to do as many iterations compared to if we had started far
   from the final solution.

   ImageGuess_prev (Floating point)
   ImageGuess_next (Floating point)

   DestinationImg
   SourceImg

   Follow these steps to implement one iteration:

   1) For every pixel p in the interior, compute two sums over the four neighboring pixels:
      Sum1: If the neighbor is in the interior then += ImageGuess_prev[neighbor]
             else if the neighbor in on the border then += DestinationImg[neighbor]

      Sum2: += SourceImg[p] - SourceImg[neighbor]   (for all four neighbors)

   2) Calculate the new pixel value:
      float newVal= (Sum1 + Sum2) / 4.f  <------ Notice that the result is FLOATING POINT
      ImageGuess_next[p] = min(255, max(0, newVal)); //clamp to [0, 255]


    In this assignment we will do 800 iterations.
   */



#include "utils.h"
#include <thrust/host_vector.h>

#define NUM_BLOCK_X 64

void computeG(const unsigned char* const channel,
    float* const g,
    const size_t numColsSource,
    const std::vector<uint2>& interiorPixelList);


__global__ void computeIteration(float* dst, //INPUT
    unsigned char* interiorPixels,
    unsigned char* borderPixels,
    const uint2* interiorPixelList,
    const size_t numColsSource,
    const size_t numVals,
    const float* const f,
    const float* const g,
    float* const f_next)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < numVals)
    {
        float blendedSum = 0.f;
        float borderSum = 0.f;

        uint2 coord = interiorPixelList[i];
        unsigned int offset = coord.x * numColsSource + coord.y;
        if (interiorPixels[offset - 1]) {
            blendedSum += f[offset - 1];
        }
        else {
            borderSum += dst[offset - 1];
        }

        if (interiorPixels[offset + 1]) {
            blendedSum += f[offset + 1];
        }
        else {
            borderSum += dst[offset + 1];
        }

        if (interiorPixels[offset - numColsSource]) {
            blendedSum += f[offset - numColsSource];
        }
        else {
            borderSum += dst[offset - numColsSource];
        }

        if (interiorPixels[offset + numColsSource]) {
            blendedSum += f[offset + numColsSource];
        }
        else {
            borderSum += dst[offset + numColsSource];
        }

        float f_next_val = (blendedSum + borderSum + g[offset]) / 4.f;

        f_next[offset] = min(255.f, max(0.f, f_next_val)); //clip to [0, 255]
    }
}

void your_blend(const uchar4* const h_sourceImg,  //IN
                const size_t numRowsSource, const size_t numColsSource,
                const uchar4* const h_destImg, //IN
                uchar4* const h_blendedImg) //OUT
{

  /* To Recap here are the steps you need to implement
  
     1) Compute a mask of the pixels from the source image to be copied
        The pixels that shouldn't be copied are completely white, they
        have R=255, G=255, B=255.  Any other pixels SHOULD be copied.

     2) Compute the interior and border regions of the mask.  An interior
        pixel has all 4 neighbors also inside the mask.  A border pixel is
        in the mask itself, but has at least one neighbor that isn't.

     3) Separate out the incoming image into three separate channels

     4) Create two float(!) buffers for each color channel that will
        act as our guesses.  Initialize them to the respective color
        channel of the source image since that will act as our intial guess.

     5) For each color channel perform the Jacobi iteration described 
        above 800 times.

     6) Create the output image by replacing all the interior pixels
        in the destination image with the result of the Jacobi iterations.
        Just cast the floating point values to unsigned chars since we have
        already made sure to clamp them to the correct range.

      Since this is final assignment we provide little boilerplate code to
      help you.  Notice that all the input/output pointers are HOST pointers.

      You will have to allocate all of your own GPU memory and perform your own
      memcopies to get data in and out of the GPU memory.

      Remember to wrap all of your calls with checkCudaErrors() to catch any
      thing that might go wrong.  After each kernel call do:

      cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

      to catch any errors that happened while executing the kernel.
  */
    // 创建Mask，非白区域
    size_t srcSize = numRowsSource * numColsSource;
    std::vector<unsigned char> mask(srcSize);

    for (int i = 0; i < srcSize; ++i) {
        mask[i] = (h_sourceImg[i].x + h_sourceImg[i].y + h_sourceImg[i].z < 3 * 255) ? 1 : 0;
    }

    // 边缘像素
    std::vector<unsigned char> borderPixels(srcSize);
    // 内部像素
    std::vector<unsigned char> strictInteriorPixels(srcSize);

    std::vector<uint2> interiorPixelList;

    //the source region in the homework isn't near an image boundary, so we can
    //simplify the conditionals a little...
    for (size_t r = 1; r < numRowsSource - 1; ++r) {
        for (size_t c = 1; c < numColsSource - 1; ++c) {
            if (mask[r * numColsSource + c]) {
                if (mask[(r - 1) * numColsSource + c] && mask[(r + 1) * numColsSource + c] &&
                    mask[r * numColsSource + c - 1] && mask[r * numColsSource + c + 1]) {
                    strictInteriorPixels[r * numColsSource + c] = 1;
                    borderPixels[r * numColsSource + c] = 0;
                    interiorPixelList.push_back(make_uint2(r, c));
                }
                else {
                    strictInteriorPixels[r * numColsSource + c] = 0;
                    borderPixels[r * numColsSource + c] = 1;
                }
            }
            else {
                strictInteriorPixels[r * numColsSource + c] = 0;
                borderPixels[r * numColsSource + c] = 0;

            }
        }
    }
    unsigned char* d_InteriorPixels;
    unsigned char* d_borderPixels;
    uint2* d_interiorPixelList;
    checkCudaErrors(cudaMalloc(&d_InteriorPixels, srcSize * sizeof(unsigned char)));
    checkCudaErrors(cudaMemcpy(d_InteriorPixels, strictInteriorPixels.data(), srcSize * sizeof(unsigned char), cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMalloc(&d_borderPixels, srcSize * sizeof(unsigned char)));
    checkCudaErrors(cudaMemcpy(d_borderPixels, borderPixels.data(), srcSize * sizeof(unsigned char), cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMalloc(&d_interiorPixelList, interiorPixelList.size() * sizeof(uint2)));
    checkCudaErrors(cudaMemcpy(d_interiorPixelList, interiorPixelList.data(), interiorPixelList.size() * sizeof(uint2), cudaMemcpyHostToDevice));

    //split the source and destination images into their respective
    //channels
    std::vector<unsigned char> red_src(srcSize);
    std::vector<unsigned char> blue_src(srcSize);
    std::vector<unsigned char> green_src(srcSize);

    for (int i = 0; i < srcSize; ++i) {
        red_src[i] = h_sourceImg[i].x;
        blue_src[i] = h_sourceImg[i].y;
        green_src[i] = h_sourceImg[i].z;
    }

    float* red_dst = new float[srcSize];
    float* blue_dst = new float[srcSize];
    float* green_dst = new float[srcSize];

    for (int i = 0; i < srcSize; ++i) {
        red_dst[i] = h_destImg[i].x;
        blue_dst[i] = h_destImg[i].y;
        green_dst[i] = h_destImg[i].z;
    }

    float* d_red_dst;
    float* d_blue_dst;
    float* d_green_dst;
    checkCudaErrors(cudaMalloc(&d_red_dst, srcSize * sizeof(float)));
    checkCudaErrors(cudaMemcpy(d_red_dst, red_dst, srcSize * sizeof(float), cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMalloc(&d_blue_dst, srcSize * sizeof(float)));
    checkCudaErrors(cudaMemcpy(d_blue_dst, blue_dst, srcSize * sizeof(float), cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMalloc(&d_green_dst, srcSize * sizeof(float)));
    checkCudaErrors(cudaMemcpy(d_green_dst, green_dst, srcSize * sizeof(float), cudaMemcpyHostToDevice));

    //next we'll precompute the g term - it never changes, no need to recompute every iteration
    std::vector<float> g_red(srcSize);
    std::vector<float> g_blue(srcSize);
    std::vector<float> g_green(srcSize);

    memset(g_red.data(), 0, srcSize * sizeof(float));
    memset(g_blue.data(), 0, srcSize * sizeof(float));
    memset(g_green.data(), 0, srcSize * sizeof(float));

    computeG(red_src.data(), g_red.data(), numColsSource, interiorPixelList);
    computeG(blue_src.data(), g_blue.data(), numColsSource, interiorPixelList);
    computeG(green_src.data(), g_green.data(), numColsSource, interiorPixelList);

    float* d_g_red;
    float* d_g_blue;
    float* d_g_green;
    checkCudaErrors(cudaMalloc(&d_g_red, srcSize * sizeof(float)));
    checkCudaErrors(cudaMemcpy(d_g_red, g_red.data(), srcSize * sizeof(float), cudaMemcpyHostToDevice));
    
    checkCudaErrors(cudaMalloc(&d_g_blue, srcSize * sizeof(float)));
    checkCudaErrors(cudaMemcpy(d_g_blue, g_blue.data(), srcSize * sizeof(float), cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMalloc(&d_g_green, srcSize * sizeof(float)));
    checkCudaErrors(cudaMemcpy(d_g_green, g_green.data(), srcSize * sizeof(float), cudaMemcpyHostToDevice));

    //for each color channel we'll need two buffers and we'll ping-pong between them
    float* blendedValsRed_1 = new float[srcSize];
    float* blendedValsRed_2 = new float[srcSize];

    float* blendedValsBlue_1 = new float[srcSize];
    float* blendedValsBlue_2 = new float[srcSize];

    float* blendedValsGreen_1 = new float[srcSize];
    float* blendedValsGreen_2 = new float[srcSize];

    //IC is the source image, copy over
    for (size_t i = 0; i < srcSize; ++i) {
        blendedValsRed_1[i] = red_src[i];
        blendedValsRed_2[i] = red_src[i];
        blendedValsBlue_1[i] = blue_src[i];
        blendedValsBlue_2[i] = blue_src[i];
        blendedValsGreen_1[i] = green_src[i];
        blendedValsGreen_2[i] = green_src[i];
    }

    float* d_blendedValsRed_1;
    float* d_blendedValsRed_2;

    float* d_blendedValsBlue_1;
    float* d_blendedValsBlue_2;

    float* d_blendedValsGreen_1;
    float* d_blendedValsGreen_2;

    checkCudaErrors(cudaMalloc(&d_blendedValsRed_1, srcSize * sizeof(float)));
    checkCudaErrors(cudaMemcpy(d_blendedValsRed_1, blendedValsRed_1, srcSize * sizeof(float), cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMalloc(&d_blendedValsRed_2, srcSize * sizeof(float)));
    checkCudaErrors(cudaMemcpy(d_blendedValsRed_2, blendedValsRed_2, srcSize * sizeof(float), cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMalloc(&d_blendedValsBlue_1, srcSize * sizeof(float)));
    checkCudaErrors(cudaMemcpy(d_blendedValsBlue_1, blendedValsBlue_1, srcSize * sizeof(float), cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMalloc(&d_blendedValsBlue_2, srcSize * sizeof(float)));
    checkCudaErrors(cudaMemcpy(d_blendedValsBlue_2, blendedValsBlue_2, srcSize * sizeof(float), cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMalloc(&d_blendedValsGreen_1, srcSize * sizeof(float)));
    checkCudaErrors(cudaMemcpy(d_blendedValsGreen_1, blendedValsGreen_1, srcSize * sizeof(float), cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMalloc(&d_blendedValsGreen_2, srcSize * sizeof(float)));
    checkCudaErrors(cudaMemcpy(d_blendedValsGreen_2, blendedValsGreen_2, srcSize * sizeof(float), cudaMemcpyHostToDevice));

    size_t numElems = interiorPixelList.size();
    size_t blockX = NUM_BLOCK_X;
    size_t gridX = (numElems + blockX - 1) / blockX;
    const dim3 blockSize(blockX, 1, 1);
    const dim3 gridSize(gridX, 1, 1);

    const size_t numIterations = 800;
    for (size_t i = 0; i < numIterations; ++i) {
        computeIteration<<<gridSize, blockSize>>>(d_red_dst, d_InteriorPixels, d_borderPixels,
            d_interiorPixelList, numColsSource, numElems, d_blendedValsRed_1, d_g_red,
            d_blendedValsRed_2);

        std::swap(d_blendedValsRed_1, d_blendedValsRed_2);
    }

    for (size_t i = 0; i < numIterations; ++i) {
        computeIteration<<<gridSize, blockSize>>>(d_blue_dst, d_InteriorPixels, d_borderPixels,
            d_interiorPixelList, numColsSource, numElems, d_blendedValsBlue_1, d_g_blue,
            d_blendedValsBlue_2);
        std::swap(d_blendedValsBlue_1, d_blendedValsBlue_2);
    }

    for (size_t i = 0; i < numIterations; ++i) {
        computeIteration<<<gridSize, blockSize>>>(d_green_dst, d_InteriorPixels, d_borderPixels,
            d_interiorPixelList, numColsSource, numElems, d_blendedValsGreen_1, d_g_green,
            d_blendedValsGreen_2);
        std::swap(d_blendedValsGreen_1, d_blendedValsGreen_2);
    }

    memcpy(h_blendedImg, h_destImg, sizeof(uchar4) * srcSize);
	checkCudaErrors(cudaMemcpy(blendedValsRed_2, d_blendedValsRed_1, srcSize * sizeof(float), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(blendedValsBlue_2, d_blendedValsBlue_1, srcSize * sizeof(float), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(blendedValsGreen_2, d_blendedValsGreen_1, srcSize * sizeof(float), cudaMemcpyDeviceToHost));


    //copy computed values for the interior into the output
    for (size_t i = 0; i < interiorPixelList.size(); ++i) {
        uint2 coord = interiorPixelList[i];

        unsigned int offset = coord.x * numColsSource + coord.y;

        h_blendedImg[offset].x = blendedValsRed_2[offset];
        h_blendedImg[offset].y = blendedValsBlue_2[offset];
        h_blendedImg[offset].z = blendedValsGreen_2[offset];
    }

    checkCudaErrors(cudaFree(d_red_dst));
    checkCudaErrors(cudaFree(d_blue_dst));
    checkCudaErrors(cudaFree(d_green_dst));

    checkCudaErrors(cudaFree(d_InteriorPixels));
    checkCudaErrors(cudaFree(d_borderPixels));
    checkCudaErrors(cudaFree(d_interiorPixelList));

    checkCudaErrors(cudaFree(d_blendedValsRed_1));
    checkCudaErrors(cudaFree(d_blendedValsBlue_1));
    checkCudaErrors(cudaFree(d_blendedValsGreen_1));

    checkCudaErrors(cudaFree(d_blendedValsRed_2));
    checkCudaErrors(cudaFree(d_blendedValsBlue_2));
    checkCudaErrors(cudaFree(d_blendedValsGreen_2));

    checkCudaErrors(cudaFree(d_g_red));
    checkCudaErrors(cudaFree(d_g_blue));
    checkCudaErrors(cudaFree(d_g_green));
}
