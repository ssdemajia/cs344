#include <cstdlib>
#include <iostream>
#include <cstdio>
#include <fstream>
#include "utils.h"
#include "timer.h"
#include <cstdio>
#include <vector>
#if defined(_WIN16) || defined(_WIN32) || defined(_WIN64)
#define NOMINMAX
#include <Windows.h>
#else
#include <sys/time.h>
#endif

#include <thrust/random/linear_congruential_engine.h>
#include <thrust/random/normal_distribution.h>
#include <thrust/random/uniform_int_distribution.h>

#include "reference_calc.h"

void computeHistogram(const unsigned int *const d_vals,
                      unsigned int* const d_histo,
                      const unsigned int numBins,
                      const unsigned int numElems);
void computeHistogram2(const unsigned int* const d_vals,
    unsigned int* d_intermediate_histo,
    unsigned int* const d_histo,
    const unsigned int numBins,
    const unsigned int numElems);

int main(void)
{
  const unsigned int numBins = 1024;
  const unsigned int numElems = 10000 * numBins;
  const float stddev = 100.f;

  std::vector<unsigned int> vals(numElems);
  unsigned int *h_vals = new unsigned int[numElems];
  std::vector<unsigned int> h_studentHisto(numBins);
  std::vector<unsigned int> h_studentHisto2(numBins);
  unsigned int *h_refHisto = new unsigned int[numBins];

#if defined(_WIN16) || defined(_WIN32) || defined(_WIN64)
  srand(GetTickCount());
#else
  timeval tv;
  gettimeofday(&tv, NULL);

  srand(tv.tv_usec);
#endif

  //make the mean unpredictable, but close enough to the middle
  //so that timings are unaffected
  unsigned int mean = rand() % 100 + 462;

  //Output mean so that grading can happen with the same inputs
  std::cout << mean << std::endl;

  thrust::minstd_rand rng;

  thrust::random::normal_distribution<float> normalDist((float)mean, stddev);

  // Generate the random values
  for (size_t i = 0; i < numElems; ++i) {
    vals[i] = std::min((unsigned int) std::max((int)normalDist(rng), 0), numBins - 1);
  }

  unsigned int *d_vals, *d_histo, *d_histo2;

  GpuTimer timer;
  size_t NUM_PARTS = 1024;
  size_t gridX = (numElems + NUM_PARTS - 1) / NUM_PARTS;
  unsigned int* d_intermediate_histo;
  size_t NumIntermediate = gridX * NUM_PARTS;
  checkCudaErrors(cudaMalloc(&d_intermediate_histo, NumIntermediate * sizeof(unsigned int)));
  checkCudaErrors(cudaMalloc(&d_vals,    sizeof(unsigned int) * numElems));
  checkCudaErrors(cudaMalloc(&d_histo,   sizeof(unsigned int) * numBins));
  checkCudaErrors(cudaMemset(d_histo, 0, sizeof(unsigned int) * numBins));
  checkCudaErrors(cudaMalloc(&d_histo2, sizeof(unsigned int) * numBins));
  checkCudaErrors(cudaMemset(d_histo2, 0, sizeof(unsigned int) * numBins));
  checkCudaErrors(cudaMemcpy(d_vals, vals.data(), sizeof(unsigned int) * numElems, cudaMemcpyHostToDevice));

  timer.Start();
  computeHistogram(d_vals, d_histo, numBins, numElems);
  timer.Stop();
  int err = printf("Your code1 ran in: %f msecs.\n", timer.Elapsed());

  if (err < 0) {
    //Couldn't print! Probably the student closed stdout - bad news
    std::cerr << "Couldn't print timing information! STDOUT Closed!" << std::endl;
    exit(1);
  }
  
  // copy the student-computed histogram back to the host
  checkCudaErrors(cudaMemcpy(h_studentHisto.data(), d_histo, sizeof(unsigned int) * numBins, cudaMemcpyDeviceToHost));

  //generate reference for the given mean
  reference_calculation(vals.data(), h_refHisto, numBins, numElems);
  //Now do the comparison
  checkResultsExact(h_refHisto, h_studentHisto.data(), numBins);

  GpuTimer timer2;
  timer2.Start();
  computeHistogram2(d_vals, d_intermediate_histo, d_histo2, numBins, numElems);
  timer2.Stop();
  err = printf("Your code2 ran in: %f msecs.\n", timer2.Elapsed());
  checkCudaErrors(cudaMemcpy(h_studentHisto2.data(), d_histo2, sizeof(unsigned int) * numBins, cudaMemcpyDeviceToHost));
  checkResultsExact(h_refHisto, h_studentHisto2.data(), numBins);


  delete[] h_vals;
  delete[] h_refHisto;

  cudaFree(d_vals);
  cudaFree(d_histo);
  cudaFree(d_histo2);
  checkCudaErrors(cudaFree(d_intermediate_histo));
  return 0;
}
