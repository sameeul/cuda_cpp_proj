/*
 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */
#ifndef CUDA_HELPER_H
#define CUDA_HELPER_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <npp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <tuple>

#include "Exceptions.h"
#include "ImagesCPU.h"
#include "ImagesNPP.h"
#include "grayscale_tiled_tiff.h"
#include "helper.h"

#ifndef MAX
#define MAX(a, b) (a > b ? a : b)
#endif

// General GPU Device CUDA Initialization
inline int gpuDeviceInit(int devID) {
  int deviceCount;
  auto err = cudaGetDeviceCount(&deviceCount);

  if (deviceCount == 0) {
    fprintf(stderr, "gpuDeviceInit() CUDA error: no devices supporting CUDA.\n");
    exit(-1);
  }

  if (devID < 0) devID = 0;

  if (devID > deviceCount - 1) {
    fprintf(stderr, "\n");
    fprintf(stderr, ">> %d CUDA capable GPU device(s) detected. <<\n", deviceCount);
    fprintf(stderr, ">> gpuDeviceInit (-device=%d) is not a valid GPU device. <<\n", devID);
    fprintf(stderr, "\n");
    return -devID;
  }

  cudaDeviceProp deviceProp;
  err = cudaGetDeviceProperties(&deviceProp, devID);

  if (deviceProp.major < 1) {
    fprintf(stderr, "gpuDeviceInit(): GPU device does not support CUDA.\n");
    exit(-1);
  }

  err = cudaSetDevice(devID);
  printf("gpuDeviceInit() CUDA Device [%d]: \"%s\n", devID, deviceProp.name);

  return devID;
}
inline int _ConvertSMVer2Cores(int major, int minor) {
  // Defines for GPU Architecture types (using the SM version to determine
  // the # of cores per SM
  typedef struct {
    int SM;  // 0xMm (hexidecimal notation), M = SM Major version,
    // and m = SM minor version
    int Cores;
  } sSMtoCores;

  sSMtoCores nGpuArchCoresPerSM[] = {{0x30, 192}, {0x32, 192}, {0x35, 192}, {0x37, 192}, {0x50, 128},
                                     {0x52, 128}, {0x53, 128}, {0x60, 64},  {0x61, 128}, {0x62, 128},
                                     {0x70, 64},  {0x72, 64},  {0x75, 64},  {0x80, 64},  {0x86, 128},
                                     {0x87, 128}, {0x89, 128}, {0x90, 128}, {-1, -1}};

  int index = 0;

  while (nGpuArchCoresPerSM[index].SM != -1) {
    if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor)) {
      return nGpuArchCoresPerSM[index].Cores;
    }

    index++;
  }

  // If we don't find the values, we default use the previous one
  // to run properly
  printf(
      "MapSMtoCores for SM %d.%d is undefined."
      "  Default to use %d Cores/SM\n",
      major, minor, nGpuArchCoresPerSM[index - 1].Cores);
  return nGpuArchCoresPerSM[index - 1].Cores;
}

// This function returns the best GPU (with maximum GFLOPS)
inline int gpuGetMaxGflopsDeviceId() {
  int current_device = 0, sm_per_multiproc = 0;
  int max_compute_perf = 0, max_perf_device = 0;
  int device_count = 0, best_SM_arch = 0;
  cudaDeviceProp deviceProp;
  cudaGetDeviceCount(&device_count);

  // Find the best major SM Architecture GPU device
  while (current_device < device_count) {
    cudaGetDeviceProperties(&deviceProp, current_device);
    if (deviceProp.major > 0 && deviceProp.major < 9999) {
      best_SM_arch = MAX(best_SM_arch, deviceProp.major);
    }
    current_device++;
  }

  // Find the best CUDA capable GPU device
  current_device = 0;
  while (current_device < device_count) {
    cudaGetDeviceProperties(&deviceProp, current_device);
    if (deviceProp.major == 9999 && deviceProp.minor == 9999) {
      sm_per_multiproc = 1;
    } else {
      sm_per_multiproc = _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor);
    }

    int compute_perf = deviceProp.multiProcessorCount * sm_per_multiproc * deviceProp.clockRate;

    if (compute_perf > max_compute_perf) {
      // If we find GPU with SM major > 2, search only these
      if (best_SM_arch > 2) {
        // If our device==dest_SM_arch, choose this, or else pass
        if (deviceProp.major == best_SM_arch) {
          max_compute_perf = compute_perf;
          max_perf_device = current_device;
        }
      } else {
        max_compute_perf = compute_perf;
        max_perf_device = current_device;
      }
    }
    ++current_device;
  }
  return max_perf_device;
}

inline int findCudaDevice(int argc, const char **argv) {
  cudaDeviceProp deviceProp;
  int devID = 0;
  devID = gpuGetMaxGflopsDeviceId();
  auto err = cudaSetDevice(devID);
  err = cudaGetDeviceProperties(&deviceProp, devID);
  printf("GPU Device %d: \"%s\" with compute capability %d.%d\n\n", devID, deviceProp.name, deviceProp.major,
         deviceProp.minor);
  return devID;
}

std::tuple<int, Npp8u *> get_scratch_buffer(const NppiSize &roi_size) {
  int max_size = 0, buf_size = 0;
  nppiSumGetBufferHostSize_16s_C1R(roi_size, &buf_size);
  max_size = MAX(max_size, buf_size);
  nppiMinMaxGetBufferHostSize_16u_C1R(roi_size, &buf_size);
  max_size = MAX(max_size, buf_size);
  nppiMeanStdDevGetBufferHostSize_16u_C1R(roi_size, &buf_size);
  max_size = MAX(max_size, buf_size);
  Npp8u *pDeviceBuffer;
  cudaMalloc((void **)(&pDeviceBuffer), max_size);

  printf("Allocating CUDA Memory of size %d bytes\n", max_size);
  return std::make_tuple(max_size, pDeviceBuffer);
}

void process_dataset(const std::string &input_dir, const std::string &output_file) {
  const std::filesystem::path image_path{input_dir};

  int nBufferSize = 0;
  Npp8u *pDeviceBuffer = nullptr;
  // pre-allocate these result swapping vars to avoid reallocating at each operation.
  Npp64f h_sum, h_std_dev, h_mean, *d_var64f_1, *d_var64f_2;
  Npp16u h_min_val, h_max_val, *d_var16u_1, *d_var16u_2;
  cudaMalloc(&d_var64f_1, sizeof(Npp64f));
  cudaMalloc(&d_var64f_2, sizeof(Npp64f));
  cudaMalloc(&d_var16u_1, sizeof(Npp16u));
  cudaMalloc(&d_var16u_2, sizeof(Npp16u));

  std::ofstream output_csv;
  output_csv.open(output_file);

  // populate header

  output_csv << "Filename,Max,Min,Sum,Mean,StdDev\n";
  std::cout << "Started processing the directory.\n";
  // loop through the image collection
  int count = 0;
  const auto start{std::chrono::steady_clock::now()};
  if (std::filesystem::exists(image_path)) {
    for (auto const &dir_entry : std::filesystem::directory_iterator{image_path}) {
      if (std::filesystem::is_regular_file(dir_entry.path())) {
        if (dir_entry.path().extension() == ".tif") {
          auto file_name = dir_entry.path().string();

          // progress indicator
          if (count % 10 == 0) std::cout << ".";
          if (count % 1000 == 0) std::cout << "\n";
          count++;

          // read data from tiff image
          auto image_ptr = std::make_unique<NyxusGrayscaleTiffTileLoader<uint16_t>>(1, file_name);
          auto tile_size = image_ptr->tileHeight(0) * image_ptr->tileWidth(0);
          auto data_ptr = std::make_shared<std::vector<uint16_t>>(tile_size);
          image_ptr->loadTileFromFile(data_ptr, 0, 0, 0, 0);

          // copy to host and then to device
          npp::ImageCPU_16u_C1 oHostSrc(image_ptr->tileWidth(0), image_ptr->tileHeight(0));
          memcpy(oHostSrc.data(), data_ptr->data(), tile_size * sizeof(uint16_t));
          npp::ImageNPP_16u_C1 oDeviceSrc(oHostSrc);

          // prep for sending to device
          NppiSize oSrcSize = {(int)oDeviceSrc.width(), (int)oDeviceSrc.height()};
          NppiSize oSizeROI = {(int)oDeviceSrc.width(), (int)oDeviceSrc.height()};

          // if no decive scratch buffer, allocate device scratch buffer
          // this also saves from reallocating at each operation
          if (!pDeviceBuffer or nBufferSize == 0) {
            auto tmp = get_scratch_buffer(oSrcSize);
            nBufferSize = std::get<0>(tmp);
            pDeviceBuffer = std::get<1>(tmp);
          }

          // do work in the device

          // calc sum
          auto stat = nppiSum_16u_C1R(oDeviceSrc.data(), oDeviceSrc.pitch(), oSizeROI, pDeviceBuffer, d_var64f_1);
          // get result
          cudaMemcpy(&h_sum, d_var64f_1, sizeof(Npp64f), cudaMemcpyDeviceToHost);

          // calc min, max
          stat = nppiMinMax_16u_C1R(oDeviceSrc.data(), oDeviceSrc.pitch(), oSizeROI, d_var16u_1, d_var16u_2,
                                    pDeviceBuffer);
          // get result
          cudaMemcpy(&h_min_val, d_var16u_1, sizeof(Npp16u), cudaMemcpyDeviceToHost);
          cudaMemcpy(&h_max_val, d_var16u_2, sizeof(Npp16u), cudaMemcpyDeviceToHost);

          // calc mean, std_dev
          stat = nppiMean_StdDev_16u_C1R(oDeviceSrc.data(), oDeviceSrc.pitch(), oSizeROI, pDeviceBuffer, d_var64f_1,
                                         d_var64f_2);
          // get result
          cudaMemcpy(&h_mean, d_var64f_1, sizeof(Npp64f), cudaMemcpyDeviceToHost);
          cudaMemcpy(&h_std_dev, d_var64f_2, sizeof(Npp64f), cudaMemcpyDeviceToHost);

          output_csv << dir_entry.path().filename() << "," << h_min_val << "," << h_max_val << "," << h_sum << ","
                     << h_mean << "," << h_std_dev << "\n";
        }
      }
    }
  }

  const auto end{std::chrono::steady_clock::now()};
  const std::chrono::duration<double> elapsed_seconds{end - start};
  std::cout << "\nFinished processing the directory.\nElapsed Time: " << elapsed_seconds.count() << " seconds\n";
  output_csv.close();

  if (d_var64f_1) cudaFree(d_var64f_1);
  if (d_var64f_2) cudaFree(d_var64f_2);
  if (d_var16u_1) cudaFree(d_var16u_1);
  if (d_var16u_2) cudaFree(d_var16u_2);

  if (pDeviceBuffer) cudaFree(pDeviceBuffer);
}

#endif