#include <npp.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <tuple>

#include "Exceptions.h"
#include "ImagesCPU.h"
#include "ImagesNPP.h"
#include "cuda_util.h"
#include "grayscale_tiled_tiff.h"

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
  std::cout << "\nFinished processing the directory (" << count
            << " images).\nElapsed Time: " << elapsed_seconds.count() << " seconds\n";
  output_csv.close();

  if (d_var64f_1) cudaFree(d_var64f_1);
  if (d_var64f_2) cudaFree(d_var64f_2);
  if (d_var16u_1) cudaFree(d_var16u_1);
  if (d_var16u_2) cudaFree(d_var16u_2);

  if (pDeviceBuffer) cudaFree(pDeviceBuffer);
}
