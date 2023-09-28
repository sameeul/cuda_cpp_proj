#include <npp.h>

#include <filesystem>
#include <iostream>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include "Exceptions.h"
#include "ImagesCPU.h"
#include "ImagesNPP.h"
#include "cuda_helper.h"
#include "grayscale_tiled_tiff.h"
#include "helper.h"

int main(int argc, char *argv[]) {
  try {
    findCudaDevice(argc, (const char **)argv);
    std::cout << "Starting..." << argv[0] << "\n";

    char *raw_input_dir;
    if (checkCmdLineFlag(argc, (const char **)argv, "inputDir")) {
      getCmdLineArgumentString(argc, (const char **)argv, "inputDir",
                               &raw_input_dir);
    } else {
      std::cout << "No input directory found, exiting.";
      exit(EXIT_SUCCESS);
    }

    std::string input_dir{raw_input_dir};
    std::cout << "Input Directory: " << input_dir << "\n";

    char *raw_output_file;
    std::string output_file;
    if (checkCmdLineFlag(argc, (const char **)argv, "outputFile")) {
      getCmdLineArgumentString(argc, (const char **)argv, "outputFile",
                               &raw_output_file);
      output_file = raw_output_file;
    } else {
      output_file = input_dir + "_features.csv";
    }

    std::cout << "Output File: " << output_file << "\n";
    const std::filesystem::path image_path{input_dir};

    // loop through the image collection
    if (std::filesystem::exists(image_path)) {
      for (auto const &dir_entry :
           std::filesystem::directory_iterator{image_path}) {
        if (std::filesystem::is_regular_file(dir_entry.path())) {
          if (dir_entry.path().extension() == ".tif") {
            std::cout << "Processing " << dir_entry.path().string() << "\n";
            // read data from tiff image
            auto image_ptr =
                std::make_unique<NyxusGrayscaleTiffTileLoader<uint16_t>>(
                    1, dir_entry.path().string());
            auto tile_size = image_ptr->tileHeight(0) * image_ptr->tileWidth(0);
            auto data_ptr = std::make_shared<std::vector<uint16_t>>(tile_size);
            image_ptr->loadTileFromFile(data_ptr, 0, 0, 0, 0);

            // copy to host and then to device
            npp::ImageCPU_16u_C1 oHostSrc(image_ptr->tileWidth(0),
                                          image_ptr->tileHeight(0));
            memcpy(oHostSrc.data(), data_ptr->data(),
                   tile_size * sizeof(uint16_t));
            npp::ImageNPP_16u_C1 oDeviceSrc(oHostSrc);

            // prep for sending to device
            NppiSize oSrcSize = {(int)oDeviceSrc.width(),
                                 (int)oDeviceSrc.height()};
            NppiSize oSizeROI = {(int)oDeviceSrc.width(),
                                 (int)oDeviceSrc.height()};

            int nBufferSize;
            nppsSumGetBufferSize_64f(tile_size, &nBufferSize);
            Npp8u *pDeviceBuffer;
            cudaMalloc((void **)(&pDeviceBuffer), nBufferSize);
            Npp64f h_sum, *d_sum;
            cudaMalloc(&d_sum, sizeof(Npp64f));

            // do work in the device
            auto stat = nppiSum_16u_C1R(oDeviceSrc.data(), oDeviceSrc.pitch(),
                                        oSizeROI, pDeviceBuffer, d_sum);
            // get result
            cudaMemcpy(&h_sum, d_sum, sizeof(Npp64f), cudaMemcpyDeviceToHost);
          }
        }
      }
    }

  } catch (npp::Exception &rException) {
    std::cerr << "Program error! The following exception occurred: \n";
    std::cerr << rException << std::endl;
    std::cerr << "Aborting." << std::endl;

    exit(EXIT_FAILURE);
  } catch (...) {
    std::cerr << "Program error! An unknow type of exception occurred. \n";
    std::cerr << "Aborting." << std::endl;

    exit(EXIT_FAILURE);
    return -1;
  }

  return 0;
}