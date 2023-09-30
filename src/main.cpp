
#include <iostream>
#include <string>

#include "Exceptions.h"
#include "cuda_util.h"

int main(int argc, char *argv[]) {
  try {
    findCudaDevice(argc, (const char **)argv);
    std::cout << "Starting..." << argv[0] << "\n";

    char *raw_input_dir;
    if (checkCmdLineFlag(argc, (const char **)argv, "inputDir")) {
      getCmdLineArgumentString(argc, (const char **)argv, "inputDir", &raw_input_dir);
    } else {
      std::cout << "No input directory found, exiting.";
      exit(EXIT_SUCCESS);
    }

    std::string input_dir{raw_input_dir};
    std::cout << "Input Directory: " << input_dir << "\n";

    char *raw_output_file;
    std::string output_file;
    if (checkCmdLineFlag(argc, (const char **)argv, "outputFile")) {
      getCmdLineArgumentString(argc, (const char **)argv, "outputFile", &raw_output_file);
      output_file = raw_output_file;
    } else {
      output_file = "features.csv";
    }

    std::cout << "Output File: " << output_file << "\n";

    process_dataset(input_dir, output_file);

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