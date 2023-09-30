# CUDA Independent Project

## Motivation
Feature extration of biological images is an important step for building shallow machine learning based image classifiers. In this project, we developed a small proof of concept to show how to use NPP library to calculate some basic image statistical features. We use the [TissueNet](https://datasets.deepcell.org/data) v1.1 image collection's traning dataset and process using our `FeatureExtractor` executable.

## How to Run
`Feature extractor` takes two command line arguments, `-inputDir` and `-outputFile`. The `inputDir` points to a directory containing `TIF` images that we want to process and `outputFile` sets the name of the csv file where we want to save the results. A typical command line execution looks like below:
```sh
(base) samee@desk:~/work/cuda_cpp_proj/build$ ./FeatureExtractor -inputDir=/home/samee/datasets/TissueNet/v1.1/standard/train/intensity -outputFile=features.csv
GPU Device 0: "Tesla T4" with compute capability 7.5

Starting Feature Extractor...
Input Directory: /home/samee/datasets/TissueNet/v1.1/standard/train/intensity
Output File: features.csv
Started processing the directory.
.
Allocating CUDA Memory of size 24576 bytes
....................................................................................................
....................................................................................................
....................................................................................................
....................................................................................................
....................................................................................................
...............
Finished processing the directory (5160 images).
Elapsed Time: 131.018 seconds
```

## Build Instruction
The following dependencies are needed to successfully build the project.
* GCC (> 9.0)
* CUDA (> 11.6)
* CMake (> 3.20)
* LibTiff (> 4.0)

These dependecnies can be freely downloaded and installed using their respective websites. Here are some sample instruction on how to clone this repository and build it.

```sh
$ git clone
$ cd cuda_cpp_proj
$ mkdir build
$ cd build
$ cmake ..
$ make 
```
One sample tif image is provided with in the `data` directory to test the executable. To check the executable, continue with the following.
```sh
$ ./FeatureExtractor -inputDir=../data -outputFile=test.csv
GPU Device 0: "Tesla T4" with compute capability 7.5

Starting Feature Extractor...
Input Directory: ../data
Output File: test.csv
Started processing the directory.
.
Allocating CUDA Memory of size 24576 bytes

Finished processing the directory (1 images).
Elapsed Time: 0.0603268 seconds
$ cat test.csv
Filename,Max,Min,Sum,Mean,StdDev
"p0_y1_r17_c0.ome.tif",0,65531,1.71294e+09,1633.58,7850.07
```

## Code Organization
The `src` directory contains the the code developed for the project. The `include` directory contains the necessary header files from the `cuda-samples` repository that are needed for using NPP objects. The `include` directory also contains code borrowed from the Lab section of the course that are used for doing common tasks like parsing the arguments and retrieving information about the GPU. The `data` directory contains one sample `TIF` image to test the executable. The `output` directory contains the `stdout` of one sample run and one sample output files containing the features calculated for the training dataset of the TissueNet v1.1 data. 


## License

MIT