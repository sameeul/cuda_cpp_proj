# CUDA Independent Project

## Motivation
Feature extraction of biological images is an important step for building shallow machine learning based image classifiers. In this project, we developed a small proof of concept to show how to use NPP library to calculate some basic image statistical features. We use the [TissueNet](https://datasets.deepcell.org/data) v1.1 image collection's training dataset (5160 images) and process using our `FeatureExtractor` executable.

## How to Run
`Feature extractor` takes two command line arguments, `-inputDir` and `-outputFile`. The `inputDir` points to a directory containing `TIF` images that we want to process and `outputFile` sets the name of the CSV file where we want to save the results. A typical command line execution looks like below:

```sh
(base) samee@desk:~/work/cuda_cpp_proj/build$ ./FeatureExtractor -inputDir=/home/samee/datasets/TissueNet/v1.1/standard/train/intensity -outputFile=features.csv

GPU Device  0:  "Tesla T4"  with  compute  capability  7.5
Starting Feature  Extractor...
Input Directory:  /home/samee/datasets/TissueNet/v1.1/standard/train/intensity
Output File:  features.csv
Started processing  the  directory.
.
Allocating CUDA  Memory  of  size  24576  bytes
....................................................................................................
....................................................................................................
....................................................................................................
....................................................................................................
....................................................................................................
...............
Finished processing  the  directory (5160 images).
Elapsed Time:  131.018  seconds
```

## Build Instruction
The following dependencies are needed to successfully build the project.
* GCC (> 9.0)
* CUDA (> 11.6)
* CMake (> 3.20)
* LibTiff (> 4.0)
  
These dependencies can be freely downloaded and installed using their respective websites. Here are some sample instruction on how to clone this repository and build it.

```sh
$ git  clone https://github.com/sameeul/cuda_cpp_proj.git
$ cd  cuda_cpp_proj
$ mkdir  build
$ cd  build
$ cmake  ..
$ make
```
One sample `TIF` image is provided with in the `sample_data` directory to test the executable. To check the executable, continue with the following.
```sh
$ ./FeatureExtractor  -inputDir=../sample_data  -outputFile=test.csv
GPU Device  0:  "Tesla T4"  with  compute  capability  7.5
Starting Feature  Extractor...
Input Directory:  ../sample_data
Output File:  test.csv
Started processing  the  directory.
.
Allocating CUDA  Memory  of  size  24576  bytes
Finished processing  the  directory (1 images).
Elapsed Time:  0.0603268  seconds
$ cat  test.csv
Filename,Max,Min,Sum,Mean,StdDev
"p0_y1_r17_c0.ome.tif",0,65531,1.71294e+09,1633.58,7850.07

```
## Code Organization
The repository is organized in the following way:

* `src:` code developed for the project 
* `include/cuda-samples:` necessary header files defining NPP objects. Reused from the `cuda-samples` repository.
* `include/cuda_lab:` code borrowed from the Lab section of the course that are used for doing common tasks like parsing the arguments and retrieving information about the GPU. 
* `sample_data:` contains one sample `TIF` image to test the executable. 
* `output:` contains the `stdout` of one sample run and one sample output file containing the features calculated for the training dataset of the TissueNet v1.1 data.
 
## License
MIT
