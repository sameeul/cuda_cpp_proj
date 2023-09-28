cmake_minimum_required(VERSION 3.20)
project(CudaFeatExt)

find_package(CUDAToolkit REQUIRED)
if(CUDAToolkit_FOUND)
    message(STATUS "CUDA Found")
    set(CUDA_ARCH_LIST "35;37;50;72;75;80;86")
    enable_language("CUDA")
	SET(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH_LIST})
	set_property(GLOBAL PROPERTY CUDA_ARCHITECTURES "${CUDA_ARCH_LIST}")
	message("Building with compute capability for ${CUDA_ARCH_LIST}.")
	set(CMAKE_CUDA_STANDARD 17)
	set(CMAKE_CUDA_STANDARD_REQUIRED ON)	
    include_directories("${CUDAToolkit_INCLUDE_DIRS}")
endif() 

find_package(TIFF REQUIRED)
if (TIFF_FOUND)
	include_directories (${TIFF_INCLUDE_DIR})
endif (TIFF_FOUND)

#==== Compiler Options
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(SOURCE src/main.cpp)

add_executable(FeatureExtractor ${SOURCE})

if (TIFF_FOUND)
	target_link_libraries(FeatureExtractor PUBLIC ${TIFF_LIBRARIES} )
endif()

if(CUDAToolkit_FOUND)
	if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
		target_compile_options(FeatureExtractor PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-fPIC>)
	endif()
    target_link_libraries(FeatureExtractor PUBLIC CUDA::cudart_static CUDA::nppc_static)
	set_target_properties(FeatureExtractor PROPERTIES CUDA_ARCHITECTURES ${CUDA_ARCH_LIST})
endif()