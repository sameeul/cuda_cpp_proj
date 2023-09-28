#!/bin/bash
# Usage: $bash install_prereq_linux.sh $INSTALL_DIR
# Default $INSTALL_DIR = ./local_install
#

export PATH=/home/jovyan/work/cmake-3.27.6-linux-x86_64/bin/:$PATH

if [ -z "$1" ]
then
      echo "No path to the Nyxus source location provided"
      echo "Creating local_install directory"
      LOCAL_INSTALL_DIR="local_install"
else
     LOCAL_INSTALL_DIR=$1
fi

mkdir -p $LOCAL_INSTALL_DIR
mkdir -p $LOCAL_INSTALL_DIR/include

git clone https://github.com/madler/zlib.git
cd zlib
mkdir build_man
cd build_man
cmake -DCMAKE_INSTALL_PREFIX=../../$LOCAL_INSTALL_DIR/ ..  
cmake --build . 
cmake --build . --target install 
cd ../../


for i in {1..5}
do
    curl -L https://github.com/ebiggers/libdeflate/releases/download/v1.19/libdeflate-1.19.tar.gz -o libdeflate-1.19.tar.gz
    if [ -f "libdeflate-1.19.tar.gz" ] ; then
        break
    fi
done

tar -xzf libdeflate-1.19.tar.gz 
cd libdeflate-1.19
PREFIX= LIBDIR=/lib64  DESTDIR=../$LOCAL_INSTALL_DIR/ make  install
cd ../

for i in {1..5}
do
    curl https://download.osgeo.org/libtiff/tiff-4.6.0.tar.gz -o tiff-4.6.0.tar.gz
    if [ -f "tiff-4.6.0.tar.gz" ] ; then
        break
    fi
done

tar -xzf tiff-4.6.0.tar.gz
cd tiff-4.6.0
mkdir build_man
cd build_man/
cmake -DCMAKE_INSTALL_PREFIX=../../$LOCAL_INSTALL_DIR/   -DCMAKE_PREFIX_PATH=../../$LOCAL_INSTALL_DIR/   ..
make install -j4
cd ../../