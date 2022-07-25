cd src/
mkdir build
cd build
cmake ../
make


cd ../cv/task1/
OPENCV_CPU_DISABLE=AVX2,AVX ../../build/cv/task1/cvtask1 tests/one_way.json
