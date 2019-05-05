# Gaussian process on GPU
Gaussian process regression project in C++, that leverage the GPU (cuda)\\

### General notes:
After installing cuda (in the following I am using cuda-9.0)
- Add the following in your ~/.bashrc file
```$xslt
export PATH=$PATH:/usr/local/cuda-9.0/bin
export LD_LIBRARY_PATH=LD_LIBRARY_PATH:/usr/local/cuda/lib64
```
Check if the following running in the terminal:
```$xslt
>> nvcc which
out: /usr/local/cuda-9.0/bin/nvcc
>> nvcc -V
out: information about  nvcc
```
- Change the GPU architecture inside the CMakeLists.txt file<br> In the line 10:
```$xslt
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_<num>,code=sm_<num>")
```
 You can find the numbers that correspond with your GPU [in this link](https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/)
- Change include directories to match your installation:
```$xslt
if ($ENV{CLION_IDE})
    include_directories(/usr/local/cuda-9.0/include)
endif ()
```

### notes when using Clion:
1. Add *.cu file extension to the C++ file type, the steps are:
```$xslt
File -> Setting -> Editor -> FileTypes -> C\C++ 
Click +
add *.cu
```
2. Build the project:
```$xslt
Build -> Build 'GP_GPU'
```
3. Edit configuration:
```$xslt
Run -> Edit Configuration
Change the executable to the file with the same as your project inside :
cmake-build-debug/CMakeFiles
```