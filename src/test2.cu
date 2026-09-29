#include <cuda_runtime.h>
#include <iostream>

__global__ void vecAdd(float* A, float* B, float* C)
{

}

int main(int argc, char* argv){
    if(argc == 1){
        std::cout << "Usage: \"./bin/test 5 5\" for 5 thread blocks, 5 thread";
    }
    else{

    }
} // int main(int argc, char* argv)