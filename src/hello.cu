#include <stdio.h>
#include <iostream>
// #include <cuda_runtime.h>
#include <chrono>

__global__ void hello()
{
    // printf("Hello from GPU!\n");
}

int main(int argc, char **argv)
{
    std::chrono::high_resolution_clock::time_point t0, t1, t2, t3;
    std::chrono::high_resolution_clock::time_point t4, t5, t6, t7;
    if(argc == 1){
        std::cout << "Usage: " << argv[0] << " NGrid NBlock" << std::endl;
        return 1;
    }
    else{
        int NGrid = atoi(argv[1]);
        int NBlock = atoi(argv[2]);
        t0 = std::chrono::high_resolution_clock::now();
        hello<<<NGrid, NBlock>>>();
        t4 = std::chrono::high_resolution_clock::now();
        hello<<<NGrid, NBlock>>>();
        t5 = std::chrono::high_resolution_clock::now();
        hello<<<NGrid, NBlock>>>();
        t6 = std::chrono::high_resolution_clock::now();

        const int nloop = 1e9;
        for(int i=0; i<nloop; ++i){
            if(i % static_cast<int>(nloop/1e2) == 0){
                std::cout << "\t" << static_cast<double>(i)/nloop*100 << "% loop of " << nloop << std::endl;
            }
        }
        t2 = std::chrono::high_resolution_clock::now();
        cudaDeviceSynchronize();
        t3 = std::chrono::high_resolution_clock::now();
        double kernel_time = std::chrono::duration<double, std::micro>(t4 - t0).count();
        double kernel_time2 = std::chrono::duration<double, std::micro>(t5 - t4).count();
        double kernel_time3 = std::chrono::duration<double, std::micro>(t6 - t5).count();
        double loop_time = std::chrono::duration<double, std::micro>(t2 - t1).count();
        double sync_time = std::chrono::duration<double, std::micro>(t3 - t2).count();
        std::cout << "NGrid: " << NGrid << ", NBlock: " << NBlock << std::endl;
        std::cout << "\tnumber of CPU for loop: " << nloop << std::endl;
        std::cout << "\tFirst Kernel function call time: " << kernel_time << " us" << std::endl;
        std::cout << "\tSecond Kernel function call time: " << kernel_time2 << " us" << std::endl;
        std::cout << "\tThird Kernel function call time: " << kernel_time3 << " us" << std::endl;
        std::cout << "\tLoop time: " << loop_time << " us" << std::endl;
        std::cout << "\tSync time: " << sync_time << " us" << std::endl;
        return 0;
    }
}
