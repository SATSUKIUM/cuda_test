#include <stdio.h>
#include <iostream>
// #include <cuda_runtime.h>
#include <chrono>

// ROOT
#include <TH1.h>
#include <TCanvas.h>
#include <TPad.h>
#include <TFile.h>
#include <TTree.h>


__global__ void hello()
{
    // printf("Hello from GPU!\n");

    return;
}

__global__ void doNothing()
{
    // do nothing

    return;
}

int main(int argc, char **argv)
{
    int width = 800;
    int height = 600;
    int xmin = 0;
    int xmax = 10;
    int nbins = 200;

    std::chrono::high_resolution_clock::time_point start_first_kernel, end_first_kernel, start_second_kernel, end_second_kernel, start_third_kernel, end_third_kernel;
    std::chrono::high_resolution_clock::time_point start_kernel_loop, mid_kernel_loop, end_kernel_loop;
    std::chrono::high_resolution_clock::time_point start_hist_fill, end_hist_fill, start_tree_fill, end_tree_fill;

    // ROOT initialization
    std::string treedir = "./tree/";
    std::string treename = "kernel_call_time.root";
    TFile *tfile = new TFile((treedir + treename).c_str(), "RECREATE");
    TTree *ttree1 = new TTree("tree1", "Kernel call time");
    Double_t kernel_call_time{0.0};
    Double_t kernel_sync_time{0.0};
    ttree1->Branch("kernel_call_time", &kernel_call_time, "kernel_call_time/D");
    ttree1->Branch("kernel_sync_time", &kernel_sync_time, "kernel_sync_time/D");

    TTree *ttree2 = new TTree("tree2", "root function call time");
    Double_t hist_fill_time{0.0};
    Double_t tree_fill_time{0.0};
    ttree2->Branch("hist_fill_time", &hist_fill_time, "hist_fill_time/D");
    ttree2->Branch("tree_fill_time", &tree_fill_time, "tree_fill_time/D");




    if(argc == 1){
        std::cout << "Usage: " << argv[0] << " NGrid NBlock" << std::endl;
        return 1;
    } // if(argc == 1)
    else{
        int NGrid = atoi(argv[1]);
        int NBlock = atoi(argv[2]);

        start_first_kernel = std::chrono::high_resolution_clock::now();
        hello<<<NGrid, NBlock>>>();
        end_first_kernel = std::chrono::high_resolution_clock::now();

        start_second_kernel = std::chrono::high_resolution_clock::now();
        hello<<<NGrid, NBlock>>>();
        end_second_kernel = std::chrono::high_resolution_clock::now();

        start_third_kernel = std::chrono::high_resolution_clock::now();
        hello<<<NGrid, NBlock>>>();
        end_third_kernel = std::chrono::high_resolution_clock::now();

        std::cout << "NGrid: " << NGrid << ", NBlock: " << NBlock << std::endl;
        double kernel_time = std::chrono::duration<double, std::micro>(end_first_kernel - start_first_kernel).count();
        double kernel_time2 = std::chrono::duration<double, std::micro>(end_second_kernel - start_second_kernel).count();
        double kernel_time3 = std::chrono::duration<double, std::micro>(end_third_kernel - start_third_kernel).count();
        std::cout << "\t1st Kernel function\"__global__ hello()\" call time: " << kernel_time << " us" << std::endl;
        std::cout << "\t2nd Kernel function\"__global__ hello()\" call time: " << kernel_time2 << " us" << std::endl;
        std::cout << "\t3rd Kernel function\"__global__ hello()\" call time: " << kernel_time3 << " us" << std::endl;

        start_first_kernel = std::chrono::high_resolution_clock::now();
        doNothing<<<NGrid, NBlock>>>();
        end_first_kernel = std::chrono::high_resolution_clock::now();
        start_second_kernel = std::chrono::high_resolution_clock::now();
        doNothing<<<NGrid, NBlock>>>();
        end_second_kernel = std::chrono::high_resolution_clock::now();
        start_third_kernel = std::chrono::high_resolution_clock::now();
        doNothing<<<NGrid, NBlock>>>();
        end_third_kernel = std::chrono::high_resolution_clock::now();

        std::cout << "NGrid: " << NGrid << ", NBlock: " << NBlock << std::endl;
        kernel_time = std::chrono::duration<double, std::micro>(end_first_kernel - start_first_kernel).count();
        kernel_time2 = std::chrono::duration<double, std::micro>(end_second_kernel - start_second_kernel).count();
        kernel_time3 = std::chrono::duration<double, std::micro>(end_third_kernel - start_third_kernel).count();
        std::cout << "\t1st Kernel function\"__global__ doNothing()\" call time: " << kernel_time << " us" << std::endl;
        std::cout << "\t2nd Kernel function\"__global__ doNothing()\" call time: " << kernel_time2 << " us" << std::endl;
        std::cout << "\t3rd Kernel function\"__global__ doNothing()\" call time: " << kernel_time3 << " us" << std::endl;

        start_first_kernel = std::chrono::high_resolution_clock::now();
        hello<<<NGrid, NBlock>>>();
        end_first_kernel = std::chrono::high_resolution_clock::now();
        start_second_kernel = std::chrono::high_resolution_clock::now();
        hello<<<NGrid, NBlock>>>();
        end_second_kernel = std::chrono::high_resolution_clock::now();

        kernel_time = std::chrono::duration<double, std::micro>(end_first_kernel - start_first_kernel).count();
        kernel_time2 = std::chrono::duration<double, std::micro>(end_second_kernel - start_second_kernel).count();
        std::cout << "\t4th Kernel function\"__global__ hello()\" call time: " << kernel_time << " us" << std::endl;
        std::cout << "\t5th Kernel function\"__global__ hello()\" call time: " << kernel_time2 << " us" << std::endl;

        start_first_kernel = std::chrono::high_resolution_clock::now();
        doNothing<<<NGrid, NBlock>>>();
        end_first_kernel = std::chrono::high_resolution_clock::now();
        start_second_kernel = std::chrono::high_resolution_clock::now();
        doNothing<<<NGrid, NBlock>>>();
        end_second_kernel = std::chrono::high_resolution_clock::now();

        kernel_time = std::chrono::duration<double, std::micro>(end_first_kernel - start_first_kernel).count();
        kernel_time2 = std::chrono::duration<double, std::micro>(end_second_kernel - start_second_kernel).count();
        std::cout << "\t4th Kernel function\"__global__ doNothing()\" call time: " << kernel_time << " us" << std::endl;
        std::cout << "\t5th Kernel function\"__global__ doNothing()\" call time: " << kernel_time2 << " us" << std::endl;

        const int nloop = 1e6;
        TH1D *h1 = new TH1D("h1", "Kernel call time", nbins, xmin, xmax);
        for(int i=0; i<nloop; ++i){
            start_kernel_loop = std::chrono::high_resolution_clock::now();
            doNothing<<<NGrid, NBlock>>>();
            mid_kernel_loop = std::chrono::high_resolution_clock::now();
            cudaDeviceSynchronize();
            end_kernel_loop = std::chrono::high_resolution_clock::now();
            kernel_call_time = std::chrono::duration<double, std::micro>(end_kernel_loop - start_kernel_loop).count();
            kernel_sync_time = std::chrono::duration<double, std::micro>(end_kernel_loop - mid_kernel_loop).count();
            start_hist_fill = std::chrono::high_resolution_clock::now();
            h1->Fill(kernel_call_time);
            end_hist_fill = std::chrono::high_resolution_clock::now();

            start_tree_fill = std::chrono::high_resolution_clock::now();
            ttree1->Fill();
            end_tree_fill = std::chrono::high_resolution_clock::now();

            hist_fill_time = std::chrono::duration<double, std::micro>(end_hist_fill - start_hist_fill).count();
            tree_fill_time = std::chrono::duration<double, std::micro>(end_tree_fill - start_tree_fill).count();

            ttree2->Fill();

            if(i % static_cast<int>(nloop/1e2) == 0){
                std::cout << "\t" << static_cast<double>(i)/nloop*100 << "% loop of " << nloop << std::endl;
            }
        } // for(int i=0; i<nloop; ++i)

        // finilize kernel function
        std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
        cudaDeviceSynchronize();
        std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();

        std::cout << "\tnumber of CPU for loop: " << nloop << std::endl;
        // ROOT Draw
        TCanvas *c1 = new TCanvas("c1", "Kernel call time", width, height);
        h1->Draw();

        // Save image
        std::string outdir = "./image/";
        std::string outname = "kernel_call_time.pdf";
        c1->SaveAs((outdir + outname).c_str());

        // write to TTree and close TFile
        ttree1->Write();
        ttree2->Write();
        tfile->Close();
        return 0;
    } // else
} // int main(int argc, char **argv)
