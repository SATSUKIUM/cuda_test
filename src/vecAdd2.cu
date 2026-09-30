#include <cuda_runtime.h>
#include "cuda_compat.h" // In this code, cuda::ceil_div()
#include <stdio.h>
#include <iostream>

#include <chrono>
#include <iomanip>

#include <string>
#include <sstream>

#include <vector>

#include <TH1.h>
#include <TCanvas.h>
#include <TPad.h>
#include <TStyle.h>
#include <TFile.h>
#include <TGraph.h>

#include <TROOT.h> // for gROOT->SetBatch(kTRUE);

__global__ void vecAdd(float* A, float* B, float* C, int length)
{
    int workerId = blockDim.x * blockIdx.x + threadIdx.x;

    if(workerId < length){
        C[workerId] = A[workerId] + B[workerId];
    }
    else{
        // printf("Worker %d is out of bounds for length %d\n", workerId, length);
    }
} // __global__ void vecAdd(float* A, float* B, float* C, int length)

__global__ void doWarmUp()
{
    // do "warm up", and nothing else
}

int main(int argc, char* argv[]){
    gROOT->SetBatch(kTRUE); // Disable interactive mode for ROOT

    std::chrono::high_resolution_clock::time_point start, end;
    start = std::chrono::high_resolution_clock::now();
    doWarmUp<<<1, 1>>>();
    cudaDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    double warmUpTime = std::chrono::duration<double, std::milli>(end - start).count();
    std::cout << "Warm-up time: " << std::fixed << std::setprecision(2) << warmUpTime << " ms" << std::endl;

    if(argc == 1){
        std::cout << "Usage: \"./bin/test 5 5\" for 5 thread blocks, 5 thread" << std::endl;
        return 1;
    }
    else{
        int arg1 = std::stoi(argv[1]);
        std::cout << "argument 1: " << arg1 << std::endl;

        // linear additon of two vectors A and B of size 65536, store the result in vector C
        const int length = 65536;
        float* A = new float[length];
        float* B = new float[length];
        float* C = new float[length];

        for(int i = 0; i < length; i++){
            A[i] = 1.0f;
            B[i] = 2.0f;
        }

        const int numThreads = std::stoi(argv[1]);
        const int numBlocks = cuda_compat::ceil_div(length, numThreads);
        std::cout << "numBlocks: " << numBlocks << std::endl;
        std::cout << "numThreads: " << numThreads << std::endl;


        TCanvas* c1 = new TCanvas("c1", "Vector Addition", 800, 600);
        TH1D* h1 = new TH1D("h1", "Vector Addition Execution Time; time (us)", 200, 0, 5000);

        TCanvas* c2 = new TCanvas("c2", "Vector Addition; cummulative elapsed time (us); time (us)", 800, 600);
        TGraph* g1 = new TGraph();

        std::vector<double> kernelTimes;
        kernelTimes.reserve(5e5); // Reserve space for 500,000 elements

        std::chrono::high_resolution_clock::time_point before_loop = std::chrono::high_resolution_clock::now();
        for(int i=0; i<5e5; ++i){
            start = std::chrono::high_resolution_clock::now();
            vecAdd<<<numBlocks, numThreads>>>(A, B, C, length);
            cudaDeviceSynchronize();
            end = std::chrono::high_resolution_clock::now();
            double kernelTime = std::chrono::duration<double, std::micro>(end - start).count();
            // std::cout << i << ": Kernel execution time: " << std::fixed << std::setprecision(2) << kernelTime << " us" << std::endl;
            // h1->Fill(kernelTime);
            // g1->SetPoint(i, i, kernelTime);
            kernelTimes.push_back(kernelTime);
        }
        std::chrono::high_resolution_clock::time_point after_loop = std::chrono::high_resolution_clock::now();
        double totalTime = std::chrono::duration<double, std::milli>(after_loop - before_loop).count();
        std::cout << "Total time for 5e5 iterations: " << std::fixed << std::setprecision(2) << totalTime << " ms" << std::endl;

        // Fill histogram and graph with kernel times
        double elapsed{0.0}, buffer{0.0};
        for(size_t i = 0; i < kernelTimes.size(); ++i){
            buffer = kernelTimes[i];
            elapsed += buffer;
            h1->Fill(buffer);
            g1->SetPoint(i, elapsed, buffer);
        }

        std::vector<double> eventTimes;
        eventTimes.reserve(5e5); // Reserve space for 500,000 elements
        elapsed = 0.0;
        for(int i=0; i<5e5; ++i){
            buffer = kernelTimes[i];
            elapsed += buffer;
            if(buffer > 30.0){
                eventTimes.push_back(elapsed);
            }
        }

        TCanvas* c3 = new TCanvas("c3", "Vector Addition; event time differences (us); entry", 800, 600);
        TH1D* h2 = new TH1D("h2", "Vector Addition; event time differences (us); entry", 400, 0, 1e5);
        double b1,b2;
        for(int i=0; i<eventTimes.size(); ++i){
            if(eventTimes[i] < 1e5){
                continue; // Skip events that are too close to the start
            }
            for(int j=i+1; j<eventTimes.size(); ++j){
                b1 = eventTimes[i];
                b2 = eventTimes[j];
                if(b2 - b1 < 1e5){ // 100 ms
                    h2->Fill(b2 - b1);
                } // if(eventTimes[j] - eventTimes[i] < 100.0){
            } // for(int j=i+1; j<eventTimes.size(); ++j){
        } // for(int i=0; i<eventTimes.size(); ++i){

        c1->cd();
        gPad->SetGrid();
        gPad->SetLogy();
        h1->Draw();

        // Save the canvas as "./${outdir}/vecAdd2.pdf"
        std::string outdir = "./image/";
        c1->SaveAs((outdir + "vecAdd2.pdf").c_str());
        // c1->SaveAs((outdir + "vecAdd2.png").c_str());

        c2->cd();
        c2->SetLeftMargin(0.15);
        gPad->SetGrid();
        g1->SetMarkerSize(0.5);
        g1->SetMarkerStyle(20);
        g1->SetTitle("Vector Addition; cummulative elapsed time (us); time (us)");
        g1->Draw("ALP");
        c2->SaveAs((outdir + "vecAdd2_graph.pdf").c_str());
        // c2->SaveAs((outdir + "vecAdd2_graph.png").c_str());

        c3->cd();
        gPad->SetGrid();
        h2->Draw();
        c3->SaveAs((outdir + "vecAdd2_event_time_differences.pdf").c_str());
        // c3->SaveAs((outdir + "vecAdd2_event_time_differences.png").c_str());



        for(int i = 0; i < length; i++){
            if(C[i] != 3.0f){
                std::cout << "Error: C[" << i << "] = " << C[i] << std::endl;
                return 1;
            }
        }
        std::cout << "Success: All values in C are correct." << std::endl;
        return 0;
    }
} // int main(int argc, char* argv)