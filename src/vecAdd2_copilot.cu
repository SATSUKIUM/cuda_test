#include <cuda_runtime.h>
#include "cuda_compat.h" // In this code, cuda::ceil_div()
#include <stdio.h>
#include <iostream>

#include <chrono>
#include <cmath>
#include <complex>
#include <iomanip>

#include <string>
#include <sstream>

#include <vector>
#include <utility>

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

void fftInPlace(std::vector<std::complex<double>>& values)
{
    const size_t size = values.size();
    for(size_t i = 1, j = 0; i < size; ++i){
        size_t bit = size >> 1;
        for(; j & bit; bit >>= 1){
            j ^= bit;
        }
        j ^= bit;
        if(i < j){
            std::swap(values[i], values[j]);
        }
    }

    const double pi = std::acos(-1.0);
    for(size_t blockSize = 2; blockSize <= size; blockSize <<= 1){
        const double angle = -2.0 * pi / blockSize;
        const std::complex<double> phase(std::cos(angle), std::sin(angle));
        for(size_t blockStart = 0; blockStart < size; blockStart += blockSize){
            std::complex<double> multiplier(1.0, 0.0);
            const size_t halfBlockSize = blockSize / 2;
            for(size_t j = 0; j < halfBlockSize; ++j){
                const std::complex<double> even = values[blockStart + j];
                const std::complex<double> odd = multiplier * values[blockStart + j + halfBlockSize];
                values[blockStart + j] = even + odd;
                values[blockStart + j + halfBlockSize] = even - odd;
                multiplier *= phase;
            }
        }
    }
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
        const int numBlocks =  cuda_compat::ceil_div(length, numThreads);
        std::cout << "numBlocks: " << numBlocks << std::endl;
        std::cout << "numThreads: " << numThreads << std::endl;


        TCanvas* c1 = new TCanvas("c1", "Vector Addition", 800, 600);
        TH1D* h1 = new TH1D("h1", "Vector Addition Execution Time; time (us)", 200, 0, 5000);

        TCanvas* c2 = new TCanvas("c2", "Vector Addition; cummulative elapsed time (us); time (us)", 800, 600);
        TGraph* g1 = new TGraph();

        constexpr int numSamples = 524288; // 2^19, suitable for a radix-2 FFT
        constexpr double eventThresholdUs = 10.0;
        std::vector<double> kernelTimes;
        kernelTimes.reserve(numSamples);

        std::chrono::high_resolution_clock::time_point before_loop = std::chrono::high_resolution_clock::now();
        for(int i=0; i<numSamples; ++i){
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
        std::cout << "Total time for " << numSamples << " iterations: "
                  << std::fixed << std::setprecision(2) << totalTime << " ms" << std::endl;

        const std::string outdir = "./image/";
        // Analyze the binary event series kernelTimes[i] > eventThresholdUs.
        std::vector<double> eventSignal(numSamples);
        size_t eventCount = 0;
        double sumKernelTime = 0.0;
        for(int i = 0; i < numSamples; ++i){
            sumKernelTime += kernelTimes[i];
            if(kernelTimes[i] > eventThresholdUs){
                eventSignal[i] = 1.0;
                ++eventCount;
            }
        }

        const double meanKernelTimeUs = sumKernelTime / numSamples;
        const double sampleFrequencyHz = 1.0e6 / meanKernelTimeUs;
        const double eventFraction = static_cast<double>(eventCount) / numSamples;
        const double pi = std::acos(-1.0);
        for(int i = 0; i < numSamples; ++i){
            // Remove the DC component and reduce spectral leakage at the edges.
            eventSignal[i] = (eventSignal[i] - eventFraction)
                           * (0.5 - 0.5 * std::cos(2.0 * pi * i / (numSamples - 1)));
        }

        std::vector<std::complex<double>> fftValues(numSamples);
        for(int i = 0; i < numSamples; ++i){
            fftValues[i] = eventSignal[i];
        }
        fftInPlace(fftValues);

        const int numFrequencyBins = numSamples / 2;
        TGraph* fftGraph = new TGraph(numFrequencyBins);
        int peakBin = 1;
        double peakPower = 0.0;
        for(int i = 0; i < numFrequencyBins; ++i){
            const double power = std::norm(fftValues[i]);
            const double frequencyHz = sampleFrequencyHz * i / numSamples;
            fftGraph->SetPoint(i, frequencyHz, power);
            if(i > 0 && power > peakPower){
                peakPower = power;
                peakBin = i;
            }
        }

        const double peakFrequencyHz = sampleFrequencyHz * peakBin / numSamples;
        std::cout << "Events above " << eventThresholdUs << " us: " << eventCount
                  << " / " << numSamples << " (" << 100.0 * eventFraction << " %)" << std::endl;
        std::cout << "FFT sample frequency: " << sampleFrequencyHz
                  << " Hz, frequency resolution: " << sampleFrequencyHz / numSamples
                  << " Hz" << std::endl;
        std::cout << "Largest non-DC event-spectrum peak: " << peakFrequencyHz
                  << " Hz" << std::endl;

        TCanvas* c3 = new TCanvas("c3", "Event Spectrum", 800, 600);
        c3->SetLeftMargin(0.15);
        gPad->SetGrid();
        fftGraph->SetTitle("FFT of kernelTimes > 10 us;frequency (Hz);power");
        fftGraph->Draw("AL");
        c3->SaveAs((outdir + "vecAdd2_fft.pdf").c_str());
        TFile fftFile((outdir + "vecAdd2_fft.root").c_str(), "RECREATE");
        fftGraph->Write("fftGraph");
        c3->Write("fftCanvas");
        fftFile.Close();

        // Fill histogram and graph with kernel times
        double elapsed{0.0}, buffer{0.0};
        for(size_t i = 0; i < kernelTimes.size(); ++i){
            buffer = kernelTimes[i];
            elapsed += buffer;
            h1->Fill(buffer);
            g1->SetPoint(i, elapsed, buffer);
        }

        c1->cd();
        gPad->SetGrid();
        gPad->SetLogy();
        h1->Draw();

        // Save the canvas as "./${outdir}/vecAdd2.pdf"
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