#include <cuda_runtime.h>
#include <cuda/cmath> // In this code, cuda::ceil_div()
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

#include <TVirtualFFT.h>
#include <TH1D.h>
#include <TCanvas.h>
#include <TMath.h>
#include <iostream>
#include <vector>
#include <cmath>

#include <numeric>

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
        const int numBlocks = cuda::ceil_div(length, numThreads);
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
        double totalTime_ = std::chrono::duration<double, std::milli>(after_loop - before_loop).count();
        std::cout << "Total time for 5e5 iterations: " << std::fixed << std::setprecision(2) << totalTime_ << " ms" << std::endl;

        // Fill histogram and graph with kernel times
        double elapsed_buf{0.0}, buffer{0.0};
        for(size_t i = 0; i < kernelTimes.size(); ++i){
            buffer = kernelTimes[i];
            elapsed_buf += buffer;
            h1->Fill(buffer);
            g1->SetPoint(i, elapsed_buf, buffer);
        }

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



        for(int i = 0; i < length; i++){
            if(C[i] != 3.0f){
                std::cout << "Error: C[" << i << "] = " << C[i] << std::endl;
                return 1;
            }
        }
        std::cout << "Success: All values in C are correct." << std::endl;

        // FFT

        double elapsed = 0.0;
        const double skipTime = 1e5; // 100 ms
        const double binWidth = 100.0;  // 100 us

        const double totalTime =
            std::accumulate(kernelTimes.begin(),
                            kernelTimes.end(), 0.0);

        const size_t numBins =
            static_cast<size_t>(
                (totalTime - skipTime) / binWidth
            );

        std::vector<double> binSum(numBins, 0.0);
        std::vector<size_t> binCount(numBins, 0);

double analysisStart = -1.0;

for (size_t i = 0; i < kernelTimes.size(); ++i) {
    const double currentTime = elapsed;
    elapsed += kernelTimes[i];

    if (currentTime < skipTime) {
        continue;
    }

    if (analysisStart < 0.0) {
        analysisStart = currentTime;
    }

    const size_t bin =
        static_cast<size_t>(
            (currentTime - analysisStart) / binWidth
        );
    
if (bin < 3) {
    std::cout << "i=" << i
              << ", currentTime=" << currentTime
              << ", bin=" << bin
              << ", count=" << binCount[bin]
              << std::endl;
}

    if (bin >= binSum.size()) {
        binSum.resize(bin + 1, 0.0);
        binCount.resize(bin + 1, 0);
    }

    binSum[bin] += kernelTimes[i];
    binCount[bin]++;
}

        // binSum, binCount は集計済みとする
        const int N = static_cast<int>(binSum.size());
        // const double binWidth = 100.0; // [us]
        const double dt = binWidth * 1e-6; // [s]
        const double fs = 1.0 / dt; // Sampling frequency [Hz]

        // FFT input
        std::vector<double> fftInput(N);

        double totalSum = 0.0;
        size_t totalCount = 0;

        // 全体の平均値
        for (int i = 0; i < N; ++i) {
            totalSum += binSum[i];
            totalCount += binCount[i];
        }

        if (N < 2 || totalCount == 0) {
            std::cerr << "Invalid FFT input." << std::endl;
            std::cout << "N: " << N << ", totalCount: " << totalCount << std::endl;
            return 1;
        }

        const double globalMean =
            totalSum / static_cast<double>(totalCount);

        // 各ビンの平均値から全体の平均値を引く
        for (int i = 0; i < N; ++i) {
            if (binCount[i] == 0) {
                std::cerr << "Empty bin: " << i << std::endl;
                return 1;
            }

            const double binMean =
                binSum[i] / static_cast<double>(binCount[i]);

            fftInput[i] = binMean - globalMean;
        }

        // ROOT FFT (real to complex)
        int n[1] = {N};
        TVirtualFFT* fft =
            TVirtualFFT::FFT(1, n, "R2C ES K");

        if (!fft) {
            std::cerr << "Failed to create FFT." << std::endl;
            return 1;
        }

        fft->SetPoints(fftInput.data());
        fft->Transform();

        // FFT output: N/2 + 1 complex values
        const int nFreq = N / 2 + 1;
        const double df = fs / N; // Frequency resolution [Hz]

        // Histogram
        TH1D* hFFT = new TH1D(
            "hFFT",
            "FFT Power Spectrum;Frequency [Hz];Power [#mus^{2}]",
            nFreq,
            -0.5 * df,
            (nFreq - 0.5) * df
        );

        for (int k = 0; k < nFreq; ++k) {
            double re, im;
            fft->GetPointComplex(k, re, im);

            const double power = re * re + im * im;
            hFFT->SetBinContent(k + 1, power);
        }

        hFFT->GetXaxis()->SetRangeUser(0.0, 300.0);

        TCanvas* cFFT = new TCanvas(
            "cFFT", "FFT Spectrum", 1000, 600
        );

        cFFT->SetGrid();
        cFFT->SetLogy();
        hFFT->Draw("HIST");

        cFFT->SaveAs("./image/vecAdd2_fft.pdf");

        delete fft;


        return 0;
    }
} // int main(int argc, char* argv)