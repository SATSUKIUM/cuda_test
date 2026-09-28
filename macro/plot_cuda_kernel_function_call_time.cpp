#include <TH1.h>
#include <TFile.h>
#include <TTree.h>
#include <TCanvas.h>
#include <TPad.h>
#include <TLegend.h>

#include <string>
#include <iostream>
#include <sstream>

void plot_cuda_kernel_function_call_time() {
    int width = 800;
    int height = 600;
    int xmin = 0;
    int xmax = 10;
    int nbins = 200;

    std::string fRootFileName = "./tree/kernel_call_time.root";
    TFile* file = TFile::Open(fRootFileName.c_str(), "READ");
    if (!file || file->IsZombie()) {
        std::cerr << "Error opening file: " << fRootFileName << std::endl;
        return;
    }

    TTree* tree = dynamic_cast<TTree*>(file->Get("tree1"));
    Double_t kernel_call_time;
    Double_t kernel_sync_time;
    tree->SetBranchAddress("kernel_call_time", &kernel_call_time);
    tree->SetBranchAddress("kernel_sync_time", &kernel_sync_time);

    TCanvas* c1 = new TCanvas("c1", "Kernel Function Call Time", width, height);
    TH1D* h1 = new TH1D("h1", "Kernel function call time", nbins, xmin, xmax);
    TH1D* h2 = new TH1D("h2", "Kernel function synchronization time", nbins, xmin, xmax);

    Long64_t nEntries = tree->GetEntries();
    for (Long64_t i = 0; i < nEntries; ++i) {
        tree->GetEntry(i);
        #if 0
        std::cout << "Entry " << i << ": kernel_call_time = " << kernel_call_time
                  << ", kernel_sync_time = " << kernel_sync_time << std::endl;
        #endif
        h1->Fill(kernel_call_time);
        h2->Fill(kernel_sync_time);
    }

    h1->SetLineColor(kBlue);
    h1->SetLineWidth(2);
    h2->SetLineColor(kRed);
    h2->SetLineWidth(2);
    c1->cd();
    h1->Draw();
    h2->Draw("SAME");
    TLegend* legend = new TLegend(0.7, 0.7, 0.9, 0.9);
    legend->AddEntry(h1, "Kernel Call Time", "l");
    legend->AddEntry(h2, "Kernel Sync Time", "l");
    legend->Draw();
    c1->SaveAs("./image/kernel_function_call_time.pdf");
    c1->Update();
    file->Close();
}