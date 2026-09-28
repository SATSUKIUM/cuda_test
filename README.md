# README.md
## build
実行する環境のGPUのCompute Capacityに合わせてオプション引数を指定する。(例: ver8.6のGPU)
```
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build
```