# NNN — Neural Network from Scratch in C++

A zero-dependency, fast neural network and byte-level BPE tokenizer built from first principles in C++17, complete with a terminal node visualizer.

## Architecture
- `tokenizer.hpp` — Byte-Pair Encoding (BPE) tokenizer using `uint64_t` bit-packed pair counting.
- `tensor.hpp` — Contiguous 1D `float` tensor engine with cache-friendly `i-k-j` matrix multiplication *(Next)*.
- `net.hpp` — 2-layer MLP with backpropagation and SGD *(Planned)*.
- `viz.hpp` — Terminal node & activation visualizer *(Planned)*.

## Build & Run
```
g++ -O3 -march=native -std=c++17 main.cpp -o nnn && ./nnn
```

