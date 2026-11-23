#include <chrono>
#include <iomanip>
#include <iostream>
#include <vector>
#include "../tensor.hpp"

using Vec2D = std::vector<std::vector<float>>;

// Your original arr_mul.cpp algorithm (vector<vector<T>> + i-j-k loop order)
Vec2D multiply_old_2d(const Vec2D& A, const Vec2D& B) {
    int rowsA = static_cast<int>(A.size());
    int colsA = static_cast<int>(A[0].size());
    int colsB = static_cast<int>(B[0].size());

    Vec2D result(rowsA, std::vector<float>(colsB, 0.0f));

    for (int i = 0; i < rowsA; ++i) {
        for (int j = 0; j < colsB; ++j) {
            for (int k = 0; k < colsA; ++k) {
                result[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    return result;
}

int main() {
    // 512 x 512 = ~134 million multiply-adds per run
    const int N = 512;
    std::cout << "Benchmarking " << N << "x" << N << " Matrix Multiplication...\n\n";

    // 1. Setup Old 2D Matrices
    Vec2D A_old(N, std::vector<float>(N, 1.5f));
    Vec2D B_old(N, std::vector<float>(N, 2.5f));

    // 2. Setup New Flat Tensors
    Tensor A_new(N, N, 1.5f);
    Tensor B_new(N, N, 2.5f);

    // Warmup CPU caches
    volatile float sink = 0.0f;

    // --- Time Old 2D (i-j-k) ---
    auto t1 = std::chrono::high_resolution_clock::now();
    Vec2D C_old = multiply_old_2d(A_old, B_old);
    auto t2 = std::chrono::high_resolution_clock::now();
    sink += C_old[0][0];

    double ms_old = std::chrono::duration<double, std::milli>(t2 - t1).count();

    // --- Time New Flat Tensor (i-k-j) ---
    auto t3 = std::chrono::high_resolution_clock::now();
    Tensor C_new = matmul(A_new, B_new);
    auto t4 = std::chrono::high_resolution_clock::now();
    sink += C_new.at(0, 0);

    double ms_new = std::chrono::duration<double, std::milli>(t4 - t3).count();

    // Calculate GFLOPS (2 * N^3 floating-point operations)
    double ops = 2.0 * N * N * N;
    double gflops_old = (ops / (ms_old / 1000.0)) / 1e9;
    double gflops_new = (ops / (ms_new / 1000.0)) / 1e9;

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Old 2D (i-j-k)      : " << std::setw(8) << ms_old << " ms  (" << gflops_old << " GFLOPS)\n";
    std::cout << "New Tensor (i-k-j)  : " << std::setw(8) << ms_new << " ms  (" << gflops_new << " GFLOPS)\n";
    std::cout << "--------------------------------------------------\n";
    std::cout << "Speedup             : " << (ms_old / ms_new) << "x faster!\n";

    return 0;
}
