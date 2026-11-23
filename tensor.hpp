#pragma once
#include <algorithm>
#include <cassert>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

struct Tensor {
    int rows;
    int cols;
    std::vector<float> data;

    Tensor(int r = 0, int c = 0, float init_val = 0.0f)
        : rows(r), cols(c), data(r * c, init_val) {}

    // Direct 2D -> 1D flat memory indexing
    inline float& at(int r, int c) { return data[r * cols + c]; }
    inline float at(int r, int c) const { return data[r * cols + c]; }

    // Fill with small random weights (breaks symmetry so neurons learn differently)
    void fill_random(float scale = 0.1f, unsigned int seed = 42) {
        std::mt19937 gen(seed);
        std::normal_distribution<float> dist(0.0f, scale);
        for (float& val : data) {
            val = dist(gen);
        }
    }

    // Pretty-print tensor values for debugging and visualization
    void print(const std::string& label = "") const {
        if (!label.empty()) std::cout << label << " (" << rows << "x" << cols << "):\n";
        for (int i = 0; i < rows; ++i) {
            std::cout << "  [ ";
            for (int j = 0; j < cols; ++j) {
                std::cout << std::fixed << std::setprecision(4) << std::setw(7) << at(i, j) << " ";
            }
            std::cout << "]\n";
        }
    }
};

// 1. FAST MATMUL (i-k-j loop order for sequential CPU cache hits & SIMD)
inline Tensor matmul(const Tensor& A, const Tensor& B) {
    assert(A.cols == B.rows && "Matmul dimension mismatch!");
    Tensor C(A.rows, B.cols, 0.0f);

    for (int i = 0; i < A.rows; ++i) {
        for (int k = 0; k < A.cols; ++k) {
            float a_ik = A.at(i, k); // Kept in a single CPU register
            const float* b_row = &B.data[k * B.cols];
            float* c_row = &C.data[i * C.cols];

            for (int j = 0; j < B.cols; ++j) {
                c_row[j] += a_ik * b_row[j];
            }
        }
    }
    return C;
}

// 2. TRANSPOSE: Swap rows and columns (A[i][j] -> T[j][i])
inline Tensor transpose(const Tensor& A) {
    Tensor T(A.cols, A.rows, 0.0f);
    for (int i = 0; i < A.rows; ++i) {
        for (int j = 0; j < A.cols; ++j) {
            T.at(j, i) = A.at(i, j);
        }
    }
    return T;
}

// 3. ADD BIAS: Broadcasts a 1xCols bias row across every row of X
inline Tensor add_bias(const Tensor& X, const Tensor& b) {
    assert(b.rows == 1 && X.cols == b.cols && "Bias shape must be (1 x cols)!");
    Tensor out = X;
    for (int i = 0; i < out.rows; ++i) {
        for (int j = 0; j < out.cols; ++j) {
            out.at(i, j) += b.at(0, j);
        }
    }
    return out;
}

// 4. ReLU ACTIVATION: max(0, x)
inline Tensor relu(const Tensor& X) {
    Tensor out(X.rows, X.cols);
    for (size_t i = 0; i < X.data.size(); ++i) {
        out.data[i] = X.data[i] > 0.0f ? X.data[i] : 0.0f;
    }
    return out;
}

// 5. SOFTMAX: Turns each row of raw logits into probabilities summing to 1.0
inline Tensor softmax(const Tensor& X) {
    Tensor out(X.rows, X.cols);
    for (int i = 0; i < X.rows; ++i) {
        // Subtract row max first so exp() never overflows to infinity
        float max_val = X.at(i, 0);
        for (int j = 1; j < X.cols; ++j) {
            if (X.at(i, j) > max_val) max_val = X.at(i, j);
        }

        float sum = 0.0f;
        for (int j = 0; j < X.cols; ++j) {
            float e = std::exp(X.at(i, j) - max_val);
            out.at(i, j) = e;
            sum += e;
        }

        for (int j = 0; j < X.cols; ++j) {
            out.at(i, j) /= sum;
        }
    }
    return out;
}
