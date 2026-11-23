#pragma once
#include <cassert>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <vector>
#include "tensor.hpp"

// Custom 2-bit packed ternary tensor: stores 4 weights {-1, 0, +1} per byte (16x smaller than float)
struct BitTensor {
    int rows;
    int cols;
    float scale;                 // Average magnitude (gamma) of original float weights
    std::vector<uint8_t> packed; // 4 weights per byte (2 bits each: 00=0, 01=+1, 10=-1)

    BitTensor(int r = 0, int c = 0)
        : rows(r), cols(c), scale(1.0f), packed((r * c + 3) / 4, 0) {}

    // Pack a single {-1, 0, +1} value into its 2-bit slot
    inline void set_trit(int idx, int8_t val) {
        uint8_t code = (val == 1) ? 0b01 : (val == -1 ? 0b10 : 0b00);
        int byte_idx = idx >> 2;          // idx / 4
        int shift = (idx & 3) << 1;       // (idx % 4) * 2
        packed[byte_idx] &= ~(0b11 << shift);
        packed[byte_idx] |= (code << shift);
    }

    // Unpack a single {-1, 0, +1} value from its 2-bit slot
    inline int8_t get_trit(int r, int c) const {
        int idx = r * cols + c;
        uint8_t code = (packed[idx >> 2] >> ((idx & 3) << 1)) & 0b11;
        if (code == 0b01) return 1;
        if (code == 0b10) return -1;
        return 0;
    }

    // BitNet AbsMean Quantization: converts a 32-bit float Tensor -> packed {-1, 0, +1}
    static BitTensor from_float(const Tensor& W) {
        BitTensor B(W.rows, W.cols);
        if (W.data.empty()) return B;

        // 1. Compute average absolute value (gamma)
        double abs_sum = 0.0;
        for (float v : W.data) {
            abs_sum += std::abs(v);
        }
        float gamma = static_cast<float>(abs_sum / W.data.size());
        if (gamma < 1e-8f) gamma = 1.0f;
        B.scale = gamma;

        // 2. Round(W / gamma) clamped to {-1, 0, +1} and pack 4 per byte
        for (size_t i = 0; i < W.data.size(); ++i) {
            float scaled = W.data[i] / gamma;
            int8_t trit = 0;
            if (scaled >= 0.5f) trit = 1;
            else if (scaled <= -0.5f) trit = -1;
            B.set_trit(static_cast<int>(i), trit);
        }
        return B;
    }

    size_t memory_bytes() const {
        return packed.size() * sizeof(uint8_t) + sizeof(scale);
    }

    void print(const std::string& label = "") const {
        if (!label.empty()) {
            std::cout << label << " (" << rows << "x" << cols
                      << ", scale=" << std::fixed << std::setprecision(4) << scale
                      << ", mem=" << memory_bytes() << " B):\n";
        }
        for (int i = 0; i < rows; ++i) {
            std::cout << "  [ ";
            for (int j = 0; j < cols; ++j) {
                int8_t t = get_trit(i, j);
                if (t == 1) std::cout << " +1 ";
                else if (t == -1) std::cout << " -1 ";
                else std::cout << "  0 ";
            }
            std::cout << "]\n";
        }
    }
};

// MULTIPLICATION-FREE MATMUL: Uses ONLY addition and subtraction!
inline Tensor bit_matmul(const Tensor& X, const BitTensor& W) {
    assert(X.cols == W.rows && "BitMatmul dimension mismatch!");
    Tensor C(X.rows, W.cols, 0.0f);

    for (int i = 0; i < X.rows; ++i) {
        for (int k = 0; k < X.cols; ++k) {
            float x_ik = X.at(i, k);
            float* c_row = &C.data[i * C.cols];

            for (int j = 0; j < W.cols; ++j) {
                int8_t w = W.get_trit(k, j);
                if (w == 1) {
                    c_row[j] += x_ik; // Just ADD!
                } else if (w == -1) {
                    c_row[j] -= x_ik; // Just SUBTRACT!
                }
                // If w == 0: do literally nothing!
            }
        }
    }

    // Apply the single scale factor once at the end
    for (float& val : C.data) {
        val *= W.scale;
    }
    return C;
}
