#include <iostream>
#include "tokenizer.hpp"
#include "tensor.hpp"
#include "bitnet.hpp"

int main() {
    // 1. Test your exact row from earlier!
    Tensor demo(1, 4);
    demo.data = {0.0032f, 0.0555f, -0.3160f, 0.5623f};
    demo.print("Original 32-bit Float Row");

    BitTensor demo_bit = BitTensor::from_float(demo);
    demo_bit.print("Quantized 2-bit BitTensor");
    std::cout << "Memory: " << demo.data.size() * sizeof(float) << " bytes -> "
              << demo_bit.memory_bytes() << " bytes!\n\n";

    // 2. Run a BitNet Forward Pass on tokenized text
    BPETokenizer tok;
    std::string corpus =
        "the neural network learns the patterns in the data. "
        "the network predicts the next token in the sequence.";
    tok.train(corpus, 20);

    std::vector<uint32_t> ids = tok.encode("the neural network");
    int vocab_size = static_cast<int>(tok.vocab.size());
    int embed_dim = 8;

    Tensor W_emb(vocab_size, embed_dim);
    W_emb.fill_random(0.5f, 101);

    // Create full-precision weights, then compress to 2-bit {-1, 0, +1}
    Tensor W_out_fp32(embed_dim, vocab_size);
    W_out_fp32.fill_random(0.5f, 202);
    BitTensor W_out_bit = BitTensor::from_float(W_out_fp32);

    std::cout << "W_out Float32 size : " << W_out_fp32.data.size() * sizeof(float) << " bytes\n";
    std::cout << "W_out BitNet size  : " << W_out_bit.memory_bytes() << " bytes ("
              << (W_out_fp32.data.size() * sizeof(float)) / W_out_bit.memory_bytes() << "x smaller)\n\n";

    // Gather embeddings -> ReLU -> Multiplication-Free bit_matmul -> Softmax
    Tensor X(static_cast<int>(ids.size()), embed_dim);
    for (size_t i = 0; i < ids.size(); ++i) {
        for (int j = 0; j < embed_dim; ++j) {
            X.at(i, j) = W_emb.at(ids[i], j);
        }
    }

    Tensor hidden = relu(X);
    Tensor logits = bit_matmul(hidden, W_out_bit); // ZERO multiplications in inner loop!
    Tensor probs = softmax(logits);

    int last_row = probs.rows - 1;
    int best_id = 0;
    for (int j = 1; j < probs.cols; ++j) {
        if (probs.at(last_row, j) > probs.at(last_row, best_id)) {
            best_id = j;
        }
    }

    std::cout << "BitNet Forward Pass guess after \"" << tok.token_str(ids.back()) << "\": "
              << "[" << best_id << ":\"" << tok.token_str(best_id) << "\"] "
              << "(prob = " << probs.at(last_row, best_id) * 100.0f << "%)\n";

    return 0;
}
