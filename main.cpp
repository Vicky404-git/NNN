#include <iostream>
#include "tokenizer.hpp"
#include "tensor.hpp"

int main() {
    // 1. Tokenize text
    BPETokenizer tok;
    std::string corpus =
        "the neural network learns the patterns in the data. "
        "the network predicts the next token in the sequence.";
    tok.train(corpus, 20);

    std::vector<uint32_t> ids = tok.encode("the neural network");
    int vocab_size = static_cast<int>(tok.vocab.size());
    int embed_dim = 4; // Each token becomes 4 numbers

    // 2. Create random Embedding Table (vocab_size x 4) and Weight Matrix (4 x vocab_size)
    Tensor W_emb(vocab_size, embed_dim);
    W_emb.fill_random(0.5f, 101);

    Tensor W_out(embed_dim, vocab_size);
    W_out.fill_random(0.5f, 202);

    Tensor b_out(1, vocab_size, 0.0f);

    // 3. Look up embeddings for our encoded tokens (num_tokens x 4)
    Tensor X(static_cast<int>(ids.size()), embed_dim);
    for (size_t i = 0; i < ids.size(); ++i) {
        for (int j = 0; j < embed_dim; ++j) {
            X.at(i, j) = W_emb.at(ids[i], j);
        }
    }
    X.print("Token Embeddings (X)");

    // 4. Forward pass: Hidden = ReLU(X) -> Logits = Hidden * W_out + b -> Probs = Softmax(Logits)
    Tensor hidden = relu(X);
    hidden.print("\nAfter ReLU (Negatives zeroed out)");

    Tensor logits = add_bias(matmul(hidden, W_out), b_out);
    Tensor probs = softmax(logits);

    // Show top predicted next-token probability for the last token
    int last_row = probs.rows - 1;
    int best_id = 0;
    for (int j = 1; j < probs.cols; ++j) {
        if (probs.at(last_row, j) > probs.at(last_row, best_id)) {
            best_id = j;
        }
    }

    std::cout << "\nUntrained guess after \"" << tok.token_str(ids.back()) << "\": "
              << "[" << best_id << ":\"" << tok.token_str(best_id) << "\"] "
              << "(prob = " << probs.at(last_row, best_id) * 100.0f << "%)\n";

    return 0;
}
