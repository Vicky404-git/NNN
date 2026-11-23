#include <iostream>
#include "tokenizer.hpp"

int main() {
    BPETokenizer tok;

    std::string corpus =
        "the neural network learns the patterns in the data. "
        "the network predicts the next token in the sequence.";

    // Learn 15 super-bricks on top of the 256 base bytes
    tok.train(corpus, 15);

    std::string sample = "the neural network predicts";
    std::vector<uint32_t> encoded = tok.encode(sample);

    std::cout << "Original bytes : " << sample.size() << "\n";
    std::cout << "Compressed IDs : " << encoded.size() << "\n\n";

    std::cout << "Visual Token Breakdown:\n";
    for (uint32_t id : encoded) {
        std::cout << "[" << id << ":\"" << tok.token_str(id) << "\"] ";
    }
    std::cout << "\n\nDecoded back   : " << tok.decode(encoded) << "\n";

    return 0;
}
