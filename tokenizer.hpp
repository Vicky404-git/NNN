#pragma once
#include <cstdint>
#include <iostream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

struct MergeRule {
    uint32_t left;
    uint32_t right;
    uint32_t new_id;
};

class BPETokenizer {
public:
    std::vector<MergeRule> merges;
    // Maps each token ID -> its raw byte sequence (for decoding back to text)
    std::vector<std::vector<uint8_t>> vocab;

    BPETokenizer() {
        // Bricks 0..255 are just the raw single bytes
        vocab.resize(256);
        for (uint32_t i = 0; i < 256; ++i) {
            vocab[i] = {static_cast<uint8_t>(i)};
        }
    }

    // 1. TRAIN: Learn 'num_merges' new super-bricks from sample text
    void train(const std::string& text, int num_merges) {
        // Convert text into initial list of byte IDs (0..255)
        std::vector<uint32_t> ids;
        ids.reserve(text.size());
        for (unsigned char c : text) {
            ids.push_back(static_cast<uint32_t>(c));
        }

        for (int step = 0; step < num_merges; ++step) {
            if (ids.size() < 2) break;

            // Count every adjacent pair using a 64-bit packed key
            std::unordered_map<uint64_t, int> counts;
            uint64_t best_pair = 0;
            int max_count = 0;

            for (size_t i = 0; i + 1 < ids.size(); ++i) {
                uint64_t pair = (static_cast<uint64_t>(ids[i]) << 32) | ids[i + 1];
                int c = ++counts[pair];
                if (c > max_count) {
                    max_count = c;
                    best_pair = pair;
                }
            }

            // Stop if no pair repeats
            if (max_count < 2) break;

            uint32_t left = static_cast<uint32_t>(best_pair >> 32);
            uint32_t right = static_cast<uint32_t>(best_pair & 0xFFFFFFFF);
            uint32_t new_id = static_cast<uint32_t>(vocab.size());

            // Save the rule and build the new super-brick's byte sequence
            merges.push_back({left, right, new_id});
            std::vector<uint8_t> merged_bytes = vocab[left];
            merged_bytes.insert(merged_bytes.end(), vocab[right].begin(), vocab[right].end());
            vocab.push_back(std::move(merged_bytes));

            // Replace all occurrences of (left, right) with new_id in-place
            ids = apply_merge(ids, left, right, new_id);
        }
    }

    // 2. ENCODE: Turn any string into token IDs using learned rules
    std::vector<uint32_t> encode(const std::string& text) const {
        std::vector<uint32_t> ids;
        ids.reserve(text.size());
        for (unsigned char c : text) {
            ids.push_back(static_cast<uint32_t>(c));
        }

        for (const auto& rule : merges) {
            ids = apply_merge(ids, rule.left, rule.right, rule.new_id);
        }
        return ids;
    }

    // 3. DECODE: Turn token IDs back into human-readable text
    std::string decode(const std::vector<uint32_t>& ids) const {
        std::string out;
        for (uint32_t id : ids) {
            if (id < vocab.size()) {
                for (uint8_t b : vocab[id]) {
                    out.push_back(static_cast<char>(b));
                }
            }
        }
        return out;
    }

    // Helper for your visual inspector later: see what a single ID looks like
    std::string token_str(uint32_t id) const {
        if (id >= vocab.size()) return "<UNK>";
        std::string s;
        for (uint8_t b : vocab[id]) {
            if (b == '\n') s += "\\n";
            else if (b == ' ') s += "·"; // Visible dot for spaces!
            else s += static_cast<char>(b);
        }
        return s;
    }

private:
    static std::vector<uint32_t> apply_merge(
        const std::vector<uint32_t>& ids, uint32_t left, uint32_t right, uint32_t new_id) {
        std::vector<uint32_t> out;
        out.reserve(ids.size());
        size_t i = 0;
        while (i < ids.size()) {
            if (i + 1 < ids.size() && ids[i] == left && ids[i + 1] == right) {
                out.push_back(new_id);
                i += 2; // Skip both glued bricks
            } else {
                out.push_back(ids[i]);
                i += 1;
            }
        }
        return out;
    }
};
