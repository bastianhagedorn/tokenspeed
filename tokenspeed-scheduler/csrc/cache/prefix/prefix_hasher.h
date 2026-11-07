// Copyright (c) 2026 LightSeek Foundation
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma once

#include <bit>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include <openssl/sha.h>

#include "utils.h"

namespace tokenspeed {

// Append each byte of [bytes, bytes+n) to out as two lowercase hex characters.
inline void AppendHexBytes(std::string& out, const uint8_t* bytes, std::size_t n) {
    static constexpr char kHex[] = "0123456789abcdef";
    for (std::size_t i = 0; i < n; ++i) {
        out.push_back(kHex[bytes[i] >> 4]);
        out.push_back(kHex[bytes[i] & 0xf]);
    }
}

// Absorb a uint32_t into the hash as 4 little-endian bytes. Encodes each token id
// and the count/length prefixes that frame extra_keys.
inline void Sha256UpdateU32LE(SHA256_CTX& ctx, uint32_t v) {
    uint8_t buf[4] = {
        static_cast<uint8_t>(v),
        static_cast<uint8_t>(v >> 8),
        static_cast<uint8_t>(v >> 16),
        static_cast<uint8_t>(v >> 24),
    };
    SHA256_Update(&ctx, buf, 4);
}

// Encode a 32-byte SHA-256 digest as a 64-char lowercase hex string.
inline std::string DigestToHex(const unsigned char* digest) {
    std::string out;
    out.reserve(SHA256_DIGEST_LENGTH * 2);
    AppendHexBytes(out, digest, SHA256_DIGEST_LENGTH);
    return out;
}

// Decode a hex string back into its raw bytes (inverse of DigestToHex).
inline std::vector<uint8_t> HexToBytes(const std::string& hex) {
    std::vector<uint8_t> bytes;
    bytes.reserve(hex.size() / 2);
    for (std::size_t i = 0; i + 1 < hex.size(); i += 2) {
        uint8_t hi = (hex[i] >= 'a')   ? static_cast<uint8_t>(hex[i] - 'a' + 10)
                     : (hex[i] >= 'A') ? static_cast<uint8_t>(hex[i] - 'A' + 10)
                                       : static_cast<uint8_t>(hex[i] - '0');
        uint8_t lo = (hex[i + 1] >= 'a')   ? static_cast<uint8_t>(hex[i + 1] - 'a' + 10)
                     : (hex[i + 1] >= 'A') ? static_cast<uint8_t>(hex[i + 1] - 'A' + 10)
                                           : static_cast<uint8_t>(hex[i + 1] - '0');
        bytes.push_back(static_cast<uint8_t>((hi << 4) | lo));
    }
    return bytes;
}

// Absorb token ids as consecutive 4-byte little-endian words on any host: each
// id's bytes go into a 1 KiB buffer, absorbed whenever it fills and at the end.
// Sha256UpdateTokensLE uses this on hosts that are not little-endian; it is a
// separate function so tests run it on little-endian hosts too.
inline void Sha256UpdateTokensBytewiseLE(SHA256_CTX& ctx, std::span<const std::int32_t> tokens) {
    uint8_t buf[1024];
    std::size_t used = 0;
    for (std::int32_t t : tokens) {
        const auto v = static_cast<uint32_t>(t);
        buf[used++] = static_cast<uint8_t>(v);
        buf[used++] = static_cast<uint8_t>(v >> 8);
        buf[used++] = static_cast<uint8_t>(v >> 16);
        buf[used++] = static_cast<uint8_t>(v >> 24);
        if (used == sizeof(buf)) {
            SHA256_Update(&ctx, buf, used);
            used = 0;
        }
    }
    if (used > 0) {
        SHA256_Update(&ctx, buf, used);
    }
}

// Absorb token ids as consecutive 4-byte little-endian words: the bytes of one
// Sha256UpdateU32LE per id, in as few updates as possible (SHA-256 does not
// depend on how its input is split). On a little-endian host these are the
// ids' own bytes, absorbed in a single update.
inline void Sha256UpdateTokensLE(SHA256_CTX& ctx, std::span<const std::int32_t> tokens) {
    if constexpr (std::endian::native == std::endian::little) {
        if (!tokens.empty()) {
            SHA256_Update(&ctx, tokens.data(), tokens.size_bytes());
        }
    } else {
        Sha256UpdateTokensBytewiseLE(ctx, tokens);
    }
}

// HashPrefixPage's digest, with the prior as raw bytes: what HexToBytes makes
// of the prior hash, so a chain can pass the previous page's digest as is.
inline void HashPrefixPageDigest(std::span<const std::int32_t> tokens, std::span<const uint8_t> prior_bytes,
                                 std::span<const std::string> extra_keys,
                                 unsigned char (&digest)[SHA256_DIGEST_LENGTH]) {
    SHA256_CTX ctx;
    SHA256_Init(&ctx);

    Sha256UpdateU32LE(ctx, static_cast<uint32_t>(prior_bytes.size()));
    if (!prior_bytes.empty()) {
        SHA256_Update(&ctx, prior_bytes.data(), prior_bytes.size());
    }

    Sha256UpdateU32LE(ctx, static_cast<uint32_t>(tokens.size()));
    Sha256UpdateTokensLE(ctx, tokens);

    // extra_keys is the terminal block, so an empty list can be skipped without
    // ambiguity (a non-empty list always writes a count >= 1 first).
    if (!extra_keys.empty()) {
        Sha256UpdateU32LE(ctx, static_cast<uint32_t>(extra_keys.size()));
        for (const std::string& key : extra_keys) {
            Sha256UpdateU32LE(ctx, static_cast<uint32_t>(key.size()));
            SHA256_Update(&ctx, key.data(), key.size());
        }
    }

    SHA256_Final(digest, &ctx);
}

// extra_keys: per-page list of distinguishing keys (e.g. LoRA name, cache salt);
// the caller encodes each value, this function owns only the framing.
//
// The whole input is prefix-framed -- [prior_len][prior][token_count][tokens]
// [extra_count][extra...] -- so every section is self-delimiting and no two
// distinct (prior, tokens, extra_keys) triples can hash the same byte stream:
// prior_len separates page 0 from a chained page, token_count keeps tokens from
// bleeding into extra_keys, and per-key length prefixes prevent re-splitting.
// Feed order is prior_hash -> tokens -> extra_keys.
inline std::string HashPrefixPage(std::span<const std::int32_t> tokens, const std::string& prior_hash,
                                  std::span<const std::string> extra_keys = {}) {
    const std::vector<uint8_t> prior_bytes = HexToBytes(prior_hash);
    unsigned char digest[SHA256_DIGEST_LENGTH];
    HashPrefixPageDigest(tokens, prior_bytes, extra_keys, digest);
    return DigestToHex(digest);
}

inline std::vector<std::string> ComputePrefixHashes(
    std::span<const std::span<const std::int32_t>> prefix_pages, const std::string& prior,
    std::span<const std::span<const std::string>> extra_keys_per_page = {}) {
    std::vector<std::string> hashes;
    hashes.reserve(prefix_pages.size());
    // Each page chains on the previous page's raw digest, which is exactly
    // what HexToBytes decodes from that page's hash. Two buffers: a page's
    // digest is written while the previous one is still its prior.
    const std::vector<uint8_t> first_prior = HexToBytes(prior);
    std::span<const uint8_t> current_prior = first_prior;
    unsigned char digests[2][SHA256_DIGEST_LENGTH];
    for (std::size_t i = 0; i < prefix_pages.size(); ++i) {
        std::span<const std::string> extra =
            (i < extra_keys_per_page.size()) ? extra_keys_per_page[i] : std::span<const std::string>{};
        unsigned char(&digest)[SHA256_DIGEST_LENGTH] = digests[i % 2];
        HashPrefixPageDigest(prefix_pages[i], current_prior, extra, digest);
        hashes.push_back(DigestToHex(digest));
        current_prior = digest;
    }
    return hashes;
}

}  // namespace tokenspeed
