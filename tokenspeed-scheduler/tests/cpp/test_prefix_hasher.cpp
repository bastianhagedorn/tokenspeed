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

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "cache/prefix/prefix_hasher.h"
#include "core/token_container.h"
#include "scheduler/request.h"

namespace tokenspeed::test {
namespace {

using token_span = std::span<const std::int32_t>;
using key_span = std::span<const std::string>;

// HashPrefixPage frames the input as [prior_len][prior][token_count][tokens][extra...];
// all-empty input is two zero u32s (8 zero bytes), whose SHA-256 is below.
constexpr const char* kEmptyFramedSha256 = "af5570f5a1810b7af78caf4bc70a660f0df51e42baf91d4de5b2328de0e83dfc";

token_span Tokens(const std::vector<std::int32_t>& v) {
    return token_span(v.data(), v.size());
}
key_span Keys(const std::vector<std::string>& v) {
    return key_span(v.data(), v.size());
}

TEST(RequestPromptHashesTest, RepeatedQueriesKeepTheFullInputPageChain) {
    Request request(RequestSpec{.request_id = "prompt", .tokens = {1, 2, 3, 4, 5, 6, 7, 8, 9}},
                    /*prefix_granularity=*/4, Role::kFused);
    const std::string first = HashPrefixPage(Tokens({1, 2, 3, 4}), "");
    const std::string second = HashPrefixPage(Tokens({5, 6, 7, 8}), first);
    const std::vector<std::string> expected{first, second};
    for (int retry = 0; retry < 3; ++retry) {
        const auto hashes = request.PromptPrefixHashes();
        EXPECT_EQ(std::vector<std::string>(hashes.begin(), hashes.end()), expected);
    }
}

TEST(RequestPromptHashesTest, AnAlignedInputKeepsItsLastFullPageForDecodeRoleMatching) {
    Request request(RequestSpec{.request_id = "decode", .tokens = {1, 2, 3, 4}},
                    /*prefix_granularity=*/2, Role::kD);
    const std::string first = HashPrefixPage(Tokens({1, 2}), "");
    const std::string second = HashPrefixPage(Tokens({3, 4}), first);
    const auto hashes = request.PromptPrefixHashes();
    EXPECT_EQ(std::vector<std::string>(hashes.begin(), hashes.end()), (std::vector<std::string>{first, second}));
}

TEST(RequestPromptHashesTest, AShortInputHasNoFullPageHashes) {
    Request request(RequestSpec{.request_id = "short", .tokens = {1, 2, 3}},
                    /*prefix_granularity=*/4, Role::kFused);
    EXPECT_TRUE(request.PromptPrefixHashes().empty());
    EXPECT_TRUE(request.PromptPrefixHashes().empty());
}

// ---- hex helpers --------------------------------------------------------

TEST(PageHasherHexTest, AppendHexBytesIsLowercaseTwoCharsPerByte) {
    std::string out;
    const uint8_t bytes[] = {0x00, 0x0f, 0xa3, 0xff};
    AppendHexBytes(out, bytes, 4);
    EXPECT_EQ(out, "000fa3ff");
}

TEST(PageHasherHexTest, HexToBytesIsInverseOfAppend) {
    const uint8_t bytes[] = {0x00, 0x01, 0x7e, 0x80, 0xab, 0xff};
    std::string hex;
    AppendHexBytes(hex, bytes, sizeof(bytes));
    std::vector<uint8_t> decoded = HexToBytes(hex);
    ASSERT_EQ(decoded.size(), sizeof(bytes));
    for (std::size_t i = 0; i < sizeof(bytes); ++i) {
        EXPECT_EQ(decoded[i], bytes[i]) << "byte " << i;
    }
}

TEST(PageHasherHexTest, HexToBytesAcceptsUppercase) {
    EXPECT_EQ(HexToBytes("ABCDEF"), HexToBytes("abcdef"));
}

// ---- HashPrefixPage -----------------------------------------------------------

TEST(HashPrefixPageTest, EmptyPageMatchesKnownSha256) {
    std::vector<std::int32_t> none;
    EXPECT_EQ(HashPrefixPage(Tokens(none), ""), kEmptyFramedSha256);
}

TEST(HashPrefixPageTest, OutputIs64HexChars) {
    std::vector<std::int32_t> toks = {1, 2, 3};
    std::string h = HashPrefixPage(Tokens(toks), "");
    EXPECT_EQ(h.size(), 64u);
    EXPECT_EQ(h.find_first_not_of("0123456789abcdef"), std::string::npos);
}

TEST(HashPrefixPageTest, Deterministic) {
    std::vector<std::int32_t> toks = {7, 8, 9};
    EXPECT_EQ(HashPrefixPage(Tokens(toks), "seed"), HashPrefixPage(Tokens(toks), "seed"));
}

TEST(HashPrefixPageTest, DifferentTokensDifferentHash) {
    std::vector<std::int32_t> a = {1, 2, 3};
    std::vector<std::int32_t> b = {1, 2, 4};
    EXPECT_NE(HashPrefixPage(Tokens(a), ""), HashPrefixPage(Tokens(b), ""));
}

TEST(HashPrefixPageTest, TokenOrderMatters) {
    std::vector<std::int32_t> a = {1, 2};
    std::vector<std::int32_t> b = {2, 1};
    EXPECT_NE(HashPrefixPage(Tokens(a), ""), HashPrefixPage(Tokens(b), ""));
}

TEST(HashPrefixPageTest, PriorHashChangesOutput) {
    std::vector<std::int32_t> toks = {5, 6};
    std::string no_prior = HashPrefixPage(Tokens(toks), "");
    std::string with_prior = HashPrefixPage(Tokens(toks), no_prior);
    EXPECT_NE(no_prior, with_prior);
}

// A tail hash (hash@T over the page's first T mod P tokens, same prior) is
// the same function as the page hash; the token_count frame keeps it apart
// from the full page and from every other offset under that prior.
TEST(HashPrefixPageTest, TailHashesAreDistinctFromThePageAndFromEachOther) {
    const std::vector<std::int32_t> page = {1, 2, 3, 4};
    const std::string prior = HashPrefixPage(Tokens({9, 9, 9, 9}), "");
    const std::string full = HashPrefixPage(Tokens(page), prior);
    std::vector<std::string> tails;
    for (std::size_t t = 1; t < page.size(); ++t) {
        tails.push_back(HashPrefixPage(token_span(page.data(), t), prior));
        EXPECT_NE(tails.back(), full);
        EXPECT_NE(tails.back(), HashPrefixPage(token_span(page.data(), t), "")) << "the prior frames the tail";
    }
    EXPECT_NE(tails[0], tails[1]);
    EXPECT_NE(tails[1], tails[2]);
    // The page's own chain hash is what a tail's prior names, so a tail of
    // the next page never collides with a tail of this one.
    EXPECT_NE(HashPrefixPage(token_span(page.data(), 1), full), tails[0]);
}

TEST(HashPrefixPageTest, EmptyExtraKeysEqualsTwoArgForm) {
    std::vector<std::int32_t> toks = {1, 2, 3};
    std::vector<std::string> empty;
    EXPECT_EQ(HashPrefixPage(Tokens(toks), "p"), HashPrefixPage(Tokens(toks), "p", Keys(empty)));
}

TEST(HashPrefixPageTest, ExtraKeysChangeOutput) {
    std::vector<std::int32_t> toks = {1, 2, 3};
    std::vector<std::string> keys = {"lora-A"};
    EXPECT_NE(HashPrefixPage(Tokens(toks), "p"), HashPrefixPage(Tokens(toks), "p", Keys(keys)));
}

TEST(HashPrefixPageTest, FramingDisambiguatesKeySplits) {
    std::vector<std::int32_t> toks = {1};
    std::vector<std::string> split_a = {"ab", "c"};
    std::vector<std::string> split_b = {"a", "bc"};
    EXPECT_NE(HashPrefixPage(Tokens(toks), "", Keys(split_a)), HashPrefixPage(Tokens(toks), "", Keys(split_b)));
}

TEST(HashPrefixPageTest, FramingDisambiguatesKeyCount) {
    std::vector<std::int32_t> toks = {1};
    std::vector<std::string> one = {"abc"};
    std::vector<std::string> two = {"a", "bc"};
    EXPECT_NE(HashPrefixPage(Tokens(toks), "", Keys(one)), HashPrefixPage(Tokens(toks), "", Keys(two)));
}

// A 32-byte prior reinterpreted as 8 LE tokens in page 0 must not produce the
// same stream as a chained page carrying that digest as prior.
TEST(HashPrefixPageTest, FramingDisambiguatesEmptyPriorFromChainedPage) {
    const std::string prior = "00112233445566778899aabbccddeeff00112233445566778899aabbccddeeff";
    std::vector<uint8_t> pb = HexToBytes(prior);
    ASSERT_EQ(pb.size(), 32u);

    std::vector<std::int32_t> toks(8);
    for (std::size_t i = 0; i < 8; ++i) {
        toks[i] = static_cast<std::int32_t>(
            static_cast<uint32_t>(pb[4 * i]) | (static_cast<uint32_t>(pb[4 * i + 1]) << 8) |
            (static_cast<uint32_t>(pb[4 * i + 2]) << 16) | (static_cast<uint32_t>(pb[4 * i + 3]) << 24));
    }
    std::vector<std::int32_t> none;
    std::string as_page0 = HashPrefixPage(Tokens(toks), "");
    std::string as_chained = HashPrefixPage(Tokens(none), prior);
    EXPECT_NE(as_page0, as_chained);
}

// Append v as 4 little-endian bytes, whatever the host's byte order.
void AppendU32LE(std::vector<uint8_t>& stream, uint32_t v) {
    for (int shift = 0; shift < 32; shift += 8) {
        stream.push_back(static_cast<uint8_t>(v >> shift));
    }
}

// The framed stream spelled out byte by byte: every token id is a 4-byte
// little-endian word whatever the host's byte order, however many updates
// absorb it.
std::string FramedStreamSha256(const std::vector<std::int32_t>& tokens, const std::string& prior_hash,
                               const std::vector<std::string>& extra_keys) {
    std::vector<uint8_t> stream;
    const std::vector<uint8_t> prior = HexToBytes(prior_hash);
    AppendU32LE(stream, static_cast<uint32_t>(prior.size()));
    stream.insert(stream.end(), prior.begin(), prior.end());
    AppendU32LE(stream, static_cast<uint32_t>(tokens.size()));
    for (const std::int32_t t : tokens) {
        AppendU32LE(stream, static_cast<uint32_t>(t));
    }
    if (!extra_keys.empty()) {
        AppendU32LE(stream, static_cast<uint32_t>(extra_keys.size()));
        for (const std::string& key : extra_keys) {
            AppendU32LE(stream, static_cast<uint32_t>(key.size()));
            stream.insert(stream.end(), key.begin(), key.end());
        }
    }
    unsigned char digest[SHA256_DIGEST_LENGTH];
    SHA256(stream.data(), stream.size(), digest);
    return DigestToHex(digest);
}

TEST(HashPrefixPageTest, TokensAreFramedAsLittleEndianWords) {
    const std::vector<std::int32_t> edge = {
        0, 1, -1, std::numeric_limits<std::int32_t>::min(), std::numeric_limits<std::int32_t>::max(), 0x01020304, 256};
    std::vector<std::int32_t> page(300);
    for (std::size_t i = 0; i < page.size(); ++i) {
        page[i] = static_cast<std::int32_t>(i * 2654435761u);
    }
    const std::string prior = HashPrefixPage(Tokens(edge), "");
    for (const std::vector<std::int32_t>* tokens : std::array<const std::vector<std::int32_t>*, 2>{&edge, &page}) {
        for (const std::string& p : {std::string{}, prior}) {
            for (const std::vector<std::string>& extra : {std::vector<std::string>{}, {"lora-A", "", "salt"}}) {
                EXPECT_EQ(HashPrefixPage(Tokens(*tokens), p, Keys(extra)), FramedStreamSha256(*tokens, p, extra))
                    << tokens->size() << " tokens, prior '" << p << "', " << extra.size() << " extra keys";
            }
        }
    }
}

// A host that is not little-endian absorbs a page's tokens with the byte-wise
// encoder, so run it here too: inside a running hash, on the edge ids, and on
// both sides of each flush of its 1 KiB (256-token) buffer.
TEST(HashPrefixPageTest, BytewiseTokenEncoderAbsorbsLittleEndianWords) {
    const std::vector<std::int32_t> edge = {
        0, 1, -1, std::numeric_limits<std::int32_t>::min(), std::numeric_limits<std::int32_t>::max(), 0x01020304, 256};
    std::vector<std::int32_t> page(513);
    for (std::size_t i = 0; i < page.size(); ++i) {
        page[i] = static_cast<std::int32_t>(i * 2654435761u);
    }
    std::vector<token_span> inputs = {Tokens(edge)};
    for (const std::size_t count : {0u, 1u, 255u, 256u, 257u, 511u, 512u, 513u}) {
        inputs.emplace_back(page.data(), count);
    }
    const std::string before = "prior";
    const std::string after = "extra";
    for (const token_span tokens : inputs) {
        std::vector<uint8_t> stream(before.begin(), before.end());
        for (const std::int32_t t : tokens) {
            AppendU32LE(stream, static_cast<uint32_t>(t));
        }
        stream.insert(stream.end(), after.begin(), after.end());
        unsigned char expected[SHA256_DIGEST_LENGTH];
        SHA256(stream.data(), stream.size(), expected);

        SHA256_CTX ctx;
        SHA256_Init(&ctx);
        SHA256_Update(&ctx, before.data(), before.size());
        Sha256UpdateTokensBytewiseLE(ctx, tokens);
        SHA256_Update(&ctx, after.data(), after.size());
        unsigned char got[SHA256_DIGEST_LENGTH];
        SHA256_Final(got, &ctx);
        EXPECT_EQ(DigestToHex(got), DigestToHex(expected)) << tokens.size() << " tokens";
    }
}

// Digests of the hasher that absorbed one token per update, before tokens
// went in as one buffer and the chain as raw digests: hashes published by
// older builds stay valid prefix-cache, KV-event and storage identities.
TEST(HashPrefixPageTest, KnownDigestsAreUnchanged) {
    const std::vector<std::int32_t> edge = {
        0, 1, -1, std::numeric_limits<std::int32_t>::min(), std::numeric_limits<std::int32_t>::max(), 256, 65536, 7};
    const std::string first = HashPrefixPage(Tokens(edge), "");
    EXPECT_EQ(first, "78cb08b7e6af3f06585a4ce00ad942759c67e9ee529331ee3d96b55b9c43ee1b");
    EXPECT_EQ(HashPrefixPage(Tokens(edge), first), "0599cee90211c7aa885f8503aa90c108893a0ec7aa9683922a438dc9f6eef9df");
    const std::vector<std::string> extra = {"lora-A", "", "salt"};
    EXPECT_EQ(HashPrefixPage(Tokens(edge), first, Keys(extra)),
              "f9213804c94467f7701b5eb1487730dc466546571bbdff1e09958e8fc3ac7ea0");

    // Three 256-token pages of negative and positive ids.
    std::vector<std::int32_t> tokens(3 * 256);
    for (std::size_t i = 0; i < tokens.size(); ++i) {
        tokens[i] = static_cast<std::int32_t>(i) * 7919 - 1000000;
    }
    std::vector<token_span> pages;
    for (std::size_t p = 0; p < 3; ++p) {
        pages.emplace_back(tokens.data() + 256 * p, 256);
    }
    EXPECT_EQ(ComputePrefixHashes(pages, ""),
              (std::vector<std::string>{"def7ea1a1a9adc0094dcbe942003a2c34513615152ffd203fd4179bdef1ce470",
                                        "1497b6aa73ab961101b9edb2f8452b8e0db05c7c0f0ba6ea45eb39a3f20ed697",
                                        "9e76a9c2b23e51a8392c015ee88d4b5d9ca7cb017ebfda234f410a6273f95f56"}));
    // An uppercase initial prior decodes like its lowercase form.
    EXPECT_EQ(ComputePrefixHashes(pages, "00112233445566778899AABBCCDDEEFF00112233445566778899AABBCCDDEEFF"),
              (std::vector<std::string>{"880bfc2f3405663cc27157a067be4bb3665fd7058fd877450956757e3657ef34",
                                        "59de7812a30536a97c7add7ff98dc6134b0aad0cffe01a4a089583fccfcc0793",
                                        "4fd45b2c634f417691f2fe9c1802977d61e95e3950030b29dae5e6b772510ee6"}));
    // Tails of the second page under the first page's hash.
    EXPECT_EQ(HashPrefixPage(token_span(tokens.data() + 256, 1), ComputePrefixHashes(pages, "")[0]),
              "57efba6ebc3e56375f9ee2a58181c2f3e0225dcdc283ddd8a5140f729ea8f404");
    EXPECT_EQ(HashPrefixPage(token_span(tokens.data() + 256, 255), ComputePrefixHashes(pages, "")[0]),
              "88072b4958ed57ae8df90a9d253a8513ac714b4791439df4e1e6e7faf65506c5");
}

// A 4-byte extra key must not produce the same stream as count(1) + len(4) +
// the key's LE int32 folded back into the token list.
TEST(HashPrefixPageTest, FramingDisambiguatesTokensFromExtraKeys) {
    std::vector<std::int32_t> short_toks = {9, 8};
    std::vector<std::string> one_key = {"wxyz"};

    // count=1, len=4, then "wxyz" (0x7a797877 little-endian) as a trailing token.
    std::vector<std::int32_t> long_toks = {9, 8, 1, 4, 0x7a797877};
    std::vector<std::string> no_keys;

    EXPECT_NE(HashPrefixPage(Tokens(short_toks), "", Keys(one_key)),
              HashPrefixPage(Tokens(long_toks), "", Keys(no_keys)));
}

// ---- ComputePrefixHashes (chaining) -------------------------------------

TEST(ComputePrefixHashesTest, MatchesManualRollingChain) {
    std::vector<std::int32_t> p0 = {1, 2};
    std::vector<std::int32_t> p1 = {3, 4};
    std::vector<std::int32_t> p2 = {5, 6};
    std::vector<token_span> pages = {Tokens(p0), Tokens(p1), Tokens(p2)};

    std::vector<std::string> got = ComputePrefixHashes(pages, "root");

    std::string h0 = HashPrefixPage(Tokens(p0), "root");
    std::string h1 = HashPrefixPage(Tokens(p1), h0);
    std::string h2 = HashPrefixPage(Tokens(p2), h1);

    ASSERT_EQ(got.size(), 3u);
    EXPECT_EQ(got[0], h0);
    EXPECT_EQ(got[1], h1);
    EXPECT_EQ(got[2], h2);
}

// The chain feeds each page the previous page's raw digest; HashPrefixPage
// decodes the same bytes from that page's hex hash.
TEST(ComputePrefixHashesTest, RawDigestChainEqualsHexPriorChain) {
    std::vector<std::int32_t> tokens(7 * 64 + 5);
    for (std::size_t i = 0; i < tokens.size(); ++i) {
        tokens[i] = static_cast<std::int32_t>(i * 40503u) - 7;
    }
    std::vector<token_span> pages;
    for (std::size_t start = 0; start + 64 <= tokens.size(); start += 64) {
        pages.emplace_back(tokens.data() + start, 64);
    }
    const std::vector<std::string> salt = {"salt"};
    const std::vector<key_span> extra = {Keys(salt), key_span{}, Keys(salt)};
    for (const std::string& prior : {std::string{}, HashPrefixPage(Tokens({3}), ""), std::string{"ABCDEF0"}}) {
        const std::vector<std::string> got = ComputePrefixHashes(pages, prior, extra);
        ASSERT_EQ(got.size(), pages.size());
        std::string expected_prior = prior;
        for (std::size_t i = 0; i < pages.size(); ++i) {
            const key_span keys = i < extra.size() ? extra[i] : key_span{};
            const std::string expected = HashPrefixPage(pages[i], expected_prior, keys);
            EXPECT_EQ(got[i], expected) << "page " << i << " under prior '" << prior << "'";
            expected_prior = expected;
        }
    }
}

TEST(ComputePrefixHashesTest, SamePageDifferentPrefixDiffers) {
    std::vector<std::int32_t> same = {9, 9};
    std::vector<std::int32_t> other = {1, 1};
    std::vector<token_span> a = {Tokens(same), Tokens(same)};
    std::vector<token_span> b = {Tokens(other), Tokens(same)};

    std::vector<std::string> ha = ComputePrefixHashes(a, "");
    std::vector<std::string> hb = ComputePrefixHashes(b, "");
    EXPECT_NE(ha[0], hb[0]);
    EXPECT_NE(ha[1], hb[1]);
}

TEST(ComputePrefixHashesTest, MissingExtraKeysPerPageTreatedAsEmpty) {
    std::vector<std::int32_t> p0 = {1};
    std::vector<std::int32_t> p1 = {2};
    std::vector<token_span> pages = {Tokens(p0), Tokens(p1)};

    std::vector<std::string> k0 = {"salt"};
    std::vector<key_span> extra = {Keys(k0)};

    std::vector<std::string> got = ComputePrefixHashes(pages, "", extra);

    std::string h0 = HashPrefixPage(Tokens(p0), "", Keys(k0));
    std::string h1 = HashPrefixPage(Tokens(p1), h0);
    EXPECT_EQ(got[0], h0);
    EXPECT_EQ(got[1], h1);
}

TEST(ComputePrefixHashesTest, IncrementalChainEqualsOneShot) {
    // 12-token stream, prefix_granularity 2 -> 6 pages.
    std::vector<std::int32_t> tokens(12);
    for (std::int32_t i = 0; i < 12; ++i) {
        tokens[i] = 100 + i;
    }
    std::vector<token_span> pages;
    for (std::size_t start = 0; start < tokens.size(); start += 2) {
        pages.push_back(token_span(tokens.data() + start, 2));
    }

    const std::vector<std::string> one_shot = ComputePrefixHashes(pages, "");

    const std::vector<token_span> head(pages.begin(), pages.begin() + 3);
    const std::vector<token_span> tail(pages.begin() + 3, pages.end());
    std::vector<std::string> incremental = ComputePrefixHashes(head, "");
    const std::vector<std::string> rest = ComputePrefixHashes(tail, incremental.back());
    incremental.insert(incremental.end(), rest.begin(), rest.end());

    EXPECT_EQ(incremental, one_shot);
}

// A step hashes the pages it completed, sliced alone rather than out of a view
// of every page of the context.
TEST(PrefixPagesTest, SlicesOnlyTheAskedCompletePages) {
    const TokenContainer tokens{std::vector<std::int32_t>{1, 2, 3, 4, 5, 6, 7}};

    const std::vector<token_span> pages = tokens.PrefixPages(/*prefix_granularity=*/2, 1, 3);

    ASSERT_EQ(pages.size(), 2u);
    EXPECT_EQ(std::vector<std::int32_t>(pages[0].begin(), pages[0].end()), (std::vector<std::int32_t>{3, 4}));
    EXPECT_EQ(std::vector<std::int32_t>(pages[1].begin(), pages[1].end()), (std::vector<std::int32_t>{5, 6}));
    EXPECT_TRUE(tokens.PrefixPages(2, 3, 3).empty());
    EXPECT_THROW(tokens.PrefixPages(2, 3, 4), std::runtime_error) << "token 7 starts a partial page";
    const TokenContainer aligned{std::vector<std::int32_t>{1, 2, 3, 4}};
    EXPECT_EQ(aligned.FullPrefixPages(2, /*except_last=*/false).size(), 2u);
    EXPECT_EQ(aligned.FullPrefixPages(2, /*except_last=*/true).size(), 1u) << "the last token's page is left out";
}

}  // namespace
}  // namespace tokenspeed::test
