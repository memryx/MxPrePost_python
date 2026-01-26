#include <gtest/gtest.h>

#include "utils.h"

#include <vector>

using MX::Pipe::Util::get_best_label;

/* ===================== get_best_label tests ===================== */

TEST(GetBestLabel, PicksBestAmongValidClasses) {
    float scores[] = {0.10f, 0.85f, 0.40f, 0.90f, 0.92f};
    std::vector<int> valid = {1, 2, 3};

    float best = -1.0f;
    int cls = get_best_label(best, scores, valid, 0.25f);

    EXPECT_EQ(cls, 3);
    EXPECT_FLOAT_EQ(best, 0.90f);
}

TEST(GetBestLabel, RespectsThreshold) {
    float scores[] = {0.10f, 0.20f, 0.24f};
    std::vector<int> valid = {0, 1, 2};

    float best = -1.0f;
    int cls = get_best_label(best, scores, valid, 0.25f);

    EXPECT_EQ(cls, -1);
    EXPECT_FLOAT_EQ(best, -1.0f);
}

TEST(GetBestLabel, EmptyValidClasses) {
    float scores[] = {0.9f, 0.1f};
    std::vector<int> valid;

    float best = -1.0f;
    int cls = get_best_label(best, scores, valid, 0.0f);

    EXPECT_EQ(cls, -1);
    EXPECT_FLOAT_EQ(best, -1.0f);
}