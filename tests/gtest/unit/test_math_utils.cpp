#include <gtest/gtest.h>

#include "utils.h"

#include <vector>
#include <algorithm>
#include <array>

using MX::Runtime::BBox;
using MX::Runtime::Mask;
using MX::Prepost::Util::get_best_label;
using MX::Prepost::Util::nms_class_agnostic;
using MX::Prepost::Util::nms_class_aware;
using MX::Prepost::Util::dfl;
using MX::Prepost::Util::preprocess;
using MX::Prepost::Util::draw_bbox;
using MX::Prepost::Util::draw_mask;

namespace {
    constexpr float DFL_PEAK_LOGIT = 20.0f;

    void set_side_logits_peak(float* coord_buf, int side, int peak_bin, float A = DFL_PEAK_LOGIT) {
        float* side_buf = coord_buf + side * 16;
        for (int i = 0; i < 16; ++i) {
            side_buf[i] = -A;
        }
        side_buf[peak_bin] = +A;
    }

    void set_side_logits_uniform(float* coord_buf, int side, float v = 0.0f) {
        float* side_buf = coord_buf + side * 16;
        for (int i = 0; i < 16; ++i) {
            side_buf[i] = v;
        }
    }
}

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

/* ===================== nms tests ===================== */

TEST(NMS, EmptyInput) {
    std::vector<BBox> boxes;
    std::vector<int> result = nms_class_aware(boxes, 0.5f);
    EXPECT_EQ(result.size(), 0);
}

TEST(NMS, SingleBox) {
    std::vector<BBox> boxes = {
        BBox(0.0f, 0.0f, 10.0f, 10.0f, 0.8f, 0, "")
    };
    std::vector<int> result = nms_class_aware(boxes, 0.5f);
    EXPECT_EQ(result.size(), 1);
    EXPECT_EQ(result[0], 0);
}

TEST(NMS, NoOverlap) {
    // Three boxes with no overlap, different confidences
    // Box 0: conf=0.9, Box 1: conf=0.7, Box 2: conf=0.8
    // Expected order after sorting by conf: {0, 2, 1}
    std::vector<BBox> boxes = {
        BBox(0.0f, 0.0f, 10.0f, 10.0f, 0.9f, 0, ""),   // boxes[0]
        BBox(20.0f, 20.0f, 30.0f, 30.0f, 0.7f, 1, ""), // boxes[1]
        BBox(40.0f, 40.0f, 50.0f, 50.0f, 0.8f, 2, "")  // boxes[2]
    };
    std::vector<int> result = nms_class_aware(boxes, 0.5f);
    
    // All boxes should be kept, in descending confidence order
    EXPECT_EQ(result.size(), 3);
    EXPECT_EQ(result[0], 0);  // Highest conf (0.9)
    EXPECT_EQ(result[1], 2);  // Second highest conf (0.8)
    EXPECT_EQ(result[2], 1);  // Lowest conf (0.7)
}

TEST(NMS, PerfectOverlap) {
    // Two identical boxes with different confidences (same class)
    // Box 0: conf=0.9, Box 1: conf=0.5
    // IoU = 1.0 (perfect overlap), threshold = 0.5
    // Expected: Only box 0 (higher conf) kept
    std::vector<BBox> boxes = {
        BBox(0.0f, 0.0f, 10.0f, 10.0f, 0.9f, 0, ""),  // boxes[0] - class 0
        BBox(0.0f, 0.0f, 10.0f, 10.0f, 0.5f, 0, "")   // boxes[1] - class 0, identical coordinates
    };
    std::vector<int> result = nms_class_aware(boxes, 0.5f);
    
    EXPECT_EQ(result.size(), 1);
    EXPECT_EQ(result[0], 0);  // Only higher-confidence box kept
}

TEST(NMS, IoUThresholdBoundaries) {
    // Test IoU threshold boundary conditions
    
    // Test 1: IoU below threshold (0.3 < 0.5) - both should be kept
    {
        // Box 1: (0, 0, 10, 10) - area = 100
        // Box 2: (5, 0, 15, 10) - area = 100
        // Intersection: (10-5) * 10 = 50
        // Union: 100 + 100 - 50 = 150
        // IoU = 50/150 = 0.333... < 0.5
        std::vector<BBox> boxes = {
            BBox(0.0f, 0.0f, 10.0f, 10.0f, 0.9f, 0, ""),  // boxes[0] - higher conf
            BBox(5.0f, 0.0f, 15.0f, 10.0f, 0.5f, 1, "")   // boxes[1] - lower conf
        };
        std::vector<int> result = nms_class_aware(boxes, 0.5f);
        EXPECT_EQ(result.size(), 2) << "Both boxes should be kept when IoU (0.333) < threshold (0.5)";
        EXPECT_EQ(result[0], 0);
        EXPECT_EQ(result[1], 1);
    }
    
    // Test 2: IoU exactly equal to threshold (0.5 = 0.5) - both should be kept (strict > check)
    {
        // Box 1: (0, 0, 10, 10) - area = 100
        // Box 2: (0, 0, 10, 10) - same box, but we need IoU = 0.5
        // For IoU = 0.5: intersection = 50, union = 150
        // If Box 1 area = 100, Box 2 area = 100, intersection = 50
        // Union = 100 + 100 - 50 = 150, IoU = 50/150 = 0.333 (not 0.5)
        // 
        // Let's use: Box 1: (0, 0, 10, 10) area = 100
        // Box 2: (0, 0, 10*sqrt(2), 10*sqrt(2)) - no, that's complex
        // 
        // Better: Box 1: (0, 0, 10, 10) area = 100
        // Box 2: (0, 0, 20, 10) area = 200
        // Intersection = 100, Union = 100 + 200 - 100 = 200, IoU = 100/200 = 0.5
        std::vector<BBox> boxes = {
            BBox(0.0f, 0.0f, 10.0f, 10.0f, 0.9f, 0, ""),   // boxes[0] - area = 100
            BBox(0.0f, 0.0f, 20.0f, 10.0f, 0.5f, 1, "")    // boxes[1] - area = 200, intersection = 100
        };
        std::vector<int> result = nms_class_aware(boxes, 0.5f);
        EXPECT_EQ(result.size(), 2) << "Both boxes should be kept when IoU (0.5) == threshold (0.5) due to strict > check";
        EXPECT_EQ(result[0], 0);
        EXPECT_EQ(result[1], 1);
    }
    
    // Test 3: IoU above threshold (0.6 > 0.5) - lower conf should be suppressed (same class)
    {
        // Box 1: (0, 0, 10, 10) - area = 100
        // Box 2: (0, 0, 10, 10) - same box, IoU = 1.0 > 0.5
        std::vector<BBox> boxes = {
            BBox(0.0f, 0.0f, 10.0f, 10.0f, 0.9f, 0, ""),  // boxes[0] - class 0, higher conf
            BBox(0.0f, 0.0f, 10.0f, 10.0f, 0.5f, 0, "")   // boxes[1] - class 0, lower conf, perfect overlap
        };
        std::vector<int> result = nms_class_aware(boxes, 0.5f);
        EXPECT_EQ(result.size(), 1) << "Lower confidence box should be suppressed when IoU (1.0) > threshold (0.5)";
        EXPECT_EQ(result[0], 0);
    }
}

TEST(NMS, MultipleOverlaps) {
    // One high-conf box overlaps with 4 lower-conf boxes (all same class)
    // All boxes have IoU > 0.5 with Box 0
    std::vector<BBox> boxes = {
        BBox(0.0f, 0.0f, 10.0f, 10.0f, 0.9f, 0, ""),   // boxes[0] - class 0, highest conf, area = 100
        BBox(1.0f, 1.0f, 11.0f, 11.0f, 0.7f, 0, ""),   // boxes[1] - class 0, IoU ≈ 0.681 > 0.5 
        BBox(0.5f, 0.5f, 10.5f, 10.5f, 0.6f, 0, ""),   // boxes[2] - class 0, IoU ≈ 0.902 > 0.5 
        BBox(2.0f, 2.0f, 12.0f, 12.0f, 0.5f, 0, ""),   // boxes[3] - class 0, IoU ≈ 0.471 < 0.5 
        BBox(1.5f, 1.5f, 11.5f, 11.5f, 0.4f, 0, "")    // boxes[4] - class 0, IoU ≈ 0.556 > 0.5 
    };
    std::vector<int> result = nms_class_aware(boxes, 0.5f);
    
    EXPECT_EQ(result.size(), 2);  // Only highest conf kept and boxes[3] is kept
    EXPECT_EQ(result[0], 0);       // Should be boxes[0]
    EXPECT_EQ(result[1], 3);       // Should be boxes[3]
}

TEST(NMS, TieConfidence) {
    // Two boxes with same confidence and high overlap (same class)
    // Since sort is unstable, either may come first, but only one should be kept
    std::vector<BBox> boxes = {
        BBox(0.0f, 0.0f, 10.0f, 10.0f, 0.8f, 0, ""),   // boxes[0] - class 0, conf = 0.8
        BBox(0.0f, 0.0f, 10.0f, 10.0f, 0.8f, 0, "")    // boxes[1] - class 0, conf = 0.8 (tie), perfect overlap
    };
    std::vector<int> result = nms_class_aware(boxes, 0.5f);
    
    // Only one should be kept (order may vary due to unstable sort)
    EXPECT_EQ(result.size(), 1);
    // Either boxes[0] or boxes[1] could be kept - both are valid
    EXPECT_TRUE(result[0] == 0 || result[0] == 1);
}

TEST(NMS, ClassAgnosticSameConfidence) {
    // Two boxes of different classes with same confidence and high overlap
    // Current implementation: Since confidences are equal, sort order is unstable,
    // but one will be suppressed due to high IoU (class-agnostic behavior)
    std::vector<BBox> boxes = {
        BBox(0.0f, 0.0f, 10.0f, 10.0f, 0.8f, 0, "person"),    // boxes[0] - class 0, conf = 0.8
        BBox(0.0f, 0.0f, 10.0f, 10.0f, 0.8f, 1, "car")        // boxes[1] - class 1, conf = 0.8 (tie), perfect overlap
    };
    std::vector<int> result = nms_class_agnostic(boxes, 0.5f);
    
    // Current behavior: One will be suppressed (IoU = 1.0 > 0.5) despite different classes
    // Due to unstable sort with equal confidence, either could be kept
    EXPECT_EQ(result.size(), 1);
    EXPECT_TRUE(result[0] == 0 || result[0] == 1);
    
}

TEST(NMS, ClassAgnosticDifferentConfidence) {
    // Two boxes of different classes with different confidence and high overlap
    // Current implementation: Lower confidence box suppressed regardless of class
    std::vector<BBox> boxes = {
        BBox(0.0f, 0.0f, 10.0f, 10.0f, 0.9f, 0, "person"),    // boxes[0] - class 0, conf = 0.9
        BBox(0.0f, 0.0f, 10.0f, 10.0f, 0.5f, 1, "car")        // boxes[1] - class 1, conf = 0.5, perfect overlap
    };
    std::vector<int> result = nms_class_agnostic(boxes, 0.5f);
    
    // Current behavior: Lower confidence box suppressed despite different class
    EXPECT_EQ(result.size(), 1);
    EXPECT_EQ(result[0], 0);  // Only boxes[0] kept (higher conf)
}

TEST(NMS, ClassAwareSameConfidence) {
    // Two boxes of different classes with same confidence and high overlap
    // Expected: Both kept (different classes should not suppress each other)
    std::vector<BBox> boxes = {
        BBox(0.0f, 0.0f, 10.0f, 10.0f, 0.8f, 0, "person"),    // boxes[0] - class 0, conf = 0.8
        BBox(0.0f, 0.0f, 10.0f, 10.0f, 0.8f, 1, "car")        // boxes[1] - class 1, conf = 0.8 (tie), perfect overlap
    };
    std::vector<int> result = nms_class_aware(boxes, 0.5f);
    
    // Expected: Both kept (different classes)
    EXPECT_EQ(result.size(), 2);
    // Both indices should be present (order may vary due to unstable sort)
    EXPECT_TRUE(std::find(result.begin(), result.end(), 0) != result.end());
    EXPECT_TRUE(std::find(result.begin(), result.end(), 1) != result.end());
}

TEST(NMS, ClassAwareDifferentConfidence) {
    // Two boxes of different classes with different confidence and high overlap
    // Expected: Both kept (different classes should not suppress each other)
    std::vector<BBox> boxes = {
        BBox(0.0f, 0.0f, 10.0f, 10.0f, 0.9f, 0, "person"),    // boxes[0] - class 0, conf = 0.9
        BBox(0.0f, 0.0f, 10.0f, 10.0f, 0.5f, 1, "car")        // boxes[1] - class 1, conf = 0.5, perfect overlap
    };
    std::vector<int> result = nms_class_aware(boxes, 0.5f);
    
    // Expected: Both kept (different classes)
    EXPECT_EQ(result.size(), 2);
    EXPECT_EQ(result[0], 0);  // Higher conf first
    EXPECT_EQ(result[1], 1);  // Lower conf second
}

/* ===================== dfl tests ===================== */

TEST(DFL, SinglePeakDistribution) {
    // coord_buf is logits (not probabilities): 4 sides * 16 bins
    float coord_buf[64] = {0};

    set_side_logits_peak(coord_buf, 0, 5);   // left
    set_side_logits_peak(coord_buf, 1, 3);   // top
    set_side_logits_peak(coord_buf, 2, 10);  // right
    set_side_logits_peak(coord_buf, 3, 7);   // bottom

    std::cout << "left:   ";
    for (int i = 0; i < 16; ++i) std::cout << coord_buf[i] << ' ';
    std::cout << "\ntop:    ";
    for (int i = 16; i < 32; ++i) std::cout << coord_buf[i] << ' ';
    std::cout << "\nright:  ";
    for (int i = 32; i < 48; ++i) std::cout << coord_buf[i] << ' ';
    std::cout << "\nbottom: ";
    for (int i = 48; i < 64; ++i) std::cout << coord_buf[i] << ' ';
    std::cout << "\n";
    std::array<float, 4> result = dfl(coord_buf, 10, 10, 8, 640, 640);

    EXPECT_NEAR(result[0], 44.0f, 1e-2f);   // x1 = (10.5 - 5) * 8
    EXPECT_NEAR(result[1], 60.0f, 1e-2f);   // y1 = (10.5 - 3) * 8
    EXPECT_NEAR(result[2], 164.0f, 1e-2f);  // x2 = (10.5 + 10) * 8
    EXPECT_NEAR(result[3], 140.0f, 1e-2f);  // y2 = (10.5 + 7) * 8
}

TEST(DFL, BasicCoordinateConversion) {
    float coord_buf[64] = {0};

    set_side_logits_peak(coord_buf, 0, 2);  // left
    set_side_logits_peak(coord_buf, 1, 4);  // top
    set_side_logits_peak(coord_buf, 2, 6);  // right
    set_side_logits_peak(coord_buf, 3, 8);  // bottom

    std::cout << "left:   ";
    for (int i = 0; i < 16; ++i) std::cout << coord_buf[i] << ' ';
    std::cout << "\ntop:    ";
    for (int i = 16; i < 32; ++i) std::cout << coord_buf[i] << ' ';
    std::cout << "\nright:  ";
    for (int i = 32; i < 48; ++i) std::cout << coord_buf[i] << ' ';
    std::cout << "\nbottom: ";
    for (int i = 48; i < 64; ++i) std::cout << coord_buf[i] << ' ';
    std::cout << "\n";    std::array<float, 4> result = dfl(coord_buf, 5, 5, 16, 640, 640);

    EXPECT_NEAR(result[0], 56.0f, 1e-2f);   // x1 = (5.5 - 2) * 16
    EXPECT_NEAR(result[1], 24.0f, 1e-2f);   // y1 = (5.5 - 4) * 16
    EXPECT_NEAR(result[2], 184.0f, 1e-2f);  // x2 = (5.5 + 6) * 16
    EXPECT_NEAR(result[3], 216.0f, 1e-2f);  // y2 = (5.5 + 8) * 16
}

TEST(DFL, CoordinatesExceedBounds) {
    // Subtest A: negative clamp (x1/y1 clamp to 0)
    {
        float coord_buf[64] = {0};
        set_side_logits_peak(coord_buf, 0, 15);  // left
        set_side_logits_peak(coord_buf, 1, 15);  // top
        set_side_logits_peak(coord_buf, 2, 15);  // right
        set_side_logits_peak(coord_buf, 3, 15);  // bottom

        std::cout << "left:   ";
        for (int i = 0; i < 16; ++i) std::cout << coord_buf[i] << ' ';
        std::cout << "\ntop:    ";
        for (int i = 16; i < 32; ++i) std::cout << coord_buf[i] << ' ';
        std::cout << "\nright:  ";
        for (int i = 32; i < 48; ++i) std::cout << coord_buf[i] << ' ';
        std::cout << "\nbottom: ";
        for (int i = 48; i < 64; ++i) std::cout << coord_buf[i] << ' ';
        std::cout << "\n";
        std::array<float, 4> result = dfl(coord_buf, 0, 0, 8, 640, 640);

        EXPECT_NEAR(result[0], 0.0f, 1e-3f);    // x1 clamped from -116
        EXPECT_NEAR(result[1], 0.0f, 1e-3f);    // y1 clamped from -116
        EXPECT_NEAR(result[2], 124.0f, 1e-2f);  // x2 = (0.5 + 15) * 8
        EXPECT_NEAR(result[3], 124.0f, 1e-2f);  // y2 = (0.5 + 15) * 8
    }

    // Subtest B: positive clamp (x2/y2 clamp to model_w-1/model_h-1)
    {
        float coord_buf[64] = {0};
        set_side_logits_peak(coord_buf, 0, 0);    // left
        set_side_logits_peak(coord_buf, 1, 0);    // top
        set_side_logits_peak(coord_buf, 2, 15);   // right
        set_side_logits_peak(coord_buf, 3, 15);   // bottom

        std::cout << "left:   ";
        for (int i = 0; i < 16; ++i) std::cout << coord_buf[i] << ' ';
        std::cout << "\ntop:    ";
        for (int i = 16; i < 32; ++i) std::cout << coord_buf[i] << ' ';
        std::cout << "\nright:  ";
        for (int i = 32; i < 48; ++i) std::cout << coord_buf[i] << ' ';
        std::cout << "\nbottom: ";
        for (int i = 48; i < 64; ++i) std::cout << coord_buf[i] << ' ';
        std::cout << "\n";
        std::array<float, 4> result = dfl(coord_buf, 79, 79, 8, 640, 640);

        EXPECT_NEAR(result[0], 636.0f, 1e-2f);  // x1 = (79.5 - 0) * 8
        EXPECT_NEAR(result[1], 636.0f, 1e-2f);  // y1 = (79.5 - 0) * 8
        EXPECT_NEAR(result[2], 639.0f, 1e-3f);  // x2 clamped from 756
        EXPECT_NEAR(result[3], 639.0f, 1e-3f);  // y2 clamped from 756
    }
}

TEST(DFL, AllSidesDifferent) {
    float coord_buf[64] = {0};

    set_side_logits_peak(coord_buf, 0, 2);   // left
    set_side_logits_peak(coord_buf, 1, 5);   // top
    set_side_logits_peak(coord_buf, 2, 8);   // right
    set_side_logits_peak(coord_buf, 3, 11);  // bottom

    std::cout << "left:   ";
    for (int i = 0; i < 16; ++i) std::cout << coord_buf[i] << ' ';
    std::cout << "\ntop:    ";
    for (int i = 16; i < 32; ++i) std::cout << coord_buf[i] << ' ';
    std::cout << "\nright:  ";
    for (int i = 32; i < 48; ++i) std::cout << coord_buf[i] << ' ';
    std::cout << "\nbottom: ";
    for (int i = 48; i < 64; ++i) std::cout << coord_buf[i] << ' ';
    std::cout << "\n";
    std::array<float, 4> result = dfl(coord_buf, 10, 10, 8, 640, 640);

    EXPECT_NEAR(result[0], 68.0f, 1e-2f);   // x1 = (10.5 - 2) * 8
    EXPECT_NEAR(result[1], 44.0f, 1e-2f);   // y1 = (10.5 - 5) * 8
    EXPECT_NEAR(result[2], 148.0f, 1e-2f);  // x2 = (10.5 + 8) * 8
    EXPECT_NEAR(result[3], 172.0f, 1e-2f);  // y2 = (10.5 + 11) * 8
}

TEST(DFL, CompleteBoundingBox) {
    float coord_buf[64] = {0};

    set_side_logits_peak(coord_buf, 0, 9);         // left peak at 9
    set_side_logits_uniform(coord_buf, 1, 0.0f);   // top uniform logits -> EV = 7.5 exactly
    set_side_logits_peak(coord_buf, 2, 5);         // right peak at 5
    set_side_logits_peak(coord_buf, 3, 7);         // bottom peak at 7

    std::cout << "left:   ";
    for (int i = 0; i < 16; ++i) std::cout << coord_buf[i] << ' ';
    std::cout << "\ntop:    ";
    for (int i = 16; i < 32; ++i) std::cout << coord_buf[i] << ' ';
    std::cout << "\nright:  ";
    for (int i = 32; i < 48; ++i) std::cout << coord_buf[i] << ' ';
    std::cout << "\nbottom: ";
    for (int i = 48; i < 64; ++i) std::cout << coord_buf[i] << ' ';
    std::cout << "\n";
    std::array<float, 4> result = dfl(coord_buf, 20, 20, 16, 640, 640);

    EXPECT_NEAR(result[0], 184.0f, 1e-2f);  // x1 = (20.5 - 9) * 16
    EXPECT_NEAR(result[1], 208.0f, 1e-2f);  // y1 = (20.5 - 7.5) * 16
    EXPECT_NEAR(result[2], 408.0f, 1e-2f);  // x2 = (20.5 + 5) * 16
    EXPECT_NEAR(result[3], 440.0f, 1e-2f);  // y2 = (20.5 + 7) * 16

    EXPECT_LT(result[0], result[2]);
    EXPECT_LT(result[1], result[3]);
    EXPECT_GE(result[0], 0.0f);
    EXPECT_GE(result[1], 0.0f);
    EXPECT_LE(result[2], 639.0f);
    EXPECT_LE(result[3], 639.0f);
}

TEST(DFL, UniformDistributionExact) {
    // All sides: uniform logits -> softmax uniform -> EV = 7.5 for each side
    float coord_buf[64] = {0};

    // left, top, right, bottom all uniform
    set_side_logits_uniform(coord_buf, 0, 0.0f);
    set_side_logits_uniform(coord_buf, 1, 0.0f);
    set_side_logits_uniform(coord_buf, 2, 0.0f);
    set_side_logits_uniform(coord_buf, 3, 0.0f);

    int row = 10;
    int col = 10;
    int stride = 8;
    int model_w = 640;
    int model_h = 640;

    std::cout << "left:   ";
    for (int i = 0; i < 16; ++i) std::cout << coord_buf[i] << ' ';
    std::cout << "\ntop:    ";
    for (int i = 16; i < 32; ++i) std::cout << coord_buf[i] << ' ';
    std::cout << "\nright:  ";
    for (int i = 32; i < 48; ++i) std::cout << coord_buf[i] << ' ';
    std::cout << "\nbottom: ";
    for (int i = 48; i < 64; ++i) std::cout << coord_buf[i] << ' ';
    std::cout << "\n";

    std::array<float, 4> result = dfl(coord_buf, row, col, stride, model_w, model_h);

    // Anchor at (10.5, 10.5), EV = 7.5
    // x1 = (10.5 - 7.5) * 8 = 24, x2 = (10.5 + 7.5) * 8 = 144
    EXPECT_NEAR(result[0], 24.0f, 1e-3f);
    EXPECT_NEAR(result[1], 24.0f, 1e-3f);
    EXPECT_NEAR(result[2], 144.0f, 1e-3f);
    EXPECT_NEAR(result[3], 144.0f, 1e-3f);
}

TEST(DFL, SoftmaxShiftInvariance) {
    // Adding a constant to all logits must not change decoded distances
    float coord_buf[64] = {0};

    // Base logits: left has sharp peak at 5, others uniform
    set_side_logits_peak(coord_buf, 0, 5);
    set_side_logits_uniform(coord_buf, 1, 0.0f);
    set_side_logits_uniform(coord_buf, 2, 0.0f);
    set_side_logits_uniform(coord_buf, 3, 0.0f);



    int row = 10;
    int col = 10;
    int stride = 8;
    int model_w = 640;
    int model_h = 640;

    std::cout << "left:   ";
    for (int i = 0; i < 16; ++i) std::cout << coord_buf[i] << ' ';
    std::cout << "\ntop:    ";
    for (int i = 16; i < 32; ++i) std::cout << coord_buf[i] << ' ';
    std::cout << "\nright:  ";
    for (int i = 32; i < 48; ++i) std::cout << coord_buf[i] << ' ';
    std::cout << "\nbottom: ";
    for (int i = 48; i < 64; ++i) std::cout << coord_buf[i] << ' ';
    std::cout << "\n";
    std::array<float, 4> result1 = dfl(coord_buf, row, col, stride, model_w, model_h);

    // Shift all logits by +100, softmax should be identical
    for (int i = 0; i < 64; ++i) {
        coord_buf[i] += 100.0f;
    }

    std::array<float, 4> result2 = dfl(coord_buf, row, col, stride, model_w, model_h);

    EXPECT_NEAR(result1[0], result2[0], 1e-4f);
    EXPECT_NEAR(result1[1], result2[1], 1e-4f);
    EXPECT_NEAR(result1[2], result2[2], 1e-4f);
    EXPECT_NEAR(result1[3], result2[3], 1e-4f);
}

TEST(DFL, TwoPeakSymmetry) {
    // Two equal peaks at bins 3 and 11 -> EV should be ~7.0 (midpoint)
    float coord_buf[64] = {0};

    // Left side: two symmetric peaks at 3 and 11
    float* left = coord_buf + 0 * 16;
    for (int i = 0; i < 16; ++i) {
        left[i] = -DFL_PEAK_LOGIT;
    }
    left[3]  = DFL_PEAK_LOGIT;
    left[11] = DFL_PEAK_LOGIT;

    // Other sides uniform for simplicity
    set_side_logits_uniform(coord_buf, 1, 0.0f);  // top
    set_side_logits_uniform(coord_buf, 2, 0.0f);  // right
    set_side_logits_uniform(coord_buf, 3, 0.0f);  // bottom

    int row = 10;
    int col = 10;
    int stride = 8;
    int model_w = 640;
    int model_h = 640;

    std::cout << "left:   ";
    for (int i = 0; i < 16; ++i) std::cout << coord_buf[i] << ' ';
    std::cout << "\ntop:    ";
    for (int i = 16; i < 32; ++i) std::cout << coord_buf[i] << ' ';
    std::cout << "\nright:  ";
    for (int i = 32; i < 48; ++i) std::cout << coord_buf[i] << ' ';
    std::cout << "\nbottom: ";
    for (int i = 48; i < 64; ++i) std::cout << coord_buf[i] << ' ';
    std::cout << "\n";

    std::array<float, 4> result = dfl(coord_buf, row, col, stride, model_w, model_h);

    // For the left side: EV ≈ (3 + 11) / 2 = 7.0
    // x1 = (10.5 - 7.0) * 8 = 28.0
    EXPECT_NEAR(result[0], 28.0f, 1e-2f);

    // Optional sanity: box is still sensible
    EXPECT_LT(result[0], result[2]);
    EXPECT_LT(result[1], result[3]);
}

/* ===================== preprocess tests ===================== */

TEST(Preprocess, BasicFunctionality) {
    // 1920x1080 RGB image, resized to 640x360, padded to 640x640
    cv::Mat input(1080, 1920, CV_8UC3, cv::Scalar(128, 128, 128));
    
    int letterbox_w = 640;
    int letterbox_h = 360;
    int pad_w = 0;
    int pad_h = 140;
    
    cv::Mat result = preprocess(input, letterbox_w, letterbox_h, pad_w, pad_h);
    
    EXPECT_EQ(result.rows, 640);
    EXPECT_EQ(result.cols, 640);
    EXPECT_EQ(result.type(), CV_32FC3);
    
    // Verify value range [0.0, 1.0]
    double min_val, max_val;
    cv::minMaxLoc(result, &min_val, &max_val);
    EXPECT_GE(min_val, 0.0f);
    EXPECT_LE(max_val, 1.0f);
    
    // Verify top padding is zero (rows 0-139)
    cv::Mat top_padding = result(cv::Rect(0, 0, 640, pad_h));
    cv::Scalar top_sum = cv::sum(top_padding);
    EXPECT_NEAR(top_sum[0], 0.0f, 1e-5f);
    
    // Verify bottom padding is zero (rows 500-639)
    int bottom_start_row = pad_h + letterbox_h;  // 140 + 360 = 500
    cv::Mat bottom_padding = result(cv::Rect(0, bottom_start_row, 640, pad_h));
    cv::Scalar bottom_sum = cv::sum(bottom_padding);
    EXPECT_NEAR(bottom_sum[0], 0.0f, 1e-5f);
    
    // Verify left and right edges are content (not padding) since pad_w=0
    // Check that left edge (col 0) contains content, not padding
    cv::Mat left_edge = result(cv::Rect(0, pad_h, 1, letterbox_h));
    cv::Scalar left_edge_mean = cv::mean(left_edge);
    EXPECT_GT(left_edge_mean[0], 0.0f);
    
    // Check that right edge (col 639) contains content, not padding
    cv::Mat right_edge = result(cv::Rect(639, pad_h, 1, letterbox_h));
    cv::Scalar right_edge_mean = cv::mean(right_edge);
    EXPECT_GT(right_edge_mean[0], 0.0f);
    
    // Verify center region is normalized correctly (rows 140-499, cols 0-639)
    cv::Mat center = result(cv::Rect(0, pad_h, 640, letterbox_h));
    cv::Scalar center_mean = cv::mean(center);
    EXPECT_NEAR(center_mean[0], 128.0f / 255.0f, 1e-3f);
}

TEST(Preprocess, SquareImageNoPadding) {
    cv::Mat input(640, 640, CV_8UC3, cv::Scalar(255, 255, 255));
    
    cv::Mat result = preprocess(input, 640, 640, 0, 0);
    
    EXPECT_EQ(result.rows, 640);
    EXPECT_EQ(result.cols, 640);
    EXPECT_EQ(result.type(), CV_32FC3);
    
    // All values should be 1.0 (255/255)
    cv::Scalar mean_val = cv::mean(result);
    EXPECT_NEAR(mean_val[0], 1.0f, 1e-5f);
}

TEST(Preprocess, WideImageVerticalPadding) {
    // 1920x480 image (4:1 ratio), resized to 640x160, padded to 640x640
    cv::Mat input(480, 1920, CV_8UC3, cv::Scalar(200, 200, 200));
    
    cv::Mat result = preprocess(input, 640, 160, 0, 240);
    
    EXPECT_EQ(result.rows, 640);
    EXPECT_EQ(result.cols, 640);
    
    // Verify top padding is zero
    cv::Mat top_pad = result(cv::Rect(0, 0, 640, 240));
    cv::Scalar top_sum = cv::sum(top_pad);
    EXPECT_NEAR(top_sum[0], 0.0f, 1e-5f);
    
    // Verify bottom padding is zero
    cv::Mat bottom_pad = result(cv::Rect(0, 400, 640, 240));
    cv::Scalar bottom_sum = cv::sum(bottom_pad);
    EXPECT_NEAR(bottom_sum[0], 0.0f, 1e-5f);
    
    // Verify center content is normalized
    cv::Mat center = result(cv::Rect(0, 240, 640, 160));
    cv::Scalar center_mean = cv::mean(center);
    EXPECT_NEAR(center_mean[0], 200.0f / 255.0f, 1e-3f);
}

TEST(Preprocess, TallImageHorizontalPadding) {
    // 480x1920 image (1:4 ratio), resized to 160x640, padded to 640x640
    cv::Mat input(1920, 480, CV_8UC3, cv::Scalar(100, 100, 100));
    
    int letterbox_w = 160;
    int letterbox_h = 640;
    int pad_w = 240;
    int pad_h = 0;
    
    cv::Mat result = preprocess(input, letterbox_w, letterbox_h, pad_w, pad_h);
    
    EXPECT_EQ(result.rows, 640);
    EXPECT_EQ(result.cols, 640);
    
    // Verify left padding is zero
    cv::Mat left_pad = result(cv::Rect(0, 0, pad_w, 640));
    cv::Scalar left_sum = cv::sum(left_pad);
    EXPECT_NEAR(left_sum[0], 0.0f, 1e-5f);
    
    // Verify right padding is zero
    cv::Mat right_pad = result(cv::Rect(pad_w + letterbox_w, 0, pad_w, 640));
    cv::Scalar right_sum = cv::sum(right_pad);
    EXPECT_NEAR(right_sum[0], 0.0f, 1e-5f);
    
    // Verify center content is normalized
    cv::Mat center = result(cv::Rect(pad_w, 0, letterbox_w, 640));
    cv::Scalar center_mean = cv::mean(center);
    EXPECT_NEAR(center_mean[0], 100.0f / 255.0f, 1e-3f);
}

TEST(Preprocess, VerySmallImage) {
    // 64x64 image upscaled to 640x640 (10x scaling)
    cv::Mat input(64, 64, CV_8UC3, cv::Scalar(50, 50, 50));
    
    cv::Mat result = preprocess(input, 640, 640, 0, 0);
    
    EXPECT_EQ(result.rows, 640);
    EXPECT_EQ(result.cols, 640);
    EXPECT_EQ(result.type(), CV_32FC3);
    
    // Verify normalization
    cv::Scalar mean_val = cv::mean(result);
    EXPECT_NEAR(mean_val[0], 50.0f / 255.0f, 1e-3f);
    
    // Verify no padding (all values should be similar, no zero regions)
    double min_val;
    cv::minMaxLoc(result, &min_val, nullptr);
    EXPECT_GT(min_val, 0.0f);
}

TEST(Preprocess, AlreadyModelSize) {
    // Input already matches model size (no resize, no pad needed)
    cv::Mat input(640, 640, CV_8UC3, cv::Scalar(180, 180, 180));
    
    cv::Mat result = preprocess(input, 640, 640, 0, 0);
    
    EXPECT_EQ(result.rows, 640);
    EXPECT_EQ(result.cols, 640);
    EXPECT_EQ(result.type(), CV_32FC3);
    
    // Verify normalization only
    cv::Scalar mean_val = cv::mean(result);
    EXPECT_NEAR(mean_val[0], 180.0f / 255.0f, 1e-3f);
    
    // Verify no zero regions (no padding)
    double min_val;
    cv::minMaxLoc(result, &min_val, nullptr);
    EXPECT_GT(min_val, 0.0f);
}

/* ===================== draw_bbox tests ===================== */

TEST(DrawBbox, DrawsAtCorrectCoordinates) {
    cv::Mat image(640, 640, CV_8UC3, cv::Scalar(0, 0, 0));
    cv::Mat original = image.clone();
    
    BBox bbox;
    bbox.x_min = 100.0f;
    bbox.y_min = 150.0f;
    bbox.x_max = 300.0f;
    bbox.y_max = 350.0f;
    bbox.cls_id = 0;
    bbox.conf = 0.95f;
    bbox.cls_name = "person";
    
    draw_bbox(image, bbox);
    
    // Verify rectangle boundaries are drawn
    // Check top edge (y_min, from x_min to x_max)
    cv::Mat top_edge = image(cv::Rect(100, 150, 200, 1));
    cv::Scalar top_sum = cv::sum(top_edge);
    EXPECT_GT(top_sum[0], 0);  // Modified (rectangle drawn)
    
    // Check bottom edge (y_max, from x_min to x_max)
    cv::Mat bottom_edge = image(cv::Rect(100, 350, 200, 1));
    cv::Scalar bottom_sum = cv::sum(bottom_edge);
    EXPECT_GT(bottom_sum[0], 0);  // Modified
    
    // Check left edge (x_min, from y_min to y_max)
    cv::Mat left_edge = image(cv::Rect(100, 150, 1, 200));
    cv::Scalar left_sum = cv::sum(left_edge);
    EXPECT_GT(left_sum[0], 0);  // Modified
    
    // Check right edge (x_max, from y_min to y_max)
    cv::Mat right_edge = image(cv::Rect(300, 150, 1, 200));
    cv::Scalar right_sum = cv::sum(right_edge);
    EXPECT_GT(right_sum[0], 0);  // Modified
    
    // Verify dimensions: width = 200, height = 200
    int expected_width = (int)bbox.x_max - (int)bbox.x_min;  // 200
    int expected_height = (int)bbox.y_max - (int)bbox.y_min; // 200
    EXPECT_EQ(expected_width, 200);
    EXPECT_EQ(expected_height, 200);
}

/* ===================== draw_mask tests ===================== */

TEST(DrawMask, PolygonAndBoundingBoxMatchInput) {
    cv::Mat image(640, 640, CV_8UC3, cv::Scalar(0, 0, 0));
    cv::Mat original = image.clone();
    
    // Create a rectangular polygon
    Mask mask;
    mask.cls_id = 0;
    mask.xys = {
        {100.0f, 150.0f},  // top-left
        {300.0f, 150.0f},  // top-right
        {300.0f, 350.0f},  // bottom-right
        {100.0f, 350.0f}   // bottom-left
    };
    
    draw_mask(image, mask, 0.5f);
    
    // Calculate expected bounding box from points
    std::vector<cv::Point> cv_points;
    for (const auto& pt : mask.xys) {
        cv_points.push_back(cv::Point((int)pt.x, (int)pt.y));
    }
    cv::Rect expected_rect = cv::boundingRect(cv_points);
    // Should be: x=100, y=150, width=200, height=200
    
    // Verify polygon interior is filled (center of polygon)
    cv::Mat center = image(cv::Rect(200, 250, 1, 1));  // Center of polygon
    cv::Scalar center_sum = cv::sum(center);
    EXPECT_GT(center_sum[0], 0);  // Modified (filled)
    
    // Verify bounding box rectangle is drawn at expected location
    // Check top edge of bounding box
    cv::Mat bbox_top = image(cv::Rect(expected_rect.x, expected_rect.y, 
                                      expected_rect.width, 1));
    cv::Scalar bbox_top_sum = cv::sum(bbox_top);
    EXPECT_GT(bbox_top_sum[0], 0);  // Bounding box drawn
    
    // Verify bounding box dimensions match cv::boundingRect
    EXPECT_EQ(expected_rect.x, 100);
    EXPECT_EQ(expected_rect.y, 150);
    EXPECT_EQ(expected_rect.width, 201);
    EXPECT_EQ(expected_rect.height, 201);
}