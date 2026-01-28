#include <gtest/gtest.h>

#include "utils.h"

#include <vector>
#include <algorithm>

using MX::Pipe::Util::get_best_label;
using MX::Pipe::Util::nms;
using MX::Pipe::BBox;
using MX::Prepost::Util::get_best_label;

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
    std::vector<int> result = nms(boxes, 0.5f);
    EXPECT_EQ(result.size(), 0);
}

TEST(NMS, SingleBox) {
    std::vector<BBox> boxes = {
        BBox(0.0f, 0.0f, 10.0f, 10.0f, 0.8f, 0, "")
    };
    std::vector<int> result = nms(boxes, 0.5f);
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
    std::vector<int> result = nms(boxes, 0.5f);
    
    // All boxes should be kept, in descending confidence order
    EXPECT_EQ(result.size(), 3);
    EXPECT_EQ(result[0], 0);  // Highest conf (0.9)
    EXPECT_EQ(result[1], 2);  // Second highest conf (0.8)
    EXPECT_EQ(result[2], 1);  // Lowest conf (0.7)
}

TEST(NMS, PerfectOverlap) {
    // Two identical boxes with different confidences
    // Box 0: conf=0.9, Box 1: conf=0.5
    // IoU = 1.0 (perfect overlap), threshold = 0.5
    // Expected: Only box 0 (higher conf) kept
    std::vector<BBox> boxes = {
        BBox(0.0f, 0.0f, 10.0f, 10.0f, 0.9f, 0, ""),  // boxes[0]
        BBox(0.0f, 0.0f, 10.0f, 10.0f, 0.5f, 1, "")   // boxes[1] - identical coordinates
    };
    std::vector<int> result = nms(boxes, 0.5f);
    
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
        std::vector<int> result = nms(boxes, 0.5f);
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
        std::vector<int> result = nms(boxes, 0.5f);
        EXPECT_EQ(result.size(), 2) << "Both boxes should be kept when IoU (0.5) == threshold (0.5) due to strict > check";
        EXPECT_EQ(result[0], 0);
        EXPECT_EQ(result[1], 1);
    }
    
    // Test 3: IoU above threshold (0.6 > 0.5) - lower conf should be suppressed
    {
        // Box 1: (0, 0, 10, 10) - area = 100
        // Box 2: (0, 0, 10, 10) - same box, IoU = 1.0 > 0.5
        std::vector<BBox> boxes = {
            BBox(0.0f, 0.0f, 10.0f, 10.0f, 0.9f, 0, ""),  // boxes[0] - higher conf
            BBox(0.0f, 0.0f, 10.0f, 10.0f, 0.5f, 1, "")   // boxes[1] - lower conf, perfect overlap
        };
        std::vector<int> result = nms(boxes, 0.5f);
        EXPECT_EQ(result.size(), 1) << "Lower confidence box should be suppressed when IoU (1.0) > threshold (0.5)";
        EXPECT_EQ(result[0], 0);
    }
}

TEST(NMS, MultipleOverlaps) {
    // One high-conf box overlaps with 4 lower-conf boxes
    // All boxes have IoU > 0.5 with Box 0
    std::vector<BBox> boxes = {
        BBox(0.0f, 0.0f, 10.0f, 10.0f, 0.9f, 0, ""),   // boxes[0] - highest conf, area = 100
        BBox(1.0f, 1.0f, 11.0f, 11.0f, 0.7f, 1, ""),   // boxes[1] - IoU ≈ 0.681 > 0.5 
        BBox(0.5f, 0.5f, 10.5f, 10.5f, 0.6f, 2, ""),   // boxes[2] - IoU ≈ 0.902 > 0.5 
        BBox(2.0f, 2.0f, 12.0f, 12.0f, 0.5f, 3, ""),   // boxes[3] - IoU ≈ 0.471 < 0.5 
        BBox(1.5f, 1.5f, 11.5f, 11.5f, 0.4f, 4, "")    // boxes[4] - IoU ≈ 0.556 > 0.5 
    };
    std::vector<int> result = nms(boxes, 0.5f);
    
    EXPECT_EQ(result.size(), 2);  // Only highest conf kept and boxes[3] is kept
    EXPECT_EQ(result[0], 0);       // Should be boxes[0]
    EXPECT_EQ(result[1], 3);       // Should be boxes[3]
}

TEST(NMS, TieConfidence) {
    // Two boxes with same confidence and high overlap
    // Since sort is unstable, either may come first, but only one should be kept
    std::vector<BBox> boxes = {
        BBox(0.0f, 0.0f, 10.0f, 10.0f, 0.8f, 0, ""),   // boxes[0] - conf = 0.8
        BBox(0.0f, 0.0f, 10.0f, 10.0f, 0.8f, 1, "")    // boxes[1] - conf = 0.8 (tie), perfect overlap
    };
    std::vector<int> result = nms(boxes, 0.5f);
    
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
    std::vector<int> result = nms(boxes, 0.5f);
    
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
    std::vector<int> result = nms(boxes, 0.5f);
    
    // Current behavior: Lower confidence box suppressed despite different class
    EXPECT_EQ(result.size(), 1);
    EXPECT_EQ(result[0], 0);  // Only boxes[0] kept (higher conf)
}

// This will fail because the current implementation is class-agnostic
TEST(NMS, ClassAwareSameConfidence) {
    // Two boxes of different classes with same confidence and high overlap
    // Expected: Both kept (different classes should not suppress each other)
    std::vector<BBox> boxes = {
        BBox(0.0f, 0.0f, 10.0f, 10.0f, 0.8f, 0, "person"),    // boxes[0] - class 0, conf = 0.8
        BBox(0.0f, 0.0f, 10.0f, 10.0f, 0.8f, 1, "car")        // boxes[1] - class 1, conf = 0.8 (tie), perfect overlap
    };
    std::vector<int> result = nms(boxes, 0.5f);
    
    // Expected: Both kept (different classes)
    EXPECT_EQ(result.size(), 2);
    // Both indices should be present (order may vary due to unstable sort)
    EXPECT_TRUE(std::find(result.begin(), result.end(), 0) != result.end());
    EXPECT_TRUE(std::find(result.begin(), result.end(), 1) != result.end());
}

// This will fail because the current implementation is class-agnostic
TEST(NMS, ClassAwareDifferentConfidence) {
    // Two boxes of different classes with different confidence and high overlap
    // Expected: Both kept (different classes should not suppress each other)
    std::vector<BBox> boxes = {
        BBox(0.0f, 0.0f, 10.0f, 10.0f, 0.9f, 0, "person"),    // boxes[0] - class 0, conf = 0.9
        BBox(0.0f, 0.0f, 10.0f, 10.0f, 0.5f, 1, "car")        // boxes[1] - class 1, conf = 0.5, perfect overlap
    };
    std::vector<int> result = nms(boxes, 0.5f);
    
    // Expected: Both kept (different classes)
    EXPECT_EQ(result.size(), 2);
    EXPECT_EQ(result[0], 0);  // Higher conf first
    EXPECT_EQ(result[1], 1);  // Lower conf second
}

