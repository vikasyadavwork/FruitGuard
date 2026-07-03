#pragma once
#include <stddef.h>
#include <stdint.h>
#ifdef _WIN32
#define FG_QUALITY_API __declspec(dllexport)
#else
#define FG_QUALITY_API
#endif

// Counts: foreground, dark, color anomaly, horizontal intersection, union.
// HSV thresholds: dark V; brown H min/max, S min/max, V min/max;
// desaturated S max, V min/max. Inclusive ranges except dark V < threshold.
// Timing output: H2D, fused kernel, D2H, reusable-buffer allocation milliseconds.
extern "C" {
FG_QUALITY_API void* fg_quality_create(int device, char* error, size_t error_size);
FG_QUALITY_API int fg_quality_analyze(void* context,
    const uint8_t* bgr, const uint8_t* mask, int width, int height,
    int tight_x, int tight_y, int tight_width, int tight_height,
    const int* thresholds, uint64_t* counts, double* timing_ms,
    char* error, size_t error_size);
FG_QUALITY_API void fg_quality_destroy(void* context);
}
