#pragma once
#include <cuda_runtime.h>
#include <cstddef>

void cpuPreprocessBgrToRgbChw(const unsigned char* input, float* output, size_t pixels);
cudaError_t launchBgrToRgbChw(const unsigned char* input, float* output,
                             size_t pixels, cudaStream_t stream = nullptr);
#if defined(_WIN32)
#define FG_EXPORT __declspec(dllexport)
#else
#define FG_EXPORT
#endif
// One context per caller/thread. Allocations grow on demand and are reused.
extern "C" {
FG_EXPORT void* fg_create(int device, char* error, size_t error_capacity);
FG_EXPORT int fg_preprocess(void* context, const unsigned char* input, float* output,
                           int width, int height, char* error, size_t error_capacity);
FG_EXPORT int fg_destroy(void* context, char* error, size_t error_capacity);
}
