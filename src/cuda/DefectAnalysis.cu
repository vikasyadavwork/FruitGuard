// FruitGuard Vision: visible defect counts, never a trained defect model.
#include "DefectAnalysis.cuh"
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <new>
#include <limits.h>
#include <chrono>

namespace {
const int THREADS = 256;
__constant__ int sat_div[256];
__constant__ int hue_div[256];

struct QualityContext {
    int device;
    cudaStream_t stream = nullptr;
    cudaEvent_t events[4] = {};
    uint8_t* bgr = nullptr;
    uint8_t* mask = nullptr;
    uint64_t* partial = nullptr;
    uint64_t* host_partial = nullptr;
    size_t capacity = 0;
    size_t block_capacity = 0;
};

int fail(char* buffer, size_t length, const char* operation, cudaError_t status) {
    if (buffer && length) {
        snprintf(buffer, length, "%s: %s", operation, cudaGetErrorString(status));
        buffer[length - 1] = '\0';
    }
    return -1;
}
int invalid(char* buffer, size_t length, const char* message) {
    if (buffer && length) {
        snprintf(buffer, length, "%s", message);
        buffer[length - 1] = '\0';
    }
    return -1;
}
void cleanup_status(const char* operation, cudaError_t status) {
    if (status != cudaSuccess)
        fprintf(stderr, "FruitGuard CUDA quality cleanup %s: %s\n", operation, cudaGetErrorString(status));
}

__global__ void count_features(const uint8_t* bgr, const uint8_t* mask,
                              int width, int height, int tx, int ty, int tw, int th,
                              int dark_v, int bhmin, int bhmax, int bsmin, int bsmax,
                              int bvmin, int bvmax, int dsmax, int dvmin, int dvmax,
                              uint64_t* partial) {
    __shared__ unsigned int sums[5][THREADS];
    const unsigned int thread = threadIdx.x;
    const int pixel = blockIdx.x * blockDim.x + thread;
    unsigned int foreground = 0, dark = 0, anomaly = 0, intersection = 0, union_count = 0;
    if (pixel < width * height) {
        const int x = pixel % width, y = pixel / width;
        foreground = mask[pixel] != 0;
        if (foreground) {
            const int b = bgr[3 * pixel], g = bgr[3 * pixel + 1], r = bgr[3 * pixel + 2];
            const int value = max(r, max(g, b));
            const int minimum = min(r, min(g, b));
            const int difference = value - minimum;
            // Fixed-point 8-bit HSV matches OpenCV's cvtColor, including ties.
            // Reference algorithm, independently implemented:
            // https://github.com/opencv/opencv/blob/4.x/modules/imgproc/src/color_hsv.simd.hpp
            const int saturation = (difference * sat_div[value] + 2048) >> 12;
            const int numerator = value == r ? g - b :
                                  value == g ? b - r + 2 * difference : r - g + 4 * difference;
            int hue = (numerator * hue_div[difference] + 2048) >> 12;
            if (hue < 0) hue += 180;
            dark = value < dark_v;
            const bool brown = hue >= bhmin && hue <= bhmax &&
                saturation >= bsmin && saturation <= bsmax && value >= bvmin && value <= bvmax;
            const bool desaturated = saturation <= dsmax && value >= dvmin && value <= dvmax;
            anomaly = brown || desaturated;
        }
        if (x >= tx && x < tx + tw && y >= ty && y < ty + th) {
            const int mirror_x = 2 * tx + tw - 1 - x;
            const bool reflected = mask[y * width + mirror_x] != 0;
            intersection = foreground && reflected;
            union_count = foreground || reflected;
        }
    }
    sums[0][thread] = foreground; sums[1][thread] = dark; sums[2][thread] = anomaly;
    sums[3][thread] = intersection; sums[4][thread] = union_count;
    __syncthreads();
    for (unsigned int stride = THREADS / 2; stride; stride >>= 1) {
        if (thread < stride) {
            #pragma unroll
            for (int feature = 0; feature < 5; ++feature)
                sums[feature][thread] += sums[feature][thread + stride];
        }
        __syncthreads();
    }
    if (!thread) {
        #pragma unroll
        for (int feature = 0; feature < 5; ++feature)
            partial[5 * blockIdx.x + feature] = sums[feature][0];
    }
}

int reserve(QualityContext* ctx, size_t pixels, size_t blocks, char* error, size_t n) {
    if (pixels <= ctx->capacity && blocks <= ctx->block_capacity) return 0;
    uint8_t* bgr = nullptr;
    uint8_t* mask = nullptr;
    uint64_t* partial = nullptr;
    uint64_t* host = nullptr;
    cudaError_t status = cudaMalloc(reinterpret_cast<void**>(&bgr), pixels * 3);
    if (status == cudaSuccess) status = cudaMalloc(reinterpret_cast<void**>(&mask), pixels);
    if (status == cudaSuccess) status = cudaMalloc(reinterpret_cast<void**>(&partial), blocks * 5 * sizeof(uint64_t));
    if (status == cudaSuccess) status = cudaMallocHost(reinterpret_cast<void**>(&host), blocks * 5 * sizeof(uint64_t));
    if (status != cudaSuccess) {
        if (bgr) cleanup_status("free temporary BGR", cudaFree(bgr));
        if (mask) cleanup_status("free temporary mask", cudaFree(mask));
        if (partial) cleanup_status("free temporary counts", cudaFree(partial));
        if (host) cleanup_status("free temporary host counts", cudaFreeHost(host));
        return fail(error, n, "allocate reusable quality buffers", status);
    }
    if (ctx->bgr) cleanup_status("free old BGR", cudaFree(ctx->bgr));
    if (ctx->mask) cleanup_status("free old mask", cudaFree(ctx->mask));
    if (ctx->partial) cleanup_status("free old counts", cudaFree(ctx->partial));
    if (ctx->host_partial) cleanup_status("free old host counts", cudaFreeHost(ctx->host_partial));
    ctx->bgr = bgr; ctx->mask = mask; ctx->partial = partial; ctx->host_partial = host;
    ctx->capacity = pixels; ctx->block_capacity = blocks;
    return 0;
}
}

void* fg_quality_create(int device, char* error, size_t error_size) {
    int count = 0;
    cudaError_t status = cudaGetDeviceCount(&count);
    if (status != cudaSuccess) { fail(error, error_size, "enumerate CUDA devices", status); return nullptr; }
    if (device < 0 || device >= count) { invalid(error, error_size, "Requested CUDA quality device is unavailable; explicitly select CPU mode"); return nullptr; }
    status = cudaSetDevice(device);
    if (status != cudaSuccess) { fail(error, error_size, "select CUDA device", status); return nullptr; }
    QualityContext* ctx = new(std::nothrow) QualityContext;
    if (!ctx) { invalid(error, error_size, "Could not allocate quality context"); return nullptr; }
    ctx->device = device;
    status = cudaStreamCreate(&ctx->stream);
    for (int i = 0; i < 4 && status == cudaSuccess; ++i) status = cudaEventCreate(&ctx->events[i]);
    int sdiv[256] = {}, hdiv[256] = {};
    for (int i = 1; i < 256; ++i) {
        sdiv[i] = static_cast<int>(lrint(static_cast<double>(255 << 12) / i));
        hdiv[i] = static_cast<int>(lrint(static_cast<double>(180 << 12) / (6 * i)));
    }
    if (status == cudaSuccess) status = cudaMemcpyToSymbol(sat_div, sdiv, sizeof(sdiv));
    if (status == cudaSuccess) status = cudaMemcpyToSymbol(hue_div, hdiv, sizeof(hdiv));
    if (status != cudaSuccess) {
        fail(error, error_size, "initialize CUDA quality context", status);
        fg_quality_destroy(ctx); return nullptr;
    }
    return ctx;
}

int fg_quality_analyze(void* context, const uint8_t* bgr, const uint8_t* mask,
    int width, int height, int tx, int ty, int tw, int th,
    const int* thresholds, uint64_t* counts, double* timing,
    char* error, size_t error_size) {
    if (!context || !bgr || !mask || !thresholds || !counts || !timing ||
        width <= 0 || height <= 0 || static_cast<int64_t>(width) * height > INT_MAX / 3 ||
        tx < 0 || ty < 0 || tw <= 0 || th <= 0 ||
        static_cast<int64_t>(tx) + tw > width || static_cast<int64_t>(ty) + th > height)
        return invalid(error, error_size, "Invalid quality input pointers, image dimensions, or tight foreground bounds");
    for (int i = 0; i < 10; ++i) {
        if (thresholds[i] < 0 || thresholds[i] > ((i == 1 || i == 2) ? 179 : 255))
            return invalid(error, error_size, "HSV thresholds outside permitted ranges");
    }
    if (thresholds[1] > thresholds[2] || thresholds[3] > thresholds[4] ||
        thresholds[5] > thresholds[6] || thresholds[8] > thresholds[9])
        return invalid(error, error_size, "HSV threshold minimum exceeds maximum");
    QualityContext* ctx = static_cast<QualityContext*>(context);
    cudaError_t status = cudaSetDevice(ctx->device);
    if (status != cudaSuccess) return fail(error, error_size, "select CUDA quality device", status);
    const size_t pixels = static_cast<size_t>(width) * height;
    const size_t blocks = (pixels + THREADS - 1) / THREADS;
    const bool allocate = pixels > ctx->capacity || blocks > ctx->block_capacity;
    const auto allocation_start = std::chrono::steady_clock::now();
    if (reserve(ctx, pixels, blocks, error, error_size)) return -1;
    timing[3] = allocate ? std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - allocation_start).count() : 0.0;
    #define FG_TRY(call) do { status = (call); if (status != cudaSuccess) { cudaStreamSynchronize(ctx->stream); return fail(error, error_size, #call, status); } } while(0)
    FG_TRY(cudaEventRecord(ctx->events[0], ctx->stream));
    FG_TRY(cudaMemcpyAsync(ctx->bgr, bgr, pixels * 3, cudaMemcpyHostToDevice, ctx->stream));
    FG_TRY(cudaMemcpyAsync(ctx->mask, mask, pixels, cudaMemcpyHostToDevice, ctx->stream));
    FG_TRY(cudaEventRecord(ctx->events[1], ctx->stream));
    count_features<<<static_cast<unsigned int>(blocks), THREADS, 0, ctx->stream>>>(
        ctx->bgr, ctx->mask, width, height, tx, ty, tw, th,
        thresholds[0], thresholds[1], thresholds[2], thresholds[3], thresholds[4],
        thresholds[5], thresholds[6], thresholds[7], thresholds[8], thresholds[9], ctx->partial);
    FG_TRY(cudaGetLastError());
    FG_TRY(cudaEventRecord(ctx->events[2], ctx->stream));
    FG_TRY(cudaMemcpyAsync(ctx->host_partial, ctx->partial, blocks * 5 * sizeof(uint64_t), cudaMemcpyDeviceToHost, ctx->stream));
    FG_TRY(cudaEventRecord(ctx->events[3], ctx->stream));
    FG_TRY(cudaEventSynchronize(ctx->events[3]));
    for (int i = 0; i < 3; ++i) {
        float ms = 0;
        FG_TRY(cudaEventElapsedTime(&ms, ctx->events[i], ctx->events[i + 1]));
        timing[i] = ms;
    }
    for (int i = 0; i < 5; ++i) counts[i] = 0;
    // GPU reduces each block; this small host reduction is in wall time only.
    for (size_t block = 0; block < blocks; ++block)
        for (int i = 0; i < 5; ++i) counts[i] += ctx->host_partial[5 * block + i];
    if (!counts[0] || !counts[4]) return invalid(error, error_size, "Foreground mask is empty");
    return 0;
    #undef FG_TRY
}

void fg_quality_destroy(void* context) {
    QualityContext* ctx = static_cast<QualityContext*>(context);
    if (!ctx) return;
    const cudaError_t selected = cudaSetDevice(ctx->device);
    if (selected != cudaSuccess) {
        cleanup_status("select device before destruction; CUDA allocations cannot be safely released", selected);
        delete ctx;
        return;
    }
    if (ctx->stream) cleanup_status("synchronize", cudaStreamSynchronize(ctx->stream));
    if (ctx->bgr) cleanup_status("free BGR", cudaFree(ctx->bgr));
    if (ctx->mask) cleanup_status("free mask", cudaFree(ctx->mask));
    if (ctx->partial) cleanup_status("free counts", cudaFree(ctx->partial));
    if (ctx->host_partial) cleanup_status("free host counts", cudaFreeHost(ctx->host_partial));
    for (int i = 0; i < 4; ++i) if (ctx->events[i]) cleanup_status("destroy event", cudaEventDestroy(ctx->events[i]));
    if (ctx->stream) cleanup_status("destroy stream", cudaStreamDestroy(ctx->stream));
    delete ctx;
}
