#include "ImagePreprocess.cuh"
#include <cstdio>
#include <limits>
#include <new>

namespace {
__global__ void bgrToRgbChw(const unsigned char* input, float* output, size_t pixels) {
    const size_t p = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (p >= pixels) return;
    output[p] = input[3 * p + 2] / 255.0f;
    output[pixels + p] = input[3 * p + 1] / 255.0f;
    output[2 * pixels + p] = input[3 * p] / 255.0f;
}
struct Context {
    int device;
    unsigned char* input = nullptr;
    float* output = nullptr;
    size_t capacity = 0;
};
int report(cudaError_t code, const char* operation, char* error, size_t capacity) {
    if (error && capacity) {
        if (code == cudaSuccess) error[0] = '\0';
        else std::snprintf(error, capacity, "%s: %s (%s)", operation,
                           cudaGetErrorName(code), cudaGetErrorString(code));
    }
    // The returned diagnostic consumes this call's error. Otherwise a handled
    // invalid-device request can incorrectly poison a later valid kernel launch.
    if (code != cudaSuccess) (void)cudaGetLastError();
    return static_cast<int>(code);
}
}

void cpuPreprocessBgrToRgbChw(const unsigned char* input, float* output, size_t pixels) {
    for (size_t p = 0; p < pixels; ++p) {
        output[p] = input[3 * p + 2] / 255.0f;
        output[pixels + p] = input[3 * p + 1] / 255.0f;
        output[2 * pixels + p] = input[3 * p] / 255.0f;
    }
}

cudaError_t launchBgrToRgbChw(const unsigned char* input, float* output,
                             size_t pixels, cudaStream_t stream) {
    if (!input || !output || !pixels || pixels > static_cast<size_t>(2147483647))
        return cudaErrorInvalidValue;
    const int threads = 256;
    const unsigned int blocks = static_cast<unsigned int>((pixels + threads - 1) / threads);
    bgrToRgbChw<<<blocks, threads, 0, stream>>>(input, output, pixels);
    return cudaGetLastError();
}

extern "C" FG_EXPORT void* fg_create(int device, char* error, size_t error_capacity) {
    cudaError_t code = cudaSetDevice(device);
    if (code != cudaSuccess) {
        report(code, "cudaSetDevice", error, error_capacity);
        return nullptr;
    }
    Context* context = new (std::nothrow) Context;
    if (!context) {
        report(cudaErrorMemoryAllocation, "allocate context", error, error_capacity);
        return nullptr;
    }
    context->device = device;
    report(cudaSuccess, "create", error, error_capacity);
    return context;
}

extern "C" FG_EXPORT int fg_preprocess(void* pointer, const unsigned char* input,
                                      float* output, int width, int height,
                                      char* error, size_t error_capacity) {
    if (!pointer || !input || !output || width <= 0 || height <= 0)
        return report(cudaErrorInvalidValue, "validate arguments", error, error_capacity);
    const size_t pixels = static_cast<size_t>(width) * static_cast<size_t>(height);
    if (pixels > 2147483647 || pixels > std::numeric_limits<size_t>::max() / (3 * sizeof(float)))
        return report(cudaErrorInvalidValue, "image too large", error, error_capacity);
    Context* context = static_cast<Context*>(pointer);
    cudaError_t code = cudaSetDevice(context->device);
    if (code != cudaSuccess) return report(code, "cudaSetDevice", error, error_capacity);
    if (pixels > context->capacity) {
        unsigned char* new_input = nullptr;
        float* new_output = nullptr;
        code = cudaMalloc(reinterpret_cast<void**>(&new_input), pixels * 3);
        if (code != cudaSuccess) return report(code, "cudaMalloc input", error, error_capacity);
        code = cudaMalloc(reinterpret_cast<void**>(&new_output), pixels * 3 * sizeof(float));
        if (code != cudaSuccess) {
            cudaError_t cleanup = cudaFree(new_input);
            return report(cleanup != cudaSuccess ? cleanup : code, "allocate output/cleanup input", error, error_capacity);
        }
        code = cudaFree(context->input);
        context->input = new_input;
        cudaError_t second = cudaFree(context->output);
        context->output = new_output;
        context->capacity = pixels;
        if (code != cudaSuccess || second != cudaSuccess)
            return report(code != cudaSuccess ? code : second, "cudaFree old buffers", error, error_capacity);
    }
    code = cudaMemcpy(context->input, input, pixels * 3, cudaMemcpyHostToDevice);
    if (code != cudaSuccess) return report(code, "host-to-device", error, error_capacity);
    code = launchBgrToRgbChw(context->input, context->output, pixels);
    if (code != cudaSuccess) return report(code, "kernel launch", error, error_capacity);
    code = cudaMemcpy(output, context->output, pixels * 3 * sizeof(float), cudaMemcpyDeviceToHost);
    return report(code, "kernel synchronization/device-to-host", error, error_capacity);
}

extern "C" FG_EXPORT int fg_destroy(void* pointer, char* error, size_t error_capacity) {
    if (!pointer) return report(cudaSuccess, "destroy", error, error_capacity);
    Context* context = static_cast<Context*>(pointer);
    cudaError_t code = cudaSetDevice(context->device);
    if (code != cudaSuccess) return report(code, "cudaSetDevice before destroy", error, error_capacity);
    cudaError_t first = cudaFree(context->input);
    cudaError_t second = cudaFree(context->output);
    delete context;
    return report(first != cudaSuccess ? first : second, "cudaFree context", error, error_capacity);
}
