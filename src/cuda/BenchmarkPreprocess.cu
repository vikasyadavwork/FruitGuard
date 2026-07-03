#include "ImagePreprocess.cuh"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using Clock = std::chrono::steady_clock;
static double ms(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}
static void check(cudaError_t code, const char* operation) {
    if (code != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorName(code) +
                                 " (" + cudaGetErrorString(code) + ")");
}
#define CHECK(call) check((call), #call)

struct Resources {
    unsigned char* input = nullptr;
    float* output = nullptr;
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    ~Resources() {
        // Destructors cannot throw; cleanup errors remain visible on stderr.
        for (cudaError_t code : {cudaEventDestroy(start), cudaEventDestroy(stop),
                                 cudaFree(input), cudaFree(output)}) {
            if (code != cudaSuccess)
                std::cerr << "CUDA cleanup error: " << cudaGetErrorString(code) << "\n";
        }
    }
};

int main(int argc, char** argv) {
    try {
        const int repeats = argc > 1 ? std::stoi(argv[1]) : 100;
        const int warmup = argc > 2 ? std::stoi(argv[2]) : 10;
        if (repeats < 1 || warmup < 1 || repeats > 100000 || warmup > 100000 || argc > 3)
            throw std::runtime_error("Usage: BenchmarkPreprocess.exe [repeats 1..100000] [warmup 1..100000]");
        auto init = Clock::now();
        CHECK(cudaSetDevice(0));
        CHECK(cudaFree(nullptr));
        const double context_ms = ms(init);
        cudaDeviceProp device;
        CHECK(cudaGetDeviceProperties(&device, 0));
        int runtime = 0, driver = 0;
        CHECK(cudaRuntimeGetVersion(&runtime));
        CHECK(cudaDriverGetVersion(&driver));
        std::cout << std::setprecision(10)
                  << "{\n  \"schema_version\": 1,\n  \"device\": \"" << device.name
                  << "\",\n  \"compute_capability\": \"" << device.major << "." << device.minor
                  << "\",\n  \"cuda_runtime_version\": " << runtime
                  << ",\n  \"cuda_driver_version\": " << driver
                  << ",\n  \"build_target\": \"sm_61\",\n  \"warmup_runs\": " << warmup
                  << ",\n  \"measured_runs\": " << repeats
                  << ",\n  \"absolute_tolerance\": 0.000001,\n  \"host_memory\": \"pageable std::vector\","
                  << "\n  \"layout\": \"uint8 BGR HWC to float32 RGB CHW divided by 255\","
                  << "\n  \"context_initialization_ms\": " << context_ms
                  << ",\n  \"timing_boundaries\": {"
                  << "\"cpu_ms\": \"steady_clock host reference, existing output allocation\","
                  << "\"h2d_ms\": \"steady_clock synchronous pageable cudaMemcpy including host staging\","
                  << "\"kernel_ms\": \"CUDA events around kernel, stop synchronized\","
                  << "\"d2h_ms\": \"steady_clock synchronous cudaMemcpy after kernel already synchronized\","
                  << "\"complete_ms\": \"separate uninstrumented H2D + launch + synchronous D2H, reused buffers; excludes allocation and context setup\","
                  << "\"allocation_setup_ms\": \"device buffers and timing event creation only\"},\n  \"cases\": [\n";
        const int sizes[][2] = {{1, 1}, {17, 19}, {640, 640}, {1920, 1080}};
        for (int case_index = 0; case_index < 4; ++case_index) {
            const int width = sizes[case_index][0], height = sizes[case_index][1];
            const size_t pixels = static_cast<size_t>(width) * height;
            const size_t input_bytes = pixels * 3, output_bytes = pixels * 3 * sizeof(float);
            std::vector<unsigned char> input(input_bytes);
            std::vector<float> cpu(pixels * 3), gpu(pixels * 3);
            for (size_t i = 0; i < input.size(); ++i) input[i] = static_cast<unsigned char>((i * 73 + 29) % 256);
            Resources r;
            auto alloc = Clock::now();
            CHECK(cudaMalloc(reinterpret_cast<void**>(&r.input), input_bytes));
            CHECK(cudaMalloc(reinterpret_cast<void**>(&r.output), output_bytes));
            CHECK(cudaEventCreate(&r.start));
            CHECK(cudaEventCreate(&r.stop));
            const double allocation_ms = ms(alloc);
            auto complete = [&]() {
                CHECK(cudaMemcpy(r.input, input.data(), input_bytes, cudaMemcpyHostToDevice));
                CHECK(launchBgrToRgbChw(r.input, r.output, pixels));
                CHECK(cudaMemcpy(gpu.data(), r.output, output_bytes, cudaMemcpyDeviceToHost));
            };
            for (int i = 0; i < warmup; ++i) {
                cpuPreprocessBgrToRgbChw(input.data(), cpu.data(), pixels);
                complete();
            }
            double cpu_sum = 0, h2d_sum = 0, kernel_sum = 0, d2h_sum = 0, complete_sum = 0;
            for (int i = 0; i < repeats; ++i) {
                auto start = Clock::now();
                cpuPreprocessBgrToRgbChw(input.data(), cpu.data(), pixels);
                cpu_sum += ms(start);
                start = Clock::now();
                CHECK(cudaMemcpy(r.input, input.data(), input_bytes, cudaMemcpyHostToDevice));
                h2d_sum += ms(start);
                CHECK(cudaEventRecord(r.start));
                CHECK(launchBgrToRgbChw(r.input, r.output, pixels));
                CHECK(cudaEventRecord(r.stop));
                CHECK(cudaEventSynchronize(r.stop));
                float kernel_ms = 0;
                CHECK(cudaEventElapsedTime(&kernel_ms, r.start, r.stop));
                kernel_sum += kernel_ms;
                start = Clock::now();
                CHECK(cudaMemcpy(gpu.data(), r.output, output_bytes, cudaMemcpyDeviceToHost));
                d2h_sum += ms(start);
                start = Clock::now();
                complete();
                complete_sum += ms(start);
            }
            double maximum_error = 0;
            for (size_t i = 0; i < cpu.size(); ++i) {
                if (!std::isfinite(gpu[i])) throw std::runtime_error("Nonfinite CUDA output");
                maximum_error = std::max(maximum_error, static_cast<double>(std::abs(cpu[i] - gpu[i])));
            }
            if (maximum_error > 1e-6) throw std::runtime_error("CPU/CUDA output comparison failed");
            std::cout << "    {\"width\": " << width << ", \"height\": " << height
                      << ", \"pixels\": " << pixels << ", \"allocation_setup_ms\": " << allocation_ms
                      << ", \"cpu_ms\": " << cpu_sum / repeats
                      << ", \"h2d_ms\": " << h2d_sum / repeats
                      << ", \"kernel_ms\": " << kernel_sum / repeats
                      << ", \"d2h_ms\": " << d2h_sum / repeats
                      << ", \"complete_ms\": " << complete_sum / repeats
                      << ", \"maximum_absolute_error\": " << maximum_error
                      << ", \"comparison_passed\": true}" << (case_index < 3 ? "," : "") << "\n";
        }
        std::cout << "  ]\n}\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "Preprocessing benchmark failed: " << error.what() << "\n";
        return EXIT_FAILURE;
    }
}
