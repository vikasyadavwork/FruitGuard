#include <cuda_runtime.h>

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <vector>

#define CUDA_CHECK(call)                                                   \
do                                                                         \
{                                                                          \
    cudaError_t error = call;                                              \
    if (error != cudaSuccess)                                              \
    {                                                                      \
        std::cerr << "CUDA error at "                                      \
                  << __FILE__ << ":" << __LINE__ << "\n"                   \
                  << "Name: " << cudaGetErrorName(error) << "\n"           \
                  << "Message: " << cudaGetErrorString(error) << "\n";      \
        std::exit(EXIT_FAILURE);                                           \
    }                                                                      \
} while (0)


// ----------------------------------------------------------
// GPU kernel
//
// Input:
//   BGR uint8 layout:
//   B G R | B G R | B G R ...
//
// Output:
//   RGB float layout:
//   R G B | R G B | R G B ...
//
// Values are normalized to [0, 1]
// ----------------------------------------------------------

__global__ void bgrToRgbNormalize(
    const unsigned char* input,
    float* output,
    int pixelCount)
{
    int pixelIndex =
        blockIdx.x * blockDim.x + threadIdx.x;

    if (pixelIndex >= pixelCount)
        return;

    int base = pixelIndex * 3;

    unsigned char B = input[base + 0];
    unsigned char G = input[base + 1];
    unsigned char R = input[base + 2];

    output[base + 0] = R / 255.0f;
    output[base + 1] = G / 255.0f;
    output[base + 2] = B / 255.0f;
}


// ----------------------------------------------------------
// CPU reference implementation
// ----------------------------------------------------------

void cpuPreprocess(
    const unsigned char* input,
    float* output,
    int pixelCount)
{
    for (int i = 0; i < pixelCount; ++i)
    {
        int base = i * 3;

        unsigned char B = input[base + 0];
        unsigned char G = input[base + 1];
        unsigned char R = input[base + 2];

        output[base + 0] = R / 255.0f;
        output[base + 1] = G / 255.0f;
        output[base + 2] = B / 255.0f;
    }
}


// ----------------------------------------------------------
// Main
// ----------------------------------------------------------

int main()
{
    // Let's simulate a 640x640 OpenCV frame
    const int width = 640;
    const int height = 640;
    const int channels = 3;

    const int pixelCount = width * height;
    const int elementCount = pixelCount * channels;

    const size_t inputBytes =
        elementCount * sizeof(unsigned char);

    const size_t outputBytes =
        elementCount * sizeof(float);


    std::cout << "Image size: "
              << width << " x "
              << height << "\n";

    std::cout << "Pixels: "
              << pixelCount << "\n\n";


    // ------------------------------------------------------
    // Create fake BGR image
    // ------------------------------------------------------

    std::vector<unsigned char> hostInput(
        elementCount
    );

    // Fill with deterministic test data
    for (int i = 0; i < elementCount; ++i)
    {
        hostInput[i] =
            static_cast<unsigned char>(i % 256);
    }


    std::vector<float> cpuOutput(
        elementCount
    );

    std::vector<float> gpuOutput(
        elementCount
    );


    // ------------------------------------------------------
    // CPU preprocessing benchmark
    // ------------------------------------------------------

    auto cpuStart =
        std::chrono::high_resolution_clock::now();

    cpuPreprocess(
        hostInput.data(),
        cpuOutput.data(),
        pixelCount
    );

    auto cpuEnd =
        std::chrono::high_resolution_clock::now();


    std::chrono::duration<double, std::milli>
        cpuDuration = cpuEnd - cpuStart;


    std::cout
        << "CPU preprocessing: "
        << cpuDuration.count()
        << " ms\n";


    // ------------------------------------------------------
    // Allocate GPU memory
    // ------------------------------------------------------

    unsigned char* deviceInput = nullptr;
    float* deviceOutput = nullptr;


    CUDA_CHECK(
        cudaMalloc(
            reinterpret_cast<void**>(&deviceInput),
            inputBytes
        )
    );


    CUDA_CHECK(
        cudaMalloc(
            reinterpret_cast<void**>(&deviceOutput),
            outputBytes
        )
    );


    // ------------------------------------------------------
    // Copy image CPU -> GPU
    // ------------------------------------------------------

    CUDA_CHECK(
        cudaMemcpy(
            deviceInput,
            hostInput.data(),
            inputBytes,
            cudaMemcpyHostToDevice
        )
    );


    // ------------------------------------------------------
    // CUDA timing events
    // ------------------------------------------------------

    cudaEvent_t startEvent;
    cudaEvent_t stopEvent;

    CUDA_CHECK(
        cudaEventCreate(&startEvent)
    );

    CUDA_CHECK(
        cudaEventCreate(&stopEvent)
    );


    const int threadsPerBlock = 256;

    const int blocks =
        (pixelCount + threadsPerBlock - 1)
        / threadsPerBlock;


    CUDA_CHECK(
        cudaEventRecord(startEvent)
    );


    // ------------------------------------------------------
    // Launch CUDA kernel
    // ------------------------------------------------------

    bgrToRgbNormalize<<<
        blocks,
        threadsPerBlock
    >>>(
        deviceInput,
        deviceOutput,
        pixelCount
    );


    CUDA_CHECK(
        cudaGetLastError()
    );


    CUDA_CHECK(
        cudaEventRecord(stopEvent)
    );

    CUDA_CHECK(
        cudaEventSynchronize(stopEvent)
    );


    float gpuMilliseconds = 0.0f;

    CUDA_CHECK(
        cudaEventElapsedTime(
            &gpuMilliseconds,
            startEvent,
            stopEvent
        )
    );


    std::cout
        << "CUDA kernel: "
        << gpuMilliseconds
        << " ms\n";


    // ------------------------------------------------------
    // Copy GPU result back
    // ------------------------------------------------------

    CUDA_CHECK(
        cudaMemcpy(
            gpuOutput.data(),
            deviceOutput,
            outputBytes,
            cudaMemcpyDeviceToHost
        )
    );


    // ------------------------------------------------------
    // Validate CPU and GPU results
    // ------------------------------------------------------

    bool correct = true;

    for (int i = 0; i < elementCount; ++i)
    {
        float difference =
            cpuOutput[i] - gpuOutput[i];

        if (difference < 0)
            difference = -difference;

        if (difference > 0.00001f)
        {
            correct = false;

            std::cout
                << "Mismatch at index "
                << i
                << "\n";

            break;
        }
    }


    if (correct)
    {
        std::cout
            << "\nCPU and GPU outputs match."
            << "\n";
    }


    // ------------------------------------------------------
    // Show first pixel
    // ------------------------------------------------------

    std::cout
        << "\nFirst BGR pixel:\n";

    std::cout
        << "B = "
        << static_cast<int>(hostInput[0])
        << "\n";

    std::cout
        << "G = "
        << static_cast<int>(hostInput[1])
        << "\n";

    std::cout
        << "R = "
        << static_cast<int>(hostInput[2])
        << "\n";


    std::cout
        << "\nFirst RGB normalized pixel:\n";

    std::cout
        << "R = "
        << gpuOutput[0]
        << "\n";

    std::cout
        << "G = "
        << gpuOutput[1]
        << "\n";

    std::cout
        << "B = "
        << gpuOutput[2]
        << "\n";


    // ------------------------------------------------------
    // Cleanup
    // ------------------------------------------------------

    CUDA_CHECK(
        cudaEventDestroy(startEvent)
    );

    CUDA_CHECK(
        cudaEventDestroy(stopEvent)
    );

    CUDA_CHECK(
        cudaFree(deviceInput)
    );

    CUDA_CHECK(
        cudaFree(deviceOutput)
    );


    return 0;
}