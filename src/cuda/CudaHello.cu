#include <cuda_runtime.h>

#include <iostream>
#include <cstdlib>

#define CUDA_CHECK(call)                                                     \
do                                                                           \
{                                                                            \
    cudaError_t error = call;                                                \
    if (error != cudaSuccess)                                                \
    {                                                                        \
        std::cerr << "CUDA error at "                                        \
                  << __FILE__ << ":" << __LINE__ << "\n"                     \
                  << "Code: " << static_cast<int>(error) << "\n"             \
                  << "Name: " << cudaGetErrorName(error) << "\n"             \
                  << "Message: " << cudaGetErrorString(error) << "\n";        \
        std::exit(EXIT_FAILURE);                                             \
    }                                                                        \
} while (0)

__global__ void doubleValues(float* data, int n)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < n)
    {
        data[index] *= 2.0f;
    }
}


int main()
{
    constexpr int N = 8;

    float hostData[N] =
    {
        1.0f,
        2.0f,
        3.0f,
        4.0f,
        5.0f,
        6.0f,
        7.0f,
        8.0f
    };


    std::cout << "Original values:\n";

    for (float value : hostData)
    {
        std::cout << value << " ";
    }

    std::cout << "\n\n";


    // ---------------------------------------
    // Check GPU
    // ---------------------------------------

    int deviceCount = 0;

    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0)
    {
        std::cerr << "No CUDA GPU found." << std::endl;
        return EXIT_FAILURE;
    }


    cudaDeviceProp properties{};

    CUDA_CHECK(
        cudaGetDeviceProperties(
            &properties,
            0
        )
    );


    std::cout
        << "GPU: "
        << properties.name
        << "\n";

    std::cout
        << "Compute capability: "
        << properties.major
        << "."
        << properties.minor
        << "\n\n";


    // ---------------------------------------
    // Allocate GPU memory
    // ---------------------------------------

    float* deviceData = nullptr;

    const size_t size =
        N * sizeof(float);


    CUDA_CHECK(
        cudaMalloc(
            reinterpret_cast<void**>(&deviceData),
            size
        )
    );


    // ---------------------------------------
    // Copy CPU -> GPU
    // ---------------------------------------

    CUDA_CHECK(
        cudaMemcpy(
            deviceData,
            hostData,
            size,
            cudaMemcpyHostToDevice
        )
    );


    // ---------------------------------------
    // Launch CUDA kernel
    // ---------------------------------------

    constexpr int threadsPerBlock = 256;

    const int blocks =
        (N + threadsPerBlock - 1)
        / threadsPerBlock;


    std::cout
        << "Launching kernel...\n";


    doubleValues<<<blocks, threadsPerBlock>>>(
        deviceData,
        N
    );


    // Check whether kernel launch itself failed
    CUDA_CHECK(
        cudaGetLastError()
    );


    // Wait until GPU finishes executing
    CUDA_CHECK(
        cudaDeviceSynchronize()
    );


    // ---------------------------------------
    // Copy GPU -> CPU
    // ---------------------------------------

    CUDA_CHECK(
        cudaMemcpy(
            hostData,
            deviceData,
            size,
            cudaMemcpyDeviceToHost
        )
    );


    // ---------------------------------------
    // Free GPU memory
    // ---------------------------------------

    CUDA_CHECK(
        cudaFree(deviceData)
    );


    // ---------------------------------------
    // Print result
    // ---------------------------------------

    std::cout
        << "\nProcessed values:\n";


    for (float value : hostData)
    {
        std::cout << value << " ";
    }


    std::cout << "\n";


    // Optional cleanup/reset
    CUDA_CHECK(
        cudaDeviceReset()
    );


    return EXIT_SUCCESS;
}