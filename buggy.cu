#include <cuda_runtime.h>
#include <vector>
#include <iostream>

// Tile size
const int N = 4;

__global__ void kernel_transpose(float* idata, float* odata, int height, int width) {
    __shared__ float tile[N][N];
    int x = threadIdx.x;
    int y = threadIdx.y;

    int idx = blockIdx.x * N + x;
    int idy = blockIdx.y * N + y;

    tile[y][x] = idata[idy * blockDim.x * gridDim.x + idx];
    __syncthreads();
    odata[idx * blockDim.x * gridDim.x + idy] = tile[x][y];
}

int main(int argc, char** argv) {
    // CHANGE THIS IF YOU WANT idata TO BE OF DIFFERENT SIZE THAN SHARED MEMORY
    // ASSUMPTION: height AND width ARE MULTIPLES OF N, SO CHANGE ACCORDINGLY (E.G. 2 * N)
    // ANOTHER ASSUMPTION FOR SIMPLICITY: SQUARE MATRIX, SO height = width
    int height = N;
    int width = height;

    std::vector<float> idata(height * width);
    std::vector<float> odata(height * width);
    for (int i = 0; i < height * width; ++i) {
        idata[i] = i;
    }

    float* device_input = nullptr;
    float* device_output = nullptr;
    
    size_t data_bytes = height * width * sizeof(float);
    
    cudaMalloc(&device_input, data_bytes);
    cudaMalloc(&device_output, data_bytes);
    
    cudaMemcpy(device_input, idata.data(), data_bytes, cudaMemcpyHostToDevice);

    dim3 threads_per_block(N, N, 1);
    dim3 blocks_per_grid(
        (width + threads_per_block.x - 1) / threads_per_block.x,
        (height + threads_per_block.y - 1) / threads_per_block.y,
        1
    );
    size_t shared_bytes = N * N * sizeof(float);
    kernel_transpose<<<blocks_per_grid, threads_per_block, shared_bytes>>>(device_input, device_output, height, width);

    cudaMemcpy(odata.data(), device_output, data_bytes, cudaMemcpyDeviceToHost);
    
    cudaFree(device_input);
    cudaFree(device_output);

    std::cout << "Input data:\n";
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            std::cout << idata[i * width + j] << ' ';
        }
        std::cout << '\n';
    }

    std::cout << '\n' << "Output data:\n";
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            std::cout << odata[i * width + j] << ' ';
        }
        std::cout << '\n';
    }
    
    return 0;
}
