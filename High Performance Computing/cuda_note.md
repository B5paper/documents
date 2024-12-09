# CUDA Note

## cache

* nvcc 编译不同的架构支持`-gencode arch=compute_70,code=sm_70`

    如果硬件是 80 的兼容性，那么这个硬件支持使用 70 兼容性编译出来的 code。

    相反，如果硬件是 70 的兼容性，那么它跑不起来使用 80 兼容性编译出来的 code.

* cuda-gdb hit 断点时，可以使用`info cuda kernels`查看当前的 kernel 函数，sm，block, grid, device 使用情况等信息。

* nvcc 加上`-g`后支持 host code 调试，加上`-G`后支持 cuda kernel code 调试

* cuda kernel example

    ```cpp
    // Kernel definition
    __global__ void VecAdd(float* A, float* B, float* C)
    {
        int i = threadIdx.x;
        C[i] = A[i] + B[i];
    }

    int main()
    {
        ...
        // Kernel invocation with N threads
        VecAdd<<<1, N>>>(A, B, C);
        ...
    }
    ```

    cuda 编程模型是 c++ 的一个扩展。

    cuda kernel 前面要加上`__global__`。（为什么不是加`__device__`？）

* 每个 cuda kernel 在运行时对应一个 cuda thread，不是 block，也不是 sm

* SMs - Streaming Multiprocessors

* 在 cuda 中，软件使用的 block 会被 cuda 根据实际的物理 block 数重新排布

    比如软件定义 8 个 block 一起计算，如果 card 0 只有 2 个 sm，那么程序会变成 2 个 block 2 个 block 执行，一共执行 4 轮；如果 card 1 有 4 个 sm，那么程序会变成 4 个 block 4 个 block 执行，一共执行 2 轮。

    这个能力被官网称作 Automatic Scalability。

    注：这里的 block 可能指的并不是 thread，一个 block 可能会包含多个 thread。

* 官网介绍的 cuda 的核心三个抽象

    At its core are three key abstractions — a hierarchy of thread groups, shared memories, and barrier synchronization — that are simply exposed to the programmer as a minimal set of language extensions.

* 假如把 pcie p2p 和 nvlink p2p 都看作 peer access 能力，那么可以使用`cudaDeviceCanAccessPeer()`判断两个 dev 是否可以通过 pcie/nvlink 进行 p2p 互联

    example:

    `main.cu`:

    ```cpp
    #include <cuda_runtime.h>
    #include <stdio.h>

    int main()
    {
        int deviceCount;
        cudaGetDeviceCount(&deviceCount);

        for (int i = 0; i < deviceCount; ++i)
        {
            for (int j = 0; j < deviceCount; ++j)
            {
                if (i != j)
                {
                    int canAccessPeer = 0;
                    cudaDeviceCanAccessPeer(&canAccessPeer, i, j);
                    if (canAccessPeer)
                    {
                        printf("dev peer access %d - %d, OK\n", i, j);
                    }
                    else 
                    {
                        printf("dev peer access %d - %d, Error\n", i, j);
                    }
                }
            }
        }

        return 0;
    }
    ```

    compile: `nvcc -g main.cu -L/usr/local/cuda-12.1/lib64 -lcudart -o main`

    run: `./main`

    output:

    ```
    dev peer access 0 - 1, OK
    dev peer access 1 - 0, OK
    ```

* a100 x 8 机器的 p2p 测速输出

    ```
    hlc@a147:~/Data/Projects/cuda-samples-12.4.1/bin/x86_64/linux/release$ ./p2pBandwidthLatencyTest 
    [P2P (Peer-to-Peer) GPU Bandwidth Latency Test]
    Device: 0, NVIDIA A100-SXM4-80GB, pciBusID: 26, pciDeviceID: 0, pciDomainID:0
    Device: 1, NVIDIA A100-SXM4-80GB, pciBusID: 2c, pciDeviceID: 0, pciDomainID:0
    Device: 2, NVIDIA A100-SXM4-80GB, pciBusID: 66, pciDeviceID: 0, pciDomainID:0
    Device: 3, NVIDIA A100-SXM4-80GB, pciBusID: 6b, pciDeviceID: 0, pciDomainID:0
    Device: 4, NVIDIA A100-SXM4-80GB, pciBusID: a4, pciDeviceID: 0, pciDomainID:0
    Device: 5, NVIDIA A100-SXM4-80GB, pciBusID: a9, pciDeviceID: 0, pciDomainID:0
    Device: 6, NVIDIA A100-SXM4-80GB, pciBusID: e1, pciDeviceID: 0, pciDomainID:0
    Device: 7, NVIDIA A100-SXM4-80GB, pciBusID: e7, pciDeviceID: 0, pciDomainID:0
    Device=0 CAN Access Peer Device=1
    Device=0 CAN Access Peer Device=2
    Device=0 CAN Access Peer Device=3
    Device=0 CAN Access Peer Device=4
    Device=0 CAN Access Peer Device=5
    Device=0 CAN Access Peer Device=6
    Device=0 CAN Access Peer Device=7
    Device=1 CAN Access Peer Device=0
    Device=1 CAN Access Peer Device=2
    Device=1 CAN Access Peer Device=3
    Device=1 CAN Access Peer Device=4
    Device=1 CAN Access Peer Device=5
    Device=1 CAN Access Peer Device=6
    Device=1 CAN Access Peer Device=7
    Device=2 CAN Access Peer Device=0
    Device=2 CAN Access Peer Device=1
    Device=2 CAN Access Peer Device=3
    Device=2 CAN Access Peer Device=4
    Device=2 CAN Access Peer Device=5
    Device=2 CAN Access Peer Device=6
    Device=2 CAN Access Peer Device=7
    Device=3 CAN Access Peer Device=0
    Device=3 CAN Access Peer Device=1
    Device=3 CAN Access Peer Device=2
    Device=3 CAN Access Peer Device=4
    Device=3 CAN Access Peer Device=5
    Device=3 CAN Access Peer Device=6
    Device=3 CAN Access Peer Device=7
    Device=4 CAN Access Peer Device=0
    Device=4 CAN Access Peer Device=1
    Device=4 CAN Access Peer Device=2
    Device=4 CAN Access Peer Device=3
    Device=4 CAN Access Peer Device=5
    Device=4 CAN Access Peer Device=6
    Device=4 CAN Access Peer Device=7
    Device=5 CAN Access Peer Device=0
    Device=5 CAN Access Peer Device=1
    Device=5 CAN Access Peer Device=2
    Device=5 CAN Access Peer Device=3
    Device=5 CAN Access Peer Device=4
    Device=5 CAN Access Peer Device=6
    Device=5 CAN Access Peer Device=7
    Device=6 CAN Access Peer Device=0
    Device=6 CAN Access Peer Device=1
    Device=6 CAN Access Peer Device=2
    Device=6 CAN Access Peer Device=3
    Device=6 CAN Access Peer Device=4
    Device=6 CAN Access Peer Device=5
    Device=6 CAN Access Peer Device=7
    Device=7 CAN Access Peer Device=0
    Device=7 CAN Access Peer Device=1
    Device=7 CAN Access Peer Device=2
    Device=7 CAN Access Peer Device=3
    Device=7 CAN Access Peer Device=4
    Device=7 CAN Access Peer Device=5
    Device=7 CAN Access Peer Device=6

    ***NOTE: In case a device doesn't have P2P access to other one, it falls back to normal memcopy procedure.
    So you can see lesser Bandwidth (GB/s) and unstable Latency (us) in those cases.

    P2P Connectivity Matrix
         D\D     0     1     2     3     4     5     6     7
         0	     1     1     1     1     1     1     1     1
         1	     1     1     1     1     1     1     1     1
         2	     1     1     1     1     1     1     1     1
         3	     1     1     1     1     1     1     1     1
         4	     1     1     1     1     1     1     1     1
         5	     1     1     1     1     1     1     1     1
         6	     1     1     1     1     1     1     1     1
         7	     1     1     1     1     1     1     1     1
    Unidirectional P2P=Disabled Bandwidth Matrix (GB/s)
       D\D     0      1      2      3      4      5      6      7 
         0 1524.39  16.69  20.20  20.58  18.53  20.63  19.52  20.58 
         1  16.98 1537.89  20.05  20.08  17.80  20.15  19.42  20.61 
         2  20.19  20.25 1534.87  16.91  17.70  20.17  19.25  20.61 
         3  20.12  20.23  17.01 1539.41  17.79  20.47  19.62  20.61 
         4  18.36  19.99  18.57  19.74 1573.51  11.79  19.22  18.40 
         5  19.46  19.00  19.69  18.76  12.10 1570.35  17.80  19.22 
         6  18.39  20.07  18.55  19.62  18.64  19.95 1571.93  11.72 
         7  19.43  18.18  19.65  18.59  19.60  18.97  11.83 1575.10 
    Unidirectional P2P=Enabled Bandwidth (P2P Writes) Matrix (GB/s)
       D\D     0      1      2      3      4      5      6      7 
         0 1545.50 270.83 274.85 273.60 274.38 272.17 273.49 274.86 
         1 270.19 1556.27 274.34 273.93 274.58 274.03 274.48 274.67 
         2 270.85 271.61 1540.93 274.54 274.32 274.63 274.53 275.05 
         3 271.98 273.56 272.55 1591.14 274.67 275.17 274.81 275.42 
         4 272.06 273.89 272.11 275.67 1587.91 274.89 276.05 275.05 
         5 272.75 273.16 273.78 275.51 275.13 1584.69 275.54 274.60 
         6 273.19 273.46 273.59 275.04 275.15 275.83 1584.69 273.68 
         7 273.62 273.78 273.35 275.28 275.49 275.44 274.88 1586.29 
    Bidirectional P2P=Disabled Bandwidth Matrix (GB/s)
       D\D     0      1      2      3      4      5      6      7 
         0 1559.38  12.03  27.82  27.85  20.26  21.38  21.41  21.52 
         1  12.23 1596.02  28.21  28.04  20.75  21.53  21.47  21.82 
         2  28.26  28.20 1597.65  12.29  20.25  21.27  21.39  21.14 
         3  28.20  28.04  12.28 1600.10  20.36  21.32  21.39  21.17 
         4  22.80  22.50  22.37  22.59 1596.02  10.55  20.90  20.69 
         5  22.13  22.51  22.39  22.52  11.88 1598.47  20.94  20.87 
         6  22.05  22.35  22.35  22.19  21.61  20.96 1599.28  10.55 
         7  22.18  22.37  22.08  22.15  21.08  21.05  12.34 1602.56 
    Bidirectional P2P=Enabled Bandwidth Matrix (GB/s)
       D\D     0      1      2      3      4      5      6      7 
         0 1556.27 423.17 423.40 423.51 424.20 424.43 424.31 425.35 
         1 424.43 1556.27 424.20 424.66 422.25 424.89 424.89 425.68 
         2 421.79 422.59 1560.16 424.08 425.81 425.35 425.35 424.08 
         3 425.52 424.08 424.31 1563.28 425.35 423.02 425.12 425.00 
         4 426.45 426.44 426.40 426.60 1607.51 518.15 518.66 521.47 
         5 425.39 426.69 426.76 427.09 517.28 1597.65 520.94 517.97 
         6 426.85 426.74 426.48 427.34 520.39 517.80 1596.02 521.47 
         7 426.40 427.50 426.43 426.02 426.93 518.39 519.35 1594.39 
    P2P=Disabled Latency Matrix (us)
       GPU     0      1      2      3      4      5      6      7 
         0   3.04  23.59  23.62  23.88  24.51  24.48  24.31  24.48 
         1  23.66   2.85  23.65  23.58  24.49  24.48  24.30  24.48 
         2  23.61  23.60   3.03  23.58  24.54  24.49  24.40  24.48 
         3  23.59  23.57  23.59   3.15  24.50  24.49  24.46  24.49 
         4  24.61  24.62  24.62  24.60   2.99  24.51  24.60  24.53 
         5  24.57  24.61  24.62  24.60  24.59   2.91  24.60  24.53 
         6  24.60  24.41  24.62  24.60  24.59  24.59   2.29  24.30 
         7  24.64  24.65  24.64  24.64  24.60  24.60  24.43   2.48 

       CPU     0      1      2      3      4      5      6      7 
         0   2.40   7.08   6.87   6.92   8.45   8.39   8.45   8.38 
         1   6.99   2.32   6.91   6.85   8.48   8.44   8.49   8.47 
         2   6.97   6.88   2.34   6.88   8.47   8.41   8.49   8.37 
         3   6.80   6.90   6.75   2.33   8.53   8.54   8.45   8.39 
         4   7.95   7.88   7.89   7.87   2.86   9.56   9.56   9.54 
         5   7.88   7.85   7.85   7.82   9.42   2.85   9.55   9.47 
         6   7.91   7.89   7.90   7.82   9.49   9.45   2.90   9.57 
         7   7.91   7.85   7.86   7.80   9.46   9.46   9.55   2.89 
    P2P=Enabled Latency (P2P Writes) Matrix (us)
       GPU     0      1      2      3      4      5      6      7 
         0   3.02   3.42   3.36   3.36   3.40   3.35   3.42   3.43 
         1   3.36   2.87   3.41   3.47   3.54   3.55   3.48   3.54 
         2   3.36   3.41   3.02   3.44   3.43   3.37   3.36   3.40 
         3   3.36   3.37   3.37   3.18   3.37   3.40   3.43   3.37 
         4   3.23   3.19   3.18   3.18   2.98   3.19   3.18   3.19 
         5   3.19   3.22   3.19   3.23   3.15   2.91   3.24   3.25 
         6   2.82   2.77   2.81   2.79   2.78   2.81   2.28   2.84 
         7   2.82   2.78   2.78   2.82   2.82   2.82   2.82   2.47 

       CPU     0      1      2      3      4      5      6      7 
         0   2.41   2.07   2.07   2.10   2.10   2.09   2.08   2.09 
         1   2.10   2.42   2.06   2.33   2.31   2.32   2.29   2.28 
         2   2.18   2.02   2.44   2.03   2.00   2.05   2.01   2.09 
         3   2.07   2.03   2.02   2.43   2.02   2.05   1.99   2.10 
         4   2.69   2.63   2.61   2.61   2.96   2.64   2.59   2.60 
         5   2.69   2.62   2.63   2.65   2.66   2.98   2.70   2.63 
         6   2.69   2.61   2.63   2.64   2.63   2.63   2.95   2.59 
         7   2.71   2.64   2.65   2.64   2.65   2.74   2.64   3.00 

    NOTE: The CUDA Samples are not meant for performance measurements. Results may vary when GPU Boost is enabled.
    ```

* 使用 cuda-sample 里提供的程序也可以测速，测出来数据和自己写代码测的差不多

    run: `(base) huliucheng@zjxj:~/Documents/Projects/cuda-samples-12.1/bin/x86_64/linux/release$ ./p2pBandwidthLatencyTest`

    output:

    ```
    [P2P (Peer-to-Peer) GPU Bandwidth Latency Test]
    Device: 0, Tesla V100-PCIE-32GB, pciBusID: b1, pciDeviceID: 0, pciDomainID:0
    Device: 1, Tesla V100-PCIE-32GB, pciBusID: e3, pciDeviceID: 0, pciDomainID:0
    Device=0 CAN Access Peer Device=1
    Device=1 CAN Access Peer Device=0

    ***NOTE: In case a device doesn't have P2P access to other one, it falls back to normal memcopy procedure.
    So you can see lesser Bandwidth (GB/s) and unstable Latency (us) in those cases.

    P2P Connectivity Matrix
         D\D     0     1
         0	     1     1
         1	     1     1
    Unidirectional P2P=Disabled Bandwidth Matrix (GB/s)
       D\D     0      1 
         0 771.60  12.22 
         1  12.19 774.28 
    Unidirectional P2P=Enabled Bandwidth (P2P Writes) Matrix (GB/s)
       D\D     0      1 
         0 708.94  11.35 
         1  11.36 773.90 
    Bidirectional P2P=Disabled Bandwidth Matrix (GB/s)
       D\D     0      1 
         0 712.33  16.60 
         1  16.42 775.05 
    Bidirectional P2P=Enabled Bandwidth Matrix (GB/s)
       D\D     0      1 
         0 711.85  22.07 
         1  22.07 776.40 
    P2P=Disabled Latency Matrix (us)
       GPU     0      1 
         0   1.96  78.17 
         1  14.45   1.97 

       CPU     0      1 
         0   2.63   6.44 
         1   6.42   2.59 
    P2P=Enabled Latency (P2P Writes) Matrix (us)
       GPU     0      1 
         0   1.91   1.74 
         1   1.74   1.97 

       CPU     0      1 
         0   2.58   2.00 
         1   1.92   2.65 

    NOTE: The CUDA Samples are not meant for performance measurements. Results may vary when GPU Boost is enabled.
    ```

* deepseek 帮忙写的检测环境中是否有 nvlink 的代码

    `main.cu`:

    ```cpp
    #include <nvml.h>
    #include <iostream>

    int main() {
        nvmlInit();

        unsigned int deviceCount;
        nvmlDeviceGetCount(&deviceCount);

        for (unsigned int i = 0; i < deviceCount; ++i) {
            nvmlDevice_t device;
            nvmlDeviceGetHandleByIndex(i, &device);

            for (unsigned int j = 0; j < deviceCount; ++j) {
                if (i != j) {
                    nvmlPciInfo_t pciInfo1, pciInfo2;
                    nvmlDeviceGetPciInfo(device, &pciInfo1);

                    nvmlDevice_t peerDevice;
                    nvmlDeviceGetHandleByIndex(j, &peerDevice);
                    nvmlDeviceGetPciInfo(peerDevice, &pciInfo2);

                    nvmlEnableState_t isEnabled;
                    nvmlDeviceGetNvLinkState(device, j, &isEnabled);

                    if (isEnabled == NVML_FEATURE_ENABLED) {
                        std::cout << "Device " << i << " is connected to Device " << j << " via NVLink." << std::endl;
                    } else {
                        std::cout << "Device " << i << " is not connected to Device " << j << " via NVLink." << std::endl;
                    }
                }
            }
        }

        nvmlShutdown();
        return 0;
    }
    ```

    compile: `nvcc -g main.cu -lnvidia-ml -o main`

    run: `./main`

    output:

    ```
    Device 0 is not connected to Device 1 via NVLink.
    Device 1 is not connected to Device 0 via NVLink.
    ```

    这个是使用 nvml 库来实现的。

* cuda pcie p2p 的一段代码

    `main.cu`:

    ```cpp
    #include <cuda_runtime.h>

    #include <iostream>
    #include <string>
    #include <vector>

    // Enhanced error checking macro
    #define checkCudaErrors(val) \
      checkCudaErrorsImpl((val), #val, __FILE__, __LINE__)

    void checkCudaErrorsImpl(cudaError_t err, const char* func, const char* file,
                             int line) {
      if (err != cudaSuccess) {
        std::cerr << "CUDA Error at " << file << ":" << line << " in " << func
                  << " - " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
      }
    }

    int main(int argc, char** argv) {
      if (argc < 3) {
        std::cerr << "Usage: " << argv[0]
                  << " <from_device1> <to_device1> [<from_device2> <to_device2> "
                     "...] [--enable-peer-access]"
                  << std::endl;
        return EXIT_FAILURE;
      }

      bool enablePeerAccess = false;
      std::vector<std::pair<int, int>> devicePairs;

      // Parse command-line arguments
      for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--enable-peer-access") {
          enablePeerAccess = true;
        } else {
          if (i + 1 < argc) {
            int fromDevice = std::stoi(argv[i]);
            int toDevice = std::stoi(argv[i + 1]);
            devicePairs.emplace_back(fromDevice, toDevice);
            ++i;  // Skip the next argument as it's part of the device pair
          }
        }
      }

      // Enable peer access between devices if possible and requested
      if (enablePeerAccess) {
        for (const auto& pair : devicePairs) {
          int fromDevice = pair.first;
          int toDevice = pair.second;

          int canAccessPeer = 0;
          checkCudaErrors(
              cudaDeviceCanAccessPeer(&canAccessPeer, fromDevice, toDevice));
          if (canAccessPeer) {
            cudaSetDevice(fromDevice);
            cudaError_t err = cudaDeviceEnablePeerAccess(toDevice, 0);
            if (err == cudaSuccess) {
              std::cout << "Peer access enabled from device " << fromDevice
                        << " to device " << toDevice << std::endl;
            } else {
              std::cout << "Failed to enable peer access from device " << fromDevice
                        << " to device " << toDevice << ": "
                        << cudaGetErrorString(err) << std::endl;
            }
          } else {
            std::cout << "Peer access not supported from device " << fromDevice
                      << " to device " << toDevice << std::endl;
          }

          checkCudaErrors(
              cudaDeviceCanAccessPeer(&canAccessPeer, toDevice, fromDevice));
          if (canAccessPeer) {
            cudaSetDevice(toDevice);
            cudaError_t err = cudaDeviceEnablePeerAccess(fromDevice, 0);
            if (err == cudaSuccess) {
              std::cout << "Peer access enabled from device " << toDevice
                        << " to device " << fromDevice << std::endl;
            } else {
              std::cout << "Failed to enable peer access from device " << toDevice
                        << " to device " << fromDevice << ": "
                        << cudaGetErrorString(err) << std::endl;
            }
          } else {
            std::cout << "Peer access not supported from device " << toDevice
                      << " to device " << fromDevice << std::endl;
          }
        }
      }

      size_t sizes[] = {
          // 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288,
          // 1048576, 2097152, 4194304, 8388608, 16777216, 33554432,
          // 67108864, 134217728, 268435456, 536870912, 1073741824, 2147483648,
        //   4294967296};           // 2KB - 4GB
        24 * 1024ULL * 1024ULL * 1024ULL,
        24 * 1024ULL * 1024ULL * 1024ULL,
        24 * 1024ULL * 1024ULL * 1024ULL,
        24 * 1024ULL * 1024ULL * 1024ULL,
        24 * 1024ULL * 1024ULL * 1024ULL,
        24 * 1024ULL * 1024ULL * 1024ULL,
        24 * 1024ULL * 1024ULL * 1024ULL,
        24 * 1024ULL * 1024ULL * 1024ULL,
        24 * 1024ULL * 1024ULL * 1024ULL,
        24 * 1024ULL * 1024ULL * 1024ULL};
      const int numRepeats = 1;  // Number of repetitions for averaging

      for (size_t size : sizes) {
        std::cout << "Testing size: " << size << " bytes" << std::endl;

        std::vector<cudaStream_t> streams(devicePairs.size());
        std::vector<void*> d_srcs(devicePairs.size());
        std::vector<void*> d_dsts(devicePairs.size());
        std::vector<float> totalMilliseconds(devicePairs.size(), 0.0f);

        for (size_t i = 0; i < devicePairs.size(); ++i) {
          int fromDevice = devicePairs[i].first;
          int toDevice = devicePairs[i].second;

          // Allocate source memory on fromDevice
          cudaSetDevice(fromDevice);
          checkCudaErrors(cudaMalloc(&d_srcs[i], size));

          // Allocate destination memory on toDevice
          cudaSetDevice(toDevice);
          checkCudaErrors(cudaMalloc(&d_dsts[i], size));

          // Create stream on fromDevice
          cudaSetDevice(fromDevice);  // Set to fromDevice before creating stream
          checkCudaErrors(cudaStreamCreate(&streams[i]));
        }

        for (int repeat = 0; repeat < numRepeats; ++repeat) {
          std::vector<cudaEvent_t> startEvents(devicePairs.size());
          std::vector<cudaEvent_t> stopEvents(devicePairs.size());

          for (size_t i = 0; i < devicePairs.size(); ++i) {
            int fromDevice = devicePairs[i].first;
            // Ensure device is set to fromDevice
            cudaSetDevice(
                fromDevice);  // Set device to fromDevice where stream[i] resides

            checkCudaErrors(cudaEventCreate(&startEvents[i]));
            checkCudaErrors(cudaEventCreate(&stopEvents[i]));

            checkCudaErrors(cudaEventRecord(startEvents[i], streams[i]));
            checkCudaErrors(cudaMemcpyAsync(d_dsts[i], d_srcs[i], size,
                                            cudaMemcpyDeviceToDevice, streams[i]));
            checkCudaErrors(cudaEventRecord(stopEvents[i], streams[i]));
          }

          // Synchronize all streams to ensure all operations for the current size
          // are completed
          for (size_t i = 0; i < devicePairs.size(); ++i) {
            int fromDevice = devicePairs[i].first;
            cudaSetDevice(fromDevice);  // Ensure device is set before synchronizing

            checkCudaErrors(cudaStreamSynchronize(streams[i]));

            float milliseconds = 0;
            checkCudaErrors(
                cudaEventElapsedTime(&milliseconds, startEvents[i], stopEvents[i]));
            totalMilliseconds[i] += milliseconds;

            checkCudaErrors(cudaEventDestroy(startEvents[i]));
            checkCudaErrors(cudaEventDestroy(stopEvents[i]));
          }
        }

        for (size_t i = 0; i < devicePairs.size(); ++i) {
          int fromDevice = devicePairs[i].first;
          int toDevice = devicePairs[i].second;

          float averageMilliseconds = totalMilliseconds[i] / numRepeats;
          double bandwidth = (size * 1e-9) / (averageMilliseconds * 1e-3);
          std::cout << "  Device Pair " << fromDevice << " -> " << toDevice
                    << ": Average Bandwidth = " << bandwidth
                    << " GB/s, milliseconds " << averageMilliseconds << std::endl;

          // Free source memory on fromDevice
          cudaSetDevice(fromDevice);
          checkCudaErrors(cudaFree(d_srcs[i]));

          // Free destination memory on toDevice
          cudaSetDevice(toDevice);
          checkCudaErrors(cudaFree(d_dsts[i]));

          // Destroy stream on fromDevice
          cudaSetDevice(
              fromDevice);  // Ensure device is set before destroying stream
          checkCudaErrors(cudaStreamDestroy(streams[i]));
        }
      }

      return EXIT_SUCCESS;
    }
    ```

    compile: `nvcc -g main.cu -L/usr/local/cuda/lib64 -lcudart -o main`

    run & output:

    * `./main 0 1`

        ```
        Testing size: 25769803776 bytes
          Device Pair 0 -> 1: Average Bandwidth = 12.4125 GB/s, milliseconds 2076.12
        Testing size: 25769803776 bytes
          Device Pair 0 -> 1: Average Bandwidth = 12.4143 GB/s, milliseconds 2075.82
        Testing size: 25769803776 bytes
          Device Pair 0 -> 1: Average Bandwidth = 12.4187 GB/s, milliseconds 2075.08
        Testing size: 25769803776 bytes
          Device Pair 0 -> 1: Average Bandwidth = 12.2919 GB/s, milliseconds 2096.49
        Testing size: 25769803776 bytes
          Device Pair 0 -> 1: Average Bandwidth = 12.4177 GB/s, milliseconds 2075.25
        Testing size: 25769803776 bytes
          Device Pair 0 -> 1: Average Bandwidth = 12.4205 GB/s, milliseconds 2074.77
        Testing size: 25769803776 bytes
          Device Pair 0 -> 1: Average Bandwidth = 12.4222 GB/s, milliseconds 2074.49
        Testing size: 25769803776 bytes
          Device Pair 0 -> 1: Average Bandwidth = 12.3017 GB/s, milliseconds 2094.81
        Testing size: 25769803776 bytes
          Device Pair 0 -> 1: Average Bandwidth = 12.4195 GB/s, milliseconds 2074.95
        Testing size: 25769803776 bytes
          Device Pair 0 -> 1: Average Bandwidth = 12.4187 GB/s, milliseconds 2075.07
        ```

    * `./main 0 1 --enable-peer-access`

        ```
        Peer access enabled from device 0 to device 1
        Peer access enabled from device 1 to device 0
        Testing size: 25769803776 bytes
          Device Pair 0 -> 1: Average Bandwidth = 11.3262 GB/s, milliseconds 2275.24
        Testing size: 25769803776 bytes
          Device Pair 0 -> 1: Average Bandwidth = 11.3263 GB/s, milliseconds 2275.22
        Testing size: 25769803776 bytes
          Device Pair 0 -> 1: Average Bandwidth = 11.3339 GB/s, milliseconds 2273.68
        Testing size: 25769803776 bytes
          Device Pair 0 -> 1: Average Bandwidth = 11.3262 GB/s, milliseconds 2275.24
        Testing size: 25769803776 bytes
          Device Pair 0 -> 1: Average Bandwidth = 11.3263 GB/s, milliseconds 2275.22
        Testing size: 25769803776 bytes
          Device Pair 0 -> 1: Average Bandwidth = 11.3262 GB/s, milliseconds 2275.23
        Testing size: 25769803776 bytes
          Device Pair 0 -> 1: Average Bandwidth = 11.3262 GB/s, milliseconds 2275.23
        Testing size: 25769803776 bytes
          Device Pair 0 -> 1: Average Bandwidth = 11.3262 GB/s, milliseconds 2275.24
        Testing size: 25769803776 bytes
          Device Pair 0 -> 1: Average Bandwidth = 11.3262 GB/s, milliseconds 2275.23
        Testing size: 25769803776 bytes
          Device Pair 0 -> 1: Average Bandwidth = 11.3262 GB/s, milliseconds 2275.23
        ```

    可以观察到下面几点：

    * 在传输数据时，无论是否开始 peer access，cpu 都会单核跑满

        看来 cpu 单核跑满与 nccl 无关，是 async copy 的结果。目前不清楚机制。

    * 假如机器止是 pcie 3.0 理论峰值速度是 16 GB/s，如果是 pcie 4.0 理论峰值速度是 32 GB/s，但是无论是否开始 peer access 实测的速度最大只有 12 GB/s 左右

        猜想：

        1. 不开启 peer access 时，host memory 只分配几个小 size buffer 作为中转，这个小 size buffer 可能是 pageable memory，此时影响速度的因素可能有：
        
            1. pageable memory 导致一直在查表；
            
            2. device 往 host memory 上写速度会拉低一些

        2. 开启 peer access 时，有可能 device memory 也只分配了一点点 buffer 映射到 bar 空间上，并不是把所有的 device memory 映射出来。

* cuda stream 在创建时，靠`cudaSetDevice()`来指定具体的 device

    这个操作看起来比较像 opengl，提前指定好上下文环境。

* 可以直接用 apt 安装 nvidia-cuda-tookit，这样可以安装上`nvcc`等开发环境

    cuda 的版本会落后一些，但是提供了提示和编译环境，可以用来跳转和写代码。

* nvcc 需要安装 g++ 编译器