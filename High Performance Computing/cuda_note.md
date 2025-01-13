# CUDA Note

## cache

* 当 cuda kernel 中 printf 过多时，会出现乱码

    下面是截取的一段逐渐乱码的过程：

    ```
    after Unroll, Unroll: 4, BytePerPack: 16
    after Unroll, Unroll: 4, BytePerPack: 16
    after Unroll, Unroll: 4, BytePerPack: 16
    after Unroll, Unroll: 4, BytePerPack: 16
    after UnMoll, Unroll: 4, BytePerPack: 16
    after UnMill, Unroll: 4, BytePerPack: 16
    after UnMinS, Unroll: 4, BytePerPack: 16
    after UnMinSrcUnroll: 4, BytePerPack: 16
    after UnMinSrcs:roll: 4, BytePerPack: 16
    after UnMinSrcs: %: 4, BytePerPack: 16
    after UnMinSrcs: 4, MinDstytePerPack: 16
    after UnMinSrcs: 4, MinDsts: PerPack: 16
    after UnMinSrcs: 4, MinDsts: 16rPack: 1702127201
    after UnMinSrcs: 4, MinDsts: 16, Byk: 1702127201
    after UnMinSrcs: 4, MinDsts: 16, ByteP1702127201
    after UnMinSrcs: 4, MinDsts: 16, BytePer
    after UnMinSrcs: 4, MinDsts: 16, BytePerPacafter UnMinSrcs: 4, MinDsts: 16, BytePerPack:after UnMinSrcs: 4, MinDsts: 16, BytePerPack: after UnMinSrcs: 4, MinDsts: 16, BytePerPack: %after UnMinSrcs: 4, MinDsts: 16, BytePerPack: 1702127201
    after UnMinSrcs: 1, MinDsts: 16, BytePerPack: 1702127201
    ^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^PMinSrcs: 1, MinDsts: 1, BytePerPack: 16
    MinSrcs: 1, MinDsts: 1, BytePerPack: 16
    MinSrcs: 1, MinDsts: 1, BytePerPack: 16
    MinSrcs: 1, MinDsts: 1, BytePerPack: 16
    ```

* cuda 线程不同步的一个现象

    `main.cu`:

    ```cpp
    #include <cuda_runtime.h>
    #include <stdio.h>
    #include <stdlib.h>

    #define N 64

    __global__ void test_kern(float *vec)
    {
        int x = threadIdx.x;

        float val = 0;
        for (int i = 0; i <= x; ++i)
        {
            val += vec[i];
        }
        vec[x] = val;
        // __syncthreads();

        printf("thread %d, val %.1f\n", x, vec[N - 1]);
    }

    int main()
    {
        // int N = 32;
        float *buf = (float*) malloc(N * sizeof(float));
        for (int i = 0; i < N; ++i)
            buf[i] = i;
        float *cubuf;
        cudaMalloc(&cubuf, sizeof(float) * N);
        cudaMemcpy(cubuf, buf, sizeof(float) * N, cudaMemcpyHostToDevice);

        test_kern<<<1, N>>>(cubuf);
        cudaDeviceSynchronize();

        free(buf);
        return 0;
    }
    ```

    compile: `nvcc -g -G main.cu -o main`

    run: `./main`

    output:

    ```
    thread 0, val 63.0
    thread 1, val 63.0
    thread 2, val 63.0
    thread 3, val 63.0
    thread 4, val 63.0
    thread 5, val 63.0
    thread 6, val 63.0
    thread 7, val 63.0
    thread 8, val 63.0
    thread 9, val 63.0
    thread 10, val 63.0
    thread 11, val 63.0
    thread 12, val 63.0
    thread 13, val 63.0
    thread 14, val 63.0
    thread 15, val 63.0
    thread 16, val 63.0
    thread 17, val 63.0
    thread 18, val 63.0
    thread 19, val 63.0
    thread 20, val 63.0
    thread 21, val 63.0
    thread 22, val 63.0
    thread 23, val 63.0
    thread 24, val 63.0
    thread 25, val 63.0
    thread 26, val 63.0
    thread 27, val 63.0
    thread 28, val 63.0
    thread 29, val 63.0
    thread 30, val 63.0
    thread 31, val 63.0
    thread 32, val 2016.0
    thread 33, val 2016.0
    thread 34, val 2016.0
    thread 35, val 2016.0
    thread 36, val 2016.0
    thread 37, val 2016.0
    thread 38, val 2016.0
    thread 39, val 2016.0
    thread 40, val 2016.0
    thread 41, val 2016.0
    thread 42, val 2016.0
    thread 43, val 2016.0
    thread 44, val 2016.0
    thread 45, val 2016.0
    thread 46, val 2016.0
    thread 47, val 2016.0
    thread 48, val 2016.0
    thread 49, val 2016.0
    thread 50, val 2016.0
    thread 51, val 2016.0
    thread 52, val 2016.0
    thread 53, val 2016.0
    thread 54, val 2016.0
    thread 55, val 2016.0
    thread 56, val 2016.0
    thread 57, val 2016.0
    thread 58, val 2016.0
    thread 59, val 2016.0
    thread 60, val 2016.0
    thread 61, val 2016.0
    thread 62, val 2016.0
    thread 63, val 2016.0
    ```

    将` // __syncthreads();`取消注释后，重新编译运行，输出为

    ```
    thread 0, val 2016.0
    thread 1, val 2016.0
    thread 2, val 2016.0
    thread 3, val 2016.0
    thread 4, val 2016.0
    thread 5, val 2016.0
    thread 6, val 2016.0
    thread 7, val 2016.0
    thread 8, val 2016.0
    thread 9, val 2016.0
    thread 10, val 2016.0
    thread 11, val 2016.0
    thread 12, val 2016.0
    thread 13, val 2016.0
    thread 14, val 2016.0
    thread 15, val 2016.0
    thread 16, val 2016.0
    thread 17, val 2016.0
    thread 18, val 2016.0
    thread 19, val 2016.0
    thread 20, val 2016.0
    thread 21, val 2016.0
    thread 22, val 2016.0
    thread 23, val 2016.0
    thread 24, val 2016.0
    thread 25, val 2016.0
    thread 26, val 2016.0
    thread 27, val 2016.0
    thread 28, val 2016.0
    thread 29, val 2016.0
    thread 30, val 2016.0
    thread 31, val 2016.0
    thread 32, val 2016.0
    thread 33, val 2016.0
    thread 34, val 2016.0
    thread 35, val 2016.0
    thread 36, val 2016.0
    thread 37, val 2016.0
    thread 38, val 2016.0
    thread 39, val 2016.0
    thread 40, val 2016.0
    thread 41, val 2016.0
    thread 42, val 2016.0
    thread 43, val 2016.0
    thread 44, val 2016.0
    thread 45, val 2016.0
    thread 46, val 2016.0
    thread 47, val 2016.0
    thread 48, val 2016.0
    thread 49, val 2016.0
    thread 50, val 2016.0
    thread 51, val 2016.0
    thread 52, val 2016.0
    thread 53, val 2016.0
    thread 54, val 2016.0
    thread 55, val 2016.0
    thread 56, val 2016.0
    thread 57, val 2016.0
    thread 58, val 2016.0
    thread 59, val 2016.0
    thread 60, val 2016.0
    thread 61, val 2016.0
    thread 62, val 2016.0
    thread 63, val 2016.0
    ```

    我们根据不同 thread 编号的大小，让 thread 执行长度不同的任务，然后让所有 thread 取一个只有当所有 thread 都算完才能得到的结果。

    可以看到，0 到 31 号 thread 取到了相同的值，但是是错的；32 到 63 号 thread 取得的值相同，并且是正确的。加上`__syncthreads();`后，所有的值都是正确的。

* `cudaLaunchKernel()` example 2

    ```cpp
    #include <cuda_runtime.h>
    #include <stdlib.h>
    #include <stdio.h>

    void assign_cubuf_rand_int(float *cubuf, size_t num_elm)
    {
        float *buf = (float*) malloc(num_elm * sizeof(float));
        for (size_t i = 0; i < num_elm; ++i)
        {
            buf[i] = rand() % 5;
        }
        cudaMemcpy(cubuf, buf, num_elm * sizeof(float), cudaMemcpyHostToDevice);
        free(buf);
    }

    void print_cubuf(float *cubuf, size_t num_elm)
    {
        float *buf = (float*) malloc(num_elm * sizeof(float));
        cudaMemcpy(buf, cubuf, num_elm * sizeof(float), cudaMemcpyDeviceToHost);
        for (size_t i = 0; i < num_elm; ++i)
        {
            printf("%.1f, ", buf[i]);
        }
        putchar('\n');
        free(buf);
    }

    __global__ void vec_add(float *A, float *B, float *C)
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        C[x] = A[x] + B[x];
    }

    int main()
    {
        float *cubuf_A, *cubuf_B, *cubuf_C;
        cudaMalloc(&cubuf_A, 8 * sizeof(float));
        cudaMalloc(&cubuf_B, 8 * sizeof(float));
        cudaMalloc(&cubuf_C, 8 * sizeof(float));

        assign_cubuf_rand_int(cubuf_A, 8);
        assign_cubuf_rand_int(cubuf_B, 8);

        puts("cubuf_A:");
        print_cubuf(cubuf_A, 8);
        puts("cubuf_B:");
        print_cubuf(cubuf_B, 8);

        // void *args[3] = {&cubuf_A, &cubuf_B, &cubuf_C};
        void **args = (void**) malloc(3 * sizeof(void*));
        args[0] = &cubuf_A;
        args[1] = &cubuf_B;
        args[2] = &cubuf_C;
        cudaLaunchKernel((const void *) vec_add, dim3(2, 1, 1), dim3(4, 1, 1), args, 0, NULL);
        // cudaStreamSynchronize(stream);
        cudaDeviceSynchronize();

        puts("cubuf_C:");
        print_cubuf(cubuf_C, 8);

        cudaFree(cubuf_A);
        cudaFree(cubuf_B);
        cudaFree(cubuf_C);

        return 0;
    }
    ```

    output:

    ```
    cubuf_A:
    3.0, 1.0, 2.0, 0.0, 3.0, 0.0, 1.0, 2.0, 
    cubuf_B:
    4.0, 1.0, 2.0, 2.0, 0.0, 4.0, 3.0, 1.0, 
    cubuf_C:
    7.0, 2.0, 4.0, 2.0, 3.0, 4.0, 4.0, 3.0,
    ```

* `cudaLaunchKernel()`的 example

    `main.cu`:

    ```cpp
    #include <cuda_runtime.h>
    #include <stdlib.h>
    #include <stdio.h>

    void assign_cubuf_rand_int(float *cubuf, size_t num_elm)
    {
        float *buf = (float*) malloc(num_elm * sizeof(float));
        for (size_t i = 0; i < num_elm; ++i)
        {
            buf[i] = rand() % 5;
        }
        cudaMemcpy(cubuf, buf, num_elm * sizeof(float), cudaMemcpyHostToDevice);
        free(buf);
    }

    void print_cubuf(float *cubuf, size_t num_elm)
    {
        float *buf = (float*) malloc(num_elm * sizeof(float));
        cudaMemcpy(buf, cubuf, num_elm * sizeof(float), cudaMemcpyDeviceToHost);
        for (size_t i = 0; i < num_elm; ++i)
        {
            printf("%.1f, ", buf[i]);
        }
        putchar('\n');
        free(buf);
    }

    __global__ void vec_add(float *A, float *B, float *C)
    {
        int x = threadIdx.x;
        C[x] = A[x] + B[x];
    }

    int main()
    {
        float *cubuf_A, *cubuf_B, *cubuf_C;
        cudaMalloc(&cubuf_A, 8 * sizeof(float));
        cudaMalloc(&cubuf_B, 8 * sizeof(float));
        cudaMalloc(&cubuf_C, 8 * sizeof(float));

        assign_cubuf_rand_int(cubuf_A, 8);
        assign_cubuf_rand_int(cubuf_B, 8);

        puts("cubuf_A:");
        print_cubuf(cubuf_A, 8);
        puts("cubuf_B:");
        print_cubuf(cubuf_B, 8);

        cudaStream_t stream;
        cudaStreamCreate(&stream);

        void *args[3] = {&cubuf_A, &cubuf_B, &cubuf_C};
        // void **args = (void**) malloc(3 * sizeof(void*));
        // args[0] = &cubuf_A;
        // args[1] = &cubuf_B;
        // args[2] = &cubuf_C;
        cudaLaunchKernel((const void *) vec_add, 1, 8, args, 0, stream);
        cudaStreamSynchronize(stream);

        puts("cubuf_C:");
        print_cubuf(cubuf_C, 8);

        cudaFree(cubuf_A);
        cudaFree(cubuf_B);
        cudaFree(cubuf_C);
        cudaStreamDestroy(stream);

        return 0;
    }
    ```

    compile:

    `nvcc -g -G main.cu -o main`

    run:

    `./main`

    output:

    ```
    cubuf_A:
    3.0, 1.0, 2.0, 0.0, 3.0, 0.0, 1.0, 2.0, 
    cubuf_B:
    4.0, 1.0, 2.0, 2.0, 0.0, 4.0, 3.0, 1.0, 
    cubuf_C:
    7.0, 2.0, 4.0, 2.0, 3.0, 4.0, 4.0, 3.0, 
    ```

    `cudaLaunchKernel()`，第一个参数是 kernel 函数地址，需要用`(const void *)`或`(void *)`类型转换一下，第二个参数是 grid dim，可以指定`dim3`，如果是一维的，直接指定 scalar 就可以了，第三个参数是 block dim，同参数二。

    第四个参数是 kernel 函数的参数列表，虽然指定的类型是`void**`，即`void*`的数组，但是实际传递的并不是 buffer 的 va，而是 buffer va 的 va。可以看到`void *args[3] = {&cubuf_A, &cubuf_B, &cubuf_C};`里，对`cubuf_A`等`float*`类型的变量又多加了一层取地址。不加这个会报 segment fault。

    第 5 个参数直接填 0 就可以，目前用不到。

    第 6 个参数可以填 stream，也可以填`NULL`，只不过这时要用`cudaDeviceSynchronize();`来阻塞等待 kernel launch。

* cuda shared memory 优化矩阵乘法的一个例子

    `main_4.cu`:

    ```cpp
    #include <stdlib.h>
    #include <stdio.h>
    #include <cuda_runtime.h>

    #define BLOCK_SIZE 4

    typedef struct {
        int width;
        int height;
        int stride;
        float* elements;
    } Matrix;

    __global__ void MatMulKernel(const Matrix, const Matrix, Matrix);

    void assign_mat_rand_int(float *m, int n_rows, int n_cols)
    {
        for (int i = 0; i < n_rows; ++i)
        {
            for (int j = 0; j < n_cols; ++j)
            {
                m[i * n_cols + j] = rand() % 5;
            }
        }
    }

    void display_mat(float *mat, int n_rows, int n_cols)
    {
        for (int i = 0; i < n_rows; ++i)
        {
            for (int j = 0; j < n_cols; ++j)
            {
                printf("%.1f, ", mat[i * n_cols + j]);
            }
            putchar('\n');
        }
    }

    int main()
    {
        Matrix A, B, C;
        A.width = 8;
        A.height = 8;
        A.elements = (float*) malloc(A.width * A.height * sizeof(float));
        B.width = 8;
        B.height = 8;
        B.elements = (float*) malloc(B.width * B.height * sizeof(float));
        C.width = 8;
        C.height = 8;
        C.elements = (float*) malloc(C.width * C.height * sizeof(float));

        assign_mat_rand_int(A.elements, A.height, A.width);
        assign_mat_rand_int(B.elements, B.height, B.width);

        puts("A:");
        display_mat(A.elements, A.height, A.width);
        puts("B:");
        display_mat(B.elements, B.height, B.width);

        Matrix d_A;
        d_A.width = d_A.stride = A.width;
        d_A.height = A.height;
        size_t size = A.width * A.height * sizeof(float);
        cudaMalloc(&d_A.elements, size);
        cudaMemcpy(d_A.elements, A.elements, size, cudaMemcpyHostToDevice);

        Matrix d_B;
        d_B.width = d_B.stride = B.width; d_B.height = B.height;
        size = B.width * B.height * sizeof(float);
        cudaMalloc(&d_B.elements, size);
        cudaMemcpy(d_B.elements, B.elements, size, cudaMemcpyHostToDevice);

        Matrix d_C;
        d_C.width = d_C.stride = C.width;
        d_C.height = C.height;
        size = C.width * C.height * sizeof(float);
        cudaMalloc(&d_C.elements, size);

        dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
        dim3 dimGrid(B.width / dimBlock.x, A.height / dimBlock.y);
        MatMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);

        cudaMemcpy(C.elements, d_C.elements, size, cudaMemcpyDeviceToHost);

        puts("C:");
        display_mat(C.elements, C.height, C.width);

        cudaFree(d_A.elements);
        cudaFree(d_B.elements);
        cudaFree(d_C.elements);
        free(A.elements);
        free(B.elements);
        free(C.elements);

        return 0;
    }

    __device__ float GetElement(const Matrix A, int row, int col)
    {
        return A.elements[row * A.stride + col];
    }

    __device__ void SetElement(Matrix A, int row, int col, float value)
    {
        A.elements[row * A.stride + col] = value;
    }

     __device__ Matrix GetSubMatrix(Matrix A, int row, int col)
    {
        Matrix Asub;
        Asub.width    = BLOCK_SIZE;
        Asub.height   = BLOCK_SIZE;
        Asub.stride   = A.stride;
        Asub.elements = &A.elements[A.stride * BLOCK_SIZE * row + BLOCK_SIZE * col];
        return Asub;
    }

     __global__ void MatMulKernel(Matrix A, Matrix B, Matrix C)
    {
        int blockRow = blockIdx.y;
        int blockCol = blockIdx.x;
        Matrix Csub = GetSubMatrix(C, blockRow, blockCol);
        float Cvalue = 0;
        int row = threadIdx.y;
        int col = threadIdx.x;
        for (int m = 0; m < (A.width / BLOCK_SIZE); ++m)
        {
            Matrix Asub = GetSubMatrix(A, blockRow, m);
            Matrix Bsub = GetSubMatrix(B, m, blockCol);
            __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
            __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];
            As[row][col] = GetElement(Asub, row, col);
            Bs[row][col] = GetElement(Bsub, row, col);
            __syncthreads();
            for (int e = 0; e < BLOCK_SIZE; ++e)
                Cvalue += As[row][e] * Bs[e][col];
            __syncthreads();
        }
        SetElement(Csub, row, col, Cvalue);
    }
    ```

    compile: `nvcc -g -G main_4.cu -o main_4`

    run: `./main_4`

    output:

    ```
    A:
    3.0, 1.0, 2.0, 0.0, 3.0, 0.0, 1.0, 2.0, 
    4.0, 1.0, 2.0, 2.0, 0.0, 4.0, 3.0, 1.0, 
    0.0, 1.0, 2.0, 1.0, 1.0, 3.0, 2.0, 4.0, 
    2.0, 0.0, 2.0, 3.0, 2.0, 0.0, 4.0, 2.0, 
    2.0, 3.0, 4.0, 2.0, 3.0, 1.0, 1.0, 2.0, 
    4.0, 3.0, 1.0, 4.0, 4.0, 2.0, 3.0, 4.0, 
    0.0, 0.0, 3.0, 1.0, 1.0, 0.0, 1.0, 3.0, 
    2.0, 0.0, 1.0, 1.0, 0.0, 0.0, 4.0, 2.0, 
    B:
    1.0, 0.0, 1.0, 4.0, 3.0, 2.0, 4.0, 0.0, 
    2.0, 0.0, 4.0, 2.0, 4.0, 4.0, 3.0, 0.0, 
    2.0, 3.0, 1.0, 3.0, 3.0, 4.0, 3.0, 1.0, 
    4.0, 4.0, 2.0, 0.0, 1.0, 3.0, 4.0, 2.0, 
    1.0, 1.0, 4.0, 4.0, 0.0, 0.0, 4.0, 3.0, 
    2.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.0, 
    1.0, 0.0, 1.0, 1.0, 0.0, 4.0, 4.0, 4.0, 
    0.0, 4.0, 1.0, 2.0, 2.0, 1.0, 1.0, 0.0, 
    C:
    13.0, 17.0, 24.0, 37.0, 23.0, 24.0, 39.0, 15.0, 
    29.0, 22.0, 26.0, 37.0, 34.0, 47.0, 54.0, 22.0, 
    19.0, 30.0, 24.0, 28.0, 25.0, 33.0, 35.0, 18.0, 
    24.0, 28.0, 24.0, 30.0, 19.0, 39.0, 52.0, 30.0, 
    30.0, 32.0, 39.0, 45.0, 38.0, 46.0, 57.0, 22.0, 
    39.0, 41.0, 52.0, 56.0, 43.0, 56.0, 80.0, 35.0, 
    12.0, 26.0, 13.0, 20.0, 16.0, 22.0, 24.0, 12.0, 
    12.0, 15.0, 11.0, 19.0, 14.0, 29.0, 33.0, 19.0,
    ```

    这个例子先按行切第一个矩阵，按列切第二个矩阵，切完后，再将第一个子矩阵竖着切成一小块一小块，每一个小块都是边长为`BLOCK_SIZE`的小正方形。用`m`来标记当前进行到了第几个小正方形。

    `__shared__`关键字来申请 shared memory，用来存放小正方形。`GetElement()`用于往小正方形中填充数据，每个线程填充一个元素。

    `e`用于标记小正方形单行/单列的各个元素序号。因为直接使用的二维坐标来标记线程，所以相当于脱掉了矩阵乘法三层循环的外面两层。

    `__syncthreads();`用于做线程同步，每次等矩阵数据加载完成，或矩阵乘法计算完成后，翥需要同步一下。

* cuda 实现矩阵乘法的一个例子

    由 cuda programming guide 里的一个 example 改编而来。

    `main_3.cu`:

    ```cpp
    #include <stdio.h>
    #include <stdlib.h>
    #include <cuda_runtime.h>

    // Matrices are stored in row-major order:
    // M(row, col) = *(M.elements + row * M.width + col)
    typedef struct {
        int width;
        int height;
        float* elements;
    } Matrix;

    #define BLOCK_SIZE 4  // threads per block

    __global__ void MatMulKernel(const Matrix mat_1, const Matrix mat_2, Matrix mat_out);

    void assign_mat_rand_int(float *m, int n_rows, int n_cols)
    {
        for (int i = 0; i < n_rows; ++i)
        {
            for (int j = 0; j < n_cols; ++j)
            {
                m[i * n_cols + j] = rand() % 5;
            }
        }
    }

    void display_mat(float *mat, int n_rows, int n_cols)
    {
        for (int i = 0; i < n_rows; ++i)
        {
            for (int j = 0; j < n_cols; ++j)
            {
                printf("%.1f, ", mat[i * n_cols + j]);
            }
            putchar('\n');
        }
    }

    int main()
    {
        Matrix A, B, C;
        A.width = 8;
        A.height = 8;
        A.elements = (float*) malloc(A.width * A.height * sizeof(float));
        B.width = 8;
        B.height = 8;
        B.elements = (float*) malloc(B.width * B.height * sizeof(float));
        C.width = 8;
        C.height = 8;
        C.elements = (float*) malloc(C.width * C.height * sizeof(float));

        assign_mat_rand_int(A.elements, A.height, A.width);
        assign_mat_rand_int(B.elements, B.height, B.width);

        puts("A:");
        display_mat(A.elements, A.height, A.width);
        puts("B:");
        display_mat(B.elements, B.height, B.width);

        Matrix d_A;
        d_A.width = A.width; d_A.height = A.height;
        size_t size = A.width * A.height * sizeof(float);
        cudaMalloc(&d_A.elements, size);
        cudaMemcpy(d_A.elements, A.elements, size, cudaMemcpyHostToDevice);

        Matrix d_B;
        d_B.width = B.width; d_B.height = B.height;
        size = B.width * B.height * sizeof(float);
        cudaMalloc(&d_B.elements, size);
        cudaMemcpy(d_B.elements, B.elements, size, cudaMemcpyHostToDevice);

        Matrix d_C;
        d_C.width = C.width; d_C.height = C.height;
        size = C.width * C.height * sizeof(float);
        cudaMalloc(&d_C.elements, size);

        dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
        dim3 dimGrid(B.width / dimBlock.x, A.height / dimBlock.y);
        MatMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);

        cudaMemcpy(C.elements, d_C.elements, size,cudaMemcpyDeviceToHost);

        puts("C:");
        display_mat(C.elements, C.height, C.width);

        cudaFree(d_A.elements);
        cudaFree(d_B.elements);
        cudaFree(d_C.elements);
        free(A.elements);
        free(B.elements);
        free(C.elements);

        return 0;
    }

    __global__ void MatMulKernel(Matrix A, Matrix B, Matrix C)
    {
        float Cvalue = 0;
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        for (int e = 0; e < A.width; ++e)
            Cvalue += A.elements[row * A.width + e] * B.elements[e * B.width + col];
        C.elements[row * C.width + col] = Cvalue;
    }
    ```

    compile: `nvcc -g -G main_3.cu -o main_3`

    run: `./main_3`

    output:

    ```
    A:
    3.0, 1.0, 2.0, 0.0, 3.0, 0.0, 1.0, 2.0, 
    4.0, 1.0, 2.0, 2.0, 0.0, 4.0, 3.0, 1.0, 
    0.0, 1.0, 2.0, 1.0, 1.0, 3.0, 2.0, 4.0, 
    2.0, 0.0, 2.0, 3.0, 2.0, 0.0, 4.0, 2.0, 
    2.0, 3.0, 4.0, 2.0, 3.0, 1.0, 1.0, 2.0, 
    4.0, 3.0, 1.0, 4.0, 4.0, 2.0, 3.0, 4.0, 
    0.0, 0.0, 3.0, 1.0, 1.0, 0.0, 1.0, 3.0, 
    2.0, 0.0, 1.0, 1.0, 0.0, 0.0, 4.0, 2.0, 
    B:
    1.0, 0.0, 1.0, 4.0, 3.0, 2.0, 4.0, 0.0, 
    2.0, 0.0, 4.0, 2.0, 4.0, 4.0, 3.0, 0.0, 
    2.0, 3.0, 1.0, 3.0, 3.0, 4.0, 3.0, 1.0, 
    4.0, 4.0, 2.0, 0.0, 1.0, 3.0, 4.0, 2.0, 
    1.0, 1.0, 4.0, 4.0, 0.0, 0.0, 4.0, 3.0, 
    2.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.0, 
    1.0, 0.0, 1.0, 1.0, 0.0, 4.0, 4.0, 4.0, 
    0.0, 4.0, 1.0, 2.0, 2.0, 1.0, 1.0, 0.0, 
    C:
    13.0, 17.0, 24.0, 37.0, 23.0, 24.0, 39.0, 15.0, 
    29.0, 22.0, 26.0, 37.0, 34.0, 47.0, 54.0, 22.0, 
    19.0, 30.0, 24.0, 28.0, 25.0, 33.0, 35.0, 18.0, 
    24.0, 28.0, 24.0, 30.0, 19.0, 39.0, 52.0, 30.0, 
    30.0, 32.0, 39.0, 45.0, 38.0, 46.0, 57.0, 22.0, 
    39.0, 41.0, 52.0, 56.0, 43.0, 56.0, 80.0, 35.0, 
    12.0, 26.0, 13.0, 20.0, 16.0, 22.0, 24.0, 12.0, 
    12.0, 15.0, 11.0, 19.0, 14.0, 29.0, 33.0, 19.0,
    ```

    这个 example 是对 A 矩阵横着切，B 矩阵竖着切，每个 kernel 只计算一行/一列。通过 grid 保证所有的行/列都会被覆盖到。

* cuda 中同一 block 中的 thread 可以有共享显存（shared memory），这个共享显存通常是 L1 cache

* 3d grid 的使用

    `main_3.cu`:

    ```cpp
    #include <cuda_runtime.h>
    #include <stdio.h>
    #include <stdlib.h>

    #define N_0 16
    #define N_1 4
    #define N_2 8

    __global__ void volume_add(float A[N_0][N_1][N_2], float B[N_0][N_1][N_2])
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        int z = blockIdx.z * blockDim.z + threadIdx.z;

        A[x][y][z] = A[x][y][z] + B[x][y][z];
    }

    int main()
    {
        float *cubuf_A, *cubuf_B;
        cudaMalloc(&cubuf_A, N_0 * N_1 * N_2 * sizeof(float));
        cudaMalloc(&cubuf_B, N_0 * N_1 * N_2 * sizeof(float));

        float buf_A[N_0][N_1][N_2];
        float buf_B[N_0][N_1][N_2];
        for (int i = 0; i < N_0; ++i)
        {
            for (int j = 0; j < N_1; ++j)
            {
                for (int k = 0; k < N_2; ++k)
                {
                    buf_A[i][j][k] = rand() % 5;
                    buf_B[i][j][k] = rand() % 5;
                }
            }
        }

        size_t num_elm = N_0 * N_1 * N_2;

        printf("buf A:\n");
        for (int i = 0; i < num_elm; ++i)
            printf("%.1f, ", buf_A[i / (N_1 * N_2)][i / N_2 % N_1][i % N_2]);
        putchar('\n');

        printf("buf B:\n");
        for (int i = 0; i < num_elm; ++i)
            printf("%.1f, ", buf_B[i / (N_1 * N_2)][i / N_2 % N_1][i % N_2]);
        putchar('\n');

        cudaMemcpy(cubuf_A, buf_A, num_elm * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(cubuf_B, buf_B, num_elm * sizeof(float), cudaMemcpyHostToDevice);
        
        dim3 thread_per_block(2, 2, 2);
        dim3 block_per_grid(8, 2, 4);
        volume_add<<<block_per_grid, thread_per_block>>>((float(*)[N_1][N_2]) cubuf_A, (float(*)[N_1][N_2]) cubuf_B);
        cudaMemcpy(buf_A, cubuf_A, num_elm * sizeof(float), cudaMemcpyDeviceToHost);
        
        printf("buf A:\n");
        for (int i = 0; i < num_elm; ++i)
            printf("%.1f, ", buf_A[i / (N_1 * N_2)][i / N_2 % N_1][i % N_2]);
        putchar('\n');

        return 0;
    }
    ```

    compile: `nvcc -g -G main_3.cu -o main_3`

    run: `./main_3`

    output:

    ```
    buf A:
    3.0, 2.0, 3.0, 1.0, 4.0, ...
    buf B:
    1.0, 0.0, 0.0, 2.0, 1.0, ...
    buf A:
    4.0, 2.0, 3.0, 3.0, 5.0, ...
    ```

    可以看到，我们的线程组总是`(2, 2, 2)`，但是每个 grid 中有`(8, 2, 4)`个 block，一共 1 个 grid。

    同一个 block 中的多个 threads 是保证并行执行的，但是 block 与 block 之间并不保证并行。

    其实 grid 中的 block 有点像 batch 的感觉。

* cuda 实现 vec add

    见`ref_32`

* block 也可以被组织为一／二／三维的 grid。这么做主要为了适配需要计算的数据。通常数据的 dim length 是会超过 gpu 中流处理器的数量的

    example:

    `main_3.cu`:

    ```cpp
    __global__ void vec_add_blk(float *A, float *B)
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        A[x] = A[x] + B[x];
        printf("%d, %d, %d\n", blockDim.x, blockDim.y, blockDim.z);
    }

    int main()
    {
        float *cubuf_A, *cubuf_B;
        cudaMalloc(&cubuf_A, 8 * sizeof(float));
        cudaMalloc(&cubuf_B, 8 * sizeof(float));
        float *buf_A, *buf_B;
        buf_A = (float*) malloc(8 * sizeof(float));
        buf_B = (float*) malloc(8 * sizeof(float));
        for (int i = 0; i < 8; ++i)
        {
            buf_A[i] = rand() % 5;
            buf_B[i] = rand() % 5;
        }

        printf("buf A:\n");
        for (int i = 0; i < 8; ++i)
            printf("%.2f, ", buf_A[i]);
        putchar('\n');

        printf("buf B:\n");
        for (int i = 0; i < 8; ++i)
            printf("%.2f, ", buf_B[i]);
        putchar('\n');

        cudaMemcpy(cubuf_A, buf_A, 8 * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(cubuf_B, buf_B, 8 * sizeof(float), cudaMemcpyHostToDevice);
        // vec_add<<<1, 8>>>(cubuf_A, cubuf_B);
        vec_add_blk<<<4, 2>>>(cubuf_A, cubuf_B);
        cudaMemcpy(buf_A, cubuf_A, 8 * sizeof(float), cudaMemcpyDeviceToHost);
        
        printf("buf A:\n");
        for (int i = 0; i < 8; ++i)
            printf("%.2f, ", buf_A[i]);
        putchar('\n');

        return 0;
    }
    ```

    output:

    ```
    buf A:
    3.00, 2.00, 3.00, 1.00, 4.00, 2.00, 0.00, 3.00, 
    buf B:
    1.00, 0.00, 0.00, 2.00, 1.00, 2.00, 4.00, 1.00, 
    2, 1, 1
    2, 1, 1
    2, 1, 1
    2, 1, 1
    2, 1, 1
    2, 1, 1
    2, 1, 1
    2, 1, 1
    buf A:
    4.00, 2.00, 3.00, 3.00, 5.00, 4.00, 4.00, 4.00,
    ```

    可以看到`blockDim`其实就是每个维度的 length。

    grid 有点像 batch 的概念，比如一个 shape 为`(20, 8, 4)`的数据，我既可以一次性处理，也可以拆分成 4 个`(5, 2, 1)`进行处理。前者的 grid 即为 1，后者的 grid 即为`(4, 4, 4)`。

    如果拆成 grid 进行处理，那么在确定数组索引时，就需要用`int x = blockIdx.x * blockDim.x + threadIdx.x;`这种方式。

    grid 与 grid 之间并不保证是并行执行的，可能是并行，也可能是串行。

    > Thread blocks are required to execute independently: It must be possible to execute them in any order, in parallel or in series.

* 一个 block (thread block) 中的 threads 数量是有限的，因为每个 block 会被绑定到一个 sm (streaming multiprocessor) 上，这个 block 中的所有 thread 都会在这个 sm 上执行。

    > On current GPUs, a thread block may contain up to 1024 threads.

    当前的 gpu，每个 block 最多有 1024 个 thread。

* `cudaPointerGetAttributes()`用法

    `main_6.cu`:

    ```cpp
    #include <cuda_runtime.h>
    #include <stdio.h>

    int main()
    {
        float *cubuf;
        cudaMalloc(&cubuf, 8 * sizeof(float));
        
        cudaPointerAttributes attr;
        cudaPointerGetAttributes(&attr, cubuf);
        printf("cubuf: %p\n", cubuf);
        printf("attr.device: %d\n", attr.device);
        printf("attr.devicePointer: %p\n", attr.devicePointer);
        printf("attr.hostPointer: %p\n", attr.hostPointer);
        printf("attr.type: %d\n", attr.type);
        
        cudaFree(cubuf);
        return 0;
    }
    ```

    compile: `nvcc -g main_6.cu -o main_6`

    run: `./main_6`

    output:

    ```
    cubuf: 0x7fd3ffa00000
    attr.device: 0
    attr.devicePointer: 0x7fd3ffa00000
    attr.hostPointer: (nil)
    attr.type: 2
    ```

    可以看到，`cudaPointerGetAttributes()`可以拿到 ptr 的 device, addr，以及 type 这三个信息。

    type 一共就 4 种：

    ```cpp
    /**
     * CUDA memory types
     */
    enum __device_builtin__ cudaMemoryType
    {
        cudaMemoryTypeUnregistered = 0, /**< Unregistered memory */
        cudaMemoryTypeHost         = 1, /**< Host memory */
        cudaMemoryTypeDevice       = 2, /**< Device memory */
        cudaMemoryTypeManaged      = 3  /**< Managed memory */
    };
    ```

    如果传递给它的是`malloc()`申请的内存，则会返回 nil：

    ```cpp
    #include <cuda_runtime.h>
    #include <stdio.h>

    int main()
    {
        float *buf = (float*) malloc(8 * sizeof(float));

        cudaPointerAttributes attr;
        cudaError_t ret = cudaPointerGetAttributes(&attr, buf);
        if (ret != cudaSuccess)
        {
            printf("fail to get pointer attr, ret: %d\n", ret);
            return -1;
        }
        printf("cubuf: %p\n", buf);
        printf("attr.device: %d\n", attr.device);
        printf("attr.devicePointer: %p\n", attr.devicePointer);
        printf("attr.hostPointer: %p\n", attr.hostPointer);
        printf("attr.type: %d\n", attr.type);
        
        free(buf);
        return 0;
    }
    ```

    output:

    ```
    cubuf: 0x561a01324520
    attr.device: -2
    attr.devicePointer: (nil)
    attr.hostPointer: (nil)
    attr.type: 0
    ```

    `cudaPointerGetAttributes()`还能对 range 进行判断：

    ```cpp
    #include <cuda_runtime.h>
    #include <stdio.h>

    int main()
    {
        float *cubuf;
        cudaMalloc(&cubuf, 8 * sizeof(float));
        
        cudaPointerAttributes attr;
        cudaPointerGetAttributes(&attr, cubuf + 7);
        printf("cubuf: %p\n", cubuf + 7);
        printf("attr.device: %d\n", attr.device);
        printf("attr.devicePointer: %p\n", attr.devicePointer);
        printf("attr.hostPointer: %p\n", attr.hostPointer);
        printf("attr.type: %d\n", attr.type);
        putchar('\n');

        cudaPointerGetAttributes(&attr, cubuf + 8);
        printf("cubuf: %p\n", cubuf + 8);
        printf("attr.device: %d\n", attr.device);
        printf("attr.devicePointer: %p\n", attr.devicePointer);
        printf("attr.hostPointer: %p\n", attr.hostPointer);
        printf("attr.type: %d\n", attr.type);
        
        cudaFree(cubuf);
        return 0;
    }
    ```

    output:

    ```
    cubuf: 0x7f764ba0001c
    attr.device: 0
    attr.devicePointer: 0x7f764ba0001c
    attr.hostPointer: (nil)
    attr.type: 2

    cubuf: 0x7f764ba00020
    attr.device: -2
    attr.devicePointer: (nil)
    attr.hostPointer: (nil)
    attr.type: 0
    ```

* 不需要 enable peer access，也可以使用`cudaMemcpy()`将数据从一个 device 复制到另一个 device

    `main_5.cu`:

    ```cpp
    #include <cuda_runtime.h>
    #include <stdio.h>

    int main()
    {
        cudaSetDevice(0);
        float *cubuf_0;
        cudaMalloc(&cubuf_0, 8 * sizeof(float));
        cudaSetDevice(1);
        float *cubuf_1;
        cudaMalloc(&cubuf_1, 8 * sizeof(float));

        // cudaSetDevice(0);
        float *buf = (float*) malloc(8 * sizeof(float));
        for (int i = 0; i < 8; ++i)
            buf[i] = 123;
        cudaMemcpy(cubuf_0, buf, 8 * sizeof(float), cudaMemcpyHostToDevice);
        cudaError_t ret;
        ret = cudaMemcpy(cubuf_1, cubuf_0, 8 * sizeof(float), cudaMemcpyDeviceToDevice);
        if (ret != cudaSuccess)
        {
            printf("fail to cuda memcpy cubuf_0 to cubuf_1\n");
            return -1;
        }

        // cudaSetDevice(1);
        cudaMemcpy(buf, cubuf_1, 8 * sizeof(float), cudaMemcpyDeviceToHost);
        for (int i = 0; i < 8; ++i)
            printf("%.1f, ", buf[i]);
        putchar('\n');

        cudaFree(cubuf_0);
        cudaFree(cubuf_1);
        free(buf);
        return 0;
    }
    ```

    compile: `nvcc -g main_5.cu -o main_5`

    run: `./main_5`

    output:

    ```
    123.0, 123.0, 123.0, 123.0, 123.0, 123.0, 123.0, 123.0,
    ```

    如果 enable peer access，则可以直接在 launch kernel 时，使用 peer device 的指针，省去了 cuda memcpy 将数据从当前 deivce 复制到 peer device 的步骤。

    说明：

    1. `cudaMemcpy()`不需要显式用`cudaSetDevice()`指定 device。看起来只要有能辨别 device 的信息（比如指针，device id），就不需要显式指定 device。

* cuda 不同进程间的 va 分配情况

    运行下面的程序 10 次，

    ```cpp
    #include <cuda_runtime.h>
    #include <stdio.h>

    int main()
    {
        cudaSetDevice(0);
        int *cubuf_0;
        cudaMalloc(&cubuf_0, 8 * sizeof(float));
        printf("cubuf 0: %p\n", cubuf_0);

        cudaSetDevice(1);
        int *cubuf_1;
        cudaMalloc(&cubuf_1, 8 * sizeof(float));
        printf("cubuf 1: %p\n", cubuf_1);

        cudaFree(cubuf_0);
        cudaFree(cubuf_1);
        return 0;
    }
    ```

    输出如下：

    ```
    (base) huliucheng@zjxj:~/Documents/Projects/cumem_test$ ./main_4
    cubuf 0: 0x7f0907a00000
    cubuf 1: 0x7f08f3a00000
    (base) huliucheng@zjxj:~/Documents/Projects/cumem_test$ ./main_4
    cubuf 0: 0x7f8a47a00000
    cubuf 1: 0x7f8a2fa00000
    (base) huliucheng@zjxj:~/Documents/Projects/cumem_test$ ./main_4
    cubuf 0: 0x7f68ada00000
    cubuf 1: 0x7f6893a00000
    (base) huliucheng@zjxj:~/Documents/Projects/cumem_test$ ./main_4
    cubuf 0: 0x7f441da00000
    cubuf 1: 0x7f4403a00000
    (base) huliucheng@zjxj:~/Documents/Projects/cumem_test$ ./main_4
    cubuf 0: 0x7f4d47a00000
    cubuf 1: 0x7f4d2fa00000
    (base) huliucheng@zjxj:~/Documents/Projects/cumem_test$ ./main_4
    cubuf 0: 0x7f8063a00000
    cubuf 1: 0x7f8049a00000
    (base) huliucheng@zjxj:~/Documents/Projects/cumem_test$ ./main_4
    cubuf 0: 0x7f8277a00000
    cubuf 1: 0x7f825fa00000
    (base) huliucheng@zjxj:~/Documents/Projects/cumem_test$ ./main_4
    cubuf 0: 0x7f1857a00000
    cubuf 1: 0x7f183fa00000
    (base) huliucheng@zjxj:~/Documents/Projects/cumem_test$ ./main_4
    cubuf 0: 0x7f8197a00000
    cubuf 1: 0x7f817fa00000
    (base) huliucheng@zjxj:~/Documents/Projects/cumem_test$ ./main_4
    cubuf 0: 0x7fefe5a00000
    cubuf 1: 0x7fefcba00000
    ```

    可以看到，所有的 va 都以`0x7f`开头，后续的两位 16 进制数是根据进程随机分配的，再往后两位不固定，再往后一位总是`a`。

* cuda 在同一个进程里分配的 va range 相同

    `main_4.cu`:

    ```cpp
    #include <cuda_runtime.h>
    #include <stdio.h>

    int main()
    {
        int num_round = 5;
        for (int i = 0; i < num_round; ++i)
        {
            printf("round: %d\n", i);

            cudaSetDevice(0);
            int *cubuf_0;
            cudaMalloc(&cubuf_0, 8 * sizeof(float));
            printf("cubuf 0: %p\n", cubuf_0);

            cudaSetDevice(1);
            int *cubuf_1;
            cudaMalloc(&cubuf_1, 8 * sizeof(float));
            printf("cubuf 1: %p\n", cubuf_1);

            cudaFree(cubuf_0);
            cudaFree(cubuf_1);

            putchar('\n');
        }

        return 0;
    }
    ```

    compile: `nvcc -g main_4.cu -o main_4`

    run: `./main_4`

    output:

    ```
    round: 0
    cubuf 0: 0x7ff44da00000
    cubuf 1: 0x7ff433a00000

    round: 1
    cubuf 0: 0x7ff433a00000
    cubuf 1: 0x7ff433c00000

    round: 2
    cubuf 0: 0x7ff433a00000
    cubuf 1: 0x7ff433c00000

    round: 3
    cubuf 0: 0x7ff433a00000
    cubuf 1: 0x7ff433c00000

    round: 4
    cubuf 0: 0x7ff433a00000
    cubuf 1: 0x7ff433c00000
    ```

    注释掉`cudaFree()`，代码变为

    ```cpp
    #include <cuda_runtime.h>
    #include <stdio.h>

    int main()
    {
        int num_round = 5;
        for (int i = 0; i < num_round; ++i)
        {
            printf("round: %d\n", i);

            cudaSetDevice(0);
            int *cubuf_0;
            cudaMalloc(&cubuf_0, 8 * sizeof(float));
            printf("cubuf 0: %p\n", cubuf_0);

            cudaSetDevice(1);
            int *cubuf_1;
            cudaMalloc(&cubuf_1, 8 * sizeof(float));
            printf("cubuf 1: %p\n", cubuf_1);

            // cudaFree(cubuf_0);
            // cudaFree(cubuf_1);

            putchar('\n');
        }

        return 0;
    }
    ```

    output:

    ```
    round: 0
    cubuf 0: 0x7fa897a00000
    cubuf 1: 0x7fa883a00000

    round: 1
    cubuf 0: 0x7fa897a00200
    cubuf 1: 0x7fa883a00200

    round: 2
    cubuf 0: 0x7fa897a00400
    cubuf 1: 0x7fa883a00400

    round: 3
    cubuf 0: 0x7fa897a00600
    cubuf 1: 0x7fa883a00600

    round: 4
    cubuf 0: 0x7fa897a00800
    cubuf 1: 0x7fa883a00800
    ```

    可以看到，地址每次都增加`0x200`，猜测 page size 为 512 Byte。

    cu mem 的最小显存分配粒度为 2M，为什么这里可以做到 512 Byte？

* cuda peer access

    `main.cu`:

    ```cpp
    #include <cuda_runtime.h>
    #include <stdio.h>

    __global__ void arr_inc_1(float *arr)
    {
        int i = threadIdx.x;
        arr[i] += 1;
    }

    int main()
    {    
        float* p0;
        cudaSetDevice(0);
        cudaMalloc(&p0, 4 * sizeof(float));
        cudaMemset(&p0, 0, 4 * sizeof(float));
        // vec_add<<<1, 4>>>(p0);

        cudaSetDevice(1);
        int canAccessPeer;
        cudaDeviceCanAccessPeer(&canAccessPeer, 0, 1);
        if (!canAccessPeer)
        {
            printf("fail to access peer 0\n");
            return -1;
        }
        cudaDeviceEnablePeerAccess(0, 0);
        arr_inc_1<<<1, 4>>>(p0);

        cudaSetDevice(0);
        float buf[4] = {0};
        cudaMemcpy(buf, p0, 4 * sizeof(float), cudaMemcpyDeviceToHost);

        for (int i = 0; i < 4; ++i)
        {
            printf("%.1f, ", buf[i]);
        }
        putchar('\n');

        cudaFree(p0);
        return 0;
    }
    ```

    compile: `nvcc -g main.cu -o main`

    run: `./main`

    output:

    ```
    1.0, 1.0, 1.0, 1.0,
    ```

    我们在 dev 0 上申请显存，然后在 dev 1 上 enable dev 0 的 peer access，再在 dev 1 上 launch kernel，用的数据是 dev 0 上的数据，最后把 dev 0 里的数据拿出来，可以看到是正确的结果。

    说明：

    1. `cudaDeviceEnablePeerAccess(0, 0);`表示从当前 device （由`cudaSetDevice(1);`设定）可以获取 remote device (dev 0) 上的数据，是单向链路。而不是 dev 0 的数据可以由任何其他 dev 获取。

    2. 根据官网资料，peer access 可能走的是 pcie 或 nvlink

        > Depending on the system properties, specifically the PCIe and/or NVLINK topology, devices are able to address each other’s memory

        是否可以走网络或者 host 中转？目前不清楚。

        这里的 peer access 似乎更关注虚拟地址的处理，而不是底层通路。

    3. 根据官网资料，一个 dev 似乎最多能 peer access 8 个其他 dev

        > On non-NVSwitch enabled systems, each device can support a system-wide maximum of eight peer connections.

* 在 50 机器上写如下程序

    `main.cu`:

    ```cpp
    #include <cuda.h>
    #include <stdlib.h>
    #include <stdio.h>

    __global__ void vec_add(float *A, float *B, float *C)
    {
        int id = blockIdx.x;
        C[id] = A[id] + B[id];
    }

    int main()
    {
        float *h_A, *h_B, *h_C;
        h_A = (float*) malloc(8 * sizeof(float));
        h_B = (float*) malloc(8 * sizeof(float));
        h_C = (float*) malloc(8 * sizeof(float));
        for (int i = 0; i < 8; ++i)
        {
            h_A[i] = rand() % 10;
            h_B[i] = rand() % 10;
        }
        float *A, *B, *C;
        cudaMalloc(&A, 8 * sizeof(float));
        cudaMalloc(&B, 8 * sizeof(float));
        cudaMalloc(&C, 8 * sizeof(float));
        cudaMemcpy(A, h_A, 8 * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(B, h_B, 8 * sizeof(float), cudaMemcpyHostToDevice);
        vec_add<<<8, 1>>>(A, B, C);
        cudaMemcpy(h_C, C, 8 * sizeof(float), cudaMemcpyDeviceToHost);
        for (int i = 0; i < 8; ++i)
        {
            printf("%.1f + %.1f = %.1f\n", h_A[i], h_B[i], h_C[i]);
        }
        return 0;
    }
    ```

    `Makefile`:

    ```makefile
    main: main.cu
    	nvcc -g -I../cuda-samples-12.1/Common -o main main.cu

    clean:
    	rm -f main
    ```

    在 vscode 中，使用如下`launch.json`:

    ```json
    {
        // Use IntelliSense to learn about possible attributes.
        // Hover to view descriptions of existing attributes.
        // For more information, visit: https://go.microsoft.com/fwlink/?linkid=830387
        "version": "0.2.0",
        "configurations": [{
            "name": "CUDA C++: Launch",
            "type": "cuda-gdb",
            "request": "launch",
            "program": "${workspaceFolder}/main"
        }]
    }
    ```

    在`vec_add()`中设置断点后，F5 运行无法 hit 断点。目前不清楚原因。

    （2024.12.20）：目前看来，应该是编译时没加`-G`。

* 224 机器上的 nccl cuda-gdb 依然很慢，起码需要半个小时以上才能 hit 断点

* 使用 b address `(cuda-gdb) b *0x7ffe85823040`会导致直接 cuda-gdb 直接报错退出。

    ```
    (cuda-gdb) b *0x7ffe85823040
    Breakpoint 3 at 0x7ffe85823040
    (cuda-gdb) c
    Continuing.
    warning: Cuda API error detected: cuLaunchKernelEx returned (0x190)


    zjxj:55959:55959 [1] enqueue.cc:1451 NCCL WARN Cuda failure 400 'invalid resource handle'
    zjxj:55959:55959 [1] NCCL INFO group.cc:231 -> 1
    zjxj:55959:55959 [1] NCCL INFO group.cc:453 -> 1
    zjxj:55959:55959 [1] NCCL INFO group.cc:546 -> 1
    zjxj:55959:55959 [1] NCCL INFO group.cc:101 -> 1
    fail to group end
    [Thread 0x7fff590fd000 (LWP 55998) exited]
    [Thread 0x7fff598fe000 (LWP 55997) exited]
    [Thread 0x7fff615ac000 (LWP 55994) exited]
    [Thread 0x7fff61dad000 (LWP 55993) exited]
    [Thread 0x7fffc0dff000 (LWP 55992) exited]
    [Thread 0x7fffc924a000 (LWP 55991) exited]
    [Thread 0x7fffc2a4f000 (LWP 55990) exited]
    [Thread 0x7fffc8a49000 (LWP 55985) exited]
    [Thread 0x7fffc9b3d000 (LWP 55983) exited]
    [Thread 0x7fffd4909000 (LWP 55963) exited]
    [Inferior 1 (process 55959) exited with code 0377]
    ```

* cuda 12.4 在编译的时候必须使用 compute 80，使用 compute 70 无法正常运行

    好像 compute 70, 80, 90 分别对应三种不同的 nv gpu 架构。

* `cudaSuccess`是 cuda runtime 里的枚举常量，`CUDA_SUCCESS`是 cuda umd 里的枚举常量

* thread id 与 thread idx 并不是同一个东西

    thread id 是 thread idx 的 flatten 版本。

    对于一维的情况，`thread_id = thread idx`

    对于二维 size 为`(dim_len_x, dim_len_y)`的情况，`thread_id = x + y * dim_len_x`

    对于三维 size 为`(dim_len_x, dim_len_y, dim_len_z)`的情况，`thread_id = x + y * dim_len_x + z * dim_len_x * dim_len_z`

    example:

    `main.cu`:

    ```cpp
    #define dim_x 3
    #define dim_y 4
    #define dim_z 5

    __global__ void vol_add(
        int A[dim_x][dim_y][dim_z], 
        int B[dim_x][dim_y][dim_z],
        int C[dim_x][dim_y][dim_z])
    {
        int i = threadIdx.x;
        int j = threadIdx.y;
        int k = threadIdx.z;

        C[i][j][k] = A[i][j][k] + B[i][j][k];
    }

    #include <cuda_runtime.h>
    #include <stdio.h>
    #include <stdlib.h>

    int main()
    {
        // int dim_x = 3, dim_y = 4, dim_z = 5;
        int num_elm = dim_x * dim_y * dim_z;
        int *host_buf_A = (int*) malloc(num_elm * sizeof(int));
        int *host_buf_B = (int*) malloc(num_elm * sizeof(int));
        int *host_buf_C = (int*) malloc(num_elm * sizeof(int));
        for (int i = 0; i < num_elm; ++i)
        {
            host_buf_A[i] = rand() % 5;
            host_buf_B[i] = rand() % 5;
        }

        int *dev_buf_A, *dev_buf_B, *dev_buf_C;
        cudaError_t ret;
        ret = cudaMalloc(&dev_buf_A, num_elm * sizeof(int));
        if (ret != cudaSuccess)
        {
            printf("fail to cuda malloc dev buf A\n");
            return -1;
        }
        printf("successfully cuda malloc dev buf A\n");

        ret = cudaMalloc(&dev_buf_B, num_elm * sizeof(int));
        if (ret != cudaSuccess)
        {
            printf("fail to cuda malloc dev buf B\n");
            return -1;
        }
        printf("successfully cuda malloc dev buf B\n");

        ret = cudaMalloc(&dev_buf_C, num_elm * sizeof(int));
        if (ret != cudaSuccess)
        {
            printf("fail to cuda malloc dev buf C\n");
            return -1;
        }
        printf("successfully cuda malloc dev buf C\n");

        ret = cudaMemcpy(dev_buf_A, host_buf_A, num_elm * sizeof(int), cudaMemcpyHostToDevice);
        if (ret != cudaSuccess)
        {
            printf("fail to cuda memcpy host buf A\n");
            return -1;
        }
        printf("successfully cuda memcpy host buf A\n");

        ret = cudaMemcpy(dev_buf_B, host_buf_B, num_elm * sizeof(int), cudaMemcpyHostToDevice);
        if (ret != cudaSuccess)
        {
            printf("fail to cuda memcpy host buf B\n");
            return -1;
        }
        printf("successfully cuda memcpy host buf B\n");

        dim3 threads_per_block(dim_x, dim_y, dim_z);
        vol_add<<<1, threads_per_block>>>(
            (int (*)[4][5]) dev_buf_A, 
            (int (*)[4][5]) dev_buf_B,
            (int (*)[4][5]) dev_buf_C);

        ret = cudaMemcpy(host_buf_C, dev_buf_C, num_elm * sizeof(int), cudaMemcpyDeviceToHost);
        if (ret != cudaSuccess)
        {
            printf("fail to cuda memcpy host buf C\n");
            return -1;
        }
        printf("successfully cuda memcpy host buf C\n");

        printf("host buf A:\n");
        for (int i = 0; i < num_elm; ++i)
        {
            printf("%d, ", host_buf_A[i]);
        }
        putchar('\n');

        printf("host buf B:\n");
        for (int i = 0; i < num_elm; ++i)
        {
            printf("%d, ", host_buf_B[i]);
        }
        putchar('\n');

        printf("host buf C:\n");
        for (int i = 0; i < num_elm; ++i)
        {
            printf("%d, ", host_buf_C[i]);
        }
        putchar('\n');

        cudaFree(dev_buf_A);
        cudaFree(dev_buf_B);
        cudaFree(dev_buf_C);
        free(host_buf_A);
        free(host_buf_B);
        free(host_buf_C);
        return 0;
    }
    ```

    compile: `nvcc -g -G -o main main.cu`

    run: `./main`

    output:

    ```
    successfully cuda malloc dev buf A
    successfully cuda malloc dev buf B
    successfully cuda malloc dev buf C
    successfully cuda memcpy host buf A
    successfully cuda memcpy host buf B
    successfully cuda memcpy host buf C
    host buf A:
    3, 2, 3, 1, 4, 2, 0, 3, 0, 2, 1, 2, 2, 2, 2, 4, 2, 4, 3, 1, 4, 1, 4, 3, 0, 3, 1, 1, 2, 1, 0, 4, 1, 1, 3, 4, 2, 4, 4, 3, 2, 1, 3, 3, 4, 2, 1, 4, 1, 4, 0, 4, 2, 2, 2, 2, 1, 1, 0, 4, 
    host buf B:
    1, 0, 0, 2, 1, 2, 4, 1, 1, 1, 3, 4, 0, 3, 0, 2, 3, 2, 1, 2, 3, 4, 2, 4, 0, 1, 0, 3, 0, 1, 0, 2, 0, 4, 2, 0, 0, 2, 4, 0, 3, 3, 4, 1, 4, 0, 3, 2, 1, 4, 0, 3, 1, 2, 2, 1, 0, 1, 4, 4, 
    host buf C:
    4, 2, 3, 3, 5, 4, 4, 4, 1, 3, 4, 6, 2, 5, 2, 6, 5, 6, 4, 3, 7, 5, 6, 7, 0, 4, 1, 4, 2, 2, 0, 6, 1, 5, 5, 4, 2, 6, 8, 3, 5, 4, 7, 4, 8, 2, 4, 6, 2, 8, 0, 7, 3, 4, 4, 3, 1, 2, 4, 8,
    ```

    看起来一个 block 是一个组织 thread 的方式。

    每个 block 最大有 1024 个 thread。

    如果我们只使用一维 thread，手动将`threadIdx.x`, `threadIdx.y`, `threadIdx.z`映射到一维的 idx 上，和直接使用 x, y, z 有什么不同？

    说明：

    1. `vol_add<<<numBlocks, threadsPerBlock>>>()`

        `threadsPerBlock`可以是一个数字，也可以是`dim3`类型的变量。

        如果数组中有 2 个元素或 3 个元素，那么必须使用`dim3`类型定义。

* thread block: cuda 中指一维的多线程，二维的多线程或三维的多线程

    vector, matrix, or volume 分别代表一维的数据，二维矩阵，三维体素

* cuda low level memory operation example

    `main.cu`:

    ```cpp
    #include <cuda.h>
    #include <stdio.h>

    int main()
    {
        cuInit(0);

        int device_id = 0;

        CUcontext ctx;
        CUresult ret;
        ret = cuCtxCreate(&ctx, 0, device_id);
        if (ret != CUDA_SUCCESS)
        {
            printf("fail to create cuda ctx\n");
            return -1;
        }
        printf("successfully create cuda ctx\n");

        CUmemGenericAllocationHandle handle;
        size_t size = 4096;
        CUmemAllocationProp prop{};
        prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        prop.location.id = device_id;
        // prop.allocFlags.compressionType = CU_MEM_ALLOCATION_COMP_NONE;
        unsigned long long flags = 0;

        size_t granularity = 0;
        cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM);

        size_t padded_size = max(size, granularity);
        printf("granularity: %lu\n", granularity);
        printf("padded size: %lu\n", padded_size);

        ret = cuMemCreate(&handle, padded_size, &prop, flags);
        if (ret != CUDA_SUCCESS)
        {
            printf("fail to create cu mem\n");
            return -1;
        }
        printf("successfully create cu mem\n");

        CUdeviceptr ptr;
        ret = cuMemAddressReserve(&ptr, padded_size, 0, 0, 0);
        if (ret != CUDA_SUCCESS)
        {
            printf("fail to reserve mem addr\n");
            return -1;
        }
        printf("successfully reserve mem addr\n");
        printf("\tptr: %p\n", (void*) ptr);
        printf("\tsizeof(unsigned long long): %lu\n", sizeof(unsigned long long));

        ret = cuMemMap(ptr, padded_size, 0, handle, 0);
        if (ret != CUDA_SUCCESS)
        {
            printf("fail to map mem\n");
            return -1;
        }
        printf("successfully map mem\n");

        ret = cuMemRelease(handle);
        if (ret != CUDA_SUCCESS)
        {
            printf("fail to release cu mem\n");
            return -1;
        }
        printf("successfully release cu mem\n");

        CUmemAccessDesc access_desc;
        access_desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        access_desc.location.id = device_id;
        access_desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        ret = cuMemSetAccess(ptr, padded_size, &access_desc, 1);
        if (ret != CUDA_SUCCESS)
        {
            printf("fail to set mem access\n");
            return -1;
        }
        printf("successfully set mem access\n");

        int *host_mem = (int*) malloc(128 * sizeof(int));
        if (!host_mem)
        {
            printf("fail to malloc host_mem\n");
            return -1;
        }
        printf("successfully malloc host_mem\n");

        ret = cuMemcpyHtoD(ptr, host_mem, 128 * sizeof(int));
        if (ret != CUDA_SUCCESS)
        {
            printf("fail to memcpy h to d\n");
            return -1;
        }
        printf("successfully memcpy h to d\n");

        // for (int i = 0; i < 128; ++i)
        // {
        //     ((int*) ptr)[i] = i;
        // }

        ret = cuMemUnmap(ptr, padded_size);
        if (ret != CUDA_SUCCESS)
        {
            printf("fail to unmap mem\n");
            return -1;
        }
        printf("successfully unmap mem\n");

        ret = cuMemAddressFree(ptr, padded_size);
        if (ret != CUDA_SUCCESS)
        {
            printf("fail to free mem addr\n");
            return -1;
        }
        printf("successfully free mem addr\n");

        ret = cuCtxDestroy(ctx);
        if (ret != CUDA_SUCCESS)
        {
            printf("fail to destroy cuda ctx\n");
            return -1;
        }
        printf("successfully destroy cuda ctx\n");

        free(host_mem);
        return 0;
    }
    ```

    compile: `nvcc -g main.cu -lcuda -o main`

    run: `./main`

    output:

    ```
    successfully create cuda ctx
    granularity: 2097152
    padded size: 2097152
    successfully create cu mem
    successfully reserve mem addr
    	ptr: 0x7f72fda00000
    	sizeof(unsigned long long): 8
    successfully map mem
    successfully release cu mem
    successfully set mem access
    successfully malloc host_mem
    successfully memcpy h to d
    successfully unmap mem
    successfully free mem addr
    successfully destroy cuda ctx
    ```

    说明：

    1. 不执行`cuInit(0);`，后面所有与 cuda 相关的函数都会执行失败

    2. `cuCtxCreate()`是必须的，不然`cuMemcpyHtoD()`会失败

    3. `cuMemMap()`之后马上就可以`cuMemRelease()`了，为什么？ release 不是释放显存的意思吗？

    4. `cuMemAddressReserve()`应该是只分配了 va 地址范围，device 这时可能还不知道这个 va 范围是多少，也没有把 va 写入到 mmu 里。

        device 在什么时候能看到其他 device 在 user space 的 va？这时候大概率就要写入 device 的 mmu 了。

    5. `cuMemSetAccess()`必须执行，不然`cuMemcpyHtoD()`会失败

    6. 只能使用`cuMemcpyHtoD()`将数据复制到 device 内，不能像 vulkan 那样 mmap 之后可以直接解引用赋值：

        ```cpp
        // for (int i = 0; i < 128; ++i)
        // {
        //     ((int*) ptr)[i] = i;  // segment fault
        // }
        ```

* 使用`cuMemCreate()`申请 device memory

    `main.cu`:

    ```cpp
    #include <cuda.h>
    #include <stdio.h>

    int main()
    {
        cuInit(0);

        int device_id = 0;

        CUmemGenericAllocationHandle handle;
        size_t size = 4096;
        CUmemAllocationProp prop{};
        prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        prop.location.id = device_id;
        // prop.allocFlags.compressionType = CU_MEM_ALLOCATION_COMP_NONE;
        unsigned long long flags = 0;

        size_t granularity = 0;
        cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM);

        size_t padded_size = max(size, granularity);
        printf("granularity: %lu\n", granularity);
        printf("padded size: %lu\n", padded_size);

        CUresult ret;
        ret = cuMemCreate(&handle, padded_size, &prop, flags);
        if (ret != CUDA_SUCCESS)
        {
            printf("fail to create cu mem\n");
            return -1;
        }
        printf("successfully create cu mem\n");

        ret = cuMemRelease(handle);
        if (ret != CUDA_SUCCESS)
        {
            printf("fail to release cu mem\n");
            return -1;
        }
        printf("successfully release cu mem\n");

        return 0;
    }
    ```

    compile:
    
    `nvcc -g main.cu -lcuda -o main`

    run:

    `./main`

    output:

    ```
    granularity: 2097152
    padded size: 2097152
    successfully create cu mem
    successfully release cu mem
    ```

    可以看到，`prop.type`, `prop.location.type`和`prop.location.id`是必填项，其他的项可以置 0。

    当前 device 支持的最小显存申请大小为 2 MB。

    说明：

    1. 必须先执行`cuInit(0);`，`cuMemCreate()`才能成功返回。

    1. `prop.type`只有一种选择。`cuda.h`中定义的 enum 如下：

        ```cpp
        /**
        * Defines the allocation types available
        */
        typedef enum CUmemAllocationType_enum {
            CU_MEM_ALLOCATION_TYPE_INVALID = 0x0,

            /** This allocation type is 'pinned', i.e. cannot migrate from its current
              * location while the application is actively using it
              */
            CU_MEM_ALLOCATION_TYPE_PINNED  = 0x1,
            CU_MEM_ALLOCATION_TYPE_MAX     = 0x7FFFFFFF
        } CUmemAllocationType;
        ```

    1. `prop.location.type`也只有一种选择，`cuda.h`中定义的 enum 如下：

        ```cpp
        /**
         * Specifies the type of location
         */
        typedef enum CUmemLocationType_enum {
            CU_MEM_LOCATION_TYPE_INVALID = 0x0,
            CU_MEM_LOCATION_TYPE_DEVICE  = 0x1,  /**< Location is a device location, thus id is a device ordinal */
            CU_MEM_LOCATION_TYPE_MAX     = 0x7FFFFFFF
        } CUmemLocationType;
        ```

* 当 src 中有`cuda.h`的函数时，无论 src 的扩展名是`.cu`还是`.cpp`，`nvcc`都不会自动链接库`-lcuda`，需要手动添加:

    `nvcc -g main.cu -lcuda -o main`

    如果使用`gcc`编译，需要将扩展名改成`.c`，并且手动指定 include 目录：

    `gcc -g main.c -I/usr/local/cuda/include -lcuda -o main`

* `cuMemCreate()`之类的函数在`cuda.h`中，不在`cuda_runtime.h`中。

    `cudaMalloc()`之类的函数在`cuda_runtime.h`中，不在`cuda.h`中。

    看起来`cuda.h`比`cuda_runtime.h`更底层一点。

* asm 基本语法

    `asm("template-string" : "constraint"(output) : "constraint"(input)"));`

    一个使用`vabsdiff4`命令的 example:

    `asm("vabsdiff4.u32.u32.u32.add" " %0, %1, %2, %3;": "=r" (result):"r" (A), "r" (B), "r" (C));`

    从例子中可以看出，前两个字符串应该是拼接在一起的完整命令：`"vabsdiff4.u32.u32.u32.add %0, %1, %2, %3;"`。根据后面的命令可以猜出，`=`表示需要写入的变量，其余的是需要读取的变量。`r`不清楚是什么意思。

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