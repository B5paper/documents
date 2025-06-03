#include <sys/socket.h>
#include <arpa/inet.h>
#include <stdio.h>
#include <unistd.h>
#include <cuda_runtime.h>
#include "cumem_hlc.h"
#include "cuda.h"

int cli_fd;

int init_sock_cli() {
    cli_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (cli_fd < 0) {
        printf("fail to create client sock fd\n");
        return -1;
    }

    uint16_t serv_port = 6543;
    const char serv_ipv4[16] = "127.0.0.1";
    struct in_addr ipv4_addr;
    int ret = inet_pton(AF_INET, serv_ipv4, &ipv4_addr);
    if (ret != 1) {
        printf("fail to convert ipvr string to u32, ret: %d\n", ret);
        return -1;
    }

    struct sockaddr_in serv_addr;
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_addr = ipv4_addr;
    serv_addr.sin_port = htons(serv_port);
    ret = connect(cli_fd, (struct sockaddr*) &serv_addr, sizeof(serv_addr));
    if (ret != 0) {
        printf("fail to connect to server, ret: %d\n", ret);
        return -1;
    }
    return 0;
}

int sock_recv(char *buf, size_t buf_len) {
    ssize_t bytes_recv = recv(cli_fd, buf, buf_len, 0);
    if (bytes_recv <= 0) {
        printf("fail to recv, buf_len: %lu, bytes_Recv: %ld\n", buf_len, bytes_recv);
        return -1;
    }
    return 0;
}

int close_sock() {
    int ret = shutdown(cli_fd, SHUT_RDWR);
    if (ret != 0) {
        printf("fail to shutdown fd: %d, ret: %d\n", cli_fd, ret);
        return -1;
    }
    return 0;
}

__global__ void kernel_p2p_copy(int *dst, int *src, int num_elm) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elm) {
        dst[idx] = src[idx];
    }
}

int main() {
    init_sock_cli();

    cudaSetDevice(0);
    cudaDeviceEnablePeerAccess(1, 0);
    int ret = cudaSetDevice(1);
    printf("%d\n", ret);
    cudaDeviceEnablePeerAccess(0, 0);

    char sock_buf[4096] = {0};
    size_t sock_buf_len = 4096;
    sock_recv(sock_buf, sizeof(cudaIpcMemHandle_t));
    // printf("recv buf: %p\n", (void*) sock_buf);

    cudaIpcMemHandle_t handle;
    handle = *(cudaIpcMemHandle_t*) sock_buf;
    int *cubuf = NULL;
    cudaIpcOpenMemHandle((void**) &cubuf, handle, cudaIpcMemLazyEnablePeerAccess);
    printf("cubuf %p\n", cubuf);

    // cudaSetDevice(0);
    print_cubuf(cubuf, 8);

    int *cubuf_2;
    cudaSetDevice(1);
    cudaMalloc(&cubuf_2, 8 * sizeof(int));
    printf("cubuf 2: %p\n", cubuf_2);
    
    kernel_p2p_copy<<<1, 8>>>(cubuf_2, cubuf, 8);
    cudaDeviceSynchronize();
    print_cubuf<int>(cubuf_2, 8);

    close_sock();

    return 0;
}
