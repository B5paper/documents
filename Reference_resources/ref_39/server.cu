#include <sys/socket.h>
#include <stdio.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <cuda_runtime.h>
#include "cumem_hlc.h"

int serv_fd;
char ipv4_addr[16] = {0};
uint16_t listen_port;
uint32_t listen_addr_ipv4;
struct sockaddr_in cli_addr;
int cli_fd;

int init_sock_server() {
    serv_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (serv_fd < 0) {
        printf("fail to create server sock fd\n");
        return -1;
    }
    
    listen_port = 6543;
    listen_addr_ipv4 = INADDR_ANY;
    
    struct sockaddr_in serv_addr;
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_addr.s_addr = listen_addr_ipv4;
    serv_addr.sin_port = htons(listen_port);
    int ret = bind(serv_fd, (struct sockaddr*) &serv_addr, sizeof(serv_addr));
    if (ret < 0) {
        printf("fail to bind serv fd: %d, ret: %d\n", serv_fd, ret);
        return -1;
    }

    ret = listen(serv_fd, 5);
    if (ret < 0) {
        printf("fail to listen\n");
        return -1;
    }
    return 0;
}

int sock_accept() {
    socklen_t cli_addr_len = sizeof(cli_addr);
    cli_fd = accept(serv_fd, (struct sockaddr*) &cli_addr, &cli_addr_len);
    if (cli_fd < 0) {
        printf("fail to accept, ret: %d\n", cli_fd);
        return -1;
    }

    const char *ret_ptr = inet_ntop(AF_INET, &cli_addr.sin_addr.s_addr, ipv4_addr, 16);
    if (ret_ptr == NULL) {
        printf("fail to connect u32 ipv4 to string\n");
        return -1;
    }
    return 0;
}

int sock_send(const char *buf, size_t len) {
    // size_t buf_len = strlen(buf) + 1;
    size_t bytes_send = send(cli_fd, buf, len, 0);
    if (bytes_send != len) {
        printf("fail to send, buf_len: %lu, bytes_send: %ld\n", len, bytes_send);
        return -1;
    }
    return 0;
}

int close_sock() {
    int ret = shutdown(cli_fd, SHUT_RDWR);
    if (ret != 0) {
        printf("fail to shutdown client fd: %d, ret: %d\n", cli_fd, ret);
        return -1;
    }

    ret = shutdown(serv_fd, SHUT_RDWR);
    if (ret != 0) {
        printf("fail to shutdown server fd: %d, ret: %d\n", serv_fd, ret);
        return -1;
    }
    return 0;
}

int main() {
    int ret = init_sock_server();
    if (ret != 0) {
        printf("fail to init sock server\n");
        return -1;
    }

    sock_accept();

    int *buf = NULL;
    int *cubuf;
    int num_elm = 8;
    
    // cudaSetDevice(0);
    // cudaDeviceEnablePeerAccess(1, 0);
    // cudaSetDevice(1);
    // cudaDeviceEnablePeerAccess(0, 0);

    cudaSetDevice(0);
    sibling_alloc_buf_assign_rand_int<int>(&buf, &cubuf, num_elm);
    print_cubuf(cubuf, num_elm);

    cudaIpcMemHandle_t handle;  // 64 bytes
    cudaIpcGetMemHandle(&handle, cubuf);

    sock_send((const char*) &handle, sizeof(handle));
    printf("sizeof handle: %lu\n", sizeof(handle));
    printf("cubuf: %p\n", cubuf);

    close_sock();

    sleep(10);

    return 0; 
}
