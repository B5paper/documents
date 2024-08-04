#include "sock.h"
#include <stdio.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <stdlib.h>
#include <string.h>

void init_serv_sock_ctx(struct serv_sock_ctx *ssctx, const uint16_t listen_port)
{
    int serv_sock_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (serv_sock_fd < 0) {
        printf("fail to create server sock fd\n");
        exit(-1);
    }
    printf("successfully create server sock fd: %d\n", serv_sock_fd);
    ssctx->serv_sock_fd = serv_sock_fd;

    uint32_t listen_addr_ipv4 = INADDR_ANY;
    struct sockaddr_in serv_sock_addr;
    serv_sock_addr.sin_family = AF_INET;
    serv_sock_addr.sin_addr.s_addr = listen_addr_ipv4;
    serv_sock_addr.sin_port = htons(listen_port);

    int ret = 0;
    ret = bind(serv_sock_fd, (struct sockaddr*)&serv_sock_addr, sizeof(serv_sock_addr));
    if (ret < 0) {
        printf("fail to bind\n");
        exit(-1);
    }
    printf("successfully bind addr: %08x, port: %u\n", listen_addr_ipv4, listen_port);
    ssctx->listen_addr = listen_addr_ipv4;
    strcpy(ssctx->listen_addr_str, "0.0.0.0");
    ssctx->listen_port = listen_port;

    ret = listen(serv_sock_fd, 5);
    if (ret < 0) {
        printf("fail to listen\n");
        exit(-1);
    }
    printf("start to listen...\n");
}

void accept_sock_cli(int serv_sock_fd, struct cli_sock_info *cs_info)
{
    struct sockaddr_in client_addr;
    int client_addr_len = sizeof(client_addr);
    int client_sock_fd = accept(serv_sock_fd, (struct sockaddr*)&client_addr, (socklen_t*)&client_addr_len);
    if (client_sock_fd < 0) {
        printf("fail to accept\n");
        exit(-1);
    }
    uint32_t client_addr_ipv4 = client_addr.sin_addr.s_addr;
    int client_port = client_addr.sin_port;

    cs_info->cli_addr = client_addr;
    cs_info->cli_sock_fd = client_sock_fd;
    cs_info->cli_addr_ipv4 = client_addr_ipv4;
    cs_info->cli_port = client_port;
    printf("successfully accept clinet, ip: %08x, port: %u\n", client_addr_ipv4, client_port);
}

void init_cli_sock_ctx(struct cli_sock_ctx *csctx, const char *serv_addr, const uint16_t serv_port)
{
    int client_sock_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (client_sock_fd < 0) {
        printf("fail to create client sock fd\n");
        exit(-1);
    }
    printf("successfully create client sock fd: %d\n", client_sock_fd);
    csctx->cli_sock_fd = client_sock_fd;

    struct in_addr inaddr;
    inet_pton(AF_INET, serv_addr, &inaddr);

    struct sockaddr_in serv_sock_addr;
    serv_sock_addr.sin_family = AF_INET;
    serv_sock_addr.sin_addr = inaddr;
    serv_sock_addr.sin_port = htons(serv_port);

    int ret = 0;
    ret = connect(client_sock_fd, (struct sockaddr*)&serv_sock_addr, sizeof(serv_sock_addr));
    if (ret < 0) {
        printf("fail to connect\n");
        exit(-1);
    }
    printf("successfully connect to server, ip: %s, port: %u\n", serv_addr, serv_port);

    strcpy(csctx->serv_addr_str, serv_addr);
    csctx->serv_port = serv_port;
}

void sock_send(int fd, const void *buf, size_t buf_len)
{
    int bytes_send = send(fd, buf, buf_len, 0);
    if (bytes_send != buf_len)
    {
        printf("fail to send data via socket\n");
        exit(-1);
    }
}

void sock_recv(int fd, void *buf, size_t buf_len)
{
    int bytes_recv = recv(fd, buf, buf_len, 0);
    if (bytes_recv != buf_len)
    {
        printf("fail to recv data via socket\n");
        exit(-1);
    }
}

int sock_serv_sync(int cli_sock_fd)
{
    sock_send(cli_sock_fd, "sync", 5);
    char buf[5] = {0};
    int bytes_recv = recv(cli_sock_fd, buf, 5, 0);
    if (bytes_recv != 5 || strcmp(buf, "sync") != 0)
    {
        printf("fail to recv sync\n");
        return -1;
    }
    return 0;
}

int sock_cli_sync(int cli_sock_fd)
{
    char buf[5] = {0};
    int bytes_recv = recv(cli_sock_fd, buf, 5, 0);
    if (bytes_recv != 5 || strcmp(buf, "sync") != 0)
    {
        printf("fail to recv sync\n");
        return -1;
    }
    sock_send(cli_sock_fd, "sync", 5);
    return 0;
}

void sock_serv_exchange_data(int fd, void *buf, size_t buf_len)
{
    sock_send(fd, buf, buf_len);
    sock_recv(fd, buf, buf_len);
}

void sock_cli_exchange_data(int fd, void *buf, size_t buf_len)
{
    void *buf_copy = malloc(buf_len);
    memcpy(buf_copy, buf, buf_len);
    sock_recv(fd, buf, buf_len);
    sock_send(fd, buf_copy, buf_len);
    free(buf_copy);
}

int exit_serv_sock_ctx(const struct serv_sock_ctx_2 *ss_ctx)
{
    struct sock_conn *p_sock_conn = ss_ctx->dummy_head->next;
    int ret;
    while (p_sock_conn)
    {
        ret = shutdown(p_sock_conn->cli_sock_fd, SHUT_RDWR);
        if (ret != 0)
        {
            printf("fail to shutdown socket\n");
            return -1;
        }
        p_sock_conn = p_sock_conn->next;
    }
    
    ret = shutdown(ss_ctx->serv_sock_fd, SHUT_RDWR);
    if (ret != 0)
    {
        printf("fail to shutdown socket\n");
        return -1;
    }
    return 0;
}
