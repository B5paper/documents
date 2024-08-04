#ifndef SOCK_H
#define SOCK_H

#include <netinet/in.h>

// -------- server --------

struct sock_conn
{
    int cli_sock_fd;
    struct sockaddr_in cli_addr;
    uint32_t cli_addr_ipv4;
    uint16_t cli_port;
    struct sock_conn *next;
};

struct serv_sock_ctx_2 {
    int serv_sock_fd;
    char listen_addr_str[16];
    uint32_t listen_addr;
    uint16_t listen_port;
    struct sock_conn *dummy_head;
};

struct serv_sock_ctx {
    int serv_sock_fd;
    char listen_addr_str[16];
    uint32_t listen_addr;
    uint16_t listen_port;
};

struct cli_sock_info
{
    struct sockaddr_in cli_addr;
    int cli_sock_fd;
    uint32_t cli_addr_ipv4;
    uint16_t cli_port;
};

void init_serv_sock_ctx(struct serv_sock_ctx *ssctx, const uint16_t listen_port);
void accept_sock_cli(int serv_sock_fd, struct cli_sock_info *cs_info);
int exit_serv_sock_ctx(const struct serv_sock_ctx_2 *ss_ctx);

// -------- client --------

struct cli_sock_ctx {
    int cli_sock_fd;
    char serv_addr_str[16];
    uint32_t serv_addr;
    uint16_t serv_port;
};

void init_cli_sock_ctx(struct cli_sock_ctx *csctx, const char *serv_addr, const uint16_t serv_port);

// -------- common --------

void sock_send(int fd, const void *buf, size_t buf_len);
void sock_recv(int fd, void *buf, size_t buf_len);
void sock_serv_exchange_data(int fd, void *buf, size_t buf_len);
void sock_cli_exchange_data(int fd, void *buf, size_t buf_len);

#endif

