# rdma note qa

[unit]
[u_0]
写一个基于 ib verbs 和 socket 的 client-server 程序，server 向 client 通过 post send 发送数据。
[u_1]
`server.c`:

```c
#include <infiniband/verbs.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include "sock.h"
#include "metadata.h"

int main(int argc, const char **argv)
{
    int num_devs;
    struct ibv_device **ib_devs = ibv_get_device_list(&num_devs);
    if (ib_devs == NULL || num_devs == 0)
    {
        printf("fail to get ibv dev list\n");
        return -1;
    }
    printf("[OK] get %d ib devices.\n", num_devs);

    const char *dev_name;
    struct ibv_device *ib_dev;
    __be64 dev_guid;
    for (int i = 0; i < num_devs; i++)
    {
        ib_dev = ib_devs[i];
        dev_name = ibv_get_device_name(ib_dev);
        dev_guid = ibv_get_device_guid(ib_dev);
        printf("\tidx: %d, name: %s, guid (BE): %016llx\n", i, dev_name, dev_guid);
    }

    int dev_idx = 0; // open idx 0 ib device
    ib_dev = ib_devs[dev_idx];
    struct ibv_context *ib_ctx = ibv_open_device(ib_dev);
    if (ib_ctx == NULL)
    {
        printf("fail to open device %d\n", dev_idx);
        return -1;
    }
    printf("[OK] open device %d.\n", dev_idx);

    struct ibv_port_attr port_attr;
    int ret = ibv_query_port(ib_ctx, 1, &port_attr);
    if (ret != 0)
    {
        printf("fail to query port\n");
        return -1;
    }
    printf("[OK] query port.\n");
    printf("\tlid: %d\n", port_attr.lid);

    int min_cqe_num = 1;
    struct ibv_cq *cq = ibv_create_cq(ib_ctx, min_cqe_num, NULL, NULL, 0);
    if (cq == NULL)
    {
        printf("fail to create cq\n");
        return -1;
    }
    printf("[OK] create cq.\n");

    struct ibv_pd *pd = ibv_alloc_pd(ib_ctx);
    if (pd == NULL)
    {
        printf("fail to allocate pd\n");
        return -1;
    }
    printf("[OK] allocate pd.\n");

    struct ibv_qp_init_attr qp_init_attr;
    qp_init_attr.qp_type = IBV_QPT_RC;
    qp_init_attr.recv_cq = cq;
    qp_init_attr.send_cq = cq;
    qp_init_attr.cap.max_send_wr = 2;
    qp_init_attr.cap.max_recv_wr = 2;
    qp_init_attr.cap.max_send_sge = 2;
    qp_init_attr.cap.max_recv_sge = 2;
    struct ibv_qp *qp = ibv_create_qp(pd, &qp_init_attr);
    if (qp == NULL)
    {
        printf("fail to create qp\n");
        return -1;
    }
    printf("[OK] create qp.\n");
    printf("\tqp num: %d\n", qp->qp_num);

    struct ibv_qp_attr qp_attr;
    int attr_mask;
    qp_attr.qp_state = IBV_QPS_INIT;
    qp_attr.pkey_index = 0;
    qp_attr.port_num = 1;
    qp_attr.qp_access_flags = IBV_ACCESS_LOCAL_WRITE;
    attr_mask = IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS;
    ret = ibv_modify_qp(qp, &qp_attr, attr_mask);
    if (ret != 0)
    {
        printf("fail to modify qp to INIT state\n");
        return -1;
    }
    printf("[OK] modify qp to INIT state.\n");

    struct serv_sock_ctx ssctx;
    init_serv_sock_ctx(&ssctx, 6543);
    struct cli_sock_info cs_info;
    accept_sock_cli(ssctx.serv_sock_fd, &cs_info);

    struct conn_info conn_info;
    conn_info.qp_num = qp->qp_num;
    conn_info.lid = port_attr.lid;
    sock_serv_exchange_data(cs_info.cli_sock_fd, &conn_info, sizeof(conn_info));
    printf("remote qp num: %u, lid: %u\n", conn_info.qp_num, conn_info.lid);

    qp_attr.qp_state = IBV_QPS_RTR;
    qp_attr.path_mtu = IBV_MTU_4096;
    qp_attr.dest_qp_num = conn_info.qp_num;
    qp_attr.rq_psn = 0;
    qp_attr.max_dest_rd_atomic = 1;
    qp_attr.min_rnr_timer = 12;
    qp_attr.ah_attr.is_global = 0;
    qp_attr.ah_attr.sl = 0;
    qp_attr.ah_attr.src_path_bits = 0;
    qp_attr.ah_attr.dlid = conn_info.lid;
    qp_attr.ah_attr.port_num = 1;
    attr_mask = IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN | IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER;
    ret = ibv_modify_qp(qp, &qp_attr, attr_mask);
    if (ret != 0)
    {
        printf("fail to modify qp to RTR state, ret: %d\n", ret);
        return -1;
    }
    printf("[OK] modify qp to RTR state.\n");

    qp_attr.qp_state = IBV_QPS_RTS;
    qp_attr.sq_psn = 0;
    qp_attr.timeout = 14;
    qp_attr.retry_cnt = 7;
    qp_attr.rnr_retry = 7;
    qp_attr.max_rd_atomic = 1;
    attr_mask = IBV_QP_STATE | IBV_QP_SQ_PSN | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY | IBV_QP_MAX_QP_RD_ATOMIC;
    ret = ibv_modify_qp(qp, &qp_attr, attr_mask);
    if (ret != 0)
    {
        printf("fail to modify qp to RTS state, ret: %d\n", ret);
        return -1;
    }
    printf("[OK] modify qp to RTS state.\n");

    const size_t buf_len = 128;
    void *buf = malloc(buf_len);
    struct ibv_mr *mr = ibv_reg_mr(pd, buf, buf_len, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ);;
    if (mr == NULL)
    {
        printf("fail to reg mr\n");
        return -1;
    }
    printf("[OK] reg mr.\n");
    printf("\tva: %p, len: %lu\n", buf, buf_len);
    strcpy(buf, "hello from server\n");

    struct mr_info mr_info;
    mr_info.buf_addr = (uint64_t) buf;
    mr_info.buf_len = buf_len;
    mr_info.rkey = mr->rkey;
    sock_serv_exchange_data(cs_info.cli_sock_fd, &mr_info, sizeof(mr_info));
    printf("remote addr: %p, len: %lu, rkey: %u\n", (void*) mr_info.buf_addr, mr_info.buf_len, mr_info.rkey);

    struct ibv_sge sge;
    sge.addr = (uint64_t) buf;
    sge.length = buf_len;
    sge.lkey = mr->lkey;
    struct ibv_send_wr send_wr, *bad_send_wr;
    send_wr.wr_id = 12345;
    send_wr.opcode = IBV_WR_SEND;
    send_wr.send_flags = IBV_SEND_SIGNALED;
    send_wr.sg_list = &sge;
    send_wr.num_sge = 1;
    send_wr.next = NULL;
    ret = ibv_post_send(qp, &send_wr, &bad_send_wr);
    if (ret != 0)
    {
        printf("fail to post send, ret: %d\n", ret);
        return -1;
    }
    printf("[OK] post send.\n");

    struct ibv_wc wc;
    int polled = 0;
    for (int i = 0; i < 3; ++i)
    {
        ret = ibv_poll_cq(cq, 1, &wc);
        if (ret == 1)
        {
            polled = 1;
            break;
        }
        sleep(1);
    }
    if (polled == 0)
    {
        printf("fail to poll cq\n");
        return -1;
    }
    printf("[OK] poll cq.\n");
    printf("\twr id: %lu, opcode: %d, qp num: %u, status: %d\n", wc.wr_id, wc.opcode, wc.qp_num, wc.status);
    
    ret = ibv_dereg_mr(mr);
    if (ret != 0)
    {
        printf("fail to dereg mr\n");
        return -1;
    }
    printf("[OK] dereg mr.\n");

    free(buf);

    ret = ibv_destroy_qp(qp);
    if (ret != 0)
    {
        printf("fail to destroy qp\n");
        return -1;
    }
    printf("[OK] destroy qp.\n");

    ret = ibv_dealloc_pd(pd);
    if (ret != 0)
    {
        printf("fail to dealloc pd\n");
        return -1;
    }
    printf("[OK] dealloc pd.\n");

    ret = ibv_destroy_cq(cq);
    if (ret != 0)
    {
        printf("fail to destroy cq\n");
        return -1;
    }
    printf("[OK] destroy cq.\n");

    ret = ibv_close_device(ib_ctx);
    if (ret != 0)
    {
        printf("fail to close device %d\n", dev_idx);
        return -1;
    }
    printf("[OK] close device %d.\n", dev_idx);

    ibv_free_device_list(ib_devs);
    ret = shutdown(cs_info.cli_sock_fd, SHUT_RDWR);
    if (ret != 0)
    {
        printf("fail to shutdown socket\n");
        return -1;
    }
    ret = shutdown(ssctx.serv_sock_fd, SHUT_RDWR);
    if (ret != 0)
    {
        printf("fail to shutdown socket\n");
        return -1;
    }
    return 0;
}
```

`client.c`:

```c
#include <infiniband/verbs.h>
#include <stdio.h>
#include <stdlib.h>
#include "sock.h"
#include "metadata.h"
#include <unistd.h>

int main(int argc, const char **argv)
{
    int num_devs;
    struct ibv_device **ib_devs = ibv_get_device_list(&num_devs);
    if (ib_devs == NULL || num_devs == 0)
    {
        printf("fail to get ibv dev list\n");
        return -1;
    }
    printf("[OK] get %d ib devices.\n", num_devs);

    const char *dev_name;
    struct ibv_device *ib_dev;
    __be64 dev_guid;
    for (int i = 0; i < num_devs; i++)
    {
        ib_dev = ib_devs[i];
        dev_name = ibv_get_device_name(ib_dev);
        dev_guid = ibv_get_device_guid(ib_dev);
        printf("\tidx: %d, name: %s, guid (BE): %016llx\n", i, dev_name, dev_guid);
    }

    int dev_idx = 0; // open idx 0 ib device
    ib_dev = ib_devs[dev_idx];
    struct ibv_context *ib_ctx = ibv_open_device(ib_dev);
    if (ib_ctx == NULL)
    {
        printf("fail to open device %d\n", dev_idx);
        return -1;
    }
    printf("[OK] open device %d.\n", dev_idx);

    struct ibv_port_attr port_attr;
    int ret = ibv_query_port(ib_ctx, 1, &port_attr);
    if (ret != 0)
    {
        printf("fail to query port\n");
        return -1;
    }
    printf("[OK] query port.\n");
    printf("\tlid: %d\n", port_attr.lid);

    int min_cqe_num = 1;
    struct ibv_cq *cq = ibv_create_cq(ib_ctx, min_cqe_num, NULL, NULL, 0);
    if (cq == NULL)
    {
        printf("fail to create cq\n");
        return -1;
    }
    printf("[OK] create cq.\n");

    struct ibv_pd *pd = ibv_alloc_pd(ib_ctx);
    if (pd == NULL)
    {
        printf("fail to allocate pd\n");
        return -1;
    }
    printf("[OK] allocate pd.\n");

    struct ibv_qp_init_attr qp_init_attr;
    qp_init_attr.qp_type = IBV_QPT_RC;
    qp_init_attr.recv_cq = cq;
    qp_init_attr.send_cq = cq;
    qp_init_attr.cap.max_send_wr = 2;
    qp_init_attr.cap.max_recv_wr = 2;
    qp_init_attr.cap.max_send_sge = 2;
    qp_init_attr.cap.max_recv_sge = 2;
    struct ibv_qp *qp = ibv_create_qp(pd, &qp_init_attr);
    if (qp == NULL)
    {
        printf("fail to create qp\n");
        return -1;
    }
    printf("[OK] create qp.\n");
    printf("\tqp num: %d\n", qp->qp_num);

    struct ibv_qp_attr qp_attr;
    int attr_mask;
    qp_attr.qp_state = IBV_QPS_INIT;
    qp_attr.pkey_index = 0;
    qp_attr.port_num = 1;
    qp_attr.qp_access_flags = IBV_ACCESS_REMOTE_WRITE;
    attr_mask = IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS;
    ret = ibv_modify_qp(qp, &qp_attr, attr_mask);
    if (ret != 0)
    {
        printf("fail to modify qp to INIT state\n");
        return -1;
    }
    printf("[OK] modify qp to INIT state.\n");

    struct cli_sock_ctx csctx;
    init_cli_sock_ctx(&csctx, "127.0.0.1", 6543);

    struct conn_info conn_info;
    conn_info.qp_num = qp->qp_num;
    conn_info.lid = port_attr.lid;
    sock_cli_exchange_data(csctx.cli_sock_fd, &conn_info, sizeof(conn_info));
    printf("remote qp num: %u, lid: %u\n", conn_info.qp_num, conn_info.lid);

    qp_attr.qp_state = IBV_QPS_RTR;
    qp_attr.path_mtu = IBV_MTU_4096;
    qp_attr.dest_qp_num = conn_info.qp_num;
    qp_attr.rq_psn = 0;
    qp_attr.max_dest_rd_atomic = 1;
    qp_attr.min_rnr_timer = 12;
    qp_attr.ah_attr.port_num = 1;
    qp_attr.ah_attr.is_global = 0;
    qp_attr.ah_attr.sl = 0;
    qp_attr.ah_attr.src_path_bits = 0;
    qp_attr.ah_attr.dlid = conn_info.lid;
    attr_mask = IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN | IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER;
    ret = ibv_modify_qp(qp, &qp_attr, attr_mask);
    if (ret != 0)
    {
        printf("fail to modify qp to RTR state, ret: %d\n", ret);
        return -1;
    }
    printf("[OK] modify qp to RTR state.\n");

    qp_attr.qp_state = IBV_QPS_RTS;
    qp_attr.sq_psn = 0;
    qp_attr.timeout = 14;
    qp_attr.retry_cnt = 7;
    qp_attr.rnr_retry = 7;
    qp_attr.max_rd_atomic = 1;
    attr_mask = IBV_QP_STATE | IBV_QP_SQ_PSN | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY | IBV_QP_MAX_QP_RD_ATOMIC;
    ret = ibv_modify_qp(qp, &qp_attr, attr_mask);
    if (ret != 0)
    {
        printf("fail to modify qp to RTS state, ret: %d\n", ret);
        return -1;
    }
    printf("[OK] modify qp to RTS state.\n");

    const size_t buf_len = 128;
    void *buf = malloc(buf_len);
    struct ibv_mr *mr = ibv_reg_mr(pd, buf, buf_len, IBV_ACCESS_LOCAL_WRITE);
    // struct ibv_mr *mr = ibv_reg_mr(pd, buf, buf_len, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ);
    if (mr == NULL)
    {
        printf("fail to reg mr\n");
        return -1;
    }
    printf("[OK] reg mr.\n");
    printf("\tva: %p, len: %lu, lkey: %u, rkey: %u\n", buf, buf_len, mr->lkey, mr->rkey);

    struct mr_info mr_info;
    mr_info.buf_addr = (uint64_t) buf;
    mr_info.buf_len = buf_len;
    mr_info.rkey = mr->rkey;
    sock_cli_exchange_data(csctx.cli_sock_fd, &mr_info, sizeof(mr_info));
    
    struct ibv_sge sge;
    sge.addr = (uint64_t) buf;
    sge.length = buf_len;
    sge.lkey = mr->lkey;
    struct ibv_recv_wr recv_wr, *bad_recv_wr;
    recv_wr.wr_id = 54321;
    recv_wr.num_sge = 1;
    recv_wr.sg_list = &sge;
    recv_wr.next = NULL;
    ret = ibv_post_recv(qp, &recv_wr, &bad_recv_wr);
    if (ret != 0)
    {
        printf("fail to post recv, ret: %d\n", ret);
        return -1;
    }
    printf("[OK] post recv.\n");

    struct ibv_wc wc;
    int polled = 0;
    for (int i = 0; i < 3; ++i)
    {
        ret = ibv_poll_cq(cq, 1, &wc);
        if (ret == 1)
        {
            polled = 1;
            break;
        }
        sleep(1);
    }
    if (polled == 0)
    {
        printf("fail to poll cq\n");
        return -1;
    }
    printf("[OK] poll cq.\n");
    printf("\twr id: %lu\n", recv_wr.wr_id);
    printf("\tbuf: %s\n", (char*) buf);

    ret = ibv_dereg_mr(mr);
    if (ret != 0)
    {
        printf("fail to dereg mr\n");
        return -1;
    }
    printf("[OK] dereg mr.\n");

    free(buf);

    ret = ibv_destroy_qp(qp);
    if (ret != 0)
    {
        printf("fail to destroy qp\n");
        return -1;
    }
    printf("[OK] destroy qp.\n");

    ret = ibv_dealloc_pd(pd);
    if (ret != 0)
    {
        printf("fail to dealloc pd\n");
        return -1;
    }
    printf("[OK] dealloc pd.\n");

    ret = ibv_destroy_cq(cq);
    if (ret != 0)
    {
        printf("fail to destroy cq\n");
        return -1;
    }
    printf("[OK] destroy cq.\n");

    ret = ibv_close_device(ib_ctx);
    if (ret != 0)
    {
        printf("fail to close device %d\n", dev_idx);
        return -1;
    }
    printf("[OK] close device %d.\n", dev_idx);

    ibv_free_device_list(ib_devs);
    ret = shutdown(csctx.cli_sock_fd, SHUT_RDWR);
    if (ret != 0)
    {
        printf("fail to shutdown socket\n");
        return -1;
    }
    return 0;
}
```

`metadata.h`:

```c
#ifndef METADATA_H
#define METADATA_H

#include <stdint.h>

struct conn_info
{
    uint16_t lid;
    uint32_t qp_num;
};

struct mr_info
{
    uint64_t buf_addr;
    uint64_t buf_len;
    uint32_t rkey;
};

#endif
```

`sock.h`:

```c
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
```

`sock.c`:

```c
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
```

`Makefile`:

```makefile
all: server client 

server: server.c sock.o
	gcc -g server.c -libverbs sock.o -o server

client: client.c sock.o
	gcc -g client.c -libverbs sock.o -o client

sock.o: sock.h sock.c
	gcc -g -c sock.c -o sock.o

clean:
	rm -f server client sock.o
```

run:

```bash
./server
./client
```

output:

server:

```
[OK] get 4 ib devices.
	idx: 0, name: mlx5_0, guid (BE): 4a34ba0003d3ebe8
	idx: 1, name: mlx5_1, guid (BE): 4b34ba0003d3ebe8
	idx: 2, name: mlx5_2, guid (BE): ba2def0003fd7010
	idx: 3, name: mlx5_3, guid (BE): bb2def0003fd7010
[OK] open device 0.
[OK] query port.
	lid: 1
[OK] create cq.
[OK] allocate pd.
[OK] create qp.
	qp num: 189
[OK] modify qp to INIT state.
successfully create server sock fd: 5
successfully bind addr: 00000000, port: 6543
start to listen...
successfully accept clinet, ip: 0100007f, port: 25789
remote qp num: 190, lid: 1
[OK] modify qp to RTR state.
[OK] modify qp to RTS state.
[OK] reg mr.
	va: 0x5628c04a6870, len: 128
remote addr: 0x56080cd36870, len: 128, rkey: 263107
[OK] post send.
[OK] poll cq.
	wr id: 12345, opcode: 0, qp num: 189, status: 0
[OK] dereg mr.
[OK] destroy qp.
[OK] dealloc pd.
[OK] destroy cq.
[OK] close device 0.
```

client:

```
[OK] get 4 ib devices.
	idx: 0, name: mlx5_0, guid (BE): 4a34ba0003d3ebe8
	idx: 1, name: mlx5_1, guid (BE): 4b34ba0003d3ebe8
	idx: 2, name: mlx5_2, guid (BE): ba2def0003fd7010
	idx: 3, name: mlx5_3, guid (BE): bb2def0003fd7010
[OK] open device 0.
[OK] query port.
	lid: 1
[OK] create cq.
[OK] allocate pd.
[OK] create qp.
	qp num: 190
[OK] modify qp to INIT state.
successfully create client sock fd: 5
successfully connect to server, ip: 127.0.0.1, port: 6543
remote qp num: 189, lid: 1
[OK] modify qp to RTR state.
[OK] modify qp to RTS state.
[OK] reg mr.
	va: 0x56080cd36870, len: 128, lkey: 263107, rkey: 263107
[OK] post recv.
[OK] poll cq.
	wr id: 54321
	buf: hello from server

[OK] dereg mr.
[OK] destroy qp.
[OK] dealloc pd.
[OK] destroy cq.
[OK] close device 0.
```
