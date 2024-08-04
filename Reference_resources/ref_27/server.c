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
