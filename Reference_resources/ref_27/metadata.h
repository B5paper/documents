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