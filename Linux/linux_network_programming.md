# Linux Network Programming

## cache

* peer connect

    `sock.h`:

    ```c
    #include <netinet/in.h>

    int sock_serv_sync(int cli_sock_fd);
    int sock_cli_sync(int cli_sock_fd);

    void sock_send(int fd, const void *buf, size_t buf_len);
    void sock_recv(int fd, void *buf, size_t buf_len);
    void sock_serv_exchange_data(int fd, void *buf, size_t buf_len);
    void sock_cli_exchange_data(int fd, void *buf, size_t buf_len);
    ```

    `sock.c`:

    ```c
    #include <stdio.h>
    #include <arpa/inet.h>
    #include <sys/socket.h>
    #include <stdlib.h>
    #include <string.h>
    #include <sys/select.h>
    #include <sys/socket.h>
    #include <arpa/inet.h>
    #include <stdlib.h>
    #include <stdio.h>
    #include <sys/time.h>
    #include <sys/types.h>
    #include <ifaddrs.h>
    #include <string.h>


    int get_my_ipv4_addr(char *my_ipv4_addr)
    {
        struct ifaddrs *if_addrs;
        int ret = getifaddrs(&if_addrs);
        if (ret < 0)
        {
            printf("fail to get if addrs\n");
            return -1;
        }

        char* if_name_black_list[] = {
            "lo",
            "docker"
        };
        int if_name_black_list_len = sizeof(if_name_black_list) / sizeof(char*);

        struct ifaddrs *ifa_node = if_addrs;
        int is_valid_ip_addr = 0;
        while (ifa_node)
        {
            int is_ifname_black_list_matched = 0;
            for (int i = 0; i < if_name_black_list_len; ++i)
            {
                char *if_name = if_name_black_list[i];
                int if_name_len = strlen(if_name);
                int ret = strncmp(ifa_node->ifa_name, if_name, if_name_len);
                if (ret == 0)
                {
                    is_ifname_black_list_matched = 1;
                    break;
                }
            }

            if (is_ifname_black_list_matched)
            {
                ifa_node = ifa_node->ifa_next;
                continue;
            }

            struct sockaddr_in *addr = (struct sockaddr_in *) ifa_node->ifa_addr;
            if (addr->sin_family != AF_INET)
            {
                ifa_node = ifa_node->ifa_next;
                continue;
            }

            char addr_str[16] = {0};
            inet_ntop(AF_INET, &addr->sin_addr.s_addr, addr_str, 16);
            // printf("name: %s, addr: %s\n", ifa_node->ifa_name, addr_str);
            strncpy(my_ipv4_addr, addr_str, 16);
            is_valid_ip_addr = 1;
            break;
        }
        freeifaddrs(if_addrs);

        if (!is_valid_ip_addr)
            return -1;
            
        return 0;
    }


    int peer_connect(int *serv_fd, int *cli_fd, int *i_am_server,
        char **node_ipv4_addrs, const uint16_t listen_port)
    {
        // length of node_ipv4_addrs is always 2

        int _serv_fd = socket(AF_INET, SOCK_STREAM, 0);
        if (_serv_fd < 0)
        {
            printf("fail to create sock fd\n");
            return -1;
        }
        printf("successfully create serv sock fd: %d\n", _serv_fd);

        int _cli_fd = socket(AF_INET, SOCK_STREAM, 0);
        if (_cli_fd < 0)
        {
            printf("fail to create fd\n");
            return -1;
        }
        printf("successfully create cli sock fd: %d\n", _cli_fd);

        uint32_t listen_addr_ipv4 = INADDR_ANY;
        struct sockaddr_in serv_sock_addr;
        serv_sock_addr.sin_family = AF_INET;
        serv_sock_addr.sin_addr.s_addr = listen_addr_ipv4;
        serv_sock_addr.sin_port = htons(listen_port);

        int ret = bind(_serv_fd, (struct sockaddr*) &serv_sock_addr, sizeof(serv_sock_addr));
        if (ret < 0)
        {
            printf("fail to bind\n");
            return -1;
        }
        printf("bind addr: %08x, port: %u\n", listen_addr_ipv4, listen_port);

        ret = listen(_serv_fd, 5);
        if (ret < 0)
        {
            printf("fail to listen\n");
            return -1;
        }
        printf("start to listen...\n");

        char my_ip_addr_str[16] = {0};
        ret = get_my_ipv4_addr(my_ip_addr_str);
        if (ret < 0)
        {
            printf("fail to get my ipv4 addr\n");
            return -1;
        }
        printf("my ipv4 addr: %s\n", my_ip_addr_str);

        char *peer_ip_addr = NULL;
        for (int i = 0; i < 2; ++i)
        {
            char *ip_addr = node_ipv4_addrs[i];
            int ret = strncmp(ip_addr, my_ip_addr_str, 16);
            if (ret == 0)
                continue;
            peer_ip_addr = ip_addr;
            break;
        }
        if (peer_ip_addr == NULL)
        {
            printf("fail to get peer node ip addr\n");
            return -1;
        }
        printf("peer node ip addr: %s\n", peer_ip_addr);

        uint64_t loop_idx = 0;
        int timeout_sec = 2;
        int timeout_ms = 100;
        while (1)
        {
            printf("start loop %lu\n", loop_idx++);

            fd_set fdset;
            FD_ZERO(&fdset);
            FD_SET(_serv_fd, &fdset);
            struct timeval timeout;
            timeout.tv_sec = timeout_sec;
            timeout.tv_usec = timeout_ms * 1000;
            ret = select(_serv_fd + 1, &fdset, NULL, NULL, &timeout);
            if (ret < 0)
            {
                printf("fail to select, ret: %d\n", ret);
                return -1;
            }

            if (FD_ISSET(_serv_fd, &fdset))
            {
                *i_am_server = 1;
                *serv_fd = _serv_fd;
                shutdown(_cli_fd, SHUT_RDWR);
                break;
            }
            printf("timeout reached, no client connect in\n");

            struct sockaddr_in peer_addr;
            peer_addr.sin_family = AF_INET;
            in_addr_t peer_ip_addr_u32net;
            inet_pton(AF_INET, peer_ip_addr, &peer_ip_addr_u32net);
            peer_addr.sin_addr.s_addr = peer_ip_addr_u32net;
            peer_addr.sin_port = htons(listen_port);
            printf("connect to peer addr: %s\n", peer_ip_addr);
            ret = connect(_cli_fd, (struct sockaddr *) &peer_addr, sizeof(peer_addr));
            if (ret == 0)
            {
                *i_am_server = 0;
                *cli_fd = _cli_fd;
                shutdown(_serv_fd, SHUT_RDWR);
                break;
            }
            printf("timeout reached, can not connect to server\n");
        }
        return 0;
    }

    void sock_send(int fd, const void *buf, size_t buf_len)
    {
        int bytes_send = send(fd, buf, buf_len, 0);
        if (bytes_send != buf_len)
        {
            printf("fail to send data via socket\n");
            printf("bytes_send: %d, buf_len: %lu\n", bytes_send, buf_len);
            exit(-1);
        }
    }

    void sock_recv(int fd, void *buf, size_t buf_len)
    {
        int bytes_recv = recv(fd, buf, buf_len, 0);
        if (bytes_recv != buf_len)
        {
            printf("fail to recv data via socket\n");
            printf("bytes_recv: %d, buf_len: %lu\n", bytes_recv, buf_len);
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

    ```

    这段代码主要使用 client, server 建立的微小时间差异来确立身份。

    可以见`ref_29`。

* 将 ipv4 地址从字符串转化为 uint32_t

    ```c
    struct in_addr inaddr;
    inet_pton(AF_INET, serv_addr, &inaddr);
    ```

    这个转换好直接就是大端序的，不需要再`htonl()`了。

    头文件：

    ```c
    #include <sys/socket.h>
    #include <netinet/in.h>
    #include <arpa/inet.h>
    ```

## topics
