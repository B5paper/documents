# Computer Network Note

## cache

* 使用`getifaddrs()`拿到当前 host 的 network interface 信息

    `main.c`:

    ```c
    #include <stdio.h>
    #include <ifaddrs.h>
    #include <arpa/inet.h>  // struct sockaddr_in

    int main()
    {
        struct ifaddrs *if_addrs;
        int ret = getifaddrs(&if_addrs);
        if (ret < 0)
        {
            printf("fail to get if addrs\n");
            return -1;
        }

        struct ifaddrs *ifa_node = if_addrs;
        while (ifa_node)
        {
            printf("ifa name: %s\n", ifa_node->ifa_name);

            struct sockaddr_in *addr = (struct sockaddr_in *) ifa_node->ifa_addr;
            printf("\tsin_family: %d\n", addr->sin_family);

            char addr_str[16] = {0};
            inet_ntop(addr->sin_family, &addr->sin_addr.s_addr, addr_str, 16);
            printf("\taddr: %s\n", addr_str);

            ifa_node = ifa_node->ifa_next;
        }

        freeifaddrs(if_addrs);
        
        return 0;
    }
    ```

    compile: `gcc -g main.c -o main`

    run: `./main`

    output:

    ```
    ifa name: lo
    	sin_family: 17
    	addr: 
    ifa name: enp0s3
    	sin_family: 17
    	addr: 
    ifa name: docker0
    	sin_family: 17
    	addr: 
    ifa name: lo
    	sin_family: 2
    	addr: 127.0.0.1
    ifa name: enp0s3
    	sin_family: 2
    	addr: 10.0.2.4
    ifa name: docker0
    	sin_family: 2
    	addr: 172.17.0.1
    ifa name: lo
    	sin_family: 10
    	addr: ::
    ifa name: enp0s3
    	sin_family: 10
    	addr: 
    ```

    `getifaddrs()`在内部创建一个链表，并返回链表头的指针`struct ifaddrs *`。我们可以用遍历链表的方式得到每个 interface 的信息。

    上面出现的 address family 数值与枚举值的对应关系：

    | enum | value |
    | - | - |
    | `AF_INET` | 2 |
    | `AF_INET6` | 10 |
    | `AF_PACKET` | 17 |

* shutdown 和 close 都无法立即重新将同一个 fd bind 到一个 address + port 上

* 假设 node 1 上有 vm 1，node 2 上有 vm 2。vm 1 无法 ping 通 vm 2 可能是因为只设置了 node 1 上的路由表，没有设置 node 2 上的路由表

    猜想：可能是 vm 1 缎带 vm 2 发送完 icmp 数据包后，vm 2 回复 icmp 包时，找不到 vm 1 所在的网段如何路由过去。

    可以将 node 1 上的路由表新添加一项：vm 2 所在网段的 gateway 为 node 2 的 ip；在 node 2 上的路由表上也新加一条：vm 1 所在网段的 gateway 为 node 1 的 ip。这样就能 ping 通了。

* linux socket programming

    `server.c`:

    ```c
    #include <sys/socket.h>
    #include <arpa/inet.h>
    #include <string.h>
    #include <stdio.h>
    #include <unistd.h>

    int main()
    {
        int serv_fd = socket(AF_INET, SOCK_STREAM, 0);
        if (serv_fd < 0)
        {
            printf("fail to create server sock fd\n");
            return -1;
        }
        printf("[OK] create server socket fd: %d\n", serv_fd);

        uint16_t listen_port = 6543;
        uint32_t listen_addr_ipv4 = INADDR_ANY;
        char ipv4_addr[16] = {0};
        const char *ret_ptr = inet_ntop(AF_INET, &listen_addr_ipv4, ipv4_addr, 16);
        if (ret_ptr == NULL)
        {
            printf("fail to convert u32 to ipv4 str\n");
            return -1;
        }

        struct sockaddr_in serv_addr;
        serv_addr.sin_family = AF_INET;
        serv_addr.sin_addr.s_addr = listen_addr_ipv4;
        serv_addr.sin_port = htons(listen_port);
        int ret = bind(serv_fd, (struct sockaddr*) &serv_addr, sizeof(serv_addr));
        if (ret < 0)
        {
            printf("fail to bind serv fd %d, ret: %d\n", serv_fd, ret);
            return -1;
        }
        printf("[OK] bind fd %d to addr %s: %u\n", serv_fd, ipv4_addr, listen_port);

        ret = listen(serv_fd, 5);
        if (ret < 0)
        {
            printf("fail to listen\n");
            return -1;
        }
        printf("[OK] start to listen...\n");

        struct sockaddr_in cli_addr;
        socklen_t cli_addr_len = sizeof(cli_addr);
        int cli_fd = accept(serv_fd, (struct sockaddr*) &cli_addr, &cli_addr_len);
        if (cli_fd < 0)
        {
            printf("fail to accept, ret: %d\n", cli_fd);
            return -1;
        }
        printf("[OK] accept 1 incoming client.\n");

        ret_ptr = inet_ntop(AF_INET, &cli_addr.sin_addr.s_addr, ipv4_addr, 16);
        if (ret_ptr == NULL)
        {
            printf("fail to convert u32 ipv4 to string\n");
            return -1;
        }
        printf("\tincoming client: ip: %s, port: %u\n", ipv4_addr, cli_addr.sin_port);

        char *buf = "hello from server";
        size_t buf_len = strlen(buf) + 1;
        ssize_t bytes_send = send(cli_fd, buf, buf_len, 0);
        if (bytes_send != buf_len)
        {
            printf("fail to send, buf_len: %lu, bytes_send: %ld\n", buf_len, bytes_send);
            return -1;
        }
        printf("[OK] send buf: %s\n", buf);

        ret = shutdown(cli_fd, SHUT_RDWR);
        if (ret != 0)
        {
            printf("fail to shutdown client fd %d, ret: %d\n", cli_fd, ret);
            return -1;
        }
        printf("[OK] shutdown client fd %d.\n", cli_fd);

        ret = close(cli_fd);
        if (ret != 0)
        {
            printf("fail to close fd %d\n", cli_fd);
            return -1;
        }
        printf("[OK] close fd %d.\n", cli_fd);

        ret = shutdown(serv_fd, SHUT_RDWR);
        if (ret != 0)
        {
            printf("fail to shutdown server fd %d, ret: %d\n", serv_fd, ret);
            return -1;
        }
        printf("[OK] shutdown server fd %d.\n", serv_fd);

        ret = close(serv_fd);
        if (ret != 0)
        {
            printf("fail to close fd %d\n", serv_fd);
            return -1;
        }
        printf("[OK] close fd %d.\n", serv_fd);

        return 0;
    }
    ```

    `client.c`:

    ```c
    #include <sys/socket.h>
    #include <arpa/inet.h>
    #include <string.h>
    #include <stdio.h>
    #include <unistd.h>

    int main()
    {
        int cli_fd = socket(AF_INET, SOCK_STREAM, 0);
        if (cli_fd < 0)
        {
            printf("fail to create client sock fd\n");
            return -1;
        }
        printf("[OK] create client socket fd: %d\n", cli_fd);

        uint16_t serv_port = 6543;
        const char serv_ipv4[16] = "127.0.0.1";
        struct in_addr ipv4_addr;
        int ret = inet_pton(AF_INET, serv_ipv4, &ipv4_addr);
        if (ret != 1)
        {
            printf("fail to convert ipv4 string to u32, ret: %d\n", ret);
            return -1;
        }

        struct sockaddr_in serv_addr;
        serv_addr.sin_family = AF_INET;
        serv_addr.sin_addr = ipv4_addr;
        serv_addr.sin_port = htons(serv_port);
        ret = connect(cli_fd, (struct sockaddr*) &serv_addr, sizeof(serv_addr));
        if (ret != 0)
        {
            printf("fail to connect to server, ret: %d\n", ret);
            return -1;
        }
        printf("[OK] connect to server %s: %u\n", serv_ipv4, serv_port);

        char buf[20] = {0};
        size_t buf_len = 20;
        ssize_t bytes_recv = recv(cli_fd, buf, buf_len, 0);
        if (bytes_recv <= 0)
        {
            printf("fail to recv, buf_len: %lu, bytes_recv: %ld\n", buf_len, bytes_recv);
            return -1;
        }
        printf("[OK] recv buf: %s\n", buf);

        ret = shutdown(cli_fd, SHUT_RDWR);
        if (ret != 0)
        {
            printf("fail to shutdown fd %d, ret: %d\n", cli_fd, ret);
            return -1;
        }
        printf("[OK] shutdown fd %d.\n", cli_fd);

        ret = close(cli_fd);
        if (ret != 0)
        {
            printf("fail to close fd %d, ret: %d\n", cli_fd, ret);
            return -1;
        }
        printf("[OK] close fd %d.\n", cli_fd);

        return 0;
    }
    ```

    `Makefile`:

    ```makefile
    all: server client

    server: server.c
        gcc -g server.c -o server

    client: client.c
        gcc -g client.c -o client

    clean:
        rm -f server client
    ```

    compile: `make`

    run:

    `./server`

    `./client`

    output:

    * server end

        ```
        [OK] create server socket fd: 3
        [OK] bind fd 3 to addr 0.0.0.0: 6543
        [OK] start to listen...
        [OK] accept 1 incoming client.
            incoming client: ip: 127.0.0.1, port: 61611
        [OK] send buf: hello from server
        [OK] shutdown client fd 4.
        [OK] close fd 4.
        [OK] shutdown server fd 3.
        [OK] close fd 3.
        ```

    * client end

        ```
        [OK] create client socket fd: 3
        [OK] connect to server 127.0.0.1: 6543
        [OK] recv buf: hello from server
        [OK] shutdown fd 3.
        [OK] close fd 3.
        ```

    说明：

    * `sys/socket.h`文件中主要包含下面几个函数：`socket()`, `bind()`, `connect()`, `send()`, `recv()`, `listen()`, `accept()`, `shutdown()`

        这些函数组成了 socket 的基本功能。

        这里的 socket 并不完全是为 internet 设计的，有一些 unix domain 或者其他的 socket 也会用到这个库。

    * 如果有 Internet 相关的需要，还需要添加头文件`#include <arpa/inet.h>`

        宏`INADDR_ANY`，`htons()`, `inet_ntop()`等函数都包含在这个头文件内。

    * `shutdown()`和`close()`并不能使刚 bind ipv4 addr 的 fd 重新 bind 相同的 ipv4 addr。

    * `inet_pton()`第三个参数注意填的是长度的指针，不是长度的值

* `select()`会更新 timeout 的值，更新的值为`total timeout - blocking time`

* 将 net 大小端的 32 位 addr 转换成 string

    ```c
    #include <arpa/inet.h>

    char addr_str[16] = {0};
    inet_ntop(AF_INET, &client_addr.sin_addr.s_addr, addr_str, 16);
    ```

* 无法确定 client / server 身份时的一个解决方案

    我们让 node 0 交替地充当 client 和 server 的角色，让 node 1 也交替地充当 clinet 和 server 的角色，如下图所示：

    <div style='text-align:center'>
    <img height=400 src='/home/hlc/Documents/documents/Reference_resources/ref_29/pic_1.png'>
    </div>

    可以看到图中有红色虚线标注的间隙，只要有这种间隙存在，就一定会一方的角色是 client，另一方的角色是 server，此时便可成功建立 socket 连接。而这种间隙出现的概率很大。（具体概率为多少？是否可以用随机过程计算出来？）

* 获取本机的 ipv4 地址

    ```c
    #include <sys/types.h>
    #include <ifaddrs.h>
    #include <arpa/inet.h>
    #include <stdio.h>
    #include <string.h>

    char my_ipv4_addr[16] = {0};

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
            printf("name: %s, addr: %s\n", ifa_node->ifa_name, addr_str);
            strncpy(my_ipv4_addr, addr_str, 16);
            is_valid_ip_addr = 1;
            break;
        }
        freeifaddrs(if_addrs);

        if (!is_valid_ip_addr)
            return -1;

        return 0;
    }

    int main()
    {
        char my_ipv4_addr[16];
        int ret = get_my_ipv4_addr(my_ipv4_addr);
        if (ret < 0)
        {
            printf("fail to get my ipv4 addr\n");
            return -1;
        }
        printf("my ip addr: %s\n", my_ipv4_addr);
        return 0;
    }
    ```

    编译：

    `gcc -g main.c -o main`

    运行：

    `./main`

    output:

    ```
    name: enp0s3, addr: 10.0.2.4
    my ip addr: 10.0.2.4
    ```

    说明：

    * `getifaddrs()`这个函数可以枚举当前系统中的所有 net interface

    * `inet_ntop()`可以将大端的 32 位整数转换成 ipv4 字符串

        看起来既可以将字符串作为函数值返回，也可以传入一段 buffer，将字符串写入到 buffer 内。

    * 一个 ipv4 字符串 buffer 只需要 16 字节就够用了

        `255.255.255.255`，`255`占 3 个字节，一共 4 组，占`3 * 4 = 12`个字节。再加上 3 个点，一共`12 + 3 = 15`个字节。额外再加一个`\0`表示字符串的末尾，所以一共 16 个字节。

* NIC 指网卡，Network interface controller

* max 地址表示为 12 个十六进制数

* MTU (maxmimum transmission unit，最大传输单元)指的是数据链路层上能通过的最大负载的大小，单位为字节

    标准心碎网的 MTU 为 1500。

    如果 IP 层有数据包要发送，而数据包的长度超过了 MTU，IP 层就要对数据包进行分片（fragmentation）操作。

* 缓存（cache）使用的是静态随机存储（static random access memory, SRAM）

* 路由表

    如果在路由表里只填网络接口，destination ip/network 和子网掩码，不填 gateway，那么说明对于指定的 ip/网段使用指定的 interface 进行收发包。

    `0.0.0.0`指任意 ip，其对应的子网掩码为`0.0.0.0`。

## note

网络字节序：

小端法：高位存高地址，低位存低地址

大端法：高位存低地址，低位存高地址

网络数据流采用大端字节序，但是 intel 的 cpu 通常采用小端字节序。

常用的一些函数：

```c++
#include <arpa/inet.h>

uint32_t htonl(uint32_t hostlong);
uint16_t htons(uint16_t hostshort);
uint32_t ntohl(uint32_t netlong);
uint16_t ntohs(uint16_t netshort);
```

h 表示 host，n 表示 network，l 表示 32 位整数，s 表示 16 位整数。

ip 字符串与二进制的转换：

```c++
#include <arpa/inet.h>
int inet_pton(int af, const char *src, void *dst);
const char *inet_ntop(int af, const void *src, char *dst, socklen_t size);
```

* `af`: `AF_INET`, `AF_INET6`

TCP 通信流程图：

三次握手：

1. 客户端：syn, 包号，每个包的最大长度
1. 服务端：syn, 包号，ack，每个包的最大长度
1. 客户端：ack 包号

四次挥手（因为每次都是半关闭）：

1. 客户端：fin, ack
1. 服务端：ack
1. 服务端：fin, ack
1. 客户端：ack

滑动窗口（TCP 流量控制）：保证数据不会丢失。

**多进程服务器**：使用`fork()`创建子进程，使用信号捕捉函数`SIGCHLD`回收子进程。

`sigaction()`

TCP 状态时序图：

1. 主动发起连接请求端：

    1. CLOSE
    1. 发送 SYN
    1. SEND_SYN
    1. 接收 ACK, SYN
    1. SEND_SYN
    1. 发送 ACK
    1. ESTABLISHED（数据通信态）

1. 主动关闭连接请求端

    1. ESTABLISHED
    1. 发送 FIN
    1. FIN_WAIT_1
    1. 接收 ACK
    1. FIN_WAIT_2（半关闭）
    1. 接收对端发送 FIN
    1. FIN_WAIT_2（半关闭）
    1. 回发 ACK
    1. TIME_WAIT（只有主动关闭连接方会有这个状态）
    1. 等 2MSL 时长
    1. CLOSE

1. 被动接收连接请求端

    1. CLOSE
    1. LISTEN
    1. 接收 SYN
    1. LISTEN
    1. 发送 ACK, SYN
    1. SYN_RCVD
    1. 接收 ack
    1. etablished

1. 被动关闭连接

    1. established
    1. 接收 fin
    1. extablished
    1. 发送 ack
    1. close_wait (说明对端处于半关闭状态)
    1. 发送 fin
    1. last_ack
    1. 接收 ack
    1. close

2MSL 意义：保证最后一个 ack 能成功被对端接收。（等待期间，对端没收到我发的 ack，对端会再发 fin）因此它一定出现在主动关闭连接请求端。

`shutdown(int sockfd, int how)`可以关闭读缓冲或写缓冲。

`shutdown()`在关闭多个文件描述符应用的文件时，采用全关闭方法。`close()`只关闭一个。

`dup2()`

## 多路 IO 转接

* select

    `int select(int nfds, fd_set *readfds, fd_set *writefds, fd_set *excepfds, struct timeval *timeout);`

    * `nfds`：最大文件描述符加 1。（`lfd+1`，每次有新连接时更新）
    * `writefds`和`exceptfds`通常设置为`NULL`。

    返回所有监听集合中，满足对应事件的总数。

    ```c++
    FD_SET();
    FD_CLR();
    FD_ZERO();
    FD_ISSET();
    ```

    `lfd`和`cfd`都可以放到`readfds`里。

    `cfd`从`lfd+1`开始，一直到最大文件描述符。

    ```c++
    for (int i = lfd + 1; i <= maxfd; ++i)
    {
        FD_ISSET(i, &rset);
        read();
        // ...
        write();
    }
    ```

    可以用一个数组来维护已经存在的文件描述符，这样可以避免轮循，提高效率。

    优点：

    1. 跨平台。

    缺点：

    1. 文件描述符最大为 1023（`FD_SETSIZE = 1024`）。

* poll

    `int poll(struct pollfd *fds, nfds_t nfds, int timeout);`

    * `fds`：要监听的文件描述符的数组

    返回满足条件的事件数。

    优点：

    1. 自带数组结构，可以将监听事件集合和返回事件集合分离。

    1. 拓展监听上限，超出 1024 限制。

    缺点：

    1. 不能跨平台。

    1. 无法直接定位满足监听条件的文件描述符。

* epoll

    `epoll_create();`

    `epoll_ctl();`

    * `EPOLL_CTL_ADD`
    * `EPOLL_CTL_MOD`
    * `EPOLL_CTL_DEL`

    `epoll_wait()`

    * `EPOLLIN`, `EPOLLOUT`, `EPOLLERR`

    `cat /proc/sys/fs/file-max`：当前计算机一个进程可以打开的文件描述符上限。默认为 1024。

    `sudo vim /etc/security/limits.conf`可修改打开的文件描述符上限。

    ```
    * soft nofile 65536
    * hard nofile 100000
    ```

    也可直接用`ulimit`来修改。

    `ulimit -a`：当前用户的进程默认打开的文件描述符上限。

    ET (Edge Triggered)：边缘触发。只有数据到来才触发，不管缓冲区中是否还有数据。

    LT (Level Triggered)：水平触发只要有数据都会触发。(默认模式)

    结论：

    epoll 的 et 模式是高效模式，但只支持非阻塞模式（非阻塞的 socket）。阻塞模式下使用`readn()`/`readline()`等函数时，可能会发生死锁。

    优点：

    高效。突破 1024 文件描述符。

    缺点：

    1. 不能跨平台，只支持 linux。

`offsetof()`

`unlink()`

## 查看 ssl 版本

查看 ssl 版本：

For TLS 1.2:

`openssl s_client -connect www.google.com:443 -tls1_2`

For TLS 1.1:

`openssl s_client -connect www.google.com:443 -tls1_1`

For TLS 1:

`openssl s_client -connect www.google.com:443 -tls1`

或者使用 nmap：`nmap --script ssl-enum-ciphers -p 443 www.google.com`

参考资料：<https://stackoverflow.com/questions/40557031/command-prompt-to-check-tls-version-required-by-a-host>

## Proxy

Materials:

1. <https://github.com/NOLFXceptMe/http>

1. <https://github.com/monkeyWie/proxyee>

1. 
        