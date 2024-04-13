# Computer Network Note

## cache

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
        