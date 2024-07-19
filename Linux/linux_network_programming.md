# Linux Network Programming

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
