# Linux Programming Note

此笔记记录 linux 下的 user space 内的各种 app 的写法。

如果涉及到 kernel space 相关的编程，可以把放到 linux driver note 中。

## cache

* `pthread_once()`的用法

    `pthread_once()`可以保证在多线程环境下，指定的函数可以只被执行一次。

    原型：

    ```c
    int pthread_once(
        pthread_once_t *once_control,
        void (*init_routine)()
    );
    ```

    example:

    `main.c`:

    ```c
    #include <pthread.h>
    #include <stdio.h>
    #include <unistd.h>

    int global_cnt = 0;

    pthread_once_t once_var = PTHREAD_ONCE_INIT;
    void thd_once_func()
    {
        ++global_cnt;
    }

    void* thd_func(void *arg)
    {
        pthread_once(&once_var, thd_once_func);
        return NULL;
    }

    int main()
    {
        pthread_t thds[5];
        int num_thds = 5;
        for (int i = 0; i < num_thds; ++i)
        {
            pthread_create(&thds[i], NULL, thd_func, NULL);
        }

        for (int i = 0; i < num_thds; ++i)
        {
            pthread_join(thds[i], NULL);
        }

        printf("global cnt: %d\n", global_cnt);

        return 0;
    }
    ```

    compile: `gcc -g main.c -o main`

    run: `./main`

    output:

    ```
    global cnt: 1
    ```

    说明：

    1. `pthread_once()`靠`once_control`来控制只执行一次`init_routine()`函数。

    1. `once_control`的初始值必须为`PTHREAD_ONCE_INIT`。

        `PTHREAD_ONCE_INIT`对应的整数值为 0。经实验，如果将`once_control`初始值设置为`1`，那么程序将卡住。如果`once_control`被设置为除了 0, 1 以外的任何值，那么`init_routine()`将一次都不会被执行。

    1. `init_routine()`的返回值为`void`，参数列表也为`void`（无参数）。

        因此这个函数主要是拿来初始化一些全局变量，比如 mutex，cond 之类的。

    1. 这个功能可以使用 mutex 和 cond 完成吗？

        首先，如果使用 mutex 或 cond，我们必须让 mutex 或 cond 在每个线程/进程中都要初始化，因为当在多台机器上启动多个进程时，我们完全无法掌控进程启动的先后顺序。

        其次，我们无法使用 cond，因为我们不知道哪个线程用来 wait，哪个线程用来 signal。这样我们只剩下 mutex 可以用了，但是事实证明 mutex 也不好使。

        我们可以写出下面的反例代码：

        ```c
        #include <pthread.h>
        #include <stdio.h>
        #include <unistd.h>
        #include <unistd.h>

        int global_cnt = 0;

        pthread_mutex_t mtx;
        int cond_var = 0;

        void* thd_func(void *arg)
        {
            pthread_mutex_init(&mtx, NULL);

            pthread_mutex_lock(&mtx);
            sleep(1);
            if (cond_var == 0)
            {
                global_cnt++;
                cond_var = 1;
            }
            pthread_mutex_unlock(&mtx);

            pthread_mutex_destroy(&mtx);  

            return NULL;
        }

        int main()
        {
            pthread_t thds[5];
            int num_thds = 5;
            for (int i = 0; i < num_thds; ++i)
            {
                pthread_create(&thds[i], NULL, thd_func, NULL);
            }
            
            for (int i = 0; i < num_thds; ++i)
            {
                pthread_join(thds[i], NULL);
            }

            printf("global cnt: %d\n", global_cnt);

            return 0;
        }
        ```

        运行程序，会直接卡死。

        当一个线程中 mtx 被 lock 后，另一个线程对 mtx 进行 init，那么第二个线程也可以顺利 lock。这样就导致了结果出错。

        这样一来，大部分线索就断了，不清楚`pthread_once()`是如何实现的。猜测可能用了`pthread_mutex_trylock()`之类的方法。

* `inet_pton()`的返回值

    返回 1 表示函数调用成功，返回 0 表示字符串不符合规范，返回 -1 表示 address family 不识别，并会设置`errno`的值。

    example:

    `main.c`:

    ```c
    #include <arpa/inet.h>
    #include <stdio.h>
    #include <errno.h>

    int main()
    {
        int ret;
        int buf;

        ret = inet_pton(AF_INET, "127.0.0.1", &buf);
        printf("test 1, ret: %d, buf: %d, errno: %d\n", ret, buf, errno);

        ret = inet_pton(AF_INET, "127.001", &buf);
        printf("test 2, ret: %d, buf: %d, errno: %d\n", ret, buf, errno);

        ret = inet_pton(123, "127.0.0.1", &buf);
        printf("test 3, ret: %d, buf: %d, errno: %d\n", ret, buf, errno);

        return 0;
    }
    ```

    output:

    ```
    test 1, ret: 1, buf: 16777343, errno: 0
    test 2, ret: 0, buf: 16777343, errno: 0
    test 3, ret: -1, buf: 16777343, errno: 97
    ```

* `recv(sockfd, buf, len, flags);`等价于`recvfrom(sockfd, buf, len, flags, NULL, NULL);`

* 一个标准的 udp socket 的写法

    `server.c`:

    ```c
    #include <sys/socket.h>
    #include <arpa/inet.h>
    #include <stdio.h>
    #include <errno.h>  // errno
    #include <unistd.h>  // close()

    int main()
    {
        int serv_fd = socket(AF_INET, SOCK_DGRAM, 0);
        if (serv_fd < 0)
        {
            printf("fail to create serv fd, ret: %d\n", serv_fd);
            return -1;
        }
        printf("successfully create serv fd %d\n", serv_fd);
        
        struct sockaddr_in serv_addr;
        serv_addr.sin_family = AF_INET;
        const char *serv_ip_str = "127.0.0.1";
        int ret = inet_pton(AF_INET, serv_ip_str, &serv_addr.sin_addr.s_addr);
        if (ret < 0)
        {
            printf("fail to convert ip str %s to int\n", serv_ip_str);
            return -1;
        }
        int serv_port = 1234;
        serv_addr.sin_port = htons(serv_port);
        ret = bind(serv_fd, (struct sockaddr*) &serv_addr, sizeof(serv_addr));
        if (ret < 0)
        {
            printf("fail to bind serv fd: %d\n", serv_fd);
            return -1;
        }
        printf("successfully bind serv fd %d to addr %s: %d\n",
            serv_fd, serv_ip_str, serv_port);

        char buf[256];
        size_t buf_len = 256;
        struct sockaddr_in cli_addr;
        socklen_t addr_len = sizeof(cli_addr);
        ssize_t bytes_recv = recvfrom(serv_fd, buf, buf_len, 0,
            (struct sockaddr*) &cli_addr, &addr_len);
        if (bytes_recv <= 0)
        {
            printf("fail to recv, ret: %ld, errno: %d\n", bytes_recv, errno);
            return -1;
        }
        char cli_ip_str[16] = {0};
        inet_ntop(AF_INET, &cli_addr.sin_addr, cli_ip_str, 16);
        uint16_t cli_port = ntohs(cli_addr.sin_port);
        printf("recv %ld bytes from %s, port %u:\n",
            bytes_recv, cli_ip_str, cli_port);
        printf("\t%s\n", buf);
        
        close(serv_fd);
        return 0;
    }
    ```

    `client.c`:

    ```c
    #include <sys/socket.h>
    #include <arpa/inet.h>
    #include <stdio.h>
    #include <errno.h>  // errno
    #include <unistd.h>  // close()

    int main()
    {
        int cli_fd = socket(AF_INET, SOCK_DGRAM, 0);
        if (cli_fd < 0)
        {
            printf("fail to create cli sock fd\n");
            return -1;
        }
        printf("create cli fd: %d\n", cli_fd);

        struct sockaddr_in serv_addr;
        serv_addr.sin_family = AF_INET;
        const char *serv_ip_str = "127.0.0.1";
        int ret = inet_pton(AF_INET, serv_ip_str, &serv_addr.sin_addr);
        if (ret < 0)
        {
            printf("fail to convert serv ip str %s to int, ret: %d\n", serv_ip_str, ret);
            return -1;
        }
        int serv_port = 1234;
        serv_addr.sin_port = htons(serv_port);

        char buf[128] = "hello from client";
        size_t buf_len = 128;
        ssize_t bytes_send = sendto(cli_fd, buf, buf_len, 0, (struct sockaddr *) &serv_addr, sizeof(serv_addr));
        if (bytes_send <= 0)
        {
            printf("fail to send, ret: %ld, errno: %d\n", bytes_send, errno);
            return -1;
        }
        printf("send %ld bytes\n", bytes_send);

        close(cli_fd);
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

    1. `./server`

    2. `./client`

    output:

    * server end

        ```
        successfully create serv fd 3
        successfully bind serv fd 3 to addr 127.0.0.1: 1234
        recv 128 bytes from 127.0.0.1, port 60160:
        	hello from client
        ```

    * client end

        ```
        create cli fd: 3
        send 128 bytes
        ```

    说明：

    1. 由于是 udp，所以 server 端不需要 listen，也不需要 accept，但是需要 bind。

    1. server 调用`recvfrom()`后，会进入阻塞状态，接收到 client 的信息后，连接即断开。因此`recvfrom()`不会返回 0. （如果 client 发送 length 为 0 的信息，这个函数会不会返回 0 呢？）

    1. 如果 server 没有调用`recvfrom()`，client 直接发送`sendto()`，那么 client 端依然会返回发送成功。并且 client 端没有办法知道`sendto()`的消息是否成功发送到 server。

    1. 如果 server 端准备的 buffer length 有限，那么 client 端的`sendto()`依然会显示所有的 buffer 都发送成功，剩余的 server 没有收到的数据会被 drop。

    1. 因为 udp 是无连接的，所以不需要`shutdown()`关闭连接，但是仍然需要`close(fd)`回收进程的 fd 资源。

    1. 因为上述的`./server`和`./client`是不同的进程，所以`fd`都是从 3 开始分配，互不影响

    1. `recvfrom()`和`sendto()`的参数 flag 对 udp 没有什么影响，通常置 0 就可以。

* close socket 的注意事项

    * server 与 client 任意一端 shutdown(cli_fd)，对端如果处于`recv()`状态，`recv()`的返回值都为 0.

    * server 端发起`shutdown(cli_fd)`，client `recv()` 0 长度 buffer 后，`shutdown(cli_fd)`，此时 server 端再`shutdown(serv_fd)`，socket 仍无法正常退出，表现为 server 重新启动时，无法立即重新绑定 ip: port。

        因此，close connection 必须由 client 端先发起，才能正常关闭 socket。

* socket 关闭后可以立即 bind 的条件

    通常情况下一个 socket server 断开连接后，如果没有正确清理资源，那么会导致 server socket fd 无法立即 bind 到同一个 address 上，需要等大概半分钟才行。但是如果资源清理得当，是可以立即 bind 的，下面是条件：

    1. server 执行`accept()`, client 执行`connect()`，此时连接建立。

    2. client 执行`shutdown(cli_fd, SHUT_RDWR);`

    3. server 执行`shutdown(serv_fd, SHUT_RDWR);`

    4. 此时若关闭 server 程序，并立即重新启动 server，那么`serv_fd`可以成功 bind 到相同的 socket address 上。

    说明：

    1. 若第一步没有执行完成，连接没有建立，那么 server 可立即重新 bind

    2. 若连接已经建立，那么要求 client 执行`shutdown()`必须要在 server 之前。若 server 在 client 之前执行`shutdown(cli_fd, SHUT_RDWR);`, `shutdown(serv_fd, SHUT_RDWR);`，那么依然会无法重新 bind

    3. server 可以执行`shutdown(cli_fd, SHUT_RDWR);`，也可以不执行，不影响结果。

    总之，需要 client 主动发起 close，server 这边才能正常处理。

* 使用 pthread cond broadcast 通知所有的 cond

    `main.c`:

    ```c
    #include <pthread.h>
    #include <stdio.h>
    #include <unistd.h>

    pthread_cond_t cond;
    pthread_mutex_t mtx;

    void* thd_func(void *arg)
    {
        pthread_t thd = pthread_self();
        printf("thd %lu in thd_func()...\n", thd);
        pthread_mutex_lock(&mtx);
        pthread_cond_wait(&cond, &mtx);
        pthread_mutex_unlock(&mtx);
        printf("thd %lu exit thd_func().\n", thd);
        return NULL;
    }

    int main()
    {
        pthread_mutex_init(&mtx, NULL);
        pthread_cond_init(&cond, NULL);

        pthread_t thds[2];
        int num_thds = 2; 
        for (int i = 0; i < num_thds; ++i)
        {
            pthread_create(&thds[i], NULL, thd_func, NULL);
        }
        
        printf("start sleep...\n");
        sleep(2);
        printf("end sleep.\n");

        pthread_mutex_lock(&mtx);
        pthread_cond_broadcast(&cond);
        pthread_mutex_unlock(&mtx);

        for (int i = 0; i < num_thds; ++i)
        {
            pthread_join(thds[i], NULL);
        }
        return 0;
    }
    ```

    compile: `gcc -g main.c -o main`

    run: `./main`

    output:

    ```
    thd 133889997669952 in thd_func()...
    thd 133889987184192 in thd_func()...
    start sleep...
    end sleep.
    thd 133889987184192 exit thd_func().
    thd 133889997669952 exit thd_func().
    ```

    如果将`pthread_cond_broadcast()`換成`pthread_cond_signal()`，那么只会通知两个线程 cond wait 的其中一个，输出如下：

    ```
    start sleep...
    thd 135955300222528 in thd_func()...
    thd 135955289736768 in thd_func()...
    end sleep.
    thd 135955300222528 exit thd_func().

    ```

    可以看到，程序在这个地方卡住。

* pthread cond 中，如果先 signal，再 wait，那么 signal 是无效的。

    `main.c`:

    ```c
    #include <pthread.h>
    #include <stdio.h>
    #include <unistd.h>

    pthread_mutex_t mtx;
    pthread_cond_t cond;

    void* thread_func(void *arg)
    {
        printf("in thread_func()...\n");
        pthread_mutex_lock(&mtx);
        pthread_cond_signal(&cond);
        pthread_mutex_unlock(&mtx);
        printf("exit thread_func().\n");
        return NULL;
    }

    int main()
    {
        pthread_mutex_init(&mtx, NULL);
        pthread_cond_init(&cond, NULL);

        pthread_t thd;
        pthread_create(&thd, NULL, thread_func, NULL);

        printf("start sleep ...\n");
        sleep(2);
        printf("end sleep.\n");

        pthread_mutex_lock(&mtx);
        pthread_cond_wait(&cond, &mtx);
        pthread_mutex_unlock(&mtx);

        pthread_join(thd, NULL);

        pthread_cond_destroy(&cond);
        pthread_mutex_destroy(&mtx);
        return 0;
    }
    ```

    compile: `gcc -g main.c -o main`

    run: `./main`

    output:

    ```
    start sleep ...
    in thread_func()...
    exit thread_func().
    end sleep.

    ```

    程序会在这里卡住。可见正常的执行顺序应该是必须保证先 wait，后 signal。

    如果是先 signal 后就算立即进入了阻塞状态，比如`listen() -> signal -> accept()`，其他线程在 signal 后 wait，也会因为无法等到 signal 而永远阻塞。

    如果有一个什么机制，可以记录 signal 已经出现过了就好了。一个最简单的想法是用一个变量：

    ```c
    #include <pthread.h>
    #include <stdio.h>
    #include <unistd.h>

    pthread_mutex_t mtx;
    pthread_cond_t cond;
    int cond_val = 0;

    void* thread_func(void *arg)
    {
        printf("in thread_func()...\n");
        pthread_mutex_lock(&mtx);
        cond_val = 1;
        pthread_cond_signal(&cond);
        pthread_mutex_unlock(&mtx);
        printf("exit thread_func().\n");
        return NULL;
    }

    int main()
    {
        pthread_mutex_init(&mtx, NULL);
        pthread_cond_init(&cond, NULL);

        pthread_t thd;
        pthread_create(&thd, NULL, thread_func, NULL);

        printf("start sleep ...\n");
        sleep(2);
        printf("end sleep.\n");

        pthread_mutex_lock(&mtx);
        if (cond_val == 0)
            pthread_cond_wait(&cond, &mtx);
        pthread_mutex_unlock(&mtx);

        pthread_join(thd, NULL);

        pthread_cond_destroy(&cond);
        pthread_mutex_destroy(&mtx);
        return 0;
    }
    ```

    output:

    ```
    start sleep ...
    in thread_func()...
    exit thread_func().
    end sleep.
    ```

    此时程序即可正常结束。

    只有当`cond_val`为 0 时才去等待，当`cond_val`为 1 时，说明 signal 已经被触发过了。这样无论是 wait 先执行，还是 signal 先执行，都能保证子线程的 mutex 创造的临界区的下一条指令，一定先于主线程临界区的下一条指令执行。

    （这里使用了一个条件变量，可以保证一个线程先于另一个线程执行，那么如果使用多个 cond，或者多个 cond_var，或者多个 cond_val 的取值，是否可以实现让两个线程到达 barrier 后，同步开始执行？）

* socket 编程时，如果 server 端在退出程序前对 serv fd 进行了`shutdown()`，那么重新启动程序后可以立即 bind 同一个 ip 和 port。

* 当 client 主动 shutdown socket 时，`poll()`会收到一个正常的`POLLIN`事件。

* linux socket 编程中，如果 client 端主动发起`shutdown()`，那么 server 端在等待`recv()`时，会收到一条长度为 0 的数据，即`recv()`的返回值为`0`。

    example:

    `server.c`:

    ```c
    #include <sys/socket.h>
    #include <arpa/inet.h>
    #include <stdio.h>
    #include <stdlib.h>
    #include <string.h>

    int main()
    {
        int serv_fd = socket(AF_INET, SOCK_STREAM, 0);
        uint16_t listen_port = 6543;
        uint32_t listen_addr_ipv4 = INADDR_ANY;

        struct sockaddr_in serv_addr;
        serv_addr.sin_family = AF_INET;
        serv_addr.sin_addr.s_addr = listen_addr_ipv4;
        serv_addr.sin_port = htons(listen_port);
        bind(serv_fd, (struct sockaddr*) &serv_addr, sizeof(serv_addr));

        listen(serv_fd, 5);
        printf("start to listen...\n");

        struct sockaddr_in cli_addr;
        socklen_t cli_addr_len = sizeof(cli_addr);
        int cli_fd = accept(serv_fd, (struct sockaddr*) &cli_addr, &cli_addr_len);

        char buf[64] = {0};
        size_t buf_len = 64;
        ssize_t bytes_recv = recv(cli_fd, buf, buf_len, 0);
        if (bytes_recv <= 0)
        {
            printf("fail to recv, bytes_recv: %ld\n", bytes_recv);
            return -1;
        }
        printf("recv buf: %s\n", buf);

        bytes_recv = recv(cli_fd, buf, buf_len, 0);
        if (bytes_recv <= 0)
        {
            printf("fail to recv, bytes_recv: %ld\n", bytes_recv);
            return -1;
        }
        printf("recv buf: %s\n", buf);

        shutdown(cli_fd, SHUT_RDWR);
        shutdown(serv_fd, SHUT_RDWR);

        return 0;
    }
    ```

    run:

    `./server`, `./client`

    server output:

    ```
    start to listen...
    recv buf: hello, world
    fail to recv, bytes_recv: 0
    ```

    client output:

    ```
    [OK] connect to server 127.0.0.1: 6543
    [OK] send buf: hello, world
    ```

* 使用 poll 接收一个 client 的 socket connection

    `main.c`:

    ```c
    #include <poll.h>
    #include <stdio.h>
    #include <sys/socket.h>
    #include <pthread.h>
    #include <arpa/inet.h>
    #include <stdlib.h>
    #include <unistd.h>

    struct client_socks_info
    {
        int *fds;
        int len;
    };

    pthread_cond_t cond;
    pthread_mutex_t mtx;
    int cond_val = 0;

    void* thd_func_serv(void *arg)
    {
        int serv_sock_fd = socket(AF_INET, SOCK_STREAM, 0);
        if (serv_sock_fd < 0)
        {
            printf("fail to create socket\n");
            return -1;
        }

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
        int ret = bind(serv_sock_fd, (struct sockaddr*) &serv_addr, sizeof(serv_addr));
        if (ret < 0)
        {
            printf("fail to bind serv fd %d, ret: %d\n", serv_sock_fd, ret);
            return -1;
        }
        printf("[OK] bind fd %d to addr %s: %u\n", serv_sock_fd, ipv4_addr, listen_port);

        ret = listen(serv_sock_fd, 5);
        if (ret < 0)
        {
            printf("fail to listen\n");
            return -1;
        }
        printf("[OK] start to listen...\n");

        pthread_mutex_lock(&mtx);
        pthread_cond_signal(&cond);
        pthread_mutex_unlock(&mtx);

        struct pollfd poll_fd;
        poll_fd.fd = serv_sock_fd;
        poll_fd.events = POLLIN;
        int num_active_fds = poll(&poll_fd, 1, -1);

        if (poll_fd.revents & POLLIN)
        {
            struct sockaddr_in cli_addr;
            socklen_t cli_addr_len = sizeof(cli_addr);
            int cli_fd = accept(serv_sock_fd, (struct sockaddr*) &cli_addr, &cli_addr_len);
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
        }

        return NULL;
    }

    int main()
    {
        pthread_cond_init(&cond, NULL);
        pthread_mutex_init(&mtx, NULL);

        pthread_t thd_serv;
        pthread_create(&thd_serv, NULL, thd_func_serv, NULL);

        int cli_sock_fd = socket(AF_INET, SOCK_STREAM, 0);
        if (cli_sock_fd < 0)
        {
            printf("fail to create cli sock fd\n");
            return -1;
        }
        printf("[OK] create client socket fd: %d\n", cli_sock_fd);

        uint16_t serv_port = 6543;
        const char serv_ipv4[16] = "127.0.0.1";
        struct in_addr ipv4_addr;
        int ret = inet_pton(AF_INET, serv_ipv4, &ipv4_addr);
        if (ret != 1)
        {
            printf("fail to convert ipv4 string to u32, ret: %d\n", ret);
            return -1;
        }

        pthread_mutex_lock(&mtx);
        pthread_cond_wait(&cond, &mtx);
        pthread_mutex_unlock(&mtx);
        pthread_cond_destroy(&cond);
        pthread_mutex_destroy(&mtx);

        struct sockaddr_in serv_addr;
        serv_addr.sin_family = AF_INET;
        serv_addr.sin_addr = ipv4_addr;
        serv_addr.sin_port = htons(serv_port);
        ret = connect(cli_sock_fd, (struct sockaddr*) &serv_addr, sizeof(serv_addr));
        if (ret != 0)
        {
            printf("fail to connect to server, ret: %d\n", ret);
            return -1;
        }
        printf("[OK] connect to server %s: %u\n", serv_ipv4, serv_port);

        pthread_join(thd_serv, NULL);
        return 0;
    }
    ```

    compile: `gcc -g main.c -o main`

    run: `./main`

    output:

    ```
    [OK] create client socket fd: 3
    [OK] bind fd 4 to addr 0.0.0.0: 6543
    [OK] start to listen...
    [OK] connect to server 127.0.0.1: 6543
    [OK] accept 1 incoming client.
    	incoming client: ip: 127.0.0.1, port: 22149
    ```

    关于同步的问题：如果 clinet 在 server poll() 之前就尝试 connect，那么会直接失败。我们希望 server 在调用 poll() 之后，client 再 connect()。
    
    我们想到的一个最简单的办法是让 client 在 connect 之前先等着，等 server poll() 就绪后再往下走。我们很容易想到使用 pthread 提供的条件变量来实现这个功能，不需要设置`int cond_val;`，我们只需要使用 cond 最基本的 signal 功能就可以了。
    
    但是由于 poll 本身就是阻塞的，所以我们不可能在 poll 之后再 signal cond。那么往前移一步是否可以呢？答案是可以的，因为只需要调用 listen() 之后，client 实际上已经可以开始 connect 了，而 listen 是非阻塞的。

    是否 linux 的设计者也考虑到了这个问题，才把非阻塞的 listen 和阻塞的 accept / poll 拆分成两个功能来写呢？这种拆分是否还有背后的计算机理论支撑呢，比如给定某种判断方法，我们就可以判断如果要引入同步机制，那么哪些函数是一定要折开写的，哪些是可以不用拆开？

* pthread 与 conditional variable

    一个最小可跑通的例子：

    `main.c`:

    ```c
    #include <pthread.h>
    #include <stdio.h>
    #include <unistd.h>

    pthread_cond_t cond;
    pthread_mutex_t mtx;

    void* thd_func(void *arg)
    {
        printf("in thd_func()...\n");
        pthread_mutex_lock(&mtx);  // without this line the program will be hanging
        pthread_cond_wait(&cond, &mtx);
        printf("exit thd_func().\n");
        return NULL;
    }

    int main()
    {
        pthread_t thd;

        pthread_mutex_init(&mtx, NULL);
        pthread_cond_init(&cond, NULL);

        pthread_create(&thd, NULL, thd_func, NULL);

        printf("start sleep...\n");
        sleep(2);
        printf("end sleep.\n");
        
        pthread_cond_signal(&cond);
        
        pthread_join(thd, NULL);
        return 0;
    }
    ```

    output:

    ```
    start sleep...
    in thd_func()...
    end sleep.
    exit thd_func().
    ```

    `pthread_cond_init()`用于初始化一个条件变量，`pthread_cond_wait()`用于等待 cond 被激活，`pthread_cond_signal()`用于激活 cond。

    `pthread_cond_wait()`需要传入一个已经 lock 的 mutex，如果在调用`pthread_cond_wait()`之前没有调用`pthread_mutex_lock(&mtx);`，那么程序会卡死。

    上面的 example 并不是经典用法，下面的才是经典用法：

    `main.c`:

    ```c
    #include <pthread.h>
    #include <stdio.h>
    #include <unistd.h>

    pthread_cond_t cond;
    pthread_mutex_t mtx;
    int cond_var = 0;

    void* thd_func(void *arg)
    {
        printf("in thd_func()...\n");
        pthread_mutex_lock(&mtx);
        while (cond_var != 2)        
            pthread_cond_wait(&cond, &mtx);
        pthread_mutex_unlock(&mtx);
        printf("exit thd_func().\n");
        return NULL;
    }

    int main()
    {
        pthread_t thd;

        pthread_mutex_init(&mtx, NULL);
        pthread_cond_init(&cond, NULL);

        pthread_create(&thd, NULL, thd_func, NULL);

        cond_var = 1;
        pthread_mutex_lock(&mtx);
        pthread_cond_signal(&cond);
        pthread_mutex_unlock(&mtx);

        printf("start sleep...\n");
        sleep(2);
        printf("end sleep.\n");

        cond_var = 2;
        pthread_mutex_lock(&mtx);
        pthread_cond_signal(&cond);
        pthread_mutex_unlock(&mtx);

        pthread_join(thd, NULL);
        return 0;
    }
    ```

    output:

    ```
    start sleep...
    in thd_func()...
    end sleep.
    exit thd_func().
    ```

    这里使用锁来保证不会出错。
    
    问题：如果删去`main()`中的锁，可能会发生什么？如果`pthread_cond_wait()`不接收锁，可能会发生什么？是否可以使用条件变量实现信号量？

* 是否可以 unlock 一个未 lock 的 mutex？

    答案是不可以。

    `main.c`:

    ```c
    #include <pthread.h>
    #include <stdio.h>
    #include <unistd.h>

    pthread_mutex_t mtx;

    void* thread_func_wait(void *arg)
    {
        printf("in thread_func_wait()...\n");
        pthread_mutex_unlock(&mtx);
        printf("exit thread_func_wait().\n");
        return NULL;
    }

    int main()
    {
        pthread_mutex_init(&mtx, NULL);
        pthread_t thd;
        pthread_create(&thd, NULL, thread_func_wait, NULL);
        printf("start sleep ...\n");
        sleep(2);
        printf("end sleep.\n");
        pthread_mutex_lock(&mtx);
        pthread_join(thd, NULL);
        return 0;
    }
    ```

    output:

    ```
    start sleep ...
    in thread_func_wait()...
    exit thread_func_wait().
    end sleep.
    ```

    程序想要使用`pthread_mutex_unlock()`做一个 wait 操作，但是 unlock 的是一个未 lock 的 mutex，此时我们根据 output 看到 thread 函数直接返回了，并没有等待。因此不可以 unlock 一个未 lock 的 mutex，目前看来其行为是直接返回。

* pthread 中使用 mutex 实现 wait 操作

    `main.c`:

    ```c
    #include <pthread.h>
    #include <stdio.h>
    #include <unistd.h>

    pthread_mutex_t mtx;

    void* thread_func_wait(void *arg)
    {
        printf("in thread_func_wait()...\n");
        pthread_mutex_lock(&mtx);
        printf("exit thread_func_wait().\n");
        pthread_mutex_unlock(&mtx);
        return NULL;
    }

    int main()
    {
        pthread_mutex_init(&mtx, NULL);
        pthread_mutex_lock(&mtx);
        pthread_t thd;
        pthread_create(&thd, NULL, thread_func_wait, NULL);
        printf("start sleep ...\n");
        sleep(2);
        printf("end sleep.\n");
        pthread_mutex_unlock(&mtx);
        pthread_join(thd, NULL);
        return 0;
    }
    ```

    output:

    ```
    start sleep ...
    in thread_func_wait()...
    end sleep.
    exit thread_func_wait().
    ```

    这种方式确实是可行的，就是有点奇怪，不知道有啥限制。

* pthread mutex 使用

    `main.c`:

    ```c
    #include <pthread.h>
    #include <stdio.h>

    pthread_mutex_t mtx = PTHREAD_MUTEX_INITIALIZER;
    int counter = 0;

    void* increase_counter(void *thd_id)
    {
        for (int i = 0; i < 5; ++i)
        {
            pthread_mutex_lock(&mtx);
            printf("thd_id: %d:, cnt: %d\n", *(pthread_t*)thd_id, counter);
            ++counter;
            pthread_mutex_unlock(&mtx);
        }
        
        return (void*) 1;
    }

    int main()
    {
        pthread_t thd_id[2];
        pthread_create(&thd_id[0], NULL, increase_counter, &thd_id[0]);
        pthread_create(&thd_id[1], NULL, increase_counter, &thd_id[1]);
        void *thd_ret = NULL;
        for (int i = 0; i < 2; ++i)
        {
            pthread_join(thd_id[i], &thd_ret);
            printf("thread %d, ret: %p\n", thd_id[i], thd_ret);
        }
        
        return 0;
    }
    ```

    compile: `gcc -g main.c -o main`

    run: `./main`

    output:

    ```
    thd_id: -1216346560:, cnt: 0
    thd_id: -1226832320:, cnt: 1
    thd_id: -1226832320:, cnt: 2
    thd_id: -1226832320:, cnt: 3
    thd_id: -1226832320:, cnt: 4
    thd_id: -1226832320:, cnt: 5
    thd_id: -1216346560:, cnt: 6
    thd_id: -1216346560:, cnt: 7
    thd_id: -1216346560:, cnt: 8
    thd_id: -1216346560:, cnt: 9
    thread -1216346560, ret: 0x1
    thread -1226832320, ret: 0x1
    ```

    如果不加 mutex，比如这样写：

    ```c
    void* increase_counter(void *thd_id)
    {
        for (int i = 0; i < 5; ++i)
        {
            printf("thd_id: %d:, cnt: %d\n", *(pthread_t*)thd_id, counter);
            ++counter;
        }
        
        return (void*) 1;
    }
    ```

    那么 output 如下：

    ```
    thd_id: 155190848:, cnt: 0
    thd_id: 155190848:, cnt: 1
    thd_id: 155190848:, cnt: 2
    thd_id: 155190848:, cnt: 3
    thd_id: 155190848:, cnt: 4
    thd_id: 144705088:, cnt: 0
    thd_id: 144705088:, cnt: 6
    thd_id: 144705088:, cnt: 7
    thd_id: 144705088:, cnt: 8
    thd_id: 144705088:, cnt: 9
    thread 155190848, ret: 0x1
    thread 144705088, ret: 0x1
    ```

    可以看到，`cnt`并不是稳定增加的。（问题：中间读取到了 0，但是为什么没有对 0 递增的结果 1？为什么中间出错了，最终的结果仍是对的？）

    `PTHREAD_MUTEX_INITIALIZER`是一个宏，展开为`{ { 0, 0, 0, 0, PTHREAD_MUTEX_TIMED_NP, 0, 0, { 0, 0 } } }`。

    问题：`pthread_mutex_init()`与`PTHREAD_MUTEX_INITIALIZER`有什么区别？

* pthread 的一个基本用法

    `main.c`:

    ```c
    #include <pthread.h>
    #include <stdio.h>

    void* print_hello(void *msg)
    {
        printf("msg from child thread: %s\n", (char*) msg);
        return (void*) 1;
    }

    int main()
    {
        pthread_t thd_id;
        pthread_create(&thd_id, NULL, print_hello, "hello, world");
        void *thd_ret = NULL;
        pthread_join(thd_id, &thd_ret);
        printf("thread ret: %p\n", thd_ret);
        return 0;
    }
    ```

    compile: `gcc -g main.c -o main`

    run: `./main`

    output:

    ```
    msg from child thread: hello, world
    thread ret: 0x1
    ```

    使用 pthread 需要添加头文件`<pthread.h>`。`pthread_create()`可以创建一个线程，创建完后线程立即执行，其原型如下：

    ```c
    int pthread_create(
        pthread_t *thread, 
        const pthread_attr_t *attr,
        void *(*start_routine) (void *),
        void *arg
    );
    ```

    第一个参数返回`pthread_t`类型的线程标识符，第二个参数不知道填啥可以填`NULL`。
    
    第三个参数是函数指针，可以看到，这个函数返回一个`void*`的值，接收一个`void*`的指针，并不支持多个输入参数。如果想给函数传入多个参数，可能需要把参数包裹成`struct`的形式。返回值也是同理。

    第四个参数是实际传给线程函数的参数。

    `pthread_join()`用于等待线程函数结束，并拿到返回值，原型如下：

    ```c
    int pthread_join(pthread_t thread, void **retval);
    ```

    `pthread_exit()`可以替换线程函数的`return`语句，其原型如下：

    ```c
    void pthread_exit(void *retval);
    ```

## note