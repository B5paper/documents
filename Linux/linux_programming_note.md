# Linux Programming Note

此笔记记录 linux 下的 user space 内的各种 app 的写法。

如果涉及到 kernel space 相关的编程，可以把放到 linux driver note 中。

## cache

* `fstat()`用于获取文件的状态信息，比如文件大小、权限、时间戳等

    头文件：`<sys/stat.h>`

    syntax:

    ```cpp
    int fstat(int fd, struct stat *buf);
    ```

    参数：

        fd：已打开文件的文件描述符（通过 open()、fileno() 等获取）。

        buf：指向 struct stat 的指针，用于存储文件状态信息。

    返回值：

        成功返回 0，失败返回 -1 并设置 errno。

    `struct stat`中的常用成员：

    * `st_mode`: 文件类型和权限（如 S_ISREG() 判断是否为普通文件）

    * `st_size`: 文件大小（字节）

    * `st_uid`: 文件所有者的用户ID
    
    * `st_gid`: 文件所属组的组ID

    * `st_atime`: 最后访问时间（Access Time）

    * `st_mtime`: 最后修改时间（Modify Time）

    * `st_ctime`: 最后状态变更时间（Change Time）

    example:

    ```cpp
    #include <fcntl.h>
    #include <unistd.h>
    #include <sys/stat.h>
    #include <stdio.h>

    int main() {
        int fd = open("msg.txt", O_RDONLY);

        struct stat my_stat;
        int ret = fstat(fd, &my_stat);
        if (ret != 0) {
            printf("fail to fstat()...\n");
            return -1;
        }

        printf("file size: %lu\n", my_stat.st_size);

        if (S_ISREG(my_stat.st_mode)) {
            printf("This is a regular file.\n");
        } else if (S_ISDIR(my_stat.st_mode)) {
            printf("This is a directory.\n");
        }

        ret = close(fd);
        if (ret != 0) {
            printf("fail to close fd: %d\n", fd);
            return -1;
        }

        return 0;
    }
    ```

    output:

    ```
    file size: 15
    This is a regular file.
    ```

* `open()`

    头文件：`#include <fcntl.h> `

    syntax:

    ```c
    int open(const char *pathname, int flags, mode_t mode);  // mode 仅在创建文件时使用
    ```

    打开文件：

    ```c
    int fd = open("msg_1.txt", O_RDONLY);
    if (fd < 0) {
        printf("fail to open file, ret: %d\n", fd);
        return -1;
    }
    ```

    创建新文件：

    ```c
    int fd = open("newfile.txt", O_CREAT, 0644); // 创建文件并设置权限 -rw-r--r--
    ```

    如果使用`O_CREAT`创建文件时没有加第三个参数设置权限，那么创建出来的文件会被加上`s`权限，导致无法正常打开。

    如果文件存在，则不会覆盖。

    如果不想使用`0644`权限创建文件，那么可以使用

    `int fd = open("msg_1.txt", O_CREAT | O_RDWR);`

    `O_RDWR`不能使用`O_RDONLY`或`O_WRONLY`，否则会加上`s`权限。同样地，如果文件存在，则不会覆盖。

    `open()`的其他 flag （未验证）：

    O_RDONLY：只读

    O_WRONLY：只写

    O_RDWR：读写

    O_APPEND：追加写入

    O_TRUNC：清空文件（如果已存在）

    O_NONBLOCK：非阻塞模式（常用于设备文件或管道）

* `read()`是 posix 标准提供的函数，是系统调用

    头文件`<unistd.h>`

    syntax:

    ```cpp
    ssize_t read(int fd, void *buf, size_t count);
    ```

    * `fd`：文件描述符（如通过 open() 打开的文件）。

    * `buf`：存储读取数据的缓冲区。

    * `count`：请求读取的字节数。

    返回值：

    返回实际读取的字节数（ssize_t），可能小于请求的 count（如文件末尾）。

    返回 -1 表示错误（需检查 errno）。

    `fread()`是 C 语言提供的函数，是对系统调用的封装
    
    头文件`<stdio.h>`

    syntax:

    ```cpp
    size_t fread(void *ptr, size_t size, size_t nmemb, FILE *stream);
    ```

    * `ptr`：存储数据的缓冲区。

    * `size`：每个数据项的字节大小（如 sizeof(int)）。

    * `nmemb`：要读取的数据项数量。

    * `stream`：`FILE*`类型的指针。

    返回值：

    返回成功读取的 数据项数量（size_t），而非字节数。

    若返回值小于 nmemb，可能到达文件末尾（feof()）或出错（ferror()）。

    * 缓冲机制（未验证）

        * `read()`

            * 无缓冲：直接调用内核接口，每次调用触发一次系统调用，效率较低（频繁小数据读取时）。

            * 适合需要精细控制或高性能的场景（如大块数据读取）。

        * `fread()`

            * 带缓冲：C 标准库在用户空间维护缓冲区，减少系统调用次数（如多次小数据读取会合并为一次系统调用）。

            * 适合常规文件操作（如文本/二进制文件逐块读取）。

* `stat()`用于获得文件属性

    example:

    ```c
    #include <sys/stat.h>

    struct stat file_info;
    stat("filename", &file_info);  // 获取文件信息
    ```

    struct stat 成员：

    st_mode → 文件类型和权限

    st_size → 文件大小

    st_uid / st_gid → 所有者/组 ID

    st_atime / st_mtime / st_ctime → 访问/修改/状态变更时间

    典型应用场景

        检查文件是否存在（stat() 返回 0 成功，-1 失败）

        监控文件变化（比较 st_mtime）

        权限管理（检查 st_mode 是否符合要求）

* 常用的 posix 函数

    这些函数由 posix 提供（比如 linux）。

    `open()`：头文件`<fcntl.h>`

    `read()`, `close()`：头文件`<unistd.h>`

    ```cpp
    #include <stdio.h>
    #include <fcntl.h>
    #include <unistd.h>
    using namespace std;

    int main() {
        int ret;
        const char *file_path = "msg.txt";
        int fd = open(file_path, O_RDONLY);
        if (fd == -1) {
            printf("fail to open %s\n", file_path);
            return -1;
        }

        char buf[128];
        ssize_t bytes_read = read(fd, buf, 128);
        if (bytes_read < 0) {
            printf("fail to read, ret: %ld\n", bytes_read);
            return -1;
        }
        printf("read bytes: %ld, msg: %s\n", bytes_read, buf);

        ret = close(fd);
        if (ret != 0) {
            printf("fail to close fd: %d\n", fd);
            return -1;
        }

        return 0;
    }
    ```

    output:

    ```
    read bytes: 15, msg: hello
    world
    123
    ```

    `lseek()`类似于`fseek()`，头文件：`<unistd.h>`, example:

    ```cpp
    off_t new_off = lseek(fd, 1, SEEK_SET);
    ```

    成功时返回新的文件偏移量（从文件开头计算的字节数）。
    
    失败时返回 -1，并设置 errno（如 EBADF 表示无效文件描述符）。

    ```cpp
    off_t pos = lseek(fd, 0, SEEK_CUR); // 返回当前位置
    ```

    管道、套接字等不支持随机访问，调用会失败。

* `getpwuid()`用法

    根据 uid 去`/etc/passwd`中查询信息。

    example:

    ```
    #include <unistd.h>
    #include <pwd.h>
    #include <stdio.h>

    int main() {
        uid_t uid = getuid();
        printf("uid: %u\n", uid);
        passwd* pwd = getpwuid(uid);
        if (pwd == NULL) {
            printf("fail to get pwuid\n");
            return -1;
        }
        printf("pw_name: %s\n", pwd->pw_name);
        printf("pw uid: %u\n", pwd->pw_uid);
        printf("pw gid: %u\n", pwd->pw_gid);
        printf("pw dir: %s\n", pwd->pw_dir);
        printf("pw shell: %s\n", pwd->pw_shell);
        printf("pw passwd: %s\n", pwd->pw_passwd);
        printf("pw gecos: %s\n", pwd->pw_gecos);
        return 0;
    }
    ```

    output:

    ```
    uid: 1000
    pw_name: hlc
    pw uid: 1000
    pw gid: 1000
    pw dir: /home/hlc
    pw shell: /bin/bash
    pw passwd: x
    pw gecos: hlc,,,
    ```

    相似地，`getpwnam()`通过用户名查询用户信息。

* `getuid()`等函数在头文件`<unistd.h>`中，返回当前用户的 uid。

    example:

    ```cpp
    #include <unistd.h>
    #include <stdio.h>

    int main() {
        uid_t uid = getuid();
        printf("uid: %u\n", uid);
        uid_t euid = geteuid();
        printf("euid: %u\n", euid);
        gid_t gid = getgid();
        printf("gid: %u\n", gid);
        gid_t egid = getegid();
        printf("egid: %u\n", egid);
        return 0;
    }
    ```

    output:

    普通运行：

    ```
    uid: 1000
    euid: 1000
    gid: 1000
    egid: 1000
    ```

    使用`sudo ./main`运行：

    ```
    uid: 0
    euid: 0
    gid: 0
    egid: 0
    ```

    可以看到，`uid`与`euid`目前没有什么区别。

    可以运行命令`id`，看到类似的输出：

    ```
    uid=1000(hlc) gid=1000(hlc) groups=1000(hlc),4(adm),24(cdrom),27(sudo),30(dip),46(plugdev),109(kvm),122(lpadmin),135(lxd),136(sambashare),137(docker),140(libvirt)
    ```

    `sudo id`输出如下：

    ```
    uid=0(root) gid=0(root) groups=0(root)
    ```

* `getenv()`, `setenv()`, `unsetenv()`用法

    这几个函数都是 c 语言中与环境变量相关的函数，在`<stdlib.h>`头文件中。

    syntax:

    ```cpp
    #include <stdlib.h>
    char *getenv(const char *name);
    int unsetenv(const char *name);
    int setenv(const char *name, const char *value, int overwrite);
    ```

    * `name`： 环境变量名。

    * `value`： 要设置的值。

    * `overwrite`： 若为 1，覆盖已存在的变量；若为 0，不覆盖。

    返回值： 成功返回`0`，失败返回`-1`。

    example:

    ```cpp
    #include <stdlib.h>
    #include <stdio.h>

    int main() {
        int ret = setenv("GREETING_MSG", "hello, world", 0);
        if (ret != 0) {
            printf("fail to set env\n");
            return -1;
        }
        const char *greeting_msg = getenv("GREETING_MSG");
        printf("greeting msg: %s\n", greeting_msg);

        ret = setenv("GREETING_MSG", "nihao", 0);
        if (ret != 0) {
            printf("fail to set env\n");
            return -1;
        }
        greeting_msg = getenv("GREETING_MSG");
        printf("greeting msg: %s\n", greeting_msg);

        ret = setenv("GREETING_MSG", "nihao", 1);
        if (ret != 0) {
            printf("fail to set env\n");
            return -1;
        }
        greeting_msg = getenv("GREETING_MSG");
        printf("greeting msg: %s\n", greeting_msg);

        return 0;
    }
    ```

    output:

    ```
    greeting msg: hello, world
    greeting msg: hello, world
    greeting msg: nihao
    ```

    这几个函数都是 POSIX 扩展，不是 C 标准，需确保系统支持。

* ai 生成的`sched_setaffinity()`的 example

    `main.c`:

    ```c
    #define _GNU_SOURCE
    #include <stdio.h>
    #include <stdlib.h>
    #include <pthread.h>
    #include <sched.h>
    #include <time.h>
    #include <stdatomic.h>

    #define MATRIX_SIZE 2048
    #define NUM_THREADS 4

    // 全局矩阵
    double A[MATRIX_SIZE][MATRIX_SIZE];
    double B[MATRIX_SIZE][MATRIX_SIZE];
    double C[MATRIX_SIZE][MATRIX_SIZE];

    // 线程参数
    typedef struct {
        int start_row;
        int end_row;
        int cpu_core; // 绑定的 CPU 核心
    } ThreadArgs;

    // 矩阵乘法（计算密集型任务）
    void* matrix_multiply(void* arg) {
        ThreadArgs* args = (ThreadArgs*)arg;
        
        // 如果指定了 CPU 核心，则绑定
        if (args->cpu_core >= 0) {
            cpu_set_t mask;
            CPU_ZERO(&mask);
            CPU_SET(args->cpu_core, &mask);
            if (sched_setaffinity(0, sizeof(mask), &mask) == -1) {
                perror("sched_setaffinity failed");
                exit(EXIT_FAILURE);
            }
        }

        // 计算矩阵乘法
        for (int i = args->start_row; i < args->end_row; i++) {
            for (int j = 0; j < MATRIX_SIZE; j++) {
                C[i][j] = 0;
                for (int k = 0; k < MATRIX_SIZE; k++) {
                    C[i][j] += A[i][k] * B[k][j];
                }
            }
        }
        return NULL;
    }

    // 初始化矩阵
    void init_matrices() {
        for (int i = 0; i < MATRIX_SIZE; i++) {
            for (int j = 0; j < MATRIX_SIZE; j++) {
                A[i][j] = (double)rand() / RAND_MAX;
                B[i][j] = (double)rand() / RAND_MAX;
            }
        }
    }

    // 运行测试（绑定或不绑定 CPU）
    void run_test(int use_affinity) {
        pthread_t threads[NUM_THREADS];
        ThreadArgs args[NUM_THREADS];
        int rows_per_thread = MATRIX_SIZE / NUM_THREADS;

        // 初始化线程参数
        for (int i = 0; i < NUM_THREADS; i++) {
            args[i].start_row = i * rows_per_thread;
            args[i].end_row = (i + 1) * rows_per_thread;
            args[i].cpu_core = use_affinity ? i : -1; // -1 表示不绑定
        }

        // 创建线程
        clock_t start = clock();
        for (int i = 0; i < NUM_THREADS; i++) {
            pthread_create(&threads[i], NULL, matrix_multiply, &args[i]);
        }

        // 等待线程完成
        for (int i = 0; i < NUM_THREADS; i++) {
            pthread_join(threads[i], NULL);
        }
        clock_t end = clock();

        // 输出结果
        double elapsed = (double)(end - start) / CLOCKS_PER_SEC;
        printf("%s CPU Affinity: Time = %.3f seconds\n",
               use_affinity ? "With" : "Without", elapsed);
    }

    int main() {
        // 初始化随机矩阵
        init_matrices();

        // 运行测试（绑定 CPU）
        run_test(1);

        // 运行测试（不绑定 CPU）
        run_test(0);

        return 0;
    }
    ```

    compile:

    `gcc main.c -o main`

    run: `./main`

    output:

    ```
    With CPU Affinity: Time = 56.594 seconds
    Without CPU Affinity: Time = 55.922 seconds
    ```

    实测绑定了 cpu 核的代码不一定比不绑定快。但是平均下来还是要快一点，设置 cpu affinity 大概能比不设置快 3%。

    绑定 CPU 亲和性（affinity）能减少线程切换开销，提高缓存命中率。

    说明：

    1. 必须使用`gcc`编译，如果使用`g++`编译可能会报错。

    1. 必须在`#include <sched.h>`前添加`#define _GNU_SOURCE`，因为`sched_setaffinity()`是 gnu 的扩展功能，不是 c 语言的标准功能。

    1. warm up 对程序的输出影响较大，第一轮跑的测试通常会慢些，可以交换两种情况做多组测试，取平均值。

    1. 如果 cpu 有超线程，将绑定的核设置为`0, 2, 4, 6`比设置为`0, 1, 2, 3`效果要好。

* linux `sched_setaffinity()`的作用

    `sched_setaffinity()`可以设置进程/线程的 cpu 亲和性。

    函数原型与头文件:

    ```c
    #include <sched.h>
    int sched_setaffinity(pid_t pid, size_t cpusetsize, const cpu_set_t *mask);
    ```

    parameters:

    * `pid`：目标进程/线程的 PID。若为 0，表示当前调用线程。

    * `cpusetsize`：mask 参数的大小（通常用`sizeof(cpu_set_t)`）。

    * `mask`：指定 CPU 亲和性的位掩码（通过`CPU_SET`等宏操作）。

    example 1:

    ```c
    cpu_set_t mask;
    CPU_ZERO(&mask);       // 清空掩码
    CPU_SET(2, &mask);     // 绑定到 CPU 核心 2

    if (sched_setaffinity(0, sizeof(mask), &mask) == -1) {
        perror("sched_setaffinity failed, errno: %d", errno);
        exit(EXIT_FAILURE);
    }
    ```

    example 2:

    ```c
    #include <sched.h>
    #include <pthread.h>

    void* thread_func(void* arg) {
        int core_id = *(int*)arg;
        cpu_set_t mask;
        CPU_ZERO(&mask);
        CPU_SET(core_id, &mask);
        if (sched_setaffinity(0, sizeof(mask), &mask) == -1) {
            perror("sched_setaffinity");
        }
        // do something
        return NULL;
    }

    int main() {
        pthread_t thread1, thread2;
        int core1 = 0, core2 = 1;
        pthread_create(&thread1, NULL, thread_func, &core1);
        pthread_create(&thread2, NULL, thread_func, &core2);
        pthread_join(thread1, NULL);
        pthread_join(thread2, NULL);
        return 0;
    }
    ```

    说明：

    1. `CPU_SET()`的第一个参数指的是 cpu 的逻辑核心编号，如果 cpu 支持超线程，那么有可能多个逻辑核心在同一个物理核心上，这样的话仍会造成资料竞争。

* pthread cond 如果先 signal，再 wait，那么无法正常运行

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