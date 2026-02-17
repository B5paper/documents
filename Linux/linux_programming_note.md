# Linux Programming Note

此笔记记录 linux 下的 user space 内的各种 app 的写法。

如果涉及到 kernel space 相关的编程，可以把放到 linux driver note 中。

## cache

* 简述 linux 中 signal() 的原型、作用及用法

    signal() 函数

    原型

    ```c
    #include <signal.h>

    void (*signal(int signum, void (*handler)(int)))(int);
    ```

    更易读的等价写法：

    ```c
    typedef void (*sighandler_t)(int);
    sighandler_t signal(int signum, sighandler_t handler);
    ```

    作用

    - **功能**：设置信号处理函数，用于捕获和处理进程收到的信号

    - **参数**：

      - `signum`：要处理的信号编号（如 SIGINT、SIGTERM）
      - `handler`：处理方式，可以是：
        - 自定义信号处理函数指针
        - `SIG_IGN`：忽略该信号
        - `SIG_DFL`：恢复默认处理

    - **返回值**：
      - 成功：返回之前的信号处理函数指针
      - 失败：返回 `SIG_ERR` 并设置 errno

    基本用法示例

    ```c
    #include <stdio.h>
    #include <signal.h>
    #include <unistd.h>

    // 自定义信号处理函数
    void sigint_handler(int signum) {
        printf("\n收到 SIGINT 信号 (%d)，按 Ctrl+C 可退出\n", signum);
    }

    int main() {
        // 注册 SIGINT 信号处理函数
        if (signal(SIGINT, sigint_handler) == SIG_ERR) {
            perror("signal 设置失败");
            return 1;
        }
        
        printf("程序运行中，按 Ctrl+C 测试信号处理\n");
        
        // 让程序持续运行
        while(1) {
            printf(".");
            fflush(stdout);
            sleep(1);
        }
        
        return 0;
    }
    ```

    其他用法示例

    ```c
    // 1. 忽略信号
    signal(SIGINT, SIG_IGN);  // 忽略 Ctrl+C

    // 2. 恢复默认处理
    signal(SIGINT, SIG_DFL);  // Ctrl+C 将终止程序

    // 3. 保存并恢复之前的处理函数
    typedef void (*sighandler_t)(int);
    sighandler_t old_handler;

    old_handler = signal(SIGINT, sigint_handler);
    // ... 使用自定义处理
    signal(SIGINT, old_handler);  // 恢复之前的处理
    ```

    注意事项

    - **可移植性**：signal() 在不同 Unix 系统上的行为可能有差异，建议使用 sigaction() 以获得更好的可移植性
    - **系统调用中断**：某些系统调用可能被信号中断，需要处理 EINTR 错误
    - **不可重入函数**：信号处理函数中应只调用异步信号安全的函数

* 简述 linux 中 pause() 的原型、作用及用法

    pause() 是 Linux/Unix 系统中的一个系统调用，用于使当前进程挂起（睡眠），直到接收到一个信号为止。

    原型

    ```c
    #include <unistd.h>

    int pause(void);
    ```

    作用

    * 挂起进程：调用 pause() 的进程会进入睡眠状态

    * 等待信号：直到进程捕获到一个信号并从信号处理函数返回后，pause() 才返回

    * 返回值：总是返回 -1，并设置 errno 为 EINTR（表示被信号中断）

    用法

    基本示例

    ```c
    #include <stdio.h>
    #include <unistd.h>
    #include <signal.h>
    #include <errno.h>

    void signal_handler(int sig) {
        printf("Received signal %d\n", sig);
    }

    int main() {
        signal(SIGINT, signal_handler);
        
        printf("Waiting for signal (PID: %d)...\n", getpid());
        
        // 进程在这里挂起，直到收到 SIGINT (Ctrl+C)
        int ret = pause();
        
        if (ret == -1 && errno == EINTR) {
            printf("pause() was interrupted by signal\n");
        }
        
        return 0;
    }
    ```

    常见用途

    1. 等待特定信号

        ```c
        #include <stdio.h>
        #include <unistd.h>
        #include <signal.h>

        volatile sig_atomic_t signal_received = 0;

        void handler(int sig) {
            signal_received = 1;
        }

        int main() {
            signal(SIGUSR1, handler);
            
            printf("Waiting for SIGUSR1...\n");
            
            while (!signal_received) {
                pause();  // 等待 SIGUSR1 信号
            }
            
            printf("SIGUSR1 received!\n");
            return 0;
        }
        ```

    2. 信号驱动服务器

        ```c
        #include <stdio.h>
        #include <unistd.h>
        #include <signal.h>
        #include <sys/types.h>

        void sigchld_handler(int sig) {
            // 处理子进程退出
            while (waitpid(-1, NULL, WNOHANG) > 0) {
                // 清理子进程资源
            }
        }

        int main() {
            signal(SIGCHLD, sigchld_handler);
            
            // 主服务器循环
            while (1) {
                // ... 处理其他任务 ...
                
                // 等待信号（这里会处理 SIGCHLD 信号）
                pause();
            }
            
            return 0;
        }
        ```

    注意事项

    1. 信号处理问题

        ```c
        // 有问题的代码
        void handler(int sig) {
            // 空处理函数
        }

        int main() {
            signal(SIGINT, handler);
            pause();  // 收到 SIGINT 后，pause() 返回，但程序继续执行
            // 问题：pause() 返回后可能执行不应该执行的代码
        }
        ```

    2. 竞态条件

        pause() 存在竞态条件问题，可能永久挂起：

        ```c
        // 不安全的用法
        void handler(int sig) {
            // 处理信号
        }

        int main() {
            // 如果在 signal() 调用后、pause() 调用前收到信号，
            // pause() 可能会永久挂起
            signal(SIGALRM, handler);
            alarm(1);  // 1秒后发送 SIGALRM
            
            // 如果信号在 alarm() 后、pause() 前到达，pause() 会永远等待
            pause();
            
            return 0;
        }
        ```

    3. 替代方案

        由于 pause() 的竞态条件问题，推荐使用以下替代方案：

        sigsuspend() - 更安全的替代

        ```c
        #include <signal.h>

        void handler(int sig) {
            printf("Signal received\n");
        }

        int main() {
            sigset_t newmask, oldmask;
            
            signal(SIGINT, handler);
            
            // 阻塞 SIGINT 信号
            sigemptyset(&newmask);
            sigaddset(&newmask, SIGINT);
            sigprocmask(SIG_BLOCK, &newmask, &oldmask);
            
            // 关键区域代码...
            
            // 原子操作：恢复信号屏蔽并挂起进程
            sigsuspend(&oldmask);
            
            // 继续处理其他任务...
            
            return 0;
        }
        ```

        使用 sigwait() 或 sigwaitinfo()

        ```c
        #include <signal.h>
        #include <stdio.h>

        int main() {
            sigset_t set;
            int sig;
            
            sigemptyset(&set);
            sigaddset(&set, SIGINT);
            sigaddset(&set, SIGTERM);
            
            // 阻塞这些信号
            sigprocmask(SIG_BLOCK, &set, NULL);
            
            printf("Waiting for signal...\n");
            sigwait(&set, &sig);  // 同步等待信号
            
            printf("Received signal: %d\n", sig);
            
            return 0;
        }
        ```

    总结

    | 特点 | 说明 |
    | - | - |
    | 功能 | 挂起进程直到收到信号 |
    | 返回值 | 总是 -1，errno 设为 EINTR |
    | 优点 | 简单易用 |
    | 缺点 | 存在竞态条件，可能永久挂起 |
    | 替代方案 | sigsuspend(), sigwait(), sigwaitinfo() |

    在实际开发中，不推荐使用 pause()，因为它存在竞态条件问题。应该优先使用 sigsuspend() 或其他更安全的信号处理机制。pause() 的主要价值在于理解 Unix 信号处理机制的基本原理。

* 如果 pause() 等待的是用户输入呢，是不是就没有竞态问题了

    不，完全不是这样的。这是一个常见的误解。让我详细解释一下：

    核心误解澄清

    pause() 不等待用户输入。它等待的是信号（signal），而不是标准输入或任何用户输入。

    信号 vs 用户输入

    1. 信号（Signal）

        ```c
        // pause() 等待的是信号，比如：
        // SIGINT    (Ctrl+C)
        // SIGTERM   (kill 命令发送的终止信号)
        // SIGUSR1   (用户自定义信号1)
        // SIGALRM   (定时器信号)
        // 等等...
        ```

    2. 用户输入

        ```c
        // 这是通过 I/O 函数完成的，比如：
        // scanf()   // 等待标准输入
        // read()    // 从文件描述符读取
        // getchar() // 获取字符
        // select()  // I/O 多路复用
        // poll()    // I/O 多路复用
        // epoll()   // Linux 高性能 I/O 多路复用
        ```

    为什么混淆了？

    很多人混淆是因为这个常见模式：

    ```c
    #include <stdio.h>
    #include <unistd.h>
    #include <signal.h>

    void handler(int sig) {
        printf("Received signal\n");
    }

    int main() {
        signal(SIGINT, handler);
        
        printf("Press Ctrl+C to send SIGINT\n");
        pause();  // 这里不是在等待键盘输入，而是在等待信号！
        
        return 0;
    }
    ```

    关键点：

    * 当你在终端按 Ctrl+C 时，终端驱动程序会向进程发送 SIGINT 信号

    * pause() 等待的是这个信号，而不是读取 Ctrl+C 这个按键输入

    * 如果只是普通的键盘输入（如输入 "hello"），pause() 完全不会响应

    正确的用户输入等待方式

    方案1：使用标准 I/O

    ```c
    #include <stdio.h>

    int main() {
        char input[100];
        
        printf("Enter something: ");
        fflush(stdout);
        
        // 这会真正等待用户输入
        if (fgets(input, sizeof(input), stdin) != NULL) {
            printf("You entered: %s", input);
        }
        
        return 0;
    }
    ```

    方案2：使用低级 I/O

    ```c
    #include <unistd.h>
    #include <stdio.h>

    int main() {
        char buffer[100];
        
        printf("Enter something: ");
        fflush(stdout);
        
        // read() 会阻塞直到有输入
        ssize_t n = read(STDIN_FILENO, buffer, sizeof(buffer)-1);
        if (n > 0) {
            buffer[n] = '\0';
            printf("You entered: %s", buffer);
        }
        
        return 0;
    }
    ```

    方案3：结合信号和输入（复杂场景）

    ```c
    #include <stdio.h>
    #include <unistd.h>
    #include <signal.h>
    #include <sys/select.h>

    volatile sig_atomic_t got_signal = 0;

    void handler(int sig) {
        got_signal = 1;
    }

    int main() {
        fd_set readfds;
        struct timeval timeout;
        
        signal(SIGINT, handler);
        
        while (!got_signal) {
            FD_ZERO(&readfds);
            FD_SET(STDIN_FILENO, &readfds);
            
            timeout.tv_sec = 5;  // 5秒超时
            timeout.tv_usec = 0;
            
            // 使用 select 同时等待输入和信号
            int ret = select(STDIN_FILENO + 1, &readfds, NULL, NULL, &timeout);
            
            if (ret == -1) {
                // 被信号中断
                if (got_signal) {
                    printf("\nReceived signal, exiting...\n");
                    break;
                }
            } else if (ret == 0) {
                printf("Timeout, still waiting...\n");
            } else if (FD_ISSET(STDIN_FILENO, &readfds)) {
                char input[100];
                fgets(input, sizeof(input), stdin);
                printf("You entered: %s", input);
            }
        }
        
        return 0;
    }
    ```

    关键区别总结

    | 特性 | pause() | 标准输入函数 |
    | - | - | - |
    | 等待什么 | 信号（signal） | 用户输入（字符流） |
    | 如何触发 | Ctrl+C、kill命令、程序异常等 | 键盘输入、管道、重定向等 |
    | 数据类型 | 信号编号（整数） | 字符/字节流 |
    | 控制方式 | 异步（可能随时发生） | 同步（按需读取） |
    | 典型函数 | pause(), sigsuspend() | scanf(), read(), fgets() |

    重要结论

    * pause() 与用户输入无关：它只等待信号，不能用于读取键盘输入

    * 竞态条件依然存在：如果 pause() 和信号的发送之间存在时间窗口，pause() 可能永久挂起

    * 正确使用场景：

        * 等待特定信号触发

        * 简单的信号同步

        * 不适用于用户交互输入

    * 用户输入的正确方式：

        * 使用 scanf(), fgets() 等标准I/O函数

        * 使用 read() 等系统调用

        * 使用 select(), poll(), epoll() 进行多路复用

    如果你需要在等待用户输入的同时响应信号，应该使用：

    * select()/poll()/epoll() 设置超时

    * 或使用信号处理配合非阻塞I/O

    * 或使用 sigwait() 同步等待信号

* 我超，我试了下，还真是

    哈哈，是的吧！ 😄 这就是 Unix/Linux 信号机制有趣（也容易让人困惑）的地方！

    你的发现验证了什么？

    ```c
    #include <stdio.h>
    #include <unistd.h>

    int main() {
        printf("Start...\n");
        
        // 你试了这个，对吗？
        pause();
        
        // 按 Ctrl+C 能退出，但输入文字完全没反应
        printf("This line never prints unless signal arrives\n");
        return 0;
    }
    ```

    你可能会试的各种情况：

    ```bash
    # 运行程序
    $ ./a.out
    Start...
    （程序卡在这里）

    # 尝试1：输入文字然后回车
    hello world   # ← 输入这个，按回车
    （没反应！）

    # 尝试2：按 Ctrl+C
    ^C            # ← 立即退出！

    # 尝试3：在另一个终端发送信号
    $ kill -INT <pid>  # 也立即退出
    ```

    为什么会有这个误解？

    其实很多人都中过招，原因有几个：

    1. 终端特性混淆

        ```c
        // 按 Ctrl+C 看起来像"键盘输入"，实际是：
        // 终端驱动 → 产生 SIGINT → 发送给前台进程
        // pause() 响应的是信号，不是按键
        ```

    2. 教学示例误导

        很多教程这样写：

        ```c
        printf("Press Ctrl+C to continue...\n");
        pause();  // 看起来像在等按键，实际在等信号
        ```

    3. 真实输入 vs 控制字符

        ```text
        键盘上的按键有两种作用：
        1. 输入数据：a, b, 1, 2, Enter 等 → 产生字符
        2. 控制功能：Ctrl+C, Ctrl+Z, Ctrl+\ 等 → 产生信号
        ```

    好玩的小实验

    实验1：看看谁在真正处理输入

    ```c
    #include <stdio.h>
    #include <unistd.h>
    #include <signal.h>

    int main() {
        // 忽略 Ctrl+C
        signal(SIGINT, SIG_IGN);
        
        printf("Try Ctrl+C now (it's ignored)\n");
        printf("But type something and press Enter: ");
        
        char buffer[100];
        fgets(buffer, sizeof(buffer), stdin);  // 真正的输入等待
        
        printf("You typed: %s", buffer);
        return 0;
    }
    ```

    实验2：信号和输入的混合效果

    ```c
    #include <stdio.h>
    #include <unistd.h>
    #include <signal.h>
    #include <string.h>

    void handler(int sig) {
        printf("\n[Signal %d received!]\n", sig);
    }

    int main() {
        signal(SIGINT, handler);  // Ctrl+C
        signal(SIGTSTP, handler); // Ctrl+Z
        
        printf("Try:\n");
        printf("1. Type normal text + Enter\n");
        printf("2. Press Ctrl+C (SIGINT)\n");
        printf("3. Press Ctrl+Z (SIGTSTP)\n");
        printf("4. Type 'exit' to quit\n\n");
        
        char input[100];
        while (1) {
            printf("> ");
            fflush(stdout);
            
            if (fgets(input, sizeof(input), stdin) == NULL) {
                break;  // EOF (Ctrl+D)
            }
            
            // 去掉换行符
            input[strcspn(input, "\n")] = 0;
            
            if (strcmp(input, "exit") == 0) {
                printf("Goodbye!\n");
                break;
            }
            
            printf("You said: %s\n", input);
        }
        
        return 0;
    }
    ```

    深入理解：终端、进程和信号的三角关系

    ```text
    用户按下键盘
        ↓
    终端驱动程序
        ├── 如果是普通键：放入输入缓冲区
        ├── 如果是 Ctrl+C：发送 SIGINT 给前台进程组
        ├── 如果是 Ctrl+Z：发送 SIGTSTP
        └── 如果是 Ctrl+\：发送 SIGQUIT
        ↓
    进程的响应
        ├── 如果有 read()/fgets()：读取缓冲区内容
        ├── 如果有 pause()：如果是信号则唤醒
        └── 如果有 signal handler：执行处理函数
    ```

    实际开发中的正确姿势

    场景1：需要同时处理输入和信号

    ```c
    #include <stdio.h>
    #include <unistd.h>
    #include <signal.h>
    #include <errno.h>

    volatile sig_atomic_t should_exit = 0;

    void sigint_handler(int sig) {
        should_exit = 1;
    }

    int main() {
        char buffer[100];
        
        signal(SIGINT, sigint_handler);
        
        while (!should_exit) {
            printf("Enter command (or Ctrl+C to exit): ");
            fflush(stdout);
            
            if (fgets(buffer, sizeof(buffer), stdin) == NULL) {
                if (errno == EINTR) {
                    // 被信号中断
                    continue;
                }
                break;
            }
            
            // 处理命令
            printf("Processing: %s", buffer);
        }
        
        printf("\nExiting gracefully...\n");
        return 0;
    }
    ```

    场景2：构建交互式 shell

    ```c
    // 类似 bash 的实现思路
    while (!exit_requested) {
        display_prompt();
        
        // 1. 设置信号处理
        setup_signals();
        
        // 2. 读取输入（可能被信号中断）
        char *line = readline_with_timeout();
        
        // 3. 解析和执行命令
        if (line != NULL) {
            execute_command(line);
            free(line);
        }
        
        // 4. 检查是否需要退出
        check_exit_conditions();
    }
    ```

    学到了什么？

    * 信号 ≠ 输入：这是 Unix 编程的重要一课

    * 终端很聪明：它知道何时生成字符，何时生成信号

    * 工具要选对：

        * 要等信号：用 pause()/sigsuspend()/sigwait()

        * 要等输入：用 read()/fgets()/scanf()

        * 两个都要等：用 select()/poll()/epoll()

    恭喜你！ 你刚刚验证了一个重要的 Unix/Linux 概念区别，这会让你成为更好的系统程序员！很多有经验的开发者都曾在这里踩过坑。😉

* 各个平台下 pause 的实现

    Windows 的几种方式：

    ```c
    // 方法1：system("pause") - 最简单
    #include <stdlib.h>
    system("pause");

    // 方法2：getch() - 需要 conio.h
    #include <conio.h>
    printf("Press any key to continue...\n");
    getch();

    // 方法3：Windows API
    #include <windows.h>
    system("pause");  // 或者用更复杂的 API
    ```

    Linux 的正确做法：

    ```c
    // 方法1：使用 getchar() - 最接近 Windows 效果
    #include <stdio.h>
    printf("Press Enter to continue...\n");
    getchar();  // 等待回车键

    // 方法2：任意键等待（需要终端特殊设置）
    #include <stdio.h>
    #include <termios.h>
    #include <unistd.h>

    void wait_for_any_key() {
        struct termios oldt, newt;
        tcgetattr(STDIN_FILENO, &oldt);
        newt = oldt;
        newt.c_lflag &= ~(ICANON | ECHO);
        tcsetattr(STDIN_FILENO, TCSANOW, &newt);
        
        printf("Press any key to continue...\n");
        getchar();
        
        tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
    }

    // 方法3：使用 read() - 低级别控制
    #include <unistd.h>
    #include <stdio.h>
    printf("Press Enter to continue...\n");
    char ch;
    read(STDIN_FILENO, &ch, 1);
    ```

    跨平台解决方案：

    ```c
    #include <stdio.h>

    void pause_program() {
        #ifdef _WIN32
            system("pause");
        #else
            printf("Press Enter to continue...\n");
            getchar();
        #endif
    }

    int main() {
        printf("程序运行中...\n");
        pause_program();  // 跨平台暂停
        printf("继续执行...\n");
        return 0;
    }
    ```

* `<sys/stat.h>`

    * 文件状态信息：包含获取文件信息的函数和数据结构

    * 定义 struct stat：用于存储文件的各种属性（大小、权限、时间戳等）

    * 文件模式常量：如 S_IRUSR（用户读权限）、S_IWUSR（用户写权限）等

    * 主要函数：stat(), fstat(), lstat() 等文件状态查询函数

* `<fcntl.h>`

    文件控制选项：定义文件操作的各种控制常量和函数

    文件打开标志：如 O_RDONLY（只读）、O_WRONLY（只写）、O_RDWR（读写）、O_CREAT（创建文件）等

    文件描述符操作：包含 open(), creat(), fcntl() 等函数的声明和相关常量

* `<sys/types.h>`

    * 定义基本系统数据类型：包含许多标准系统数据类型的定义

    * 提供类型别名：如 pid_t（进程ID）、uid_t（用户ID）、gid_t（组ID）、off_t（文件偏移）、size_t（大小类型）等

* `alarm()`

    为当前进程设置一个定时器（闹钟），在指定的时间到期后，内核会向该进程发送一个 SIGALRM 信号。

    syntax:

    ```c
    #include <unistd.h>

    unsigned int alarm(unsigned int seconds);
    ```

    如果一个进程之前已经通过 alarm() 设置了一个尚未触发的闹钟，再次调用 alarm() 将会重置（覆盖） 之前的闹钟。

    函数的返回值是前一个闹钟的剩余秒数。如果之前没有设置闹钟，则返回0。

    SIGALRM 信号的默认操作是终止进程。通常，我们不会使用默认操作，而是使用 signal() 或 sigaction() 函数来捕获这个信号，并为其注册一个信号处理函数，以便在定时器到期时执行自定义的操作（例如超时处理、周期性任务等）。

    example:

    ```c
    #include <stdio.h>
    #include <unistd.h>
    #include <signal.h>

    // 信号处理函数
    void alarm_handler(int signum) {
        printf("Ring! Alarm received. Time's up!\n");
    }

    int main() {
        // 注册信号处理函数，捕获SIGALRM信号
        signal(SIGALRM, alarm_handler);

        printf("Setting alarm for 3 seconds...\n");
        alarm(3); // 设置3秒后发送SIGALRM信号

        // 暂停进程，等待信号
        pause();

        printf("Program continues after alarm.\n");
        return 0;
    }
    ```

    output:

    ```
    Setting alarm for 3 seconds...
    Ring! Alarm received. Time's up!
    Program continues after alarm.
    ```

    alarm() 的定时精度是秒，对于需要更高精度（如毫秒、微秒）的定时任务，应该使用 setitimer() 或更现代的 timer_create() 等函数。

* `GRUB_CMDLINE_LINUX="console=ttyS0"`

    将系统的第一个串行端口（ttyS0） 设置为主要控制台（console）

    qemu 虚拟机中，在`/etc/default/grub`中修改`GRUB_CMDLINE_LINUX`为`GRUB_CMDLINE_LINUX="console=ttyS0"`，使配置生效：`sudo update-grub`，重启后可以看到 console 中显示整个开机过程的 log，随机进入登陆提示。部分输出如下：

    ```
    ...
    [  OK  ] Finished Permit User Sessions.
    systemd-user-sessions.service
             Starting Hold until boot process finishes up...
             Starting Terminate Plymouth Boot Screen...

    Ubuntu 22.04.4 LTS Ubuntu22 ttyS0

    Ubuntu22 login: 
    ```

    `console=`是一个内核参数，用于指定内核和系统消息（包括启动信息、登录提示、系统错误等）输出到哪个设备。

    `ttyS`是 Linux 中对串行端口（Serial Port，也叫 COM 端口）的命名。`ttyS0`对应第一个串行端口（即 Windows 系统中的 COM1 口）。

    还可以将`console=ttyS0`改为`console=ttyS0,115200n8`，其中

    * `115200`：波特率（Baud Rate），为 115200 bps（比特每秒），表示数据传输的速度。

    * `n`：奇偶校验（Parity），n 代表 “none”，即无奇偶校验。

    * `8`：数据位（Data Bits），为 8 个数据位。

* select / poll 底层机制并不是轮询（Busy Polling），只有在处理事件 fd 时才是线性查找

* epoll examples

    1. example 1

        ```c
        // 1. 创建 socket，bind，listen
        int listen_sock = setup_listening_socket();

        // 2. 创建 epoll 实例
        int epfd = epoll_create1(0);

        // 3. 将监听 socket 添加到 epoll，关注其可读事件（有新连接）
        struct epoll_event event;
        event.events = EPOLLIN;
        event.data.fd = listen_sock;
        epoll_ctl(epfd, EPOLL_CTL_ADD, listen_sock, &event);

        while (1) {
            // 4. 等待事件
            struct epoll_event events[MAX_EVENTS];
            int n = epoll_wait(epfd, events, MAX_EVENTS, -1);

            for (int i = 0; i < n; i++) {
                // 5. 处理事件
                if (events[i].data.fd == listen_sock) {
                    // 监听socket可读，说明有新连接到来
                    int conn_sock = accept(listen_sock, ...);
                    // 将新连接的 socket 也加入 epoll 监控
                    set_nonblocking(conn_sock); // ET模式必须设为非阻塞
                    event.events = EPOLLIN | EPOLLET; // 使用ET模式
                    event.data.fd = conn_sock;
                    epoll_ctl(epfd, EPOLL_CTL_ADD, conn_sock, &event);
                } else {
                    // 普通客户端socket可读，进行数据读写
                    handle_connection(events[i].data.fd);
                }
            }
        }
        ```

    2. example 2

        一个完整的、可编译运行的 epoll 示例，它是一个简单的回显（Echo）服务器。

        ```c
        #include <stdio.h>
        #include <stdlib.h>
        #include <string.h>
        #include <unistd.h>
        #include <arpa/inet.h>
        #include <sys/socket.h>
        #include <sys/epoll.h>
        #include <fcntl.h>
        #include <errno.h>

        #define PORT 8080
        #define MAX_EVENTS 10
        #define BUFFER_SIZE 1024

        // 设置文件描述符为非阻塞模式
        void set_nonblocking(int fd) {
            int flags = fcntl(fd, F_GETFL, 0);
            fcntl(fd, F_SETFL, flags | O_NONBLOCK);
        }

        // 创建监听socket
        int create_listen_socket() {
            int listen_fd;
            struct sockaddr_in server_addr;

            // 创建socket
            if ((listen_fd = socket(AF_INET, SOCK_STREAM, 0)) == -1) {
                perror("socket");
                exit(EXIT_FAILURE);
            }

            // 设置SO_REUSEADDR选项
            int opt = 1;
            if (setsockopt(listen_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt)) == -1) {
                perror("setsockopt");
                close(listen_fd);
                exit(EXIT_FAILURE);
            }

            // 绑定地址
            server_addr.sin_family = AF_INET;
            server_addr.sin_addr.s_addr = INADDR_ANY;
            server_addr.sin_port = htons(PORT);

            if (bind(listen_fd, (struct sockaddr*)&server_addr, sizeof(server_addr)) == -1) {
                perror("bind");
                close(listen_fd);
                exit(EXIT_FAILURE);
            }

            // 开始监听
            if (listen(listen_fd, SOMAXCONN) == -1) {
                perror("listen");
                close(listen_fd);
                exit(EXIT_FAILURE);
            }

            printf("Server listening on port %d...\n", PORT);
            return listen_fd;
        }

        // 处理客户端连接
        void handle_client(int client_fd) {
            char buffer[BUFFER_SIZE];
            ssize_t bytes_read;

            // 读取数据
            while ((bytes_read = read(client_fd, buffer, BUFFER_SIZE - 1)) > 0) {
                buffer[bytes_read] = '\0';
                printf("Received from client %d: %s", client_fd, buffer);
                
                // 回显数据
                if (write(client_fd, buffer, bytes_read) == -1) {
                    perror("write");
                    break;
                }
            }

            // 客户端断开连接或读取出错
            if (bytes_read == 0) {
                printf("Client %d disconnected\n", client_fd);
            } else if (bytes_read == -1) {
                if (errno != EAGAIN && errno != EWOULDBLOCK) {
                    perror("read");
                }
            }
        }

        int main() {
            int listen_fd, epoll_fd;
            struct epoll_event event, events[MAX_EVENTS];

            // 创建监听socket
            listen_fd = create_listen_socket();

            // 创建epoll实例
            if ((epoll_fd = epoll_create1(0)) == -1) {
                perror("epoll_create1");
                close(listen_fd);
                exit(EXIT_FAILURE);
            }

            // 添加监听socket到epoll，关注可读事件
            event.events = EPOLLIN;
            event.data.fd = listen_fd;
            if (epoll_ctl(epoll_fd, EPOLL_CTL_ADD, listen_fd, &event) == -1) {
                perror("epoll_ctl");
                close(listen_fd);
                close(epoll_fd);
                exit(EXIT_FAILURE);
            }

            printf("Epoll server started. Waiting for connections...\n");

            while (1) {
                // 等待事件发生
                int nfds = epoll_wait(epoll_fd, events, MAX_EVENTS, -1);
                if (nfds == -1) {
                    perror("epoll_wait");
                    break;
                }

                // 处理所有就绪的事件
                for (int i = 0; i < nfds; i++) {
                    // 有新连接到来
                    if (events[i].data.fd == listen_fd) {
                        struct sockaddr_in client_addr;
                        socklen_t client_len = sizeof(client_addr);
                        int client_fd;

                        // 接受新连接
                        client_fd = accept(listen_fd, (struct sockaddr*)&client_addr, &client_len);
                        if (client_fd == -1) {
                            perror("accept");
                            continue;
                        }

                        // 设置客户端socket为非阻塞模式
                        set_nonblocking(client_fd);

                        // 添加客户端socket到epoll，关注可读事件（使用边缘触发模式）
                        event.events = EPOLLIN | EPOLLET;
                        event.data.fd = client_fd;
                        if (epoll_ctl(epoll_fd, EPOLL_CTL_ADD, client_fd, &event) == -1) {
                            perror("epoll_ctl");
                            close(client_fd);
                            continue;
                        }

                        printf("New client connected: %d\n", client_fd);

                    } 
                    // 客户端socket可读
                    else if (events[i].events & EPOLLIN) {
                        handle_client(events[i].data.fd);
                        
                        // 注意：在实际应用中，你可能需要更复杂的连接管理
                        // 这里简单处理，读取完成后就关闭连接
                        printf("Closing connection for client %d\n", events[i].data.fd);
                        close(events[i].data.fd);
                    }
                }
            }

            // 清理资源
            close(listen_fd);
            close(epoll_fd);
            return 0;
        }
        ```

        compile: `gcc -o epoll_server epoll_server.c`

        run: `./epoll_server`

        test:

        ```bash
        # 使用 telnet 测试
        telnet localhost 8080

        # 或者使用 netcat
        nc localhost 8080

        # 或者使用多个终端同时连接测试
        ```

        输入一些文字，服务器会回显你输入的内容

* epoll 的两种工作模式

    1. 水平触发 (Level-Triggered, LT) (默认模式)

        条件满足就持续通知：只要一个文件描述符还有数据可读，每次调用 epoll_wait 都会返回它的事件。

        优点：编码简单，不容易遗漏事件。

        注意：如果不读完数据，会一直被通知。

    2. 边缘触发 (Edge-Triggered, ET)

        状态变化时只通知一次：只有当文件描述符从不可读变为可读（即新数据到来）时，才会收到一次通知。

        优点：性能更高，减少了事件被重复触发的次数。

        要求：必须使用非阻塞 I/O，并且必须一次性把缓冲区里的数据全部读完（直到 read 返回 EAGAIN 错误），否则可能会永远丢失这次事件。

* epoll

    epoll 是 Linux 系统上一种高效的多路复用 I/O 机制，用于同时监控大量的文件描述符（如网络套接字），看它们是否可读、可写或出现异常。它非常适合处理高并发网络服务器。

    与其不断地轮询所有连接（像 select/poll 那样），epoll 采用了“事件通知”的方式。当内核检测到某个被监控的描述符就绪时，它会通知应用程序，从而避免了无效的检查。

    使用 epoll 的方法：

    1. 创建一个 epoll 实例 (epoll_create)

        ```c
        int epfd = epoll_create1(0); // 参数通常传0
        ```

        `epfd`是一个文件描述符，代表这个 epoll 实例。后续所有操作都要用到它。

    2. 管理 epoll 监控列表 (epoll_ctl)

        通过 epoll_ctl 向这个实例（epfd）中添加、修改或删除需要监控的文件描述符。

        ```c
        // 添加一个 socket fd 到监控列表中，关注其可读事件
        struct epoll_event event;
        event.events = EPOLLIN; // 监控可读事件
        event.data.fd = sockfd; // 当事件发生时，我们知道是哪个fd触发的

        epoll_ctl(epfd, EPOLL_CTL_ADD, sockfd, &event);
        ```

        * `EPOLL_CTL_ADD`: 添加

        * `EPOLL_CTL_MOD`: 修改

        * `EPOLL_CTL_DEL`: 删除

        events 常用标志：

        * EPOLLIN: 文件描述符可读（有数据到来）

        * EPOLLOUT: 文件描述符可写

        * EPOLLET: 设置为边缘触发（Edge-Triggered）模式（默认为水平触发 Level-Triggered）

    3. 等待事件发生 (epoll_wait)

        调用 epoll_wait 来等待事件发生。这个函数会阻塞，直到有一个或多个被监控的描述符就绪。

        ```c
        #define MAX_EVENTS 10
        struct epoll_event events[MAX_EVENTS];

        // 等待事件发生，超时时间设为 -1 表示一直阻塞
        int nfds = epoll_wait(epfd, events, MAX_EVENTS, -1);

        // 处理所有就绪的事件
        for (int i = 0; i < nfds; i++) {
            if (events[i].events & EPOLLIN) { // 如果是可读事件
                int ready_fd = events[i].data.fd;
                // 对这个 ready_fd 进行读操作（如 recv, accept）
            }
            // 可以检查其他事件，如 EPOLLOUT
        }
        ```

        * epoll_wait 返回就绪的事件数量 nfds。

        * 事件数组 events 会被填充，我们可以遍历这个数组来处理所有就绪的 I/O 操作。

* `openat()`

    在一个特定的目录文件描述符所指向的目录下，打开或创建一个文件。

    syntax:

    ```c
    #include <fcntl.h>

    int openat(int dirfd, const char *pathname, int flags, ... /* mode_t mode */);
    ```

    参数说明：

    * `dirfd`：一个指向目录的文件描述符。它也可以是一些特殊值：

        一个普通的目录文件描述符（通过 open 某个目录获得）。

        AT_FDCWD：一个特殊值，表示“相对于当前工作目录”。如果指定这个值，openat() 的行为就完全等同于传统的 open()，但它仍然为其他 *at() 系列函数（如 fstatat）提供一致性。

    * `pathname`：要打开的文件路径。它可以是：

        绝对路径（如 /tmp/file）：此时 dirfd 参数会被忽略。

        相对路径（如 file.txt）：此时路径是相对于 dirfd 所指向的目录来解释的。

    * `flags` 和 `mode`：与 open() 函数的参数完全相同，用于指定打开标志（如 O_RDONLY, O_CREAT）和创建文件时的权限。

    返回值：

    成功时：返回一个新打开的文件描述符（一个非负整数）。

    失败时：返回 -1，并设置全局变量 errno 来指示具体的错误原因。

    它是对经典 open() 系统调用的扩展，解决了 open() 在某些场景下的两个关键问题：

    * 竞态条件（Race Conditions）

        竞态条件： 在多线程程序中，如果一个线程在 chdir() 之后、open() 之前，另一个线程也调用了 chdir()，那么第一个线程就会打开错误的文件。这是一个非常经典的TOCTOU（检查时间与使用时间）竞态条件漏洞。

    * 维护进程的“当前工作目录”状态

        使用 openat() 的现代方法：

        * `dirfd = open("/a/b/c", O_DIRECTORY)`: 只打开目录，获取其文件描述符 dirfd

        * `fd = openat(dirfd, "file.txt", ...)`: 在 dirfd 指向的目录下打开文件`

* `sync()`

    将内核缓冲区中所有未写入磁盘的数据（包括文件数据、元数据如inode等）立即写入到硬盘。

    syntax:

    ```c
    #include <unistd.h>

    void sync(void);
    ```

    `sync()`调用本身是异步的。它只是启动写入操作，不会等待所有数据实际写完才返回。

    sync() 会立即触发一个流程，通知内核将所有脏页（被修改过但未写入磁盘的缓冲区内容）排队写入磁盘。它作用于整个系统，刷新所有内核缓冲区，而不仅仅是调用它的那个进程的缓冲区。

* `getpid()`

    获取当前的进程 id（PID）。

    `getppid()`: 获取当前进程的父进程的进程ID (PPID)。

    除了系统启动时的第一个进程（init 或 systemd，PID 通常为 1），每个进程都有父进程。

    如果父进程先于子进程结束，子进程就会变成“孤儿进程”，并被 init 进程（PID 1）收养。此时，子进程调用 getppid() 将返回 1。

* 写时复制（Copy-On-Write, COW）

    只有在真正需要写入（修改）数据时，才会去复制一份副本。在此之前，所有对象（或进程）共享同一份原始数据。

    `fork()`使用了 cow 机制，因此可快速创建一个新进程。

    在调用`fork()`时，内核会把当前进程的所有内存 page 改成只读权限，如果旧进程或新进程尝试往内存写入数据，那么会触发页错误（Page Fault），此时内核会把这一页数据复制一份新的，供尝试写入数据的进程使用。

    每个进程尝试写入数据，都会触发一次 page fault。所以如果新旧进程都写入了数据，那么目前会有三份数据：

    1. P：最初的共享数据（如果再无其他进程共享，它可能会被回收）。

    2. P_father：父进程的私有副本，包含了父进程的修改。

    3. P_child：子进程的私有副本，包含了子进程的修改（基于最初的数据，而非父进程修改后的数据）。

    写时复制（COW）的操作粒度通常是一个内存页（Page）。

* `munmap()`主要用于释放进程的虚拟地址。

    如果 mmap() 映射的是文件，那么`munmap()`会在解除映射时把数据写回文件。

* `mmap()`内部原理

    将进程的一段 va　映射到某个对象上（文件，或内存），程序访问这段虚拟内存时，操作系统通过缺页异常（Page Fault）来自动完成数据的加载和同步。

    mmap 可以实现延迟加载（Lazy Loading）：调用 mmap 时，操作系统并不会立即将整个文件内容读入物理内存。它只是在内核中为进程创建一个数据结构（Linux 中是 `vm_area_struct`），记录下这个映射关系（例如：虚拟地址范围 0x4000 - 0x5000 对应文件 a.txt 的偏移 0 - 4096 字节）。这个过程非常快，消耗资源极少，并且与文件大小无关。真正的数据加载发生在程序首次访问对应的内存地址时。

    虚拟内存区域（VMA - Virtual Memory Area）

    `vm_area_struct`:

    vm_start, vm_end: 这段映射的起始和结束虚拟地址。

    vm_file: 被映射的文件。

    vm_pgoff: 文件中的偏移量（以页为单位）。

    vm_flags: 权限标志（如可读、可写、私有映射、共享映射）。

    进程访问 va 时，MMU 触发一个 缺页异常（Page Fault），CPU 从用户态陷入内核态。内核找到 va 对应的 vma，然后根据 vma 找到对应的文件，将文件内容按 page size （4KB）读到 page 中（数据在物理内存里），然后更新进程的页表，建立 virtual page 到 physical page 的映射。

    此时返回到用户态，并重新执行那条触发异常的指令。

    此后进程读写的都是 physical page 中的内容，此物理页被内核标记为脏页（dirty），意味着它比磁盘上的文件内容更新。

    最终，内核的 pdflush（页回写）守护进程会在后台自动将“脏页”写回到磁盘文件中，以保持数据同步。应用程序也可以主动调用 msync() 来强制立即同步数据。

* 多路复用（select/poll/epoll）中的多路指的是独立的I/O流或连接通道，通常指指大量的网络 socket 连接，复用指的是复用同一个线程/进程。

    多路复用 - multiplexing

* 如果在 fork 前父进程打开了一个文件，拿到一个 fd，那么在 fork 后，父进程的 fd 和子进程的 fd 相同，并且共享同一组状态数据，比如 offset 等

    但是一个进程 close 了 fd，并不会使另一个进程 read 数据失败，因为 fd 采用引用计数机制。

    example:

    ```c
    #include <stdio.h>
    #include <fcntl.h>
    #include <unistd.h>

    int main() {
        int fd = open("msg.txt", O_RDONLY);
        if (fd < 0) {
            printf("fail to open file\n");
            return -1;
        }
        char buf[16] = {0};

        pid_t pid = fork();

        if (pid < 0) {
            fprintf(stderr, "fail to fork\n");
            return -1;
        } else if (pid > 0) {
            sleep(2);  // 主进程等待子进程读取文件数据
        }

        ssize_t bytes_read = read(fd, buf, 10);
        if (bytes_read < 10) {
            printf("fail to read\n");
            return -1;
        }

        printf("buf: [%s]\n", buf);

        if (pid == 0) {
            int ret = close(fd);
            if (ret != 0) {
                printf("fail to close fd\n");
                return -1;
            }
        } else {
            sleep(2);  // 主进程等子进程关闭 fd 后，再尝试读数据
            bytes_read = read(fd, buf, 5);
            buf[5] = '\0';
            printf("buf from parent process: [%s]\n", buf);
            int ret = close(fd);
            if (ret != 0) {
                printf("fail to close fd\n");
                return -1;
            }
        }

        return 0;
    }
    ```

    output:

    ```
    buf: [hello, wor]
    buf: [ld, nihao,]
    buf from parent process: [ zaij]
    ```

* `read(int fd, void *buf, size_t count);`读取的是 count 个字节，不会把`bus[count - 1]`设置为`\0`。

* `fsync()`

    将系统缓冲区中的内容写入到磁盘，阻塞等待。

    syntax:

    ```c
    #include <unistd.h>

    int fsync(int fd);
    ```

* fwrite() 相比系统调用 write() 增加了缓冲区，write() 在操作系统中也使用了缓冲区，这两个缓冲区还是一回事

    `fwrite()`的缓冲区是 C 运行时库（比如 glibc）设置的缓冲区，目的是减少系统调用。存在于进程的地址空间中，进程结束即消失。与此相关的`fflush()`本质是强制进行一次`write()`系统调用。

    `write()`的缓冲区 (page cache) 由操作系统提供，由所有进程、所有 fd 共享。

* `fcntl()`

    对文件描述符 fd 下发各种 control 命令。

    syntax:

    ```c
    #include <fcntl.h>

    int fcntl(int fd, int cmd, ... /* arg */ );
    ```

    常用功能（未验证）：

    * 复制文件描述符 (F_DUPFD, F_DUPFD_CLOEXEC)

        复制一个已有的文件描述符，创建一个新的描述符指向同一个文件。

        `fcntl(old_fd, F_DUPFD, new_fd);`

        比较智障的是，这个新复制的 new_fd 和 old_fd 共享同一个 fd struct，因此文件的 offset 仍是共享的。

    * 获取/设置文件描述符标志 (F_GETFD, F_SETFD)

        FD_CLOEXEC（Close-on-Execute），设置此标志后，当进程执行 exec() 系列函数加载新程序时，该文件描述符会被自动关闭，防止它被意外继承到新程序中。

        如果一个文件在打开时是 read only，那么后续不可以通过 fcntl() 改成 rdwr。fcntl() 的 F_SETFL 命令无法改变文件的访问模式（Access Mode）。

        可以改变的几个标记：

        O_APPEND：强制每次写入都追加到文件末尾。

        O_NONBLOCK：设置为非阻塞模式。

        O_ASYNC：启用信号驱动I/O。

        O_DIRECT：尝试最小化缓存效应。

    * 获取/设置文件状态标志 (F_GETFL, F_SETFL)

    * 管理文件锁 (F_GETLK, F_SETLK, F_SETLKW)

        作用：对文件区域施加建议性锁 (Advisory Lock)。

        F_GETLK：检查是否可以加锁。

        F_SETLK：尝试加锁（非阻塞，如果冲突立即返回错误）。

        F_SETLKW：尝试加锁（阻塞，如果冲突则等待直到锁可用）。

        这是一种“建议性”锁，意味着它只对同样使用 fcntl() 检查锁的进程有效。如果一个进程不检查锁直接读写，锁是无法阻止它的。

    * 信号驱动I/O (F_SETOWN, F_GETOWN, F_SETSIG, F_GETSIG)

        设置当文件描述符上发生I/O事件（例如数据可读）时，应该接收信号的进程或进程组。这是实现异步I/O的一种传统方式。

    整体看下来，`fcntl()`用处不大，处理的基本都是边角料情况。等用到了再说。

* `fork()`创建的是新的进程，不是新的线程，所以父进程与子进程的内存都是独立的

    example:

    ```c
    #include <stdio.h>
    #include <unistd.h>

    int main() {
        int val = 0;

        pid_t pid = fork();

        if (pid < 0) {
            fprintf(stderr, "fail to fork\n");
            return -1;
        } else if (pid == 0) {
            val = 456;
        } else {
            val = 123;
        }

        printf("val is %d\n", val);
        return 0;
    }
    ```

    output:

    ```
    val is 123
    val is 456
    ```

* `fork()`

    复制当前进程的资源，创建一个新的子进程。

    syntax:

    ```c
    #include <unistd.h>

    pid_t fork(void);
    ```

    如果返回值为 0，那么说明当前的进程已经来到了子进程，如果返回值为非 0，那么说明当前的进程仍是父进程。

    example:

    ```c
    #include <stdio.h>
    #include <unistd.h>

    int main() {
        pid_t pid = fork();

        if (pid < 0) {
            fprintf(stderr, "fail to fork\n");
            return -1;
        } else if (pid == 0) {  // pic == 0 means this is a child process
            printf("my pid: %d, my parent pid :%d\n", getpid(), getppid());
        } else {  // parent process
            printf("my pid: %d, my child pid: %d\n", getpid(), pid);
        }

        printf("a greeting from parent process and child process\n");
        return 0;
    }
    ```

    output:

    ```
    my pid: 886283, my child pid: 886284
    a greeting from parent process and child process
    my pid: 886284, my parent pid :886283
    a greeting from parent process and child process
    ```

    操作系统内核会为子进程创建一个新的 PCB，用于调度。

* `msync()`

    （未验证）

    将内存中的内容写回到文件。

    操作系统会不定期将`mmap()`内存中的内容写回到文件，但是如果我们对进程间同步的要求较高，那么就需要手动`msync()`。

    syntax:

    ```c
    #include <sys/mman.h>

    int msync(void *addr, size_t length, int flags);
    ```

    其中`flags`可取值如下：

    * `MS_SYNC`：回写完成后函数返回。

    * `MS_ASYNC`：发出回写命令，函数立即返回。

    * `MS_INVALIDATE`：通知其他进程的映射副本失效，使其他进程重新读取文件内容。

    如果是匿名映射，那么`msync()`无意义。

    进程 A 和 B 同时以 shared 模式 mmap 一个文件，进程 A 修改文件，进程 B 并不会定期重新读取文件，除非遇到`MS_INVALIDATE`的`msync()`。

* `mmap()`匿名映射

    匿名映射 Anonymous Mapping

    匿名映射不与磁盘文件关联，直接分配虚拟内存供进程使用

    example:

    ```cpp
    #include <sys/mman.h>
    #include <stdio.h>
    #include <string.h>

    int main() {
        void *buf = mmap(NULL, 1024, PROT_READ | PROT_WRITE,
            MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
        if (buf == NULL) {
            printf("fail to mmap\n");
            return -1;
        }

        strcpy((char*) buf, "hello, world");

        printf("msg: %s\n", (char*) buf);
      
        int ret = munmap(buf, 1024);
        if (ret != 0) {
            printf("fail to munmap\n");
            return -1;
        }
        return 0;
    }
    ```

    output:

    ```
    msg: hello, world
    ```

    这个似乎可以拿来替换`malloc()`，如果是 shared 模式还可以在进程间通信。

* `mmap()`的`MAP_SHARED`模式与`MAP_PRIVATE`模式

    在 shared 模式中，对映射内存的修改会同步到实际文件（如果映射的是文件），其他进程映射同一文件时能看到变更。内存的写操作可能触发文件系统的 I/O（取决于回写策略）。

    在 private 模式中，对映射内存的修改永远不会同步到文件，而是创建进程私有的写时复制（Copy-on-Write, COW）副本。

    （因为不会写多进程程序，所以这里先不写 example）

    MAP_SHARED 的同步可能引入 I/O 延迟，MAP_PRIVATE 的 COW 机制可能导致内存分裂。（什么是内存分裂？）

* `stat()`

    头文件：`#include <sys/stat.h>`

    syntax:

    ```c
    int stat(const char *restrict pathname,
            struct stat *restrict statbuf);
    ```

    返回文件信息。

    这个函数和`fstat()`唯一区别是，`fstat()`使用的是`fd`，而`stat()`使用的是文件路径。

* `lstat()`

    如果路径指向符号链接，`lstat()`返回的是符号链接本身的信息（如链接文件的大小、权限等），而`stat()`会处理链接指向的文件。

    syntax:

    ```c
    #include <sys/stat.h>
    int lstat(const char *pathname, struct stat *statbuf);
    ```

    example:

    ```c
    #include <sys/stat.h>
    #include <stdio.h>
    #include <unistd.h>

    int check_link_file(const char *file_path) {
        struct stat my_stat;
        int ret = lstat(file_path, &my_stat);
        if (ret != 0) {
            printf("fail to run fstat()\n");
            return -1;
        }

        if (S_ISLNK(my_stat.st_mode)) {
            printf("%s is a link file\n", file_path);
        } else {
            printf("%s is not a link file\n", file_path);
        }

        return 0;
    }

    int main() {
        const char *file_paths[2] = {
            "msg.txt",
            "msg_link.txt"
        };

        int ret = check_link_file(file_paths[0]);
        if (ret != 0) {
            printf("fail to check link file: %s\n", file_paths[0]);
            return -1;
        }

        ret = check_link_file(file_paths[1]);
        if (ret != 0) {
            printf("fail to check link file: %s\n", file_paths[1]);
            return -1;
        }

        return 0;
    }
    ```

    output:

    ```
    msg.txt is not a link file
    msg_link.txt is a link file
    ```

* `open()`的文件覆盖问题

    使用 `open()`函数创建新文件时，在旧文件存在的情况下，如果 flag 中仅有`O_CREAT`，那么不会覆盖旧文件，直接打开现有文件。如果 flag 为`O_CREAT | O_EXCL`，则打开失败，如果文件不存在，则创建新文件。如果 flag 为`O_CREAT | O_TRUNC`，则会覆盖旧文件。

    总结：

    * 默认不覆盖：仅用`O_CREAT`会保留旧文件内容。

    * 禁止覆盖：`O_EXCL`确保文件不存在时才创建。

    * 显式覆盖：`O_TRUNC`强制清空旧文件。

* `memmem()`

    用于在一段内存中搜索指定内容的位置。

    syntax:

    ```c
    #include <string.h>

    void *memmem(const void *haystack, size_t haystacklen,
                 const void *needle, size_t needlelen);
    ```

    example:

    ```cpp
    #include <string.h>
    #include <stdio.h>

    int main() {
        char buf[128] = {'n', 'i', '\0', 'h', 'a', 'o', '\0', 1, 2, 3};
        char sub[3] = {'o', '\0', 1};
        char *pos = (char*) memmem(buf, 128, sub, 3);
        for (int i = 0; i < 3; ++i) {
            printf("%d, ", *(pos+i));
        }
        putchar('\n');
        return 0;
    }
    ```

    output:

    ```
    111, 0, 1,
    ```

* `mmap()`

    `mmap()`是 linux 的系统调用，可以把文件映射到内存空间。

    syntax:

    ```cpp
    #include <sys/mman.h>
    void *mmap(void *addr, size_t length, int prot, int flags, int fd, off_t offset);
    ```

    参数说明：

    * `addr`：建议映射的起始地址（通常为 NULL，由内核决定）。

    * `length`：映射区域的长度。

    * `prot`：保护权限（如 PROT_READ | PROT_WRITE）。

    * `flags`：映射类型（如 MAP_SHARED 或 MAP_PRIVATE）。

    * `fd`：文件描述符（匿名映射时设为 -1）。

    * `offset`：文件偏移量（通常为 0）。

    example:

    ```cpp
    #include <sys/mman.h>
    #include <fcntl.h>
    #include <unistd.h>
    #include <errno.h>
    #include <stdio.h>
    #include <string.h>

    int main() {
        int fd = open("msg.txt", O_RDWR);
        if (fd < 0) {
            printf("fail to open file, errno: %d\n", errno);
            return -1;
        }

        int ret;
        const char *msg = "hello, world\n";
        size_t msg_len;

        void *addr = mmap(NULL, 4096, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        if (addr == NULL) {
            printf("fail to mmap\n");
            goto MMAP_FAILED;
        }

        msg_len = strlen(msg);

        memcpy(addr, msg, msg_len);

        munmap(addr, 4096);

    MMAP_FAILED:
        ret = close(fd);
        if (ret != 0) {
            printf("fail to close fd\n");
            return -1;
        }
        return 0;
    }
    ```

    `msg.txt`:

    ```
    hello
    world
    nihao
    zaijian
    ```

    运行代码后的`msg.txt`:

    ```
    hello, world
    ihao
    zaijian
    ```

    如果将`memcpy()`改成`strcpy()`，则会在文件中写入`\0`，导致文件被写入不可解析的字符。

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
