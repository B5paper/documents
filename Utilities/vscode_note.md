# VSCode Note

## cache

* c++ 中的 string & 在 vscode 中，debug 断点模式下，鼠标悬停不显示内容

    下面是实测结果：

    ```cpp
    #include <string>
    #include <unordered_map>
    using namespace std;

    int main() {
        string str = "hello, world";  // 显示
        string &str_2 = str;  // 不显示
        const string &str_3 = str;  // 不显示
        const string &str_4 = "hello, world";  // 不显示

        unordered_map<string, string> umap {
            {"hello", "world"},
            {"nihao", "zaijian"}
        };
        string &str_val = umap["hello"];  // 不显示
        const string &con_str_val = umap["nihao"];  // 不显示
        return 0;
    }
    ```

* vscode 中 gdb pretty print

    ```json
    "configurations": [{
        "name": "(gdb) Launch",
        // ...
        "setupCommands": [
            {
                "description": "Enable pretty-printing for gdb",
                "text": "-enable-pretty-printing",
            },
            // ...
        ]
    }]
    ```

    必须设置`-enable-pretty-printing`这个才可以直接显示 string 的内容。否则显示的都是原始的指针。

* vscode sync scroll

    vscode 中同时滚动左右两个分栏

    extension name: `Sync Scroll`

    安装完后，`Ctrl` + `Shift` + `P`打开命令栏，搜索 sync scroll，会看到一个 change sync scroll mode，选择之后，会有三个选项:

    * normal

        保持左右两栏行数相同。

    * offset

        保持左右两栏滚动相同的距离

    * off

    在 vscode 右下角也会有当前 sync scroll 模式的状态。

* vscode 快捷键

    * 移动到下一个词的开头	Ctrl + →

    * 移动到上一个词的开头	Ctrl + ←

    * 选择到下一个词的开头	Ctrl + Shift + →

    * 选择到上一个词的开头	Ctrl + Shift + ←

    * 删除前一个词	Ctrl + Backspace

    * 删除后一个词	Ctrl + Delete

    * 打开设置  Ctrl + ,

* vscode 在`launch.json`中没有设置`cwd`时，程序中的`./`表示用户目录。比如`/share_data/users/hlc`

* 关于 vscode 中 gdb 调试 c++ 时，无法鼠标悬停显示`const string &str`的调研

    1. ds 的输出为 GDB 显示的是 std::string 在 libstdc++（GCC 的标准库实现）中的内部结构，而非直接显示字符串内容。这是因为 GDB 默认以“结构体/类成员”的形式显示对象，而没有自动调用 std::string 的字符串解码逻辑。

    2. 只有 local 变量窗口和 watch 变量窗口可以正确显示`const string &str`的内容，鼠标悬停无法直接显示。试了下默认的 lldb，比 gdb 更差，鼠标悬停时根本不解析`const string&`，只解析一些基本的 C 语言的数据结构。

        重新看了下，只有`const string &str = xxx;`这一行里，`str`无法正常显示，在其他行里`str`在鼠标悬停时可以正常显示。

* vscode 中，使用`-exec p`打印字符串数组，会把`\000`也打印出来

    ```cpp
    #include <stdio.h>

    int main()
    {
        char str[16] = "hello";
        printf("%s\n", str);
        return 0;
    }
    ```

    在`return 0;`前下断点，在 vscode debug console 中调用`-exec p str`，输出如下：

    ```
    $1 = "hello\000\000\000\000\000\000\000\000\000\000"
    ```

    而正常的 terminal 中，`printf()`的输出如下：

    ```
    hello
    ```

* vscode 可以使用 ctrl + shift + B 编译，使用 F5 启动 debug。

* `\bold`无法被 vscode 和 markdown preview 渲染插件识别，可以使用`\mathbf`来指定正粗体。

* vscode 中，取消了 tab stop 后，还是会有 tab 缩进 2 个空格的现象，这时候还需要取消 Detect Indentation

* 如果 vscode 中在编辑 makefile 时，tab 键总是插入 4 个空格而不是 tab，可以在 vscode setting 里把 Detect indentation 选项关了再打开一次就好了

* vscode 的 pipe 模式

    vscode 可以在`launch.json`中设置 pipe 模式，通过 ssh 在 remote host 上调用 gdb，再把 gdb 的输入输出重定向到 local host，从而不需要在 remote host 上安装 vscode，也不需要安装任何 vscode 插件，即可远程调试程序。

    配置方法：

    1. 在 remote host 上新建一个工程文件夹`mkdir -p /home/hlc/Documents/Projects/vscode_test`

        创建程序文件`main.c`:

        ```c
        #include <stdio.h>

        int main()
        {
            printf("hello world\n");
            return 0;
        }
        ```

        编译：`gcc -g main.c -o main`

    2. remote host 上还需要安装 ssh server 和 gdb，此时即可满足最小调试要求。

    3. 在 local host 上创建`launch.json`：

        ```json
        {
            // Use IntelliSense to learn about possible attributes.
            // Hover to view descriptions of existing attributes.
            // For more information, visit: https://go.microsoft.com/fwlink/?linkid=830387
            "version": "0.2.0",
            "configurations": [
                {
                    "name": "(gdb) Pipe Launch",
                    "type": "cppdbg",
                    "request": "launch",
                    "program": "/remote_host/path/to/main",
                    "args": [],
                    "stopAtEntry": false,
                    "cwd": "${fileDirname}",
                    "environment": [],
                    "externalConsole": false,
                    "pipeTransport": {
                        "debuggerPath": "/usr/bin/gdb",
                        "pipeProgram": "/usr/bin/ssh",
                        "pipeArgs": [
                            // "-pw",
                            // "password",
                            "hlc@10.0.2.4"
                        ],
                        "pipeCwd": "/usr/bin"  // this line
                    },
                    "MIMode": "gdb",
                    "setupCommands": [
                        {
                            "description": "Enable pretty-printing for gdb",
                            "text": "-enable-pretty-printing",
                            "ignoreFailures": true
                        },
                        {
                            "description": "Set Disassembly Flavor to Intel",
                            "text": "-gdb-set disassembly-flavor intel",
                            "ignoreFailures": true
                        }
                    ]
                }
            ]
        }
        ```

        标注为`// this line`的那一行可以为空字符串，不影响。

        `-pw password`实测不支持。不清楚为什么 vscode 官网写了这个用法。

    4. 在 local host 上创建一份和 remote host 上一模一样的`main.c`文件，并打上断点

        按 F5 运行，此时会显示要求安装一个交互式输入 ssh 密码的小工具`ssh-askpass`。执行`sudo apt install ssh-askpass`。

        按照提示输入 ssh fingerprint 和密码后，即可正常 hit 断点。如果嫌麻烦可以在 local host 上运行`ssh-copy-id <user>@<remote_host_ip>`把 local host 的 ssh public key 添加到 remote host。

        debug console 的输出如下：

        ```
        =thread-group-added,id="i1"
        GNU gdb (Ubuntu 15.0.50.20240403-0ubuntu1) 15.0.50.20240403-git
        Copyright (C) 2024 Free Software Foundation, Inc.
        License GPLv3+: GNU GPL version 3 or later <http://gnu.org/licenses/gpl.html>
        This is free software: you are free to change and redistribute it.
        There is NO WARRANTY, to the extent permitted by law.
        Type "show copying" and "show warranty" for details.
        This GDB was configured as "x86_64-linux-gnu".
        Type "show configuration" for configuration details.
        For bug reporting instructions, please see:
        <https://www.gnu.org/software/gdb/bugs/>.
        Find the GDB manual and other documentation resources online at:
            <http://www.gnu.org/software/gdb/documentation/>.

        For help, type "help".
        Type "apropos word" to search for commands related to "word".
        Warning: Debuggee TargetArchitecture not detected, assuming x86_64.
        =cmd-param-changed,param="pagination",value="off"
        [Thread debugging using libthread_db enabled]
        Using host libthread_db library "/lib/x86_64-linux-gnu/libthread_db.so.1".

        Breakpoint 1, main () at main.c:5
        5	    printf("hello from ubuntu 2404\n");
        Loaded '/lib64/ld-linux-x86-64.so.2'. Symbols loaded.
        Loaded '/lib/x86_64-linux-gnu/libc.so.6'. Symbols loaded.
        Warning: Source file '/home/hlc/Documents/Projects/vscode_test/main.c' is newer than module file '/home/hlc/Documents/Projects/vscode_test/main'.
        Execute debugger commands using "-exec <command>", for example "-exec info registers" will list registers in use (when GDB is the debugger)
        ```

    说明：

    1. local host 的`main.c`文件与 remote host 不一致不影响远程调试，因为 remote gdb 在打断点时只关心文件名和第几行。

        如下图所示：

        <div style='text-align:center'>
        <img src='../../Reference_resources/ref_30/pic_1.png'>
        </div>

    refs:

    1. Pipe transport

        <https://code.visualstudio.com/docs/cpp/pipe-transport>

    2. VS Code Remote Development

        <https://code.visualstudio.com/docs/remote/remote-overview>

    3. How to debug with VSCode and pipe command

        <https://stackoverflow.com/questions/54052909/how-to-debug-with-vscode-and-pipe-command>

    4. How to specify password in ssh command

        <https://superuser.com/questions/1605215/how-to-specify-password-in-ssh-command>

* vscode `Ctrl` + `Shift` + `O` 可以跳转到 symbol，也可以使用 `Ctrl` + `T` 跳转。

    `Ctrl` + `Shift` + `O`只搜索当前文件，`ctrl` + `t`会搜索 work space 下的所有文件。

    ref: <https://code.visualstudio.com/docs/editor/editingevolved>

* vscode 中，取消了 tab stop 后，还是会有 tab 缩进 2 个空格的现象，这时候还需要取消 Detect Indentation

* 防止 vscode 里 tab 键做太多段落自动对齐的工作：

    取消 use tab stops

* vscode 关闭自动补全引号：

    auto closing quotes 设置成 never

* vscode 关闭自动补全括号：

    auto closing brackets 设置成 never

* 使用 vscode 调试 sudo 程序

    核心是需要 gdb 由 sudo 启动。

    可以在`launch.json`里加一行：

    `"miDebuggerPath": "/home/hlc/.local/bin/sudo_gdb.sh"`

    `sudo_gdb.sh`里只要写一行：

    ```bash
    #!/bin/bash
    sudo gdb "$@"
    ```

    然后`sudo chmod +x sudo_gdb.sh`。

    接着在 vscode 的 intergrated terminal 里输入`sudo echo 1`，正常输入密码。此时这个 terminal 里，root 权限会持续开启一段时间，使用`sudo`运行其他程序不需要再输入密码。

    这个时候就可以在 vscode 里运行 F5 调试程序了。

* vscode attach 不需要输入 root 的方法

    ```bash
    echo 0 | sudo tee /proc/sys/kernel/ya/ptrace_scope
    ```

    sudo 和 tee 连用，就可以让 echo 写字符到需要 root 权限的文件里？

* cuda vscode debug

    ```cu
    #include <stdio.h>
    #include <cuda_runtime.h>

    __global__ void vectorAdd(const float *A, const float *B, float *C, int numElements) 
    {
        int i = blockDim.x * blockIdx.x + threadIdx.x;

        if (i < numElements)
        {
            C[i] = A[i] + B[i] + 0.0f;
        }
    }

    int main()
    {
        int numElements = 1024;
        size_t size = numElements * sizeof(float);
        float *h_A = (float *) malloc(size);
        float *h_B = (float *) malloc(size);
        float *h_C = (float *) malloc(size);
        for (int i = 0; i < numElements; ++i)
        {
            h_A[i] = rand() / (float) RAND_MAX;
            h_B[i] = rand() / (float) RAND_MAX;
        }

        float *d_A = NULL;
        cudaMalloc((void**) &d_A, size);
        float *d_B = NULL;
        cudaMalloc((void**) &d_B, size);
        float *d_C = NULL;
        cudaMalloc((void**) &d_C, size);

        cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

        int threadsPerBlock = 256;
        int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
        vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);

        cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

        for (int i = 0; i < 4; ++i)
        {
            printf("%.2f + %.2f = %.2f\n", h_A[i], h_B[i], h_C[i]);
        }

        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        free(h_A);
        free(h_B);
        free(h_C);

        return 0;
    }
    ```

    compile:

    ```bash
    nvcc -g main.cu -o main
    ```

    run:

    ```
    ./main
    ```

    output:

    ```
    0.84 + 0.39 = 1.23
    0.78 + 0.80 = 1.58
    0.91 + 0.20 = 1.11
    0.34 + 0.77 = 1.10
    ```

    vscode debug:

    1. install vscode extension: `Nsight Visual Studio Code Edition`

    2. add debug type: cuda-gdb

    3. press F5 start debugging

    也可以直接使用`gdb <exe_file>`去调试。

* vscode 可以使用`ctrl + p`搜索文件名

1. invoke call stack window: <https://stackoverflow.com/questions/1607065/how-to-show-call-stack-immediates-and-other-hidden-windows-in-visual-studio>

1. <https://learn.microsoft.com/en-us/visualstudio/debugger/how-to-use-the-call-stack-window?view=vs-2022>

* vscode 使用 lldb 调试的技巧：<https://www.codenong.com/cs105826090/>

    感觉有用的内容不多，有时间了看看。

* ssh 多级跳转

    ```ssh
    Host jump1
    HostName ip of jump1
    Port port id
    User user name
    IdentityFile key_path

    Host jump2
    HostName ip of jump2
    Port port id
    User user name
    IdentityFile key_path
    # command line,setting jump1
    ProxyCommand ssh -q -W %h:%p jump1


    # Target machine with private IP addressHost target
    Host target
    HostName ip_of target
    Port port id
    User user name
    IdentityFile key_pathProxyCommand ssh -q -W %h:%p jump2
    ```

    比如一台 server 要经过 jump1 连到 jump2，再通过 jump2 连到 target，那么就可以用上面的配置。

    Ref: <https://www.doc.ic.ac.uk/~nuric/coding/how-to-setup-vs-code-remote-ssh-with-a-jump-host.html>

    <https://support.cs.wwu.edu/home/survival_guide/tools/VSCode_Jump.html>

    也可以直接用`proxyjump`，example:

    ```s
    Host 10.211.10.67
    HostName 10.211.10.67
    ProxyJump liucheng@172.18.25.248
    User test

    Host 192.168.122.75
    HostName 192.168.122.75
    ProxyJump liucheng@172.18.25.248
    User ubuntu
    ```

* vscode debug with a root privilege

    一种方式是创建一个`gdb`的脚本：

    ```bash
    #!/bin/bash
    pkexec /usr/bin/gdb "$@"
    ```

    然后在 vscode 中将`miDebuggerPath`设置成 gdb 脚本的 path 就好了。

    另一种方法是设置权限：`user_name ALL=(ALL) NOPASSWD:/usr/bin/gdb`，可以直接让 gdb 获得 root 权限。

    Ref: <https://stackoverflow.com/questions/40033311/how-to-debug-programs-with-sudo-in-vscode>

## note

