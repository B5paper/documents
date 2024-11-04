# VSCode Note

## cache

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

