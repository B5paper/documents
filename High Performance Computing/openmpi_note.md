# OpenMPI Note

## cache

* 一个动态分配内存的矩阵乘法 c 语言程序 demo

    ```c
    #include <stdlib.h>
    #include <stdio.h>

    typedef enum bool
    {
        false = 0,
        true
    } bool;

    void alloc_matrix(float ***M, size_t nrow, size_t ncol, bool assign_rand_val)
    {
        *M = malloc(nrow * sizeof(float *));
        for (size_t i = 0; i < nrow; ++i)
            *(*M + i) = malloc(ncol * sizeof(float));

        if (assign_rand_val == false)
            return;

        for (size_t i = 0; i < nrow; ++i)
        {
            for (size_t j = 0; j < ncol; ++j)
            {
                // (*M)[i][j] = (float) (rand() % 5);
                *(*(*M + i) + j) = (float) (rand() % 5);
            }
        }
    }

    void free_matrix(float **M, size_t ncol)
    {
        for (size_t i = 0; i < ncol; ++i)
            free(M[i]);
        free(M);
    }

    void print_matrix(float **M, size_t nrow, size_t ncol)
    {
        for (size_t i = 0; i < nrow; ++i)
        {
            for (size_t j = 0; j < ncol; ++j)
            {
                printf("%.0f, ", *(*(M + i) + j));
            }
            putchar('\n');
        }
    }

    int main()
    {
        float **A, **B, **C;
        size_t nrow_A = 4, ncol_A = 2;
        size_t nrow_B = 2, ncol_B = 3;
        size_t nrow_C = nrow_A, ncol_C = ncol_B;

        alloc_matrix(&A, nrow_A, ncol_A, true);
        printf("matrix A:\n");
        print_matrix(A, nrow_A, ncol_A);
        putchar('\n');

        alloc_matrix(&B, nrow_B, ncol_B, true);
        printf("matrix B:\n");
        print_matrix(B, nrow_B, ncol_B);
        putchar('\n');

        alloc_matrix(&C, nrow_C, ncol_C, false);

        for (size_t row = 0; row < nrow_C; ++row)
        {
            for (size_t col = 0; col < ncol_C; ++col)
            {
                C[row][col] = 0;
                for (size_t k = 0; k < ncol_A; ++k)
                {
                    C[row][col] += A[row][k] * B[k][col];
                }
            }
        }

        printf("matrix C = A.dot(B):\n");
        print_matrix(C, nrow_C, ncol_C);

        return 0;
    }
    ```

    说明：

    1. 为什么`(*M)[i][j]`仍然可以访问到元素，但是以前的例子都是在给函数传参数的时候，使用`M[][3]`之类的方式，才能正常访问数组的元素，这两者有什么不同？

    2. 是否可以定义指向数组的指针，比如

        ```c
        int arr[3];
        int[] *parr = &arr;
        ```

    3. c 语言中的`rand()`函数在`stdlib.h`中声明。`size_t`类型在`stddef.h`中定义。
    
        由于`stdlib.h`包含了`stddef.h`，所以如果定义了`stdlib.h`，就不需要再写`#include <stddef.h>`了。

* mpi status example code

    `main.c`:

    ```c
    #include <mpi.h>
    #include <stdio.h>
    #include <string.h>

    int main()
    {
        int ret = MPI_Init(NULL, NULL);
        if (ret != 0)
        {
            printf("fail to init mpi, ret: %d\n", ret);
            return -1;
        }
        printf("[OK] init mpi.\n");

        int world_size;
        ret = MPI_Comm_size(MPI_COMM_WORLD, &world_size);
        if (ret != 0)
        {
            printf("fail to get comm world isze, ret: %d\n", ret);
            return -1;
        }
        printf("[OK] get world size: %d.\n", world_size);

        int rank;
        ret = MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        if (ret != 0)
        {
            printf("fail to get rank, ret: %d\n", ret);
            return -1;
        }
        printf("[OK] get rank: %d.\n", rank);

        char buf[20] = {0};
        if (rank == 0)
        {
            char *msg = "hello from rank 0";
            size_t msg_len = strlen(msg);
            size_t min_len;
            if (msg_len > 19)
                min_len = 19;
            else
                min_len = msg_len;
            strncpy(buf, msg, min_len);
            ret = MPI_Send(buf, 20, MPI_CHAR, 1, 0, MPI_COMM_WORLD);
            if (ret != 0)
            {
                printf("fail to send data, ret: %d\n", ret);
                return -1;
            }
            printf("[OK] send buf to rank 1:\n" "\t%s\n", buf);
        }
        else if (rank == 1)
        {
            MPI_Status status;
            ret = MPI_Recv(buf, 20, MPI_CHAR, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
            if (ret != 0)
            {
                printf("fail to recv data, ret: %d\n", ret);
                return -1;
            }
            int count;
            MPI_Get_count(&status, MPI_CHAR, &count);
            printf("recv status:\n" "\tsource: %d, tag: %d, count: %d\n",
                status.MPI_SOURCE,
                status.MPI_TAG,
                count
            );
            printf("[OK] recv buf from rank 0:\n" "\t%s\n", buf);
        }
        else
        {
            printf("unknown rank: %d\n", rank);
            return -1;
        }

        ret = MPI_Finalize();
        if (ret != 0)
        {
            printf("fail to finalize mpi, ret: %d\n", ret);
            return -1;
        }
        printf("[OK] finalize mpi.\n");
        
        return 0;
    }
    ```

    `Makefile`:

    ```makefile
    main: main.c
        mpicc -g main.c -o main

    clean:
        rm -f main

    ```

    compile: `make`

    run: `mpirun -n 2 --host 10.0.2.15,10.0.2.4 ./main`

    output:

    ```
    [OK] init mpi.
    [OK] get world size: 2.
    [OK] get rank: 0.
    [OK] init mpi.
    [OK] get world size: 2.
    [OK] get rank: 1.
    [OK] send buf to rank 1:
        hello from rank 0
    recv status:
        source: 0, tag: 0, count: 20
    [OK] recv buf from rank 0:
        hello from rank 0
    [OK] finalize mpi.
    [OK] finalize mpi.
    ```

    说明：

    * 这段代码仅使用了 status 中的 source, tag, count 这三个信息，没有用到 error 信息。error 信息可以用于 recv 未知数量的数据。

* openmpi send recv code example

    `main.c`:

    ```c
    #include <mpi.h>
    #include <stdio.h>
    #include <string.h>

    int main()
    {
        int ret = MPI_Init(NULL, NULL);
        if (ret != 0)
        {
            printf("fail to init mpi, ret: %d\n", ret);
            return -1;
        }
        printf("[OK] init mpi.\n");

        int world_size;
        ret = MPI_Comm_size(MPI_COMM_WORLD, &world_size);
        if (ret != 0)
        {
            printf("fail to get comm world isze, ret: %d\n", ret);
            return -1;
        }
        printf("[OK] get world size: %d.\n", world_size);

        int rank;
        ret = MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        if (ret != 0)
        {
            printf("fail to get rank, ret: %d\n", ret);
            return -1;
        }
        printf("[OK] get rank: %d.\n", rank);

        char buf[20] = {0};
        if (rank == 0)
        {
            char *msg = "hello from rank 0";
            size_t msg_len = strlen(msg);
            size_t min_len;
            if (msg_len > 19)
                min_len = 19;
            else
                min_len = msg_len;
            strncpy(buf, msg, min_len);
            ret = MPI_Send(buf, 20, MPI_CHAR, 1, 0, MPI_COMM_WORLD);
            if (ret != 0)
            {
                printf("fail to send data, ret: %d\n", ret);
                return -1;
            }
            printf("[OK] send buf to rank 1:\n" "\t%s\n", buf);
        }
        else if (rank == 1)
        {
            ret = MPI_Recv(buf, 20, MPI_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if (ret != 0)
            {
                printf("fail to recv data, ret: %d\n", ret);
                return -1;
            }
            printf("[OK] recv buf from rank 0:\n" "\t%s\n", buf);
        }
        else
        {
            printf("unknown rank: %d\n", rank);
            return -1;
        }

        ret = MPI_Finalize();
        if (ret != 0)
        {
            printf("fail to finalize mpi, ret: %d\n", ret);
            return -1;
        }
        printf("[OK] finalize mpi.\n");
        
        return 0;
    }
    ```

    `Makefile`:

    ```makefile
    main: main.c
        mpicc -g main.c -o main

    clean:
        rm -f main
    ```

    compile: `make`

    run: `mpirun -n 2 --host 10.0.2.4,10.0.2.15 ./main`

    output:

    ```
    [OK] init mpi.
    [OK] get world size: 2.
    [OK] get rank: 0.
    [OK] init mpi.
    [OK] get world size: 2.
    [OK] get rank: 1.
    [OK] send buf to rank 1:
        hello from rank 0
    [OK] recv buf from rank 0:
        hello from rank 0
    [OK] finalize mpi.
    [OK] finalize mpi.
    ```

    说明：

    * 代码中`printf("[OK] send buf to rank 1:\n" "\t%s\n", buf);`将两行的 printf 合成一行写，是因为两台 node 的 printf 并没有执行顺序的保证，为了避免输出混乱，就把 buf 和 prompt 使用同一个 printf 输出了。
    
        如果将 printf 写成下面的形式：

        ```c
        // ...
                printf("[OK] send buf to rank 1:\n");
                printf("\t%s\n", buf);
        // ...
                printf("[OK] recv buf from rank 0:\n");
                printf("\t%s\n", buf);
        // ...
        ```

        那么运行程序可能会得到下面的输出：

        ```
        [OK] init mpi.
        [OK] get world size: 2.
        [OK] get rank: 1.
        [OK] recv buf from rank 0:
        [OK] init mpi.
        [OK] get world size: 2.
        [OK] get rank: 0.
        [OK] send buf to rank 1:
            hello from rank 0
            hello from rank 0
        [OK] finalize mpi.
        [OK] finalize mpi.
        ```

* 一个基于 LAN 的 openmpi hello world 程序

    `main.c`:

    ```c
    #include <mpi.h>
    #include <stdio.h>

    int main()
    {
        int ret = MPI_Init(NULL, NULL);
        if (ret != 0)
        {
            printf("fail to init mpi, ret: %d\n", ret);
            return -1;
        }
        printf("[OK] init mpi.\n");

        int world_size;
        ret = MPI_Comm_size(MPI_COMM_WORLD, &world_size);
        if (ret != 0)
        {
            printf("fail to get comm world isze, ret: %d\n", ret);
            return -1;
        }
        printf("[OK] get world size: %d.\n", world_size);

        int rank;
        ret = MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        if (ret != 0)
        {
            printf("fail to get rank, ret: %d\n", ret);
            return -1;
        }
        printf("[OK] get rank: %d.\n", rank);

        ret = MPI_Finalize();
        if (ret != 0)
        {
            printf("fail to finalize mpi, ret: %d\n", ret);
            return -1;
        }
        printf("[OK] finalize mpi.\n");
        
        return 0;
    }
    ```

    `Makefile`:

    ```makefile
    main: main.c
        mpicc -g main.c -o main

    clean:
        rm -f main
    ```

    compile:

    `make`

    run & output:

    ```
    (base) hlc@hlc-VirtualBox:~/nfs_shared/mpi_hello_world$ mpirun --tag-output -n 2 --host 10.0.2.4,10.0.2.15 ./main
    [1,0]<stdout>:[OK] init mpi.
    [1,0]<stdout>:[OK] get world size: 2.
    [1,0]<stdout>:[OK] get rank: 0.
    [1,1]<stdout>:[OK] init mpi.
    [1,1]<stdout>:[OK] get world size: 2.
    [1,1]<stdout>:[OK] get rank: 1.
    [1,1]<stdout>:[OK] finalize mpi.
    [1,0]<stdout>:[OK] finalize mpi.
    (base) hlc@hlc-VirtualBox:~/nfs_shared/mpi_hello_world$ 
    ```

    说明：

    * world size 指的是所有 node 上的所有进程的数量总和，rand 指的是当前进程在 world size 中的索引。

    * `-n 2`可以省略，此时每个 host 上默认只起一个进程

    * 可以使用`pirun -n 3 --host 10.0.2.4:2,10.0.2.15 ./main`指定`10.0.2.4`上起 2 个进程，`10.0.2.15`上默认起一个进程。当然，此处的`-n 3`也可以省略。

        对于`-n <N> --host <ip_1>:<N_1>,<ip_2>:<N_2>`形式指定的进程数量，如果`N != N_1 + N_2`，那么会报错。如果`<N_1>`或`<N_2>`没被指定，那么默认值为 1.

* 关于`mpirun`的参数`--host`无法使用 ip 的问题

    在有些机器上，使用`mpirun -n 2 --host 10.0.2.4,10.0.2.15 ./main`运行 mpi app 时，程序会无反应，最终连接超时。

    这是因为当前机器没有配置好 ssh 相关信息。

    解决方案：

    假如现在有 node 1, node 2 两台机器，
    
    1. 首先在 node 1 上使用 ssh 登陆 node 2，使得 node 2 的信息被记录在 node 1 的`~/.ssh/known_hosts`文件中。

    2. 在 node 2 上执行`ssh-copy-id <user>@<node_1_ipv4_addr>`，将 node 2 的 ssh public key 复制到 node 1 上（具体是添加到 node 1 的`~/.ssh/authorized_keys`文件里）。

    此时再运行``mpirun -n 2 --host 10.0.2.4,10.0.2.15 ./main``，程序即可正常执行。

* openmpi app 的 vscode 配置

    使用`sudo apt install libopenmpi-dev`安装的 openmpi 包，其库文件和头文件在`/usr/lib/x86_64-linux-gnu/openmpi`目录下，这个目录不是`gcc`的默认搜索目录，因此直接在`main.c`中写`#include <mpi.h>`，会提示找不到头文件。

    但是由于 openmpi 的 app 通常不使用 gcc 编译，而使用`mpicc`编译，因此只需要在 vscode 的 c/c++ 配置文件中，把编译器路径改成`/usr/bin/mpicc`即可。

    配置文件如下：

    `c_cpp_properties.json`:

    ```json
    {
        "configurations": [
            {
                "name": "Linux",
                "includePath": [
                    "${workspaceFolder}/**"
                ],
                "defines": [],
                "compilerPath": "/usr/bin/mpicc",
                "cStandard": "c17",
                "cppStandard": "gnu++17",
                "intelliSenseMode": "linux-gcc-x64"
            }
        ],
        "version": 4
    }
    ```

    此时再返回`main.c`文件中，可以看到`#include <mpi.h>`已经没有报错。

* vscode + gdb attach + mpi program debugging

    1. add `launch.json`:

        ```json
        {
            // Use IntelliSense to learn about possible attributes.
            // Hover to view descriptions of existing attributes.
            // For more information, visit: https://go.microsoft.com/fwlink/?linkid=830387
            "version": "0.2.0",
            "configurations": [{
                "name": "(gdb) Attach",
                "type": "cppdbg",
                "request": "attach",
                "program": "${workspaceFolder}/main_hlc",
                "processId":"${command:pickProcess}",
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
            }]
        }
        ```

        type `gdb`, select `attach` rather than `launch`.

        add `processId`, fill the field with `"${command:pickProcess}"`.

    2. add a fragment of code at the beginning of the main function

        ```c
        int main(int argc, char** argv)
        {
            int in_loop = 1;
            while (in_loop)
            {
                sleep(1);
            }

            // ....
        }
        ```

    3. compile the mpi program

        `mpicc -g main.c -o main_hlc`

        do not forget `-g` option.

    4. run the program in the **terminal**

        `mpirun -np 2 --host node1,node2 --mca btl_tcp_if_include enp0s3 ./main_hlc`

    5. press `F5` in the vscode

        search `main_hlc` in the prompt, select the program name `main_hlc`:

        <div style='text-align:center'>
        <img width=700 src='../../Reference_resources/ref_27/pics/pic_1.png'>
        </div>

    6. take a notice of the terminal panel below, enter `y` and press Enter

        <div style='text-align:center'>
        <img width=700 src='../../Reference_resources/ref_27/pics/pic_2.png'>
        </div>

        enter the password of root.

    7. add a breakpoint at the `sleep(1)` line, the program will stop at this breakpoint

        modify the value of `in_loop` in the variables window, then the program can step out the while loop.

        double click `in_loop` variable, and set it to `0`, then press Enter

        <div style='text-align:center'>
        <img width=700 src='../../Reference_resources/ref_27/pics/pic_3.png'>
        </div>

        Press F5, the program will continue to next breakpoint.


* mpi hello world program

    ```c
    #include <mpi.h>
    #include <stdio.h>

    int main(int argc, char** argv)
    {
        MPI_Init(NULL, NULL);
        int world_size;
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);
        int world_rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
        char processor_name[MPI_MAX_PROCESSOR_NAME];
        int name_len;
        MPI_Get_processor_name(processor_name, &name_len);

        printf("Hello world from processor %s, rank %d out of %d processors\n",
            processor_name, world_rank, world_size);

        MPI_Finalize();
        return 0;
    }
    ```

    compile:

    `mpicc main.c -o main`

    run:

    `mpirun -np 2 --host node1,node2 ./main`

    output:

    ```
    Hello world from processor hlc-VirtualBox, rank 0 out of 2 processors
    Hello world from processor hlc-VirtualBox, rank 1 out of 2 processors
    ```

    说明：

    * `MPI_Init()`

        syntax:

        ```c
        int MPI_Init(int *argc, char ***argv);
        ```

        这个函数用于初始化一些 mpi 的内部变量，然后接收参数，处理`argc`和`argv`。目前并不知道具体怎么处理这些 args。

        一个常见的 example：

        ```c
        int main(int argc, char **argv)
        {
            /* declare variables */
            MPI_Init(&argc, &argv);
            /* parse arguments */
            /* main program */
            MPI_Finalize();
            return 0;
        }
        ```

        如果不需要额外处理 args，可以将`argc`和`argv`都填`NULL`。

    * mpi api reference: <https://www.open-mpi.org/doc/v4.1/>

    * `MPI_Comm_size()`

        syntax:

        ```c
        int MPI_Comm_size(MPI_Comm comm, int *size);
        ```

        Returns the size of the group associated with a communicator.

        `comm`是一个枚举值，指定 commicator，通常指定`MPI_COMM_WORLD`就可以。

        For `MPI_COMM_WORLD`, it indicates the total number of processes available.

        这个函数似乎等价于先调用`MPI_Comm_group()`拿到 group，再调用`MPI_Group_size()`拿到 group size，最后再调用`MPI_Group_free()`释放 group 相关的信息资源。

    * `MPI_Comm_rank()`

        syntax:

        ```c
        int MPI_Comm_rank(MPI_Comm comm, int *rank);
        ```

        Determines the rank of the calling process in the communicator. 

        返回当前进程在通信世界中的索引。

        猜想：等价于依次调用下面三个函数：

        1. `MPI_Comm_group()`

        2. `MPI_Group_rank()`

        3. `MPI_Group_free()`

        目前不清楚 return value 作为 error number 时，在哪查询具体的含义。

    * `MPI_Get_processor_name()`

        syntax:

        ```c
        int MPI_Get_processor_name(char *name, int *resultlen);
        ```

        Gets the name of the processor. 

        这个函数实际返回的是操作系统的 machine name。

        猜想：这是 mpi 中唯一一个可以查看物理 node 信息的函数。

        vulkan 的函数一般都是调用两遍，第一遍拿到 length，第二遍才写缓冲区。不清楚这个函数是否支持这种操作。

* mpi 使用 nfs

    * master
    
        1. 安装 nfs server: `sudo apt install nfs-kernel-server`

        2. 创建一个普通目录：

            ```bash
            cd ~
            mkdir nfs_shared
            ```

        3. 配置 nfs

            `sudo vim /etc/exports`

            添加一行：

            ```
            /home/hlc/nfs_shared *(rw,sync,no_root_squash,no_subtree_check)
            ```

            应用配置：

            `sudo exportfs -a`

        4. 把可执行文件或者工程目录放到`nfs_shared`目录下

            `cp -r ~/Documents/Projects/mpi_test ~/nfs_shared`

    * worker
    
        1. 安装 nfs: `sudo apt install nfs-common`

        2. 创建空目录

            ```bash
            cd ~
            mkdir nfs_shared
            ```

        3. mount

            ```bash
            sudo mount -t nfs master_node:/home/hlc/nfs_shared ~/nfs_shared
            ```

        说明：

        * mount 时，remote 路径必须用绝对路径，既不能用`master_node:nfs_shared`，也不能用`master_node:~/nfs_shared`

        * 创建空目录`nfs_shared`时，其所在的目录必须和 master 保持一致，不然在 mpirun 时会找不到可执行程序

        * `master_node`可以是 hostname，也可以是 ip 地址，但不能是`<user_name>@<hostname>`或者`<user_name>@<ip_addr>`，因为 nfs 用的根本不是 ssh 协议。

* mpi test case

    目前可以跑通的一个 hello world 用例：

    install:

    ```bash
    sudo apt install openmpi-bin openmpi-common libopenmpi-dev
    ```

    进入项目目录，没有的话创建一个：

    `cd /home/hlc/Documents/Projects/mpi_test`

    创建文件：`mpi_hello_world.c`

    ```c
    #include <mpi.h>
    #include <stdio.h>

    int main(int argc, char** argv) {
        // Initialize the MPI environment
        MPI_Init(NULL, NULL);

        // Get the number of processes
        int world_size;
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);

        // Get the rank of the process
        int world_rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

        // Get the name of the processor
        char processor_name[MPI_MAX_PROCESSOR_NAME];
        int name_len;
        MPI_Get_processor_name(processor_name, &name_len);

        // Print off a hello world message
        printf("Hello world from processor %s, rank %d out of %d processors\n",
            processor_name, world_rank, world_size);

        // Finalize the MPI environment.
        MPI_Finalize();
    }
    ```

    编译：

    `mpicc mpi_hello_world.c`

    此时会生成一个`a.out`文件。

    本机生成 ssh key （已经有了的话就不需要了）:

    `ssh-keygen`

    一路回车就行，密码为空。

    此时本机 ip 为`10.0.2.4`，另一台 node 的 ip 为`10.0.2.15`。

    把本地的 public key 复制到其他 node 上：

    `ssh-copy-id 10.0.2.15` （默认使用当前用户名）

    然后编辑`/etc/hosts`文件，添加下面两行：

    ```
    10.0.2.4 node1
    10.0.2.15 node2
    ```

    将`mpi_test`文件夹复制到 node2 相同的位置：

    ```bash
    scp -r /home/hlc/Documents/Projects/mpi_test node2:/home/hlc/Documents/Projects/
    ```

    在 node2 上也需要用`mpicc`编译出`a.out`。

    此时在 node1 上运行

    `mpirun -np 2 --host node1,node2 /home/hlc/Documents/Projects/mpi_test/a.out`

    输出：

    ```
    Hello world from processor hlc-VirtualBox, rank 0 out of 2 processors
    Hello world from processor hlc-VirtualBox, rank 1 out of 2 processors
    ```

    说明局域网 mpi 环境搭建成功。

    注：

    * `--host`参数只接收 hostname，不接收 ip 地址。因此配置`/etc/hosts`文件是必需的。

        注意这个参数是`--host`，后面不加`s`

        2024/08/09: 又验证了一遍，确实是这样，`--host`不接收 ip 地址。｀hlc@hlc-VirtualBox:~/Documents/Projects/mpi_test$ mpirun -np 2 --host 10.0.2.15,10.0.2.4 ./main｀执行后长时间没反应。

    * 运行程序的路径必须是绝对路径

        也有可能是相对路径是相对用户 host 目录的？

        2024/08/09: 可以是相对路径，但是这个路径只在本 node 被解析为相对路径，在其他 node 被解析为绝对路径。因此无论使用相对路径还是绝对路径，都要求 executable 文件在不同 node 的路径都完全一致。

    * 如果不同 node 的系统/处理器相同，那么二进制可执行文件不需要再`mpicc`编译一遍

* openmpi 可以直接用 apt 安装，不需要专门下载源代码编译

## 安装

### 使用 apt 安装

```bash
sudo apt-get install libopenmpi-dev
```

### 从源码编译安装

官网下载：<https://www.open-mpi.org/software/ompi/v5.0/>

配置：

```bash
./configure --prefix=<path>
```

编译：

```bash
make
```

安装：

```bash
make install
```

这里 install 的路径就是前面`--prefix`指定的路径。

如果不执行`make install`直接使用编译好的库，需要注意有些文件名不太一样。比如`mpicc`其实是`opal_wrapper`，在`openmpi-5.0.2/opal/tools/wrappers`文件夹下，非常地难找。因此建议还是执行一下`make install`。

将 binary 添加到 path：

```bash
export PATH=$PATH:/home/hlc/Softwares/openmpi/bin
```

## hello world

`main.cpp`:

```cpp
#include <mpi.h>
#include <stdio.h>

int main(int argc, char** argv) {
    MPI_Init(NULL, NULL); // Initialize MPI

    // get number of processes
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // get my process's rank
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    printf("Hello, world. - Love, process %d/%d \n",
           world_rank, world_size);

    MPI_Finalize();  // Clean-up
}
```

编译：

```bash
mpicc main.cpp -o main
```

运行：

```bash
mpirun ./main
```

输出：

```
Hello, world. - Love, process 0/16 
Hello, world. - Love, process 7/16 
Hello, world. - Love, process 8/16 
Hello, world. - Love, process 6/16 
Hello, world. - Love, process 9/16 
Hello, world. - Love, process 10/16 
Hello, world. - Love, process 2/16 
Hello, world. - Love, process 3/16 
Hello, world. - Love, process 4/16 
Hello, world. - Love, process 12/16 
Hello, world. - Love, process 11/16 
Hello, world. - Love, process 15/16 
Hello, world. - Love, process 5/16 
Hello, world. - Love, process 13/16 
Hello, world. - Love, process 14/16 
Hello, world. - Love, process 1/16
```

## run in a LAN

resource:

* <https://www.eecis.udel.edu/~youse/openmpi/#/7>

* <https://mpitutorial.com/tutorials/running-an-mpi-cluster-within-a-lan/>

## send recv

### send recv

`main.c`:

```c
#include <mpi.h>
#include <stdio.h>

int main(int argc, char** argv)
{
    MPI_Init(NULL, NULL);
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    int number;
    if (world_rank == 0) {
        number = 54321;
        MPI_Send(&number, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
        printf("process 0 sent number %d\n", number);
    } else if (world_rank == 1) {
        MPI_Recv(&number, 1, MPI_INT, 0, 0, MPI_COMM_WORLD,
                MPI_STATUS_IGNORE);
        printf("process 1 received number %d from process 0\n",
            number);
    }

    MPI_Finalize();
    return 0;
}
```

compile:

```bash
mpicc main.c -o main
```

run:

```
mpirun -np 2 --host node1,node2 --mca btl_tcp_if_include enp0s3 ./main
```

output:

```
process 0 sent number 54321
process 1 received number 54321 from process 0
```

说明：

* 在运行时必须指定`--mca btl_tcp_if_include enp0s3`才能执行成功。否则 mpi 会找`ifconfig`列出来的第一个 network interface 尝试数据传输。

* `MPI_Send()`, `MPI_Recv()`

    syntax:

    ```c
    int MPI_Send(const void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm);

    int MPI_Recv(void *buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm, MPI_Status *status);
    ```

    `count`指的是元素的个数，不是 buffer length。`datatype`是 mpi 提前定义好的一些基本数据类型，常用的有

    | MPI datatype | C equivalent |
    | - | - |
    | MPI_INT | int |
    | MPI_LONG | long int |
    | MPI_FLOAT | float |
    | MPI_BYTE | char |

    `dest`指的是 destination 的 rank。

    `tag`目前不知道是什么意思，直接填 0 就行。

* send 和 recv 都是阻塞式的。

### ping pong

* mpi ping pong

    `main.c`:

    ```c
    #include <mpi.h>
    #include <stdio.h>

    int main(int argc, char** argv)
    {
        MPI_Init(NULL, NULL);
        int world_rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

        // suppose there are only two nodes 
        int ping_pong_id = 0;
        const int max_ping_pong_id = 5;
        int peer_rank;
        if (world_rank == 0)
            peer_rank = 1;
        else
            peer_rank = 0;
        while (ping_pong_id <= max_ping_pong_id)
        {
            if (world_rank == ping_pong_id % 2)
            {
                MPI_Send(&ping_pong_id, 1, MPI_INT, peer_rank, 0, MPI_COMM_WORLD);
                printf("process %d sent number %d\n", world_rank, ping_pong_id);
            }
            else
            {
                MPI_Recv(&ping_pong_id, 1, MPI_INT, peer_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                printf("process %d received number %d\n", world_rank, ping_pong_id);
            }
            ping_pong_id++;
        }

        MPI_Finalize();
        return 0;
    }
    ```

    output:

    ```
    process 0 sent number 0
    process 0 received number 1
    process 0 sent number 2
    process 1 received number 0
    process 0 received number 3
    process 0 sent number 4
    process 1 sent number 1
    process 1 received number 2
    process 0 received number 5
    process 1 sent number 3
    process 1 received number 4
    process 1 sent number 5
    ```

    说明：

    * 使用`ping_pong_id`去控制 node 是进入 send 模式还是进入 recv 模式。

        由于`ping_pong_id`每个进程有一份独立的值，不是进程间共享，所以不用担心加锁之类的问题。

    * 输出并不是先打印完 process 0 才打印 process 1，因此可以排除并不是先打印 self node 的 output，再从别的 node 把 output 传输过来，append 到已经打印的输出上。

        由于输出也不是严格按照 node 0, node 1 的交替顺序，所以也可以排除 printf 是严格按照代码顺序输出的。

        猜测：mpi 每隔随机的一段时间，就去别的 node 上把标准输出传输到当前 node 上并输出。

### ring

猜想：ring 本质上是实现了一种流水线，除了最后一次 message 需要 pass 所有的 node，其他的 message 总是可以流水线（pipeline）的方式在 node 间被传递。

猜想 2：message 的数量越多，ring 处理的效率越高。

```c
#include <mpi.h>
#include <stdio.h>
#include <unistd.h>

int main(int argc, char** argv) {
    MPI_Init(NULL, NULL);

    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    int token = 0;
    int ret;
    if (world_rank == 0)
    {
        token = 12345;
    }
    else
    {
        token = -1;
        ret = MPI_Recv(&token, 1, MPI_INT, world_rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        if (ret != 0)
        {
            printf("rank %d fail to recv\n", world_rank);
            return -1;
        }
        printf("rank %d received token %d from %d\n", world_rank, token, world_rank - 1);
    }

    ret = MPI_Send(&token, 1, MPI_INT, (world_rank + 1) % world_size, 0, MPI_COMM_WORLD);
    if (ret != 0)
    {
        printf("rank %d fail to send\n", world_rank);
        return -1;
    }

    if (world_rank == 0)
    {
        token = -1;
        ret = MPI_Recv(&token, 1, MPI_INT, world_size - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        if (ret != 0)
        {
            printf("rank %d fail to recv\n", world_rank);
            return -1;
        }
        printf("rank %d received token %d from %d\n", world_rank, token, world_size - 1);
    }

    MPI_Finalize();
    return 01;
}
```

run:

```bash
mpirun -np 2 --host node1,node2 ./main
```

output:

```
rank 0 received token 12345 from 1
rank 1 received token 12345 from 0
```

说明：

* rank 0 会首先在`MPI_Send()`处等其他进程，其他进程会在`MPI_Recv()`处等上一个进程。

    等 rank 0 send, rank 1 recv 成功后，rank 0 在`MPI_Recv()`处等待，rank 1 则在`MPI_Send()`处开始和 rank 2 同步。以此类推。

    这个过程如下图所示：

    <div style='text-align:center'>
    <img width=700 src='../../Reference_resources/ref_28/pic_1.png'>
    </div>

* 下面这段代码似乎也能实现 ring 功能，并且更简洁

    ```c
    #include <mpi.h>
    #include <stdio.h>
    #include <unistd.h>

    int main(int argc, char** argv) {
        MPI_Init(NULL, NULL);

        int world_size;
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);

        int world_rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

        int token = 0;
        int ret;
        if (world_rank == 0)
        {
            token = 12345;
            MPI_Send(&token, 1, MPI_INT, (world_rank + 1) % world_size, 0, MPI_COMM_WORLD);
        }
        else
        {
            MPI_Recv(&token, 1, MPI_INT, world_rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            printf("rank %d received token %d from rank %d\n", world_rank, token, world_rank - 1);
        }

        if (world_rank == 0)
        {
            MPI_Recv(&token, 1, MPI_INT, world_size - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            printf("rank %d received token %d from rank %d\n", world_rank, token, world_size - 1);
        }
        else
        {
            MPI_Send(&token, 1, MPI_INT, (world_rank + 1) % world_size, 0, MPI_COMM_WORLD);
        }

        MPI_Finalize();
        return 0;
    }
    ```

## Examples

### hello world

### send recv
