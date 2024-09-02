# OpenMPI Note

## cache

* mpi ring

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

* mpi send and recv

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

