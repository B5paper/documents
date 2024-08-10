# OpenMPI Note

## cache

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

