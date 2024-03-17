# OpenMPI Note

## 安装：

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

```python
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

