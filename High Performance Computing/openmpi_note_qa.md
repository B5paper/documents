# open mpi note qa

[unit]
[u_0]
写一个 openmpi hello world 程序。
[u_1]
`main.c`:

```c
#include <openmpi/mpi.h>
#include <stdio.h>

int main()
{
    int ret = MPI_Init(NULL, NULL);
    if (ret != 0)
    {
        printf("fail to init mpi\n");
        return -1;
    }
    printf("successfully init mpi\n");

    int rank;
    ret = MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (ret != 0)
    {
        printf("fail to get world rank\n");
        return -1;
    }
    printf("successfully get world rank: %d\n", rank);

    ret = MPI_Finalize();
    if (ret != 0)
    {
        printf("fail to finalize mpi\n");
        return -1;
    }
    printf("successfully finalize mpi\n");
    
    return 0;
}
```

`Makefile`:

```
main: main.c
	mpicc -g main.c -o main
```

编译：`make`

运行：`mpirun -np 4 ./main`

output:

```
successfully init mpi
successfully get world rank: 1
successfully init mpi
successfully get world rank: 2
successfully init mpi
successfully get world rank: 3
successfully init mpi
successfully get world rank: 0
successfully finalize mpi
successfully finalize mpi
successfully finalize mpi
successfully finalize mpi
```