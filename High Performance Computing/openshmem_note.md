# OpenSHMEM Note

* openmpi 版的 openshmem 的 api reference

    <https://docs.open-mpi.org/en/v5.0.x/man-openshmem/man3/index.html>

    <https://docs.open-mpi.org/en/main/man-openshmem/man3/OpenSHMEM.3.html>

* 官网: <http://openshmem.org/site/>

## cache

* openshmem example: broadcast

    `main.c`:

    ```c
    #include <stdio.h>
    #include <stdlib.h>
    #include <shmem.h>

    long pSync[SHMEM_BCAST_SYNC_SIZE];

    int main()
    {
        shmem_init();
        int me = shmem_my_pe();
        int npes = shmem_n_pes();

        long *source = (long *) shmem_malloc(npes * sizeof(long));
        for (int i = 0; i < npes; i += 1)
            source[i] = me * 4 + i;

        long *target = (long *) shmem_malloc(npes * sizeof(long));
        for (int i = 0; i < npes; i += 1)
            target[i] = -1;

        for (int i = 0; i < SHMEM_BCAST_SYNC_SIZE; i += 1)
            pSync[i] = SHMEM_SYNC_VALUE;
        shmem_barrier_all();

        printf("before, pe %d: source: %ld %ld %ld %ld, target: %ld %ld %ld %ld\n", me, source[0], source[1], source[2], source[3], target[0], target[1], target[2], target[3]);

        shmem_broadcast64(target, source, npes, 0, 0, 0, npes, pSync);

        printf("after, pe %d: source: %ld %ld %ld %ld, target: %ld %ld %ld %ld\n", me, source[0], source[1], source[2], source[3], target[0], target[1], target[2], target[3]);
        shmem_barrier_all();

        shmem_free(target);
        shmem_free(source);
        shmem_finalize();
        return 0;
    }
    ```

    compile: `oshcc -g main.c -o main`

    run: `oshrun -np 4 ./main`

    output:

    ```
    before, pe 3: source: 12 13 14 15, target: -1 -1 -1 -1
    after, pe 3: source: 12 13 14 15, target: 0 1 2 3
    before, pe 0: source: 0 1 2 3, target: -1 -1 -1 -1
    after, pe 0: source: 0 1 2 3, target: -1 -1 -1 -1
    before, pe 1: source: 4 5 6 7, target: -1 -1 -1 -1
    after, pe 1: source: 4 5 6 7, target: 0 1 2 3
    before, pe 2: source: 8 9 10 11, target: -1 -1 -1 -1
    after, pe 2: source: 8 9 10 11, target: 0 1 2 3
    ```

    `shmem_broadcast64()`可以将某个 pe 上指定的一块数组广播到其他 pe 上。

    output 中并不是先输出 before，再统一输出 after，说明`shmem_broadcast64()`并不是一个 barrier。

    又自己写了个 example，感觉这个更能突出 broadcast 的作用：

    ```c
    #include <stdio.h>
    #include <shmem.h>

    long pSync[SHMEM_BCAST_SYNC_SIZE];

    int main()
    {
        shmem_init();
        int me = shmem_my_pe();
        int npes = shmem_n_pes();
        
        int arr_len = 3;
        long *arr = (long *) shmem_malloc(arr_len * sizeof(long));
        for (int i = 0; i < arr_len; ++i)
            arr[i] = -1;

        if (me == 0)
        {
            for (int i = 0; i < arr_len; ++i)
                arr[i] = i;
        }

        for (int i = 0; i < SHMEM_BCAST_SYNC_SIZE; i += 1)
            pSync[i] = SHMEM_SYNC_VALUE;
        shmem_barrier_all();

        printf("before, pe %d: arr: %ld %ld %ld\n", me, arr[0], arr[1], arr[2]);

        shmem_broadcast64(arr, arr, arr_len, 0, 0, 0, npes, pSync);

        printf("after, pe %d: arr: %ld %ld %ld\n", me, arr[0], arr[1], arr[2]);
        shmem_barrier_all();

        shmem_free(arr);
        shmem_finalize();
        return 0;
    }
    ```

    output:

    ```
    before, pe 2: arr: -1 -1 -1
    before, pe 3: arr: -1 -1 -1
    after, pe 3: arr: 0 1 2
    before, pe 1: arr: -1 -1 -1
    after, pe 1: arr: 0 1 2
    before, pe 0: arr: 0 1 2
    after, pe 0: arr: 0 1 2
    after, pe 2: arr: 0 1 2
    ```
    
* openshmem example: barrier

    `main.c`:

    ```c
    #include <stdio.h>
    #include <shmem.h>

    long pSync[SHMEM_BARRIER_SYNC_SIZE];
    int x = 123;

    int main()
    {
        for (int i = 0; i < SHMEM_BARRIER_SYNC_SIZE; i += 1)
            pSync[i] = SHMEM_SYNC_VALUE;

        shmem_init();
        int me = shmem_my_pe();
        int npes = shmem_n_pes();

        if (me == 0)
            shmem_int_p(&x, 456, 1);
        shmem_barrier(0, 0, npes, pSync);

        printf("%d: x = %d\n", me, x);

        shmem_finalize();
        return 0;
    }
    ```

    compile: `oshcc -g main.c -o main`

    run: `oshrun -np 4 ./main`

    output:

    ```
    0: x = 123
    3: x = 123
    1: x = 456
    2: x = 123
    ```

    如果不添加`shmem_barrier(0, 0, npes, pSync);`这一行，输出会变为：

    ```
    2: x = 123
    3: x = 123
    1: x = 123
    0: x = 123
    ```

    说明：

    1. `long pSync[SHMEM_BARRIER_SYNC_SIZE];`这个数组可以设置为 main 的局部变量，不影响 barrier 功能。

* openshmem example: bar pair

    `main.c`:

    ```c
    #include <stdio.h>
    #include <unistd.h>
    #include <shmem.h>

    #define NPES 4

    long pSync[SHMEM_BARRIER_SYNC_SIZE];
    int x = 10101;

    int main()
    {
        for (int i = 0; i < SHMEM_BARRIER_SYNC_SIZE; i += 1)
            pSync[i] = SHMEM_SYNC_VALUE;

        shmem_init();
        int me = shmem_my_pe();

        if (me == 0)
            shmem_int_p(&x, 4, 1);
        shmem_barrier_all();  // this line

        if (me == 2)
        {
            printf("Process %d going to sleep\n", me);
            sleep(3);
            printf("Process %d out from sleep\n", me);
        }

        printf("Process %d before barrier\n", me);
        if (me == 2 || me == 3)
            shmem_barrier(2, 0, 2, pSync);
        printf("Process %d after barrier\n", me);

        printf("%d: x = %d\n", me, x);
        
        shmem_finalize();
        return 0;
    }
    ```

    compile: `oshcc -g main.c -o main`

    run: `oshrun -np 4 ./main`

    output:

    ```
    Process 2 going to sleep
    Process 3 before barrier
    Process 0 before barrier
    Process 0 after barrier
    0: x = 10101
    Process 1 before barrier
    Process 1 after barrier
    1: x = 4

    Process 2 out from sleep
    Process 2 before barrier
    Process 2 after barrier
    Process 3 after barrier
    3: x = 10101
    2: x = 10101
    ```

    这个程序在运行时会 sleep 3 秒，为了区分 sleep 前和 sleep 后的输出，在 output 里添加了一个换行。

    这段代码做了一个综合测试，pe 0 修改 pe 1 里的数据，然后 pe 0 和 pe 1 正常结束。代码中有一行注释了`// this line`，这一行原 example 中是没有的。如果没有这一行，pe 1 中的`x`的值不会被修改成功。

    pe 2 进入 sleep 状态，pe 3 会在`shmem_barrier()`处等待 pe 2。当 pe 2 运行到 barrier 处后，pe 2 和 pe 3 才会恢复运行。

    `SHMEM_BARRIER_SYNC_SIZE`在程序中的实际值是 1，猜测这个值的含义是能同时监视的值的数量。

    `SHMEM_SYNC_VALUE`的值是`-1`，但如果给`pSync[i]`赋其他值，不影响 barrier 的功能。

* openshmem example: ring put

    `main.c`:

    ```c
    #include <stdio.h>
    #include <sys/utsname.h>
    #include <shmem.h>

    #define N 7

    int main(void)
    {
        struct utsname u;
        int su = uname(&u);
        if (su != 0)
        {
            printf("fail to get uname\n");
            return -1;
        }

        shmem_init();
        int me = shmem_my_pe();
        int npes = shmem_n_pes();

        long src[N];
        for (int i = 0; i < N; i += 1)
            src[i] = (long) me;

        long *dest = (long *) shmem_malloc(N * sizeof(long));
        int next_pe = (me + 1) % npes;
        shmem_long_put(dest, src, N, next_pe);
        shmem_barrier_all();

        printf("%d @ %s: dest: ", me, u.nodename);
        for (int i = 0; i < N; i += 1)
            printf("%ld, ", dest[i]);
        putchar('\n');

        shmem_free(dest);
        shmem_finalize();
        return 0;
    }
    ```

    compile: `oshcc -g main.c -o main`

    run: `oshrun -np 4 ./main`

    output:

    ```
    3 @ ubuntu2404: dest: 2, 2, 2, 2, 2, 2, 2, 
    1 @ ubuntu2404: dest: 0, 0, 0, 0, 0, 0, 0, 
    2 @ ubuntu2404: dest: 1, 1, 1, 1, 1, 1, 1, 
    0 @ ubuntu2404: dest: 3, 3, 3, 3, 3, 3, 3, 
    ```

    这个程序用`put`循环地将一个数组（一段内存）从当前 pe 搬移到下一个 pe。

* openshmem example: amo

    ```c
    #include <shmem.h>
    #include <stdio.h>
    #include <stdlib.h>

    const int tries = 10000;

    #ifdef TEST64BIT
    typedef long locktype;

    #else /*  */
    typedef int locktype;

    #endif /*  */

    int main()
    {
        int tpe, other;
        long i;
        struct
        {
            locktype a;
            locktype b;
        } *twovars;
        int numfail = 0;
        shmem_init();
        tpe = 0;
        other = shmem_n_pes() - 1;
        twovars = shmem_malloc(sizeof(*twovars));
        if (shmem_my_pe() == 0) {
            printf("Element size: %ld bytes\n", sizeof(locktype));
            printf("Addresses: 1st element %p\n", (void *) &twovars->a);
            printf("           2nd element %p\n", (void *) &twovars->b);
            printf("Iterations: %d   target PE: %d   other active PE: %d\n",
                tries, tpe, other);
        }
        twovars->a = 0;
        twovars->b = 0;
        shmem_barrier_all();
        if (shmem_my_pe() == 0) {

            // put two values alternately to the 1st 32 bit word
            long expect, check;
            for (i = 0; i < tries; i++) {
                expect = 2 + i % 2;
                if (sizeof(locktype) == sizeof(int)) {
                    shmem_int_p((void *) &twovars->a, expect, tpe);
                    check = shmem_int_g((void *) &twovars->a, tpe);
                }
                else if (sizeof(locktype) == sizeof(long)) {
                    shmem_long_p((void *) &twovars->a, expect, tpe);
                    check = shmem_long_g((void *) &twovars->a, tpe);
                }
                if (check != expect) {
                    printf("error: iter %ld get returned %ld expected %ld\n", i,
                        check, expect);
                    numfail++;
                    if (numfail > 10) {
                        printf("FAIL\n");
                        abort();
                    }
                }
            }
            printf("PE %d done doing puts and gets\n", shmem_my_pe());
        }
        else if (shmem_my_pe() == other) {

            // keep on atomically incrementing the 2nd 32 bit word
            long oldval;
            for (i = 0; i < tries; i++) {
                if (sizeof(locktype) == sizeof(int)) {
                    oldval =
                        shmem_int_atomic_fetch_inc((void *) &twovars->b, tpe);
                }
                else if (sizeof(locktype) == sizeof(long)) {
                    oldval =
                        shmem_long_atomic_fetch_inc((void *) &twovars->b, tpe);
                }
                if (oldval != i) {
                    printf("error: iter %ld finc got %ld expect %ld\n", i,
                        oldval, i);
                    numfail++;
                    if (numfail > 10) {
                        printf("FAIL\n");
                        abort();
                    }
                }
            }
            printf("PE %d done doing fincs\n", shmem_my_pe());
        }
        shmem_barrier_all();
        if (numfail) {
            printf("FAIL\n");
        }
        shmem_barrier_all();
        if (shmem_my_pe() == 0) {
            printf("test complete\n");
        }
        shmem_finalize();
        return 0;
    }
    ```

    这个看着有点像综合数据正确性测试。pe 0 不断地向 pe 1 中轮替写入数字 2 和数字 3，然后再从 pe 1 中把数据拿回来看是否正确。pe 1 则不断把 pe 0 中的数字做加 1 操作，然后检查结果是否正确。

    output:

    ```
    Element size: 4 bytes
    PE 1 done doing fincs
    Addresses: 1st element 0xff0000d0
            2nd element 0xff0000d4
    Iterations: 10000   target PE: 0   other active PE: 1
    PE 0 done doing puts and gets
    test complete
    ```

* openshmem example: set

    `main.c`:

    ```c
    #include <stdio.h>
    #include <shmem.h>

    int g_val;

    int main()
    {
        shmem_init();
        int me = shmem_my_pe();

        g_val = 123;
        shmem_barrier_all();

        if (me == 0)
        {
            shmem_int_atomic_set(&g_val, 456, 1);
        }
        shmem_barrier_all();

        printf("pe %d: g_val = %d\n", me, g_val);

        shmem_finalize();
        return 0;
    }
    ```

    compile: `oshcc -g main.c -o main`

    run: `oshrun -np 2 ./main`

    output:

    ```
    pe 1: g_val = 456
    pe 0: g_val = 123
    ```

    这个 example 说明 set 函数可以修改远程 pe 里的变量。

* openshmem example: fetch

    `main.c`:

    ```c
    #include <stdio.h>
    #include <shmem.h>

    int g_val;

    int main()
    {
        shmem_init();
        int me = shmem_my_pe();
        if (me == 0)
            g_val = 123;
        else if (me == 1)
            g_val = 234;
        printf("pe %d: g_val = %d\n", me, g_val);

        if (me == 0)
        {
            const int fetched = shmem_int_atomic_fetch(&g_val, 1);
            printf("pe %d: g_val = %d, fetched val = %d\n", me, g_val, fetched);
        }
        shmem_barrier_all();

        printf("pe %d: g_val = %d\n", me, g_val);

        shmem_finalize();
        return 0;
    }
    ```

    compile: `oshcc -g main.c -o main`

    run: `oshrun -np 2 ./main`

    output:

    ```
    pe 1: g_val = 234
    pe 0: g_val = 123
    pe 1: g_val = 234
    pe 0: g_val = 123, fetched val = 234
    pe 0: g_val = 123
    ```

    看起来 fetch 的作用就是从远程 pe 拿一个数据。

    这个例子可以看出`shmem_barrier_all()`只影响有内存依赖关系的代码的执行顺序，不影响其他代码（比如`printf`）的执行顺序。

    猜想：通常在修改内存值的代码后加一个 barrier，可以保证后面的数据都是正确的。

* openshmem example atomic add

    `main.c`:

    ```c
    #include <stdio.h>
    #include <shmem.h>

    int g_val;

    int main()
    {
        shmem_init();
        int me = shmem_my_pe();

        g_val = 0;
        shmem_barrier_all();

        if (me == 0)
        {
            shmem_int_atomic_add(&g_val, 2, 1);
        }
        shmem_barrier_all();

        printf("%d: g_val = %d\n", me, g_val);

        shmem_finalize();
        return 0;
    }
    ```

    compile: `oshcc -g main.c -o main`

    run: `oshrun -np 2 ./main`

    output:

    ```
    0: g_val = 0
    1: g_val = 2
    ```

    看起来`shmem_int_atomic_add()`可以让指定 pe 上的指定 int 值加一个数。

    说明：

    1. 如果把这个函数替换成

        ```c
        if (me == 1)
        {
            g_val += 2;
        }
        ```

        那么有什么不同？

        答：注意到，原代码中`if (pe == 0)`表示当前的进程是 0，而`shmem_int_atomic_add()`里指定修改的是进程 1 里的数据。这个函数说明可以跨进程修改数据。

    2. 如果把`g_val`在`main()`函数内声明，那么程序会报错。说明全局变量默认就在 shmem 中，而局部变量一定是 private 的。

* openshmem example: variable accessibility test

    `main.c`:

    ```c
    #include <stdio.h>
    #include <shmem.h>

    long global_var;
    static int global_static_var;

    int main()
    {
        long local_var;
        static int local_static_var;

        shmem_init();
        int num_pes = shmem_n_pes();
        if (num_pes < 2)
        {
            puts("Please run at least 2 pes.\n");
            return -1;
        }
        printf("There are totally %d pes.\n", num_pes);
        int me = shmem_my_pe();

        int *shm_pvar = (int *) shmem_malloc(sizeof(int));

        if (me == 0)
        {
            if (shmem_addr_accessible(&global_var, 1))
                puts("pe 0 can get global var from pe 1.");
            else
                puts("pe 0 can NOT get global var from pe 1.");

            if (shmem_addr_accessible(&global_static_var, 1))
                puts("pe 0 can get global static var from pe 1.");
            else
                puts("pe 0 can NOT get global static var from pe 1.");

            if (shmem_addr_accessible(&local_var, 1))
                puts("pe 0 can get local var from pe 1.");
            else
                puts("pe 0 can NOT get local var from pe 1.");

            if (shmem_addr_accessible(&local_static_var, 1))
                puts("pe 0 can get local static var from pe 1.");
            else
                puts("pe 0 can NOT get local static var from pe 1.");            

            if (shmem_addr_accessible(shm_pvar, 1))
                puts("pe 0 can get shm_pvar from pe 1.");
            else
                puts("pe 0 can NOT get shm_pvar from pe 1.");
        }

        shmem_free(shm_pvar);
        shmem_finalize();
        return 0;
    }
    ```

    compile: `oshcc -g main.c -o main`

    run: `oshrun -np 2 ./main`

    output:

    ```
    There are totally 2 pes.
    pe 0 can get global var from pe 1.
    pe 0 can get global static var from pe 1.
    pe 0 can NOT get local var from pe 1.
    pe 0 can get local static var from pe 1.
    pe 0 can get shm_pvar from pe 1.
    There are totally 2 pes.
    ```

    看起来全局变量，static 局部变量，以及`shmem_malloc()`申请的内存都可以被跨 pe 访问，但是函数的局部变量不行。

* 尝试在跑通 openshmem hello world 的环境下继续跑下面的程序，没跑通

    ```c
    #include <shmem.h>
    #include <stdio.h>
    #include <stdlib.h>

    int main(void) {
        static long lock = 0;
        shmem_init();
        int mype = shmem_my_pe();
        int npes = shmem_n_pes();
        int my_nelem = mype + 1; /* linearly increasing number of elements with PE */
        int total_nelem = (npes * (npes + 1)) / 2;
        int *source = (int *)shmem_malloc(npes * sizeof(int)); /* symmetric alloc */
        int *dest = (int *)shmem_malloc(total_nelem * sizeof(int));
        for (int i = 0; i < my_nelem; i++)
            source[i] = (mype * (mype + 1)) / 2 + i;
        for (int i = 0; i < total_nelem; i++)
            dest[i] = -9999;
        /* Wait for all PEs to initialize source/dest: */
        shmem_team_sync(SHMEM_TEAM_WORLD);
        shmem_int_collect(SHMEM_TEAM_WORLD, dest, source, my_nelem);
        shmem_set_lock(&lock); /* Lock prevents interleaving printfs */
        printf("%d: %d", mype, dest[0]);
        for (int i = 1; i < total_nelem; i++)
            printf(", %d", dest[i]);
        printf("\n");
        shmem_clear_lock(&lock);
        shmem_finalize();
        return 0;
    }
    ```

    报错：

    ```
    [ubuntu2404][[14096,1],0][shmem_team.c:47:pshmem_team_sync] Internal error is appeared rc = -7
    [ubuntu2404][[14096,1],0][shmem_collect.c:171:pshmem_int_collect] Internal error is appeared rc = -7
    [ubuntu2404][[14096,1],1][shmem_team.c:47:pshmem_team_sync] Internal error is appeared rc = -7
    [ubuntu2404][[14096,1],1][shmem_collect.c:171:pshmem_int_collect] Internal error is appeared rc = -7
    [ubuntu2404][[14096,1],3][shmem_team.c:47:pshmem_team_sync] Internal error is appeared rc = -7
    [ubuntu2404][[14096,1],3][shmem_collect.c:171:pshmem_int_collect] Internal error is appeared rc = -7
    [ubuntu2404][[14096,1],2][shmem_team.c:47:pshmem_team_sync] Internal error is appeared rc = -7
    [ubuntu2404][[14096,1],2][shmem_collect.c:171:pshmem_int_collect] Internal error is appeared rc = -7
    0: -9999, -9999, -9999, -9999, -9999, -9999, -9999, -9999, -9999, -9999
    2: -9999, -9999, -9999, -9999, -9999, -9999, -9999, -9999, -9999, -9999
    3: -9999, -9999, -9999, -9999, -9999, -9999, -9999, -9999, -9999, -9999
    1: -9999, -9999, -9999, -9999, -9999, -9999, -9999, -9999, -9999, -9999
    ```

    不清楚原因。

* openmpi 5.0 项目中的 example 可以跑通：

    ```
    root@3eacd89b1206:~/openmpi-5.0.5/examples# oshrun --allow-run-as-root -np 4 ./ring_oshmem
    --------------------------------------------------------------------------
    PMIx was unable to find a usable compression library
    on the system. We will therefore be unable to compress
    large data streams. This may result in longer-than-normal
    startup times and larger memory footprints. We will
    continue, but strongly recommend installing zlib or
    a comparable compression library for better user experience.

    You can suppress this warning by adding "pcompress_base_silence_warning=1"
    to your PMIx MCA default parameter file, or by adding
    "PMIX_MCA_pcompress_base_silence_warning=1" to your environment.
    --------------------------------------------------------------------------
    Process 0 puts message 10 to 1 (4 processes in ring)
    Process 0 decremented value: 9
    Process 0 decremented value: 8
    Process 0 decremented value: 7
    Process 0 decremented value: 6
    Process 0 decremented value: 5
    Process 0 decremented value: 4
    Process 0 decremented value: 3
    Process 0 decremented value: 2
    Process 0 decremented value: 1
    Process 0 decremented value: 0
    Process 0 exiting
    Process 1 exiting
    Process 3 exiting
    Process 2 exiting
    ```

    看来 openshmem 的 API 变动也比较大。有时间了看下文档详细研究下。

    openmpi 里自带的 openshmem 可以看 openmpi 的 doc。

* 最新版的 ucx 自带了测速工具

    ```
    (base) hlc@ubuntu2404:~/Documents/Projects/ucx/build/bin$ ls
    io_demo  ucx_info  ucx_perftest  ucx_perftest_daemon  ucx_read_profile
    ```

    试了下，`ucx_perftest`主要能测 lo, tcp, posix, sysv 这几种，有的还跑不通。能跑通的测试结果都不稳定，有时候测出来 200，有时候测出来 900，根本没法作为参考。

    目前能跑通的命令：

    server 端：`/ucx_perftest -t put_bw -d memory -x posix`

    client 端：`./ucx_perftest 127.0.0.1 -t put_bw -d memory -x posix`

* `openshmem-examples` repo <https://github.com/openshmem-org/openshmem-examples.git> 中的代码目前测试的都能跑通。可以以这个为参考。

* 看起来`shmemcc`和`shmemrun`趋向于淘汰了，可以使用`oshcc`，`oshrun`取代。

* 猜测 apt 自带的 openshmem 不能跑通的原因是，apt 中的 openmpi 版本是 4.1，其在编译的时候，使用了比较新的 ucx 版本。但是 apt 的 ucx 版本仍然是比较旧的，导致出现错误。

* openshmem 可以跑通的两种方式

    * 下载 openmpi 3.0 版本

        编译配置：

        `./configure --prefix=/root/openmpi-3.0.6/build --exec-prefix=/root/openmpi-3.0.6/bin --with-ucx`

        编译：

        `make -j8`

        编译安装：

        `make install`

        说明：

        1. 这里配置`prefix`是为了后面执行`make install`。只执行`make`不会编译出`xxx.so`之类的文件。

        2. `--with-ucx`表示使用本机系统自带的 ucx 进行编译。通常这个 ucx 是用 apt 安装的。

        编译 app:

        `./shmemcc ~/shmem_test/main.c -o main`

        运行：

        `./shmemrun --allow-run-as-root -np 4 ./main`

        此时即可获得稳定输出：

        ```
        Hello from 1 of 4
        Hello from 2 of 4
        Hello from 3 of 4
        Hello from 0 of 4
        ```

    * 下载 openmpi 5.0 版本

        此时如果使用系统自带的 ucx 会报错，提示必须使用 1.9 及以上版本的 ucx。可以去 github <https://github.com/openucx/ucx> 找到最新版 ucx。

        openmpi 编译配置：

        ```bash
        ./configure --prefix=/home/hlc/Downloads/openmpi_505/openmpi-5.0.5/build --exec-prefix=/home/hlc/Downloads/openmpi_505/openmpi-5.0.5/bin --enable-oshmem --enable-oshmem-compat --enable-oshmem-profile --with-oshmem-param-check --with-ucx=/home/hlc/Documents/Projects/ucx/build
        ```

        编译：

        `make -j8`

        编译安装：

        `make install`

        说明：

        1. 不写`--enable-oshmem --enable-oshmem-compat --enable-oshmem-profile --with-oshmem-param-check`有可能不编译完整的 openshmem，只编译出一个`oshrun`文件。

        2. `--with-ucx=/home/hlc/Documents/Projects/ucx/build`这里填的是我们自己编译的 ucx 目录。

        编译 app:

        `./shmemcc /home/hlc/Documents/Projects/shmem_test/main.c -o main`

        运行：

        `./oshrun -np 4 ./main`

        此时即可获得稳定输出：

        ```
        Hello from 1 of 4
        Hello from 3 of 4
        Hello from 2 of 4
        Hello from 0 of 4
        ```

        说明：

        1. `./oshrun -np 4 ./main`，注意编译出来的可执行文件已经没有`oshmemrun`了，只有`oshrun`。

* 在 ubuntu 22.04 上尝试了 openmpi 版本的 shmem，全是报错，没有成功过

    ```
    (base) hlc@hlc-VirtualBox:~/Documents/Projects/shmem_test$ shmemrun -np 4 ./main
    [hlc-VirtualBox:06026] Error ../../../../../oshmem/mca/memheap/base/memheap_base_static.c:249 - _load_segments() too many segments (max = 32): skip 76ebb7b88000-76ebb7b99000 rw-p 00000000 00:00 0 
    [hlc-VirtualBox:06026] Error ../../../../../oshmem/mca/memheap/base/memheap_base_static.c:249 - _load_segments() too many segments (max = 32): skip 76ebb7c88000-76ebb7c9a000 rw-p 00000000 00:00 0 
    [hlc-VirtualBox:06031] Error ../../../../../oshmem/mca/memheap/base/memheap_base_static.c:249 - _load_segments() too many segments (max = 32): skip 71087b26f000-71087b281000 rw-p 00000000 00:00 0 
    [hlc-VirtualBox:06031] ../../../../../../oshmem/mca/spml/ucx/spml_ucx.c:127  Error: Failed to get new mkey for segment: max number (32) of segment descriptor is exhausted
    [hlc-VirtualBox:06031] ../../../../../../oshmem/mca/spml/ucx/spml_ucx.c:223  Error: mca_spml_ucx_ctx_mkey_new failed
    [hlc-VirtualBox:06031] ../../../../../../oshmem/mca/spml/ucx/spml_ucx.c:736  Error: mca_spml_ucx_ctx_mkey_cache failed
    [hlc-VirtualBox:6031 :0:6031] Caught signal 11 (Segmentation fault: address not mapped to object at address 0x49)
    ==== backtrace (tid:   6031) ====
    0  /lib/x86_64-linux-gnu/libucs.so.0(ucs_handle_error+0x2e4) [0x710878154fc4]
    1  /lib/x86_64-linux-gnu/libucs.so.0(+0x24fec) [0x710878158fec]
    2  /lib/x86_64-linux-gnu/libucs.so.0(+0x251aa) [0x7108781591aa]
    3  /lib/x86_64-linux-gnu/libucp.so.0(+0x35e52) [0x7108781c3e52]
    4  /lib/x86_64-linux-gnu/libucp.so.0(ucp_mem_unmap+0x66) [0x7108781c3f96]
    5  /usr/lib/x86_64-linux-gnu/openmpi/lib/openmpi3/mca_spml_ucx.so(mca_spml_ucx_register+0x247) [0x710879083e07]
    6  /lib/x86_64-linux-gnu/liboshmem.so.40(mca_memheap_base_reg+0x92) [0x71087b238a92]
    7  /lib/x86_64-linux-gnu/liboshmem.so.40(mca_memheap_base_select+0xe4) [0x71087b239824]
    8  /lib/x86_64-linux-gnu/liboshmem.so.40(oshmem_shmem_init+0x238) [0x71087b1b0508]
    9  /lib/x86_64-linux-gnu/liboshmem.so.40(shmem_init+0x50) [0x71087b1b0760]
    10  ./main(+0x11da) [0x58079e7bb1da]
    11  /lib/x86_64-linux-gnu/libc.so.6(+0x29d90) [0x71087ae29d90]
    12  /lib/x86_64-linux-gnu/libc.so.6(__libc_start_main+0x80) [0x71087ae29e40]
    13  ./main(+0x1105) [0x58079e7bb105]
    =================================
    [hlc-VirtualBox:06031] *** Process received signal ***
    [hlc-VirtualBox:06031] Signal: Segmentation fault (11)
    [hlc-VirtualBox:06031] Signal code:  (-6)
    [hlc-VirtualBox:06031] Failing at address: 0x3e80000178f
    [hlc-VirtualBox:06031] [ 0] /lib/x86_64-linux-gnu/libc.so.6(+0x42520)[0x71087ae42520]
    [hlc-VirtualBox:06031] [ 1] /lib/x86_64-linux-gnu/libucp.so.0(+0x35e52)[0x7108781c3e52]
    [hlc-VirtualBox:06031] [ 2] /lib/x86_64-linux-gnu/libucp.so.0(ucp_mem_unmap+0x66)[0x7108781c3f96]
    [hlc-VirtualBox:06031] [ 3] /usr/lib/x86_64-linux-gnu/openmpi/lib/openmpi3/mca_spml_ucx.so(mca_spml_ucx_register+0x247)[0x710879083e07]
    [hlc-VirtualBox:06031] [ 4] /lib/x86_64-linux-gnu/liboshmem.so.40(mca_memheap_base_reg+0x92)[0x71087b238a92]
    [hlc-VirtualBox:06031] [ 5] /lib/x86_64-linux-gnu/liboshmem.so.40(mca_memheap_base_select+0xe4)[0x71087b239824]
    [hlc-VirtualBox:06031] [ 6] /lib/x86_64-linux-gnu/liboshmem.so.40(oshmem_shmem_init+0x238)[0x71087b1b0508]
    [hlc-VirtualBox:06031] [ 7] /lib/x86_64-linux-gnu/liboshmem.so.40(shmem_init+0x50)[0x71087b1b0760]
    [hlc-VirtualBox:06031] [ 8] ./main(+0x11da)[0x58079e7bb1da]
    [hlc-VirtualBox:06031] [ 9] /lib/x86_64-linux-gnu/libc.so.6(+0x29d90)[0x71087ae29d90]
    [hlc-VirtualBox:06031] [10] /lib/x86_64-linux-gnu/libc.so.6(__libc_start_main+0x80)[0x71087ae29e40]
    [hlc-VirtualBox:06031] [11] ./main(+0x1105)[0x58079e7bb105]
    [hlc-VirtualBox:06031] *** End of error message ***
    --------------------------------------------------------------------------
    Primary job  terminated normally, but 1 process returned
    a non-zero exit code. Per user-direction, the job has been aborted.
    --------------------------------------------------------------------------
    --------------------------------------------------------------------------
    shmemrun noticed that process rank 3 with PID 0 on node hlc-VirtualBox exited on signal 11 (Segmentation fault).
    --------------------------------------------------------------------------
    ```

    ubuntu 2404 上编译器的版本为

    ```
    (base) hlc@ubuntu2404:~$ shmemcc --version
    gcc (Ubuntu 13.2.0-23ubuntu4) 13.2.0
    Copyright (C) 2023 Free Software Foundation, Inc.
    This is free software; see the source for copying conditions.  There is NO
    warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

    ```

    ubuntu 2204 上编译器版本为

    ```
    (base) hlc@hlc-VirtualBox:~$ shmemcc --version
    gcc (Ubuntu 12.3.0-1ubuntu1~22.04) 12.3.0
    Copyright (C) 2022 Free Software Foundation, Inc.
    This is free software; see the source for copying conditions.  There is NO
    warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
    ```

    不同的编译器、不同的操作系统下有不同的行为，有可能是 openmpi 库有问题。

    两个探索方向：

    1. 換其他版本的 openshmem 实现试一试

    2. 搜报错信息，看看有无解决方案

* `sudo apt install libucx-dev`这个好像没啥用

    `shmem.h`可以通过`sudo apt install libopenmpi-dev`获得。

    在 openmpi 的 include 目录下：`/usr/lib/x86_64-linux-gnu/openmpi/include/shmem.h`

* openshmem hello world

    `main.c`:

    ```c
    #include <shmem.h>
    #include <stdio.h>

    int main()
    {
        int me, npes;
        shmem_init();
        npes = num_pes();
        me = my_pe();
        printf("Hello from %d of %d\n", me, npes);
        shmem_finalize();
        return 0;
    }
    ```

    compile:

    `shmemcc -g main.c -o main`

    run:

    `shmemrun -np 4 ./main`

    output:

    ```
    Hello from 1 of 4
    Hello from 2 of 4
    Hello from 3 of 4
    Hello from 0 of 4
    ```

    有时候会报错，不清楚原因：

    ```
    [ubuntu2404:04160] Warning ../../../../../oshmem/mca/memheap/base/memheap_base_alloc.c:113 - mca_memheap_base_allocate_segment() too many segments are registered: 33. This may cause performance degradation. Pls try adding --mca memheap_base_max_segments <NUMBER> to mpirun/oshrun command line to suppress this message
    Hello from 0 of 4
    Hello from 2 of 4
    Hello from 3 of 4
    [ubuntu2404:4160 :0:4160] Caught signal 11 (Segmentation fault: address not mapped to object at address 0x5ccfd6b0dc80)
    ==== backtrace (tid:   4160) ====
    0  /lib/x86_64-linux-gnu/libucs.so.0(ucs_handle_error+0x2ec) [0x76d66003e64c]
    1  /lib/x86_64-linux-gnu/libucs.so.0(+0x2d34d) [0x76d66004034d]
    2  /lib/x86_64-linux-gnu/libucs.so.0(+0x2d51d) [0x76d66004051d]
    3  /lib/x86_64-linux-gnu/liboshmem.so.40(+0xbcc98) [0x76d663026c98]
    4  /lib/x86_64-linux-gnu/liboshmem.so.40(mca_memheap_modex_recv_all+0x38b) [0x76d663027cbb]
    5  /lib/x86_64-linux-gnu/liboshmem.so.40(oshmem_shmem_init+0x263) [0x76d662f98d33]
    6  /lib/x86_64-linux-gnu/liboshmem.so.40(shmem_init+0x4f) [0x76d662f98f3f]
    7  ./main(+0x11da) [0x5ccfd601e1da]
    8  /lib/x86_64-linux-gnu/libc.so.6(+0x2a1ca) [0x76d662c2a1ca]
    9  /lib/x86_64-linux-gnu/libc.so.6(__libc_start_main+0x8b) [0x76d662c2a28b]
    10  ./main(+0x1105) [0x5ccfd601e105]
    =================================
    [ubuntu2404:04160] *** Process received signal ***
    [ubuntu2404:04160] Signal: Segmentation fault (11)
    [ubuntu2404:04160] Signal code:  (-6)
    [ubuntu2404:04160] Failing at address: 0x3e800001040
    [ubuntu2404:04160] [ 0] /lib/x86_64-linux-gnu/libc.so.6(+0x45320)[0x76d662c45320]
    [ubuntu2404:04160] [ 1] /lib/x86_64-linux-gnu/liboshmem.so.40(+0xbcc98)[0x76d663026c98]
    [ubuntu2404:04160] [ 2] /lib/x86_64-linux-gnu/liboshmem.so.40(mca_memheap_modex_recv_all+0x38b)[0x76d663027cbb]
    [ubuntu2404:04160] [ 3] /lib/x86_64-linux-gnu/liboshmem.so.40(oshmem_shmem_init+0x263)[0x76d662f98d33]
    [ubuntu2404:04160] [ 4] /lib/x86_64-linux-gnu/liboshmem.so.40(shmem_init+0x4f)[0x76d662f98f3f]
    [ubuntu2404:04160] [ 5] ./main(+0x11da)[0x5ccfd601e1da]
    [ubuntu2404:04160] [ 6] /lib/x86_64-linux-gnu/libc.so.6(+0x2a1ca)[0x76d662c2a1ca]
    [ubuntu2404:04160] [ 7] /lib/x86_64-linux-gnu/libc.so.6(__libc_start_main+0x8b)[0x76d662c2a28b]
    [ubuntu2404:04160] [ 8] ./main(+0x1105)[0x5ccfd601e105]
    [ubuntu2404:04160] *** End of error message ***
    --------------------------------------------------------------------------
    Primary job  terminated normally, but 1 process returned
    a non-zero exit code. Per user-direction, the job has been aborted.
    --------------------------------------------------------------------------
    --------------------------------------------------------------------------
    shmemrun noticed that process rank 1 with PID 0 on node ubuntu2404 exited on signal 11 (Segmentation fault).
    --------------------------------------------------------------------------
    ```

    即使在代码中加上 barrier 也一样会报错，根据报错信息看，有 3 个 pe 正常打印，有 1 个 pe 报错，猜测可能是有的 pe 已经 init 完成，有的 pe 还未 init 完成，已经完成 init 的 pe 调用所有 shmem 相关的函数都会报错。

    假如是因为有的 pe 未完成 init，那么在 init 函数后加一个 sleep()，应该可以保证所有 pe 都完成 init。但是加了 sleep 后仍然会报错。说明可能不是 init 未完成的原因。

    尝试在程序的各个地方都加上 sleep，仍然报错，说明不像是同步导致的问题。

    另外一个报错的信息：

    ```
    [ubuntu2404:05038] Warning ../../../../../oshmem/mca/memheap/base/memheap_base_alloc.c:113 - mca_memheap_base_allocate_segment() too many segments are registered: 33. This may cause performance degradation. Pls try adding --mca memheap_base_max_segments <NUMBER> to mpirun/oshrun command line to suppress this message
    Hello from 0 of 4
    Hello from 1 of 4
    Hello from 2 of 4
    Hello from 3 of 4
    ```

    这个报错完全没有头绪。

* openshmem terms collection

    * Processing Elements (PEs)

    * PGAS - Partitioned Global Address Space

    * Single Program Multiple Data (SPMD) 

    * Remote Memory Access (RMA)

    * Atomic Memory Operations (AMOs)

    * Remotely accessible data objects are called Symmetric Data Objects

    * dynamic shared object (DSO)

    * Symmetric Heap

    * Symmetric Hierarchical MEMory library (SHMEM)

## note
