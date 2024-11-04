# OpenSHMEM Note

## cache

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
