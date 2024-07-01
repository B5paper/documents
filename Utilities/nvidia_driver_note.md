# Nvidia driver note

## cache

* 查看 nvidia driver 的版本

    （未验证）

    `nvidia-smi` 

    `cat /proc/driver/nvidia/version `

    `modinfo nvidia|grep version:`

    `nvidia-smi --query-gpu=driver_version --format=csv,noheader --id=0`

* 安装顺序

    kernel version: 5.19.0-50-generic

    install headers: `sudo apt install linux-headers-5.19.0-50-generic`

    nvidia driver version: 520

    nvidia kernel source -> [firmware,] kernel common -> [dkms open ->] dkms

    libnvidia-compute -> nvidia utils

    libnvidia-common -> libnvidia-gl -> nvidia-driver

    cuda-libraries, cuda-drivers -> cuda-runtime, cuda-toolkit -> cuda

    cuda-compiler, cuda-libraries-dev, cuda-tools, cuda-codumentation, cuda-demo-suite -> cuda-toolkit

    cuda-command-line-tools, cuda-visual-tools -> cuda-tools

    cuda-nsight-compute, cuda-nsight-systems, cuda-nsight, cuda-nvml-dev, cuda-nvvp -> cuda-visual-tools

    cuda-cupti -> cuda-cupti-dev, cuda-gdb, cuda-memcheck, cuda-nvidisasm, cuda-nvprof, cuda-nvtx, cuda-sanitizer -> cuda-command-line-tools

    cuda-nvdisasm -> cuda-gdb

    cuda-nvrtc-dev, cuda-cublas-dev, libcufft-dev, libcufile-dev, libcurand-dev, libcusolver-dev, libcusparse-dev, libnpp-dev, libnvjpeg-dev, cuda-profiler-api -> cuda-libraries-dev

    cuda-cuobjdump, cuda-cuxxfilt, cuda-nvcc, cuda-nvprune -> cuda-compiler

    cuda-cudart-dev -> cuda-nvcc

    cuda-cccl, cuda-driver-dev -> cuda-cudart-dev

    cuda-cudart, cuda-nvrtc, libcublas, libcufft, libcufile, libcurand, libcusolver, libcusparse, libnpp, libnvjpeg -> cuda-libraries

    cuda-toolkit-config-common -> cuda-cudart

    libnvidia-common, libnvidia-decode, libnvidia-encode, libnvidia-fbc, libnvidia-gl, nvidia-compute-utils, nvidia-driver, nvidia-modprobe, nvidia-settings -> cuda-drivers 

    libopengl0 -> libnvidia-gl

    libnvidia-extra, libnvidia-cfg, xserver-xorg-video-nvidia -> nvidia-driver

    screen-resolution-extra, libvdpau1 -> nvidia-settings

* 无论 cuda 11.8 还是 12.1，都需要 gcc 12

* `update-alternatives --get-selections`可以得到所有的默认程序列表

    同样使用`update-alternatives`命令可以更改默认的程序。

* `ln -s <target_file> <new_link>`

* 安装 nvidia driver 时，需要切换到文字操作系统

    cuda 的 `.run` installer file 里包含了 nvidia driver，因此也需要在文字操作系统下安装。