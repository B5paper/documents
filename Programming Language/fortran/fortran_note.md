* ubuntu 如何安装 fortran 编译环境

    在 Ubuntu 系统中安装 Fortran 编译环境非常简单。以下是几种常见的安装方法：

    方法一：安装 GFortran（推荐）

    GFortran 是 GCC 的一部分，是最常用的 Fortran 编译器。

    1. 更新包列表

        ```bash
        sudo apt update
        sudo apt upgrade
        ```

    2. 安装 GFortran

        ```bash
        sudo apt install gfortran
        ```

    3. 验证安装

        ```bash
        gfortran --version
        ```

    方法二：安装完整的 GCC 编译套件

    如果你还需要 C/C++ 等其他编译器：

    ```bash
    sudo apt install build-essential gfortran
    ```

    `build-essential` 包含了 GCC、G++、make 等常用编译工具。

    方法三：安装其他 Fortran 编译器

    **安装 Intel Fortran Compiler (ifort)**

    ```bash
    # 需要先安装 Intel oneAPI 基础工具包
    wget https://registrationcenter-download.intel.com/akdlm/irc_nas/19079/l_BaseKit_p_2023.0.0.25537_offline.sh
    sudo sh l_BaseKit_p_2023.0.0.25537_offline.sh
    # 在安装界面选择 Fortran 编译器组件
    ```

    **安装 Flang (LLVM Fortran)**

    ```bash
    sudo apt install flang
    ```

    **测试 Fortran 环境**

    创建一个简单的 Fortran 程序 `hello.f90`：

    ```fortran
    program hello
      implicit none
      print *, 'Hello, Fortran!'
    end program hello
    ```

    编译运行：

    ```bash
    gfortran hello.f90 -o hello
    ./hello
    ```

    **安装开发工具（可选）**

    ```bash
    # 安装调试器
    sudo apt install gdb

    # 安装性能分析工具
    sudo apt install valgrind

    # 安装构建工具
    sudo apt install cmake
    ```

    这样就完成了 Fortran 编译环境的安装！最常用的是第一种方法，安装 GFortran 即可满足大多数需求。

