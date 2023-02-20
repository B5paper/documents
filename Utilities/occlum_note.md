# Occlum Note

## Installation

1. occlum 的安装依赖于 sgx。目前 occlum v0.28.0 依赖于 sgx v2.16。为了方便，occlum 把 sgx fork 了一份，放到了自己的仓库里：<https://github.com/occlum/linux-sgx>，可以直接从这里 clone。

    从`intel/linux-sgx`安装最新版本的 sgx （v2.17.0）会导致 occlum 安装不成功。

1. 安装 sgx 时，首先按照说明安装依赖：

    `sudo apt-get install build-essential ocaml ocamlbuild automake autoconf libtool wget python-is-python3 libssl-dev git cmake perl`

    `sudo apt-get install libssl-dev libcurl4-openssl-dev protobuf-compiler libprotobuf-dev debhelper cmake reprepro unzip pkgconf libboost-dev libboost-system-dev protobuf-c-compiler libprotobuf-c-dev lsb-release`

    然后运行：

    `make preparation`

    此时会从 git 上 clone 一些仓库，可能需要设置一下代理：`http_proxy=http://<ip_addr>:<port> https_proxy=http://<ip_addr>:<port> make preparation`

    接着把一些文件 copy 到 bin 里：`sudo cp external/toolset/{current_distr}/* /usr/local/bin`。这里把`{current_distr}`换成`Ubuntu20.04`

    最后开始编译：`make sdk`。中间缺什么库再使用`sudo apt install`安装就行了。这一步通常问题不大。

    编译安装包：`make sdk_install_pkg`

    后面的`make psw`编译容易出错，后面 occlum 好像也用不上这个库，所以就没编译。

    编译完后，单独把编译好的库拿出来，放到某个目录里（通常是`/opt/intel/sgxsdk`，也可以自己设置自定义路径）：

    `cd linux/installer/bin`

    `./sgx_linux_x64_sdk_${version}.bin --prefix {SDK_INSTALL_PATH_PREFIX}`

    （可以把`{SDK_INSTALL_PATH_PREFIX}`这里换成`/opt/intel/sgxsdk`，也可以找个自定义的路径）

    最后执行一个不知道有啥用的命令：`source ${sgx-sdk-install-path}/environment`

1. 在 occlum 目录下，执行`SGX_SDK=/path/to/intel_sgx_sdk make submodule`

    如果前面安装到`/opt/intel/sgxsdk`，那么这里就不用设置环境变量了。如果前面安装到了自定义路径，那么这里需要替换成自定义的路径。

    编译到后面会出现问题，`../occlum/tools/toolchains`里的各种语言编译器，会被放到`/opt/occlum/toolchains`里，这个目录必须有 root 权限才能创建文件夹，但是如是使用`sudo`来执行`sudo ./build.sh`，会有一些环境变量问题不好解决。`./build.sh`中的前面一部分命令需要用普通用户权限执行，后面有关目录权限的问题需要`sudo`手动执行。

1. 