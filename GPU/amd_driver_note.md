# amd driver note

首先去官网下载 amdgpu-install 工具：

<https://www.amd.com/en/support/linux-drivers>

下载完后得到的是一个 deb 包，安装完后执行

`amdgpu-install -y --accept-eula --usecase=workstation --vulkan=pro --opencl=rocr --no-32`

说明：

* `-y` 应该是 yes 的意思吧，

* `--accept-eula`必须要加上，这个好像是一个什么条款。

    如果不加这个，`amdgpu-install`就会把 apt 的 sources 里的一个搜索路径给注释掉，导致无法搜索到 amd 的闭源驱动。

* `--usecase=workstation`

    加这个主要是为了安装 amd 的核心驱动。如果只写`--vulkan=pro`之类的，会说找不到依赖。

* `--vulkan=pro`

    指定 vulkan 版本。amd 的 vulkan 有两个版本，一个是开源驱动 amdvlk，一个是闭源驱动 pro。这里 pro 指的是 proprietary（专有的），即 amd 的闭源驱动。

    这俩版本性能感觉差不多，这里选 pro 了。

* `--opencl=rocr`

    在 ubuntu 22.04.4，kernel 版本 6.5.0 中，amd 似乎完全放弃了对`--opencl=legacy`的支持。这里只能填这个。

* `--no-32`

    ubuntu 22.04.4，kernel 版本 6.5.0 似乎完全抛弃了对 32 位程序的支持，如果不加这个参数，amd 还是会去找 32 位库的支持，导致有些 32 位依赖包在 apt 库中找不到。

安装过程可能需要 dkms 编译，默认会开所有可用的线程。我的机器可能供电有问题，瞬时开所有线程会导致系统崩溃，编译失败。

解决办法是在 BIOS 里关掉超线程，或者关掉几个 cpu 核心。目前测试的是关闭超线程后，可以成功安装。

剩下的问题基本可以看 log 输出搞定，比如编译时候需要 gcc-12，可以手动安装一下。

如果需要卸载 amd 驱动，可以执行`amdgpu-install --uninstall`，这样会卸载所有驱动。

安装完成后，运行`clinfo`会看到 platform，但是 device 数量显示是 0，因为 amd 给 device 创建的用户组是`render`，需要把当前用户加入到这个 group 里才有读取信息的权限。

```bash
ls -l /dev/dri/render*  # 查看 device 权限
sudo usermod -a -G render $LOGNAME  # 将当前用户加入到用户组 render 中，获取 opencl 权限
sudo usermod -a -G video $LOGNAME  # 将当前用户加入到用户组 video 中，获取视频编解码的权限
```

设置完成后，需要`logout`重新登陆当前用户，或者直接重启系统，使得用户组的变更生效。
