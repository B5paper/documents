# Apt note

## cache

* apt instlal pyhton3-xx 安装的包的位置

    * Python 模块（纯 Python）:

        /usr/lib/python3/dist-packages/ (Ubuntu/Debian 的标准位置)

        /usr/local/lib/python3.x/dist-packages/ (某些情况)

    * Python 模块（含 C 扩展）:

        /usr/lib/python3.x/site-packages/ (某些情况)

    * 可执行脚本:

        /usr/bin/ (例如 pip, black, flake8 等命令行工具)

    * 配置文件:

        /etc/ (某些 Python 包会安装配置文件在这里)

    * 共享库:

        /usr/share/ (文档、示例文件等)

    查看具体的包：

    `dpkg -L python3-xxx`

    pip install xxx → 默认用户级安装 (~/.local/lib/python3.x/site-packages/)

* isoinfo 在 genisoimage 包中

* `http://security.ubuntu.com/ubuntu/ jammy-security restricted multiverse universe main`的 ip 为`1.1.1.3`，属于 cloudflare 的机器，国内不一定能访问到。

    如果在`apt update`时无法访问这个 ip 的 80 端口，可以考虑在`/etc/apt/source.list`里把这一行注释掉。

* apt 只下载不安装 deb 包：

    apt 下载的包会被默认放到`/var/cache/apt/archives`目录下。

    首先清空这个目录：`sudo apt clean`

    然后下载指定包以及依赖：

    `sudo apt install --download-only <package_name>`

    再把缓存目录下的 deb 包都复制出来：

    `cp /var/cache/apt/archives/*.deb ./my_debs`

    如果需要安装，可以用 dpkg：

    `sudo dpkg -i ./my_debs/*.deb`

## note

查看已经安装的包的版本：`apt policy <package-name>`

安装指定版本的包：`apt install <package-name>=<version>`

也可以用`apt list <package-name>`查看已经安装的包

查看已安装的软件：`apt list --installed  <package_name>`

