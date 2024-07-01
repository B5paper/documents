# Apt note

## cache

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

