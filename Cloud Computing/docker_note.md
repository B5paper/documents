# Docker Note

Learning materials:

1. <https://www.simplilearn.com/tutorials/docker-tutorial/what-is-dockerfile>

1. get the manifest using bash: <https://stackoverflow.com/questions/57316115/get-manifest-of-a-public-docker-image-hosted-on-docker-hub-using-the-docker-regi/72574987#72574987>

1. docker handbook: <https://docker-handbook.farhan.dev/table-of-contents>

1. awesome docker: <https://github.com/veggiemonk/awesome-docker>



官网：<https://docs.docker.com/>

docker hub:

username: hhlc
email: liucheng.hu@intel.com
password: hlc695230

安装：`sudo apt install docker.io`

卸载：

```
systemctl stop docker
apt purge docker.io
rm -rf /var/lib/docker
rm -rf /var/lib/containerd
```

registry mirror configuration:

```
aaaa
```

## 有关 docker service

一些命令：

* 启动：`systemctl start docker`
* 停止：`systemctl stop docker`
* 重启：`systemctl restart docker`
* 查看 docker 状态：`systemctl status docker`
* 开机启动：`systemctl enable docker`
* 查看 docker 概要信息：`docker info`
* 查看 docker 总体帮助文件：`docker --help`
* 查看 docker 命令帮助文档：`docker 具体命令 --help`

TAG 指的是镜像 image 的版本。默认

## 有关镜像（image）

* 列出主机本地上的镜像：`docker images`（等同于`sudo docker image ls`, `sudo docker image list`）

    * `docker images -a`：列出本地所有镜像（含历史映像层）
    * `docker images -q`：只显示镜像 ID

* 从[docker.io](docker.io)里搜索镜像：`docker search [option] <image_name>`

    * `docker search --limit N <img_name>`

        只列出 N 个镜像。默认是 25 个。

* 拉取镜像：`docker pull <image_name>[:TAG]`

    没有写 tag 就表示下载最新版：`docker pull <img_name>:latest`

* 列出整体信息：`docker system df`

    输出大概长这样：

    ```
    TYPE            TOTAL     ACTIVE    SIZE      RECLAIMABLE
    Images          1         1         13.26kB   0B (0%)
    Containers      1         0         0B        0B
    Local Volumes   0         0         0B        0B
    Build Cache     0         0         0B        0B
    ```

* 删除镜像：`docker rmi <ID>`

    强制删除：`docker rmi -f <ID>`

    删除多个 image：`docker rmi -f <img_name_1>:TAG <img_name_2>:TAG`

    删除所有 images：`docker rmi -f $(docker images -qa)`

虚悬镜像：仓库名，标签都是`<none>`的镜像。俗称 dangling image。

## 有关容器（container）

* 从镜像启动一个容器：`docker run [options] <image_name> [command]`

    常用的 options:

    * `--name=<container_name>`：为容器指定一个名称。不指定的话会随机分配。

    * `-d`：后台运行并返回容器 ID，即启动守护式容器（后台运行）

        docker 容器后台运行时必须有一个前台进程。容器运行的命令如果不是一直扶起的命令，比如（top, tail），就会自动退出

    * `-i`：以交互模式运行容器，通常与`-t`共用

    * `-t`：为容器重新分配一个伪输入端，通常与`-i`同时使用。

    * `-P`：随机端口映射

    * `-p`：指定端口映射

        `-p hostPort:containerPort`

        `-p ip:hostPort:containerPort`

    * `command`：打开端口时要运行的命令

        `docker run -it ubuntu /bin/bash`

    * `-e ENVION_NAME=VALUE`

        启动容器时配置好环境变量。

* 列出当前的所有**正在运行**容器：`docker ps [options]`

    * `-a`：列出所有容器。包括当前所有正在运行的容器和历史上运行过的。

    * `-l`：显示最近创建的容器

    * `-n`：显示最近 n 个创建的容器

    * `-q`：静默模式，只显示容器 ID

* 退出容器

    两种方式：

    1. 在交互模式中输入`exit`，退出当前 shell。如果所有的 shell 都已经退出，那么容器会停止。

    1. 按下使用`ctrl + p + q`退出，不会停止当前 shell。（按下`ctrl`，然后先按`p`，再按`q`）

* 重新启动已经停止运行的容器：`docker start [options] <container_id>`

    重新启动已经停止的容器时，会执行在创建容器时使用的`<command>`。

    如果一个已经停止运行的容器，它在创建时候的 command 会很快执行结束，并且不是`bash`之类的，那么似乎不可以重新启动后直接进入 bash。

    常用的 options:

    `-i`: Attach container's STDIN
    `-a`: Attach STDOUT/STDERR and forward signals

    如果在使用 image 创建 container 时，使用的 command 是`bash`，那么可以使用

    ```bash
    sudo docker start -ia <container_id>
    ```

    重新启动容器，并进入 bash 界面。（测试好像只用`-i`也可以，但是只用`-a`的话会有 bug）

* 重新进入后台运行的容器：

    * `docker exec -it <container_id> <command>`

    * `docker attach <container_id>`

    `attach`直接进入容器启动命令的终端，不会启动新的进程，用`exit`退出会导致容器的终止。

    `exec`是在容器中打开新的终端，并且可以启动新的进程，使用`exit`退出不会导致容器终止。

    Example:

    ```bash
    sudo docker exec -it c8dadfb9dc6c bash
    ```

* 停止容器：`docker stop <container_id>`

    （正在运行中的容器似乎无法直接 stop，不知道为什么）

* 强制停止容器：`docker kill <container_id>`

* 删除已停止的容器：`docker rm <container_id>`

    一次删除多个容器实例：

    * `docker rm -f $(docker ps -a -q)`

    * `docker ps -a -q | xargs docker rm`

* 查看日志：`docker logs <img>`

* 查看容器内部细节：`docker inspect <img | container>`

* 复制 container 中的文件到主机上：`docker cp <container_id>:<path> <dst_path_on_host>`

* 导出某个容器：`docker export <container_id> > abcd.tar`

* 导入某个容器：`cat <tar_file> | docker import - <usr>/<img>:<tag>`

容器与系统交互使用的是 socket，通过端口与系统交互。

* 重新命名一个 container

    ```bash
    docker rename CONTAINER NEW_NAME
    ```

    注：
    
    1. docker 的各种常用命令，有时间了系统地学一下：<https://docs.docker.com/engine/reference/run/>

## 镜像仓库（docker image registry）

UnionFS（联合文件系统）：是一种分层，轻量级并且高性能的文件系统，它支持**对文件系统的修改作为一次提交来一层层地叠加**。镜像可以通过分层叠加。

docker 镜像的最底层是引导文件系统 bootfs。在 bootfs 之上是 rootfs(root file system)，即`/dev`，`/bin`，`/proc`，`/etc`等等。

提交一个新镜像：

`docker commit -m="some info" -a="author" <container_id> <repo_name>/<img_name>[:tag]`

docker hub: <https://hub.docker.com/>

私有库搭建：

先下载`registry`这个镜像：`docker pull registry`

运行：`docker run -d -p 5000:5000 -v /zzyyuse/myregistry/:/tmp/registry --privileged=true registry`

`-v host_path:container_path`可以将宿主机上的路径挂载到容器内部的指定路径。

可以指定多个`-v`。如果`container_path`不存在，则会自动创建。

默认情况下使用的是`rw`权限：`-v /host_abs_path:/container_path:rw`

只读的话，可以把`rw`改成`ro`。

容器2继承容器1的卷规则：`docker run -it --privileged=true --volumes-from <img_name> --name u2 ubuntu`

此时容器1，容器2，宿主机之间都能数据共享。

查看当前私服仓库中的镜像：`curl -XGET http://192.168.111.162:5000/v2/_catalog`

修改本地镜像 tag：`docker tag zzyyubuntu:1.2 192.168.111.162:5000/zzyyubuntu:1.2`

配置允许 http 协议：在`cat /etc/docker/daemon.json`文件中添加：

`"insecure-registries": ["192.168.111.162:5000"]`

推送：`docker push 192.168.111.167:5000/zzyyubuntu:1.2`

docker 挂载主机目录访问如果出现`cannot open directory: Permisson denied`，可以使用在挂载目录后加一个`--privileged=true`解决。不开这个的话，container内部的 root 只是外部的一个普通用户权限。

## 其他

1. mysql 中文乱码

    需要配置文件：

    ```
    [client]
    default_character_set=utf8
    [mysqld]
    collation_server = utf8_general_ci
    character_set_server = utf8
    ```

1. proxy

    在 registry 中 pull image 时，代理的设置需要在 docker service 中配置，因为摘取镜像这个动作是 service 干的，不是 client 干的。官网也给出了具体教程：<https://docs.docker.com/config/daemon/systemd/#httphttps-proxy>。

    The configs in `~/.docker/config.json` will impact on the proxy of all containers. Besides, the proxy address in the file should be the ip address of virtual ethernet interface `docker0` rather than `127.0.0.1`, because the `127.0.0.1` inside the container is different from `127.0.0.1` on the host.

    如果要设置 build image 以及 run container 时的代理，可以参考：<https://docs.docker.com/network/proxy/>

1. restart a stopped docker container

    Ref: <https://stackoverflow.com/questions/39666950/how-restart-a-stopped-docker-container>

1. Removing Dangling and Unused Images

    <https://www.baeldung.com/ops/docker-remove-dangling-unused-images#:~:text=If%20we%20do%20not%20want,can%20use%20the%20%2Da%20flag.&text=The%20command%20will%20return%20the,the%20space%20that%20was%20freed.>

1. 进入 container 时选择 user: <https://blog.csdn.net/q1248807225/article/details/113754472?spm=1001.2101.3001.6661.1&utm_medium=distribute.pc_relevant_t0.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-1-113754472-blog-124169267.pc_relevant_multi_platform_whitelistv3&depth_1-utm_source=distribute.pc_relevant_t0.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-1-113754472-blog-124169267.pc_relevant_multi_platform_whitelistv3&utm_relevant_index=1>

1. docker port introduction: <https://nickjanetakis.com/blog/docker-tip-59-difference-between-exposing-and-publishing-ports>

## Dockerfile

Dockerfile 用于构建镜像。

```dockerfile
FROM <base_img>:<tag>  # base image
MAINTAINER author<mail_addres>  # 维护者的姓名和邮箱
RUN <command>  # 在构建的时候就执行命令
EXPOSE <port>  # 端口暴露
WORKDIR  在创建容器后，终端默认登陆的工作目录
USER 指定该镜像以什么样的用户去执行，默认为 root
ENV <env_name> <value>  容器中的环境变量
ADD <xxx.tar.gz> <container_dir>  # 将宿主机目录下的文件拷贝进镜像且会自动处理 URL 和解压 tar 压缩包
COPY 只复制文件或目录，不解压
VOLUME 数据目录挂载
CMD ['command', '参数1', ...]  指定启动容器后要运行的指令
ENTRYPOINT ['comman', 'param1', 'param2', ...]  指定启动容器后要运行的命令
```

Dockerfile 中可以有多个 CMD 指令，但只有最后一个生效。CMD 会被 docker run 之后的参数替换。`ENTRYPOINT`不会被`docker run`之后的命令覆盖，而且`CMD`或者`docker run`的指令会被当作参数送给`ENDTRYPOINT`指令指定的程序.

例子：

```dockerfile
entrypoint ["nginx", "-c"]
cmd ["/etc/nginx/nginx.conf"]
```

`docker build -t <img_name>:<tag> <dockerfile_dir>`

* `COPY [--chown=<user>:<group>] <源路径>... <目标路径>`
* `COPY [--chown=<user>:<group>] ["<源路径1>",... "<目标路径>"]`

如果 copy 的是一个文件夹，那么并不会自动在目标路径中创建文件夹，而是会把源路径文件夹中的文件都复制到目标路径文件夹中。

因此 copy 的正确写法通常是：

```dockerfile
COPY ./<dir_name> /root/<dir_name>
```

查看所有的虚悬镜像：

`docker image ls -f dangling=true`

通常会把虚悬镜像删除：

`docker image prune`（删除所有虚悬镜像）

使用自己签名的证书，将创建的 registry 变成 https 接口：<https://docs.docker.com/registry/deploying/#get-a-certificate>

docker uninstall dangling image: <https://stackoverflow.com/questions/33913020/docker-remove-none-tag-images>，有空了再看看这个。