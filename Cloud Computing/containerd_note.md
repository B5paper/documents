# Containerd Note

可以使用`apt install docker.io`，在安装 docker 时，把 containerd 和 runc、ctr 等工具都安装上。

systemd sercice:

file path: `/usr/local/lib/systemd/system/containerd.service`

使用`apt install docker.io`安装时，配置文件的地址：`/usr/lib/systemd/system/containerd.service`

content:

```service
# Copyright The containerd Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this /usr/local/lib/systemd/system/containerd.service
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

[Unit]
Description=containerd container runtime
Documentation=https://containerd.io
After=network.target local-fs.target

[Service]
#uncomment to enable the experimental sbservice (sandboxed) version of containerd/cri integration
#Environment="ENABLE_CRI_SANDBOXES=sandboxed"
ExecStartPre=-/sbin/modprobe overlay
ExecStart=/usr/local/bin/containerd

Type=notify
Delegate=yes
KillMode=process
Restart=always
RestartSec=5
# Having non-zero Limit*s causes performance problems due to accounting overhead
# in the kernel. We recommend using cgroups to do container-local accounting.
LimitNPROC=infinity
LimitCORE=infinity
LimitNOFILE=infinity
# Comment TasksMax if your systemd version does not supports it.
# Only systemd 226 and above support this version.
TasksMax=infinity
OOMScoreAdjust=-999

[Install]
WantedBy=multi-user.target
```

如果需要设置代理，可以加上：

```service
[Service]
Environment="HTTP_PROXY=http://9.21.61.141:3128/"
```

启动：

```bash
systemctl daemon-reload
systemctl enable --now containerd
```

测试：

```bash
ctr images pull docker.io/library/redis:alpine
ctr run docker.io/library/redis:alpine redis
```

常用的与 containerd 交互的 ctr 有三种：

1. `ctr`

    由 containerd 社区开发，主要用于调试。

* 停止一个 container

    首先列出所有正在运行的 container: `ctr task ls`

    然后使用`kill`终止 container: `ctr task kill v0`

    或者发送 signal: `ctr task kill -s SIGKILL v0`

* 如果查不到某些 container 或 image，那么需要查看 namespace 设置的是否正确

    比如：`ctr images list -n k8s.io`

## ctr

ctr man page: <https://www.mankier.com/8/ctr#>

ctr 本身没有 manual，这个 man page 似乎是仿照着 man page 的格式写的文档。

## Problem shooting

1. login

    * 方法一：在命令行直接加用户名和密码

        `ctr --debug image pull -u {username}:{password} --plain-http docker.xxx.cn:5000/maxfaith/miop_ui:development`

    * 方法二：在配置文件里把写好

        ```yaml
        [plugins."io.containerd.grpc.v1.cri".registry.configs]
            [plugins.cri.registry.mirrors]
                [plugins.cri.registry.mirrors."docker.xxx.cn:5000"]
                    endpoint = ["http://docker.xxx.cn:5000"]

            [plugins.cri.registry.configs."docker.xxx:5000".auth]
                username = "xxxxxx"
                password = "xxxxxxxx"
        ```

    其他参考资料：<https://serverfault.com/questions/1086316/how-do-you-login-to-docker-hub-when-using-containerd>

1. 查看 log：`sudo journalctl -xu containerd`

    换行：`journalctl -xn | less`

    `journalctl -xn --no-pager`

    Ref: <https://unix.stackexchange.com/questions/229188/journalctl-how-to-prevent-text-from-truncating-in-terminal>