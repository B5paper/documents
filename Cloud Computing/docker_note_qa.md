# Docker Note QA

[unit]
[u_0]
docker run 启动一个镜像，并进入 bash
[u_1]
`docker run -it <image>:<tag> <command>`

[unit]
[u_0]
重新进入 stop 的容器，并进入 bash
[u_1]
`docker start -ia <container_name>`

[unit]
[u_0]
重新进入后台运行的容器
[u_1]
`docker exec -it <container_name> <command>`

[unit]
[u_0]
列出当前正在运行的容器
[u_1]
`docker ps`

[unit]
[u_0]
列出所有的容器，包含已经停止的容器
[u_1]
`docker ps -a`