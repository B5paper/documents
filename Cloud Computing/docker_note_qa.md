# Docker Note QA

[unit]
[idx]
0
[id]
6362350047929645109
[u_0]
docker run 启动一个镜像，并进入 bash
[u_1]
`docker run -it <image>:<tag> <command>`

[unit]
[idx]
1
[id]
11182387183404132538
[u_0]
重新进入 stop 的容器，并进入 bash
[u_1]
`docker start -ia <container_name>`

[unit]
[idx]
2
[id]
15644948504462002486
[u_0]
重新进入后台运行的容器
[u_1]
`docker exec -it <container_name> <command>`

[unit]
[idx]
3
[id]
4420774242792228039
[u_0]
列出当前正在运行的容器
[u_1]
`docker ps`

[unit]
[idx]
4
[id]
16774027880631328205
[u_0]
列出所有的容器，包含已经停止的容器
[u_1]
`docker ps -a`

