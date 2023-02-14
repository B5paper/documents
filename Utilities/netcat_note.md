# Netcat note

windows exe: <https://joncraton.org/blog/46/netcat-for-windows/>

netcat for windows: <https://github.com/diegocr/netcat>


传输二进制文件：

在接收端使用命令：`nc -l 1234 > filetxt`

在发送端使用命令：`nc <ip_addr> 1234 < file.txt` 

文件完成传输后，接收文件的 nc 进程会自动退出。

scanning:

`nc -v -w 2 -z target 20-30`

* `-z`：Only scan for listening daemons, without sending any data to them. 

* `-i`可以用于增加端口之间的扫描间隔。

Banner grabbing:

`nc -v -n 192.168.1.90 80`

* `-d`：禁止从 stdin 输入。此时无论从 terminal 输入，还是使用`<`进行重定向输入，都是不可以的。

文件传输：

可以使用`<`向 nc 通过 stdin 传输文件：

server: `nc -l -p 12345 < hello.txt`

client: `nc 127.0.0.1 > hello.txt`