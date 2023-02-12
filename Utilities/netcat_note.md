# Netcat note

windows exe: <https://joncraton.org/blog/46/netcat-for-windows/>

netcat for windows: <https://github.com/diegocr/netcat>


传输二进制文件：

在接收端使用命令：`nc -l 1234 > filetxt`

在发送端使用命令：`nc <ip_addr> 1234 < file.txt` 

文件完成传输后，接收文件的 nc 进程会自动退出。