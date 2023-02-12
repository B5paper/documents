# OpenSSH

首先需要安装 openssh-server:

`sudo apt install openssh-server`

查看服务列表：

`service --status-all`

里面应该有一个`ssh`。

然后启动服务：

`sudo service ssh start`

1. 启动服务`ssh`时，报错`sshd: no hostkeys available -- exiting`

    需要进入`/etc/ssh`文件夹下生成 rsa 的公钥：

    `ssh-keygen -A`

    然后再启动`ssh`就可以了：`sudo service ssh start`。

1. 客户端登录`ssh`时，报错`no hostkeys available`

    需要在服务器端修改配置文件`/etc/ssh/sshd_config`，把`PasswordAuthentication`改成`yes`。

    此时用`sudo service ssh restart`重启服务好像也没用，可能需要重启系统才行。