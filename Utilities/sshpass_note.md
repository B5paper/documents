# sshpass note

* `sshpass`

    sshpass 可以通过非交互的方式登陆 ssh。

    安装：`apt install sshpass`

    * 使用`-p`参数在命令行输入密码

        `sshpass -p <password> ssh <user>@<host>`

    * 使用`-f`从文件读取密码

        `sshpass -f <passwd_file> ssh <user>@<host>`

        注：

        1. `<passwd_file>`中的密码，sshpass 只会读取第一行的内容，不包括换行符。

    * 使用`-e`从环境变量`SSHPASS`读取密码

        ```bash
        export SSHPASS='123456'
        sshpass -e ssh user@192.168.1.1
        ```

    当 ssh 第一次连接 host 或者 host 的信息与本地存储的 known host 不符时，ssh 会弹出一个交互式的对话要求更新 host 信息。如果需要跳过这个交互式对话，可以使用`-o StrictHostKeyChecking=no`:

    ```bash
    sshpass -p '123456' ssh -o StrictHostKeyChecking=no user@192.168.1.1
    ```

    推荐使用`-f`，这样密码不会被记录在 bash history 中。