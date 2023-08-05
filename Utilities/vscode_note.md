# VSCode Note

1. invoke call stack window: <https://stackoverflow.com/questions/1607065/how-to-show-call-stack-immediates-and-other-hidden-windows-in-visual-studio>

1. <https://learn.microsoft.com/en-us/visualstudio/debugger/how-to-use-the-call-stack-window?view=vs-2022>

* vscode 使用 lldb 调试的技巧：<https://www.codenong.com/cs105826090/>

    感觉有用的内容不多，有时间了看看。

* ssh 多级跳转

    ```ssh
    Host jump1
    HostName ip of jump1
    Port port id
    User user name
    IdentityFile key_path

    Host jump2
    HostName ip of jump2
    Port port id
    User user name
    IdentityFile key_path
    # command line,setting jump1
    ProxyCommand ssh -q -W %h:%p jump1


    # Target machine with private IP addressHost target
    Host target
    HostName ip_of target
    Port port id
    User user name
    IdentityFile key_pathProxyCommand ssh -q -W %h:%p jump2
    ```

    比如一台 server 要经过 jump1 连到 jump2，再通过 jump2 连到 target，那么就可以用上面的配置。

    Ref: <https://www.doc.ic.ac.uk/~nuric/coding/how-to-setup-vs-code-remote-ssh-with-a-jump-host.html>

    <https://support.cs.wwu.edu/home/survival_guide/tools/VSCode_Jump.html>

    也可以直接用`proxyjump`，example:

    ```s
    Host 10.211.10.67
    HostName 10.211.10.67
    ProxyJump liucheng@172.18.25.248
    User test

    Host 192.168.122.75
    HostName 192.168.122.75
    ProxyJump liucheng@172.18.25.248
    User ubuntu
    ```