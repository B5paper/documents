# fail2ban note

## cache

* fail2ban 手动往 sshd jail 里添加一个 ip

    ```bash
    # 永久禁止 IP（直到手动移除）
    sudo fail2ban-client set sshd banip 192.168.1.100

    # 禁止带时间限制（例如 1 天）
    sudo fail2ban-client set sshd banip --time 86400 192.168.1.100
    ```

    查看和管理已禁止的 IP:

    ```bash
    # 查看 sshd jail 状态
    sudo fail2ban-client status sshd

    # 查看已禁止的 IP 列表
    sudo fail2ban-client get sshd banned

    # 解禁 IP
    sudo fail2ban-client set sshd unbanip 192.168.1.100
    ```

## topics