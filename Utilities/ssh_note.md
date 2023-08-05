# Ssh Note

Learning materials:

1. <https://www.ssh.com/academy/ssh/config>

1. <https://linuxize.com/post/using-the-ssh-config-file/>

正向转发：假如现在有两台机器 A 和 B，B 上装有 ssh server，A 想在访问本机的某个端口时，变成访问机器 B 上的某个端口，那么就称为正向代理。

此时在`A`机器上运行：

`ssh -L [A_addr:]<A_port>:<B_addr>:<B_port> user_name@addr`

登陆就可以了。

注意，`addr`不一定和`B_addr`相同。若，则通过`addr`转发到`B_addr`上。

反向代理：假如现在有机器`A`和`B`，`B`上装有 ssh server，目标是在`B`上访问`B_port`端口时，相当于访问`A_port`端口。

命令：

`ssh -R B_addr:B_port:A_addr:A_port user@addr`

此时在`B`上访问`B_addr:B_port`就当于访问`A_addr:A_port`。

同理，`addr`不一定和`B_addr`相同。

这种形式相当于内网穿透。

生成一个密钥对：`ssh-keygen -t rsa`

在远程机吕上把公钥写到`authorized_keys`文件里面：`cat ~/id_rsa.pub >> ~/.ssh/authorized_keys`

## connect to a ssh server through a jump/intermediary server

Ref: <https://www.cyberciti.biz/faq/linux-unix-ssh-proxycommand-passing-through-one-host-gateway-server/>

(This artical did not mentioned the usage of ProxyCommand `nc xxx`.)

The simplest way to use a imtermediary to connect to another ssh server is

```bash
ssh -J <use_1@address_1[:port]> <user_2@address_2>
```

* What do the `%h` and `%p` mean in the proxy command?

    <https://unix.stackexchange.com/questions/183951/what-do-the-h-and-p-do-in-this-command>

## ssh config file

The `~/.ssh` directory is automatically created when the user runs the ssh command for the first time. If the directory doesn’t exist on your system, create it using the command below:

```bash
mkdir -p ~/.ssh && chmod 700 ~/.ssh
```

By default, the SSH configuration file may not exist, so you may need to create it using the touch command :

```bash
touch ~/.ssh/config
```

This file must be readable and writable only by the user and not accessible by others:

```bash
chmod 600 ~/.ssh/config
```

The SSH Config File takes the following structure:

```ssh
Host hostname1
    SSH_OPTION value
    SSH_OPTION value

Host hostname2
    SSH_OPTION value

Host *
    SSH_OPTION value
```

Indentation is not required but is recommended since it makes the file easier to read.

The `Host` directive can contain one pattern or a whitespace-separated list of patterns. Each pattern can contain zero or more non-whitespace character or one of the following pattern specifiers:

* `*` - Matches zero or more characters. For example, Host * matches all hosts, while 192.168.0.* matches hosts in the 192.168.0.0/24 subnet.

* `?` - Matches exactly one character. The pattern, Host 10.10.0.? matches all hosts in 10.10.0.[0-9] range.

* `!` - When used at the start of a pattern, it negates the match. For example, Host 10.10.0.* !10.10.0.5 matches any host in the 10.10.0.0/24 subnet except 10.10.0.5.

The SSH client reads the configuration file stanza by stanza, and if more than one patterns match, the options from the first matching stanza take precedence. Therefore more host-specific declarations should be given at the beginning of the file, and more general overrides at the end of the file.

You can find a full list of available ssh options by typing `man ssh_config` in your terminal or visiting the `ssh_config` `man` page <http://man.openbsd.org/OpenBSD-current/man5/ssh_config.5>.

The SSH config file is also read by other programs such as scp , sftp , and rsync .

Example:

```ssh
Host dev
    HostName dev.example.com
    User john
    Port 2322
```

connect: `ssh dev`

Shared SSH Config File Example:

```ssh
Host targaryen
    HostName 192.168.1.10
    User daenerys
    Port 7654
    IdentityFile ~/.ssh/targaryen.key

Host tyrell
    HostName 192.168.10.20

Host martell
    HostName 192.168.10.50

Host *ell
    user oberyn

Host * !martell
    LogLevel INFO

Host *
    User root
    Compression yes
```

When you type ssh targaryen, the ssh client reads the file and apply the options from the first match, which is Host targaryen. Then it checks the next stanzas one by one for a matching pattern. The next matching one is Host * !martell (meaning all hosts except martell), and it will apply the connection option from this stanza. The last definition Host * also matches, but the ssh client will take only the Compression option because the User option is already defined in the Host targaryen stanza.

The full list of options used when you type ssh targaryen is as follows:

```ssh
HostName 192.168.1.10
User daenerys
Port 7654
IdentityFile ~/.ssh/targaryen.key
LogLevel INFO
Compression yes
```

When running ssh tyrell the matching host patterns are: Host tyrell, Host *ell, Host * !martell and Host *. The options used in this case are:

```ssh
HostName 192.168.10.20
User oberyn
LogLevel INFO
Compression yes
```

The ssh client reads its configuration in the following precedence order:

1. Options specified from the command line.
1. Options defined in the ~/.ssh/config.
1. Options defined in the /etc/ssh/ssh_config.

If you want to override a single option, you can specify it on the command line. For example, if you have the following definition:

```ssh
Host dev
    HostName dev.example.com
    User john
    Port 2322
```

and you want to use all other options but to connect as user root instead of john simply specify the user on the command line:

`ssh -o "User=root" dev`

The -F (configfile) option allows you to specify an alternative per-user configuration file.

To tell the ssh client to ignore all of the options specified in the ssh configuration file, use:

`ssh -F /dev/null user@example.com`

## Problem shooting

1. `debug1: Local version string SSH-2.0-OpenSSH_8.9p1 Ubuntu-3ubuntu0.1`

    `Connection closed by UNKNOWN port 65535`

    Usually it's the network problem. Please check `~/.ssh/config` if there is any unexpected config.

    Also please check proxy configs.