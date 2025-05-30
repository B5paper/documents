# sftp note

## cache

* sshfs 与 sftp 的关系

    Sshfs is a network file system for Linux that runs over the SFTP protocol. 

* ftps 与 sftp 的关系

    FTPS is basically the old ftp protocol run over SSL (Secure Sockets Layer) or TLS (Transport Layer Security).

* sftp implements

    * `pysftp` is a Python implementation

        <https://pypi.python.org/pypi/pysftp>

    * `Paramiko` is another Python implementation

        <http://docs.paramiko.org/en/2.2/api/sftp.html>

    * `pkg/sftp` is a Go language implementation

        <https://github.com/pkg/sftp>

    * `libssh` is a C implementation of the protocol

        <https://www.libssh.org/>

    * `libssh2` is another C implementation of the protocol

        <https://www.libssh2.org/>

    * `Rebex SFTP` is a .NET (C#) implementation

        <https://www.rebex.net/sftp.net/default.aspx>

    * `phpseclib` is another PHP implementation

        <http://phpseclib.sourceforge.net/>

    * `SmartFTP` is an ActiveX component

        <https://www.smartftp.com/en-us/ftplib>

    * `JCraft JSch` is a Java implementation

        <http://www.jcraft.com/jsch/>

    * `SSHJ` is another Java implementation

        <https://github.com/hierynomus/sshj>

    * `List of SFTP Client Libraries =nofollow`

        <http://www.sftp.net/client-libraries>

    * `Comparison of Commons VFS, SSHJ and JSch Libraries for SFTP Support`

        <https://medium.com/@ldclakmal/comparison-of-commons-vfs-sshj-and-jsch-libraries-for-sftp-support-cd5a0db2fbce>

* sftp

    SSH File Transfer Protocol (SFTP), or Secure File Transfer Protocol

    SFTP runs over SSH in the standard SSH port.

    login: `sftp username@hostname`

    local commands:

    ```
    lls lpwd lcd lmkdir lumask
    ```

    remote commands:

    ```
    ls pwd cd mkdir rename rm rmdir chmod chown dir ln symlink
    ```

    transfer commands:

    ```
    get put
    ```

    control commands:

    ```
    exit quit bye help ? ! version
    ```

    * `get`的 usage 与 parameters：

        * `get <file_name>`: 将远程的文件`<file_name>`复制到当前文件夹。当前文件夹可以用`lpwd`查看，并可以使用`lcd`更改。

        * `get -R <dir_name>`: 递归地传输文件夹。

            * `<dir_name>`也可以是一个文件，`<file_name>`

            * `<dir_name>`无论是 symbol link，还是原文件（夹），sftp 都可以正常处理。

            * 如果`<dir_name>`文件夹中又有 symbol link，那么 sftp 不会处理，并且会有如下输出：

                ```
                sftp> get -R Videos
                Fetching /home/hlc/Videos/ to Videos
                Retrieving /home/hlc/Videos
                download "/home/hlc/Videos/use-proxy-link.sh": not a regular file
                ```

        * `get -p <file_name>`: 将远程文件的 privilege 和 modified date 都复制过来。

        * `get -a <file_name>`：如果前面复制了一半，网络中断了，那么可以用这个参数 resume transfer。

        * `get -f <file_name>`: 在文件传输完成后，会执行`fsync`将文件写回磁盘。

    * `!`

        `!`命令有两种使用方式，一种是直接执行`!`，此时会打开一个 bash 界面，在 bash 界面中执行的命令会在 local 机器上执行，另一种是`!<command>`，此时会在 local 机器上新打开一个 bash 脚本执行`<command>`命令。

    说明：

    * 使用`ls -lh <file_name>`无法辨别远程文件是 symbol link 还是原文件。

    * `!<command>`可以在 local 执行命令，但是`!cd <path>`并不能改变`lls`显示的目录，只能用`lcd`来修改。

        `!<command>`有点像在 local 开了一个 bash 脚本，并执行一些命令。

    * 没有`ldir`命令，只有`dir`命令。

* sftp protocol, requests and responses

    Operations or packet types supported by the protocol include:

    INIT: sends client version numbers and extensions to the server

    VERSION: returns server version number and extensions to the client

    OPEN: opens or creates a file, returning a file handle

    CLOSE: closes a file handle

    READ: reads data from a file

    WRITE: writes data to a file

    OPENDIR: opens a directory for reading, returning a directory handle

    READDIR: reads file names and attributes from a directory handle

    MKDIR: creates a directory

    RMDIR: removes a directory

    REMOVE: removes a file

    RENAME: renames a file

    STAT: returns file attributes given a path, following symlinks

    LSTAT: returns file attributes given a path, without following symlinks

    FSTAT: returns file attributes given a file handle

    SETSTAT: modifies file attributes given a path

    FSETSTAT: modifies file attributes given a file handle

    READLINK: reads the value of a symbolic link

    SYMLINK: creates a symbolic link

    REALPATH: canonicalizes server-size relative path to an absolute path

    The following response packets are returned by the server:

    STATUS: indicates success or failure of an operation

    HANDLE: returns a file handle upon success

    DATA: returns data upon success

    ATTRS: returns file attributes upon success

    There is also an extension mechanism for arbitrary vendor-specific extensions. The extensions that are supported are negotiated using the INIT and VERSION packets.

    EXTENDED: sends a vendor-specific request from client to server

    EXTENDED_REPLY: sends a vendor-specific response from server to client.

## note