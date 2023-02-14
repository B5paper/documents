# GPG Note

gpg 是 GNU 提供的一个工具，全称 GnuPG。主要用于实现 OpenPGP 格式的签名和验证。

windows 下安装：<https://gpg4win.org/download.html>

linux 下安装：

`sudo apt-get install gpg`

local host operation:

导入一个密钥（目前已经验证可以导入公钥，不知道能不能导入私钥）：`gpg --import /path/to/pubkey_or_privkey`

可以在 keyserver 上搜索一个邮箱对应的 key：

`gpg --keyserver pgp.mit.edu --search-keys them@something.com`

或者通过 keyid 直接拿到它：

`gpg --recv-keys FOODDEAD`

查看某个邮箱对应的 key：

`gpg --fingerprint some@email.com`

显示本地存储的 public key：

`gpg --list-keys`或`gpg --list-public-keys`

显示本地存储的 private keys：

`gpg --list-secret-keys`

创建一个新的私钥：

`gpg --gen-key`

这时需要填姓名和邮箱。

导出私钥：

`gpg --export-secret-keys --armor <key_id> > ./my-priv-gpg-key.asc`

其中参数`--armor`指的是 armored (ASCII) 模式，它把密钥的二进制模式转换成人类可读模式，从而可以打印出来到纸上。不指定这个参数，会导出二进制文件格式，这样文件的大小会小一些。

导出公钥：

`gpg --export <key_id>`

删除一个秘钥：

`gpg --delete-secret-keys <key_id>`

删除一个公钥：

`gpg --delete-keys <key_id>`

import a public key from a server:

`gpg --keyserver pgp.mit.edu  --recv C104CDF0EDA54C82`

publish a public key to a server:

`gpg --keyserver hkp://pgp.mit.edu --send-keys XXXXXXXX`

encrypt a message with a public key:

`gpg --encrypt message.txt`

此时会让用户输入关键字，从 public key 库中选择一个来加密。这里的关键字可以是 key ID，可以是公钥拥有者的姓名，也可以是公钥拥有者的邮箱。显示出可用的公钥后，要再按一下回车才行。

命令执行结果是在当前文件夹下生成一个`message.txt.gpg`文件。

也可以直接使用`-r`参数或`--recipient`指定关键字：

`gpg --recipient nanodano@devdungeon.com --encrypt message.txt`

可以指定`--armor`参数，生成可打印的密文。这时会生成一个`message.txt.asc`文件。

可以指定`--output`参数，指定输出文件。

解密：

`gpg -d message.txt.gpg`

将解密后的文件输出到指定文件中：

`gpg --decrypt message.txt.gpg > decrypted.txt`

也可以同时加密和签名：

`gpg --encrypt --sign --recipient nanodano@devdungeon.com message.txt`

Signatures:

只 sign，不 encrypt：

`gpg --sign message.txt`：生成一个`message.txt.gpg`文件。可以加上`--armor`参数，生成`message.txt.asc`参数。

`gpg --clearsign message.txt`，等价于`--sign --armor`，生成`asc`文件。

只 verify，不 decrypt：

`gpg --verify message.txt.gpg`

说明：

1. 按道理`--sign`和`--clearsign`都只 sign，不 decrypt，但是我看了看生成的文件，里面的内容已经被替换成无法辨别原文的字符串了。因此`--verify`其实并不常用，使用`--decrypt`会直接解密出内容。

1. 可以使用`gpg --output message.txt message.txt.asc`得到和`--decrypt`同样的效果。

同时`sign`和`encrypt`：

`gpg --sign --symmetric message.txt`：使用 aes256 进行加密（需要输入一个密码），然后使用 private key 进行 sign。

`gpg --sign --encrypt --recipient nanodano@devdungeon.com message.txt`：非对称签名和加密。

对明文进行签名后，将签名文件和明文分离保存：

`gpg --detach-sign message.txt`，执行命令后会创建一个`message.txt.sig`文件。此时可以使用`gpg --verify message.txt.sig`验证签名。也可以指定签名对原文件进行验证：`gpg --verify some_signature.sig ./message.txt`。

Reference:

1. <https://www.devdungeon.com/content/gpg-tutorial>