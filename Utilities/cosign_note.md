# Cosign Note

Homepage: <https://github.com/sigstore/cosign>

安装：下载 github release 页面的压缩包，解压后的文件就是二进制文件。可以把它软链接到`/usr/local/bin`目录下。

* generate key pair: `cosign generate-key-pair`

    在当前目录生成一对`.key`文件和`.pub`文件，分别是私钥和公钥。

* sign an image: `cosign sign --key cosign.key quay.io/swhlc/test@sha256:2fd65ff14822b0b8095bbd6dc457661fa7a54d3a11a0428baf202269988bd62d`

    这个操作会在 registry 中 push 一个`sha256-2fd65ff14822b0b8095bbd6dc457661fa7a54d3a11a0428baf202269988bd62d.sig`文件。

* verify an image

    `cosign verify --key cosign.pub quay.io/swhlc/test:latest`

    输出：

    ```
    Verification for quay.io/swhlc/test:latest --
    The following checks were performed on each of these signatures:
    - The cosign claims were validated
    - The signatures were verified against the specified public key

    [{"critical":{"identity":{"docker-reference":"quay.io/swhlc/test"},"image":{"docker-manifest-digest":"sha256:2fd65ff14822b0b8095bbd6dc457661fa7a54d3a11a0428baf202269988bd62d"},"type":"cosign container image signature"},"optional":null}]
    ```