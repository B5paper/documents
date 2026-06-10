# Frp Note

## cache

* frpc 是否有配置 public key 的方法？

    好问题。简单结论是：**frp / frpc** 本身 —— 在当前官方文档主线中 —— **没有“用公钥 (public key) 的认证方式 (public-key authentication)”**，而主要依赖 **token 或 OIDC**。 ([GoFRP][1])

    frpc 目前不支持 public key

    官方支持的认证方式

    * frp / frpc 支持两种认证方法：**token**（默认）和 **OIDC (OpenID Connect)**。 

    * token 的方式，只需在服务端 (frps) 与客户端 (frpc) 的配置文件里设置相同的 token。 

    * 从 v0.64.0 起，还支持通过 `tokenSource` 从文件读取 token，以避免把 token 明文写到配置文件中。 

* 那么 frp 或 linux 是否有产生随机长序列 token 的方式或者小工具？

    一行命令生成强随机 token（推荐）

    1. 使用 /dev/urandom + base64

        ```bash
        head -c 32 /dev/urandom | base64
        ```

        生成约 43 字节的 token，非常安全，可直接用于 frp 的 `auth.token`。

        若想更短或更长，调整 `head -c N`：

        * 16 字节 → `head -c 16 /dev/urandom | base64`
        * 32 字节 → 推荐
        * 64 字节 → 更强

    2. 使用 openssl（最常用）

        ```bash
        openssl rand -base64 32
        ```

        生成高质量随机 token。

        输出类似：

        ```
        QJLGhTzzwY5S7FTj0n8Xk2h0S8Cqk4cZqHdo7w==
        ```

        也可以生成十六进制：

        ```bash
        openssl rand -hex 32
        ```

    3. 使用 uuid（不够强，但可读性好）

        ```bash
        uuidgen
        ```

        不适合作为高强度 token，但简单方便。

    4. python 一行（系统自带）

        ```bash
        python3 -c 'import secrets; print(secrets.token_urlsafe(32))'
        ```

        安全程度极高，可用于生产环境。

    **直接为 frp 生成并写入 token**

    如果你想直接生成 32 字节 token 到 frpc.ini：

    ```bash
    TOKEN=$(openssl rand -base64 32)
    echo "auth.token = $TOKEN" >> frpc.ini
    ```
## topics
