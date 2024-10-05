* v2ray http config

    v2ray 配置文件的模板：

    ```json
    {
        "log": {},
        "api": {},
        "dns": {},
        "stats": {},
        "routing": {},
        "policy": {},
        "reverse": {},
        "inbounds": [],
        "outbounds": [],
        "transport": {}
    }
    ```

    比较重要的是`inbounds`和`outbounds`这两个，剩下的都可以不写。

    inbounds 的模板：

    ```json
    {
        "port": 1080,
        "listen": "127.0.0.1",
        "protocol": "协议名称",
        "settings": {},
        "streamSettings": {},
        "tag": "标识",
        "sniffing": {
            "enabled": false,
            "destOverride": ["http", "tls"]
        },
        "allocate": {
            "strategy": "always",
            "refresh": 5,
            "concurrency": 3
        }
    }
    ```

    比较重要的是前四个，`port`, `listen`, `protocol`, `settings`。

    其中`settings`与`protocol`相关，比如 http 的 settings 如下：

    ```json
    {
        "timeout": 0,
        "accounts": [
            {
                "user": "my-username",
                "pass": "my-password"
            }
        ],
        "allowTransparent": false,
        "userLevel": 0
    }
    ```

    这里以 inbound 为 http 协议，outbound 为 freedom 为例：

    `vpoint_http_freedom.json`:

    ```json
    {
        "inbounds": [
            {
                "port": 1080,
                "listen": "0.0.0.0",
                "protocol": "http"
            }
        ],
        "outbounds": [
            {
                "protocol": "freedom"
            }
        ]
    }
    ```

    启动：`./v2ray run -c vpoint_http_freedom.json`

    使用：

    `curl -x http://<server_ip>:1080 www.baidu.com`

    或

    `http_proxy=http://<server_ip>:1080 https_proxy=http://<server_ip>:1080 curl www.baidu.com`

    说明：

    1. http 代理是明文传输，在经过 gfw 时可能会被过滤而丢包。因此 http 协议通常在本地或局域网内开启，为本地的 http 提供代理服务。

* v2ray vmess config

    vmess protocol inbounds settings 的模板：

    ```json
    {
        "clients": [
            {
                "id": "27848739-7e62-4138-9fd3-098a63964b6b",
                "level": 0,
                "alterId": 4,
                "email": "love@v2ray.com"
            }
        ],
        "default": {
            "level": 0,
            "alterId": 4
        },
        "detour": {
            "to": "tag_to_detour"
        },
        "disableInsecureEncryption": false
    }
    ```

    其中，`clients`中的`id`和`alterId`是必填项，剩下的都可以没有。

    v2ray 的 outbounds 模板：

    ```json
    {
        "sendThrough": "0.0.0.0",
        "protocol": "协议名称",
        "settings": {},
        "tag": "标识",
        "streamSettings": {},
        "proxySettings": {
            "tag": "another-outbound-tag"
        },
        "mux": {}
    }
    ```

    其中，`protocol`和`settings`是必填，其他的都可以不填。如果`protocol`是`freedom`，`settings`也可以不填。

    vmess outbounds settings 的模板：

    ```json
    {
        "vnext": [
            {
                "address": "127.0.0.1",
                "port": 37192,
                "users": [
                    {
                        "id": "27848739-7e62-4138-9fd3-098a63964b6b",
                        "alterId": 4,
                        "security": "auto",
                        "level": 0
                    }
                ]
            }
        ]
    }
    ```

    可以看到，对 vmess 协议而言，`settings`中只有`vnext`这一个字段。

    其中`address`，`port`，`users`中的`id`和`alterId`是必填，剩下的都是选填。

    比较常见的一个代理过程是 http -> vmess client -> vmess server -> freedom，其中 http -> vmess client 工作于本地，vmess server -> freedom 工作于远程服务器，这里分别给出这两个最简配置文件：

    * `vpoint_http_vmess.json`，放在本地

        ```json
        {
            "inbounds": [
                {
                    "port": 8822,
                    "listen": "0.0.0.0",
                    "protocol": "http"
                }
            ],
            "outbounds": [
                {
                    "protocol": "vmess",
                    "settings": {
                        "vnext": [
                            {
                                "address": "10.10.10.10",
                                "port": 1080,
                                "users": [
                                    {
                                        "id": "27848739-7e62-4138-9fd3-098a63964b6b",
                                        "alterId": 16
                                    }
                                ]
                            }
                        ]
                    }
                }
            ]
        }
        ```

    * `vpoint_vmess_freedom.json`，放在远程服务器

        ```json
        {
            "inbounds": [
                {
                    "port": 1080,
                    "listen": "0.0.0.0",
                    "protocol": "vmess",
                    "settings": {
                        "clients": [
                            {
                                "id": "27848739-7e62-4138-9fd3-098a63964b6b",
                                "alterId": 16
                            }
                        ]
                    }
                }
            ],
            "outbounds": [
                {
                    "protocol": "freedom"
                }
            ]
        }
        ```

    启动代理：

    1. 首先在 server 端启动

        `./v2ray run -c vpoint_vmess_freedom.json`

        说明：

        1. 2024.10.05：如果直接这样启动，会无法成功接收 client 的 vmess 连接，目前需要在启动时增加一个环境变量：

            `V2RAY_VMESS_AEAD_FORCED=false ./v2ray run -c vpoint_vmess_freedom.json`

            不清楚这个环境变量是干嘛用的。

            ref: <https://github.com/v2fly/v2ray-core/discussions/1514>

        2. 配置文件`vpoint_http_vmess.json`中的 ip `"10.10.10.10"`是远程服务器的 ip。

    2. 在本地启动 http 代理

        `./v2ray run -c vpoint_http_vmess.json`

    3. 测试代理是否正常工作

        `curl -x http://127.0.0.1:8822 www.baidu.com`