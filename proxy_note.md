# proxy note

这里存放与 proxy 相关的技巧，软件，笔记。

## cache

* v2ray + http 代理是否可以代理 udp？如果不可以那么如何代理 udp？

    **V2Ray 的 HTTP 代理协议本身不支持代理 UDP 流量**。根据官方文档的明确说明，`http` 协议**只能代理 TCP 协议，所有基于 UDP 的协议均无法通过**。

    如果你需要代理 UDP 流量（例如用于 DNS 查询、在线游戏、QUIC 协议等），V2Ray/Xray 提供了多种替代方案，核心思路是**放弃使用 HTTP 代理协议，改用其他原生支持 UDP 的协议或入站方式**。

    ### 方案一：使用原生支持 UDP 的代理协议（推荐）

    这是最直接的方式。你需要将出站协议从 `http` 更换为支持 UDP 的协议。目前，主流的支持 UDP 转发的协议包括 **VMess、VLESS、Trojan、Shadowsocks (SS)、SSR 以及 TUIC** 等。

    以 **Socks** 协议为例，配置相对简单。你可以在入站（Inbound）配置中显式开启 UDP 支持：
    ```json
    {
      "inbounds": [{
        "port": 1080,
        "protocol": "socks",
        "settings": {
          "udp": true  // 关键设置：开启 UDP 转发
        }
      }]
    }
    ```
    开启后，V2Ray 需要知道本机的 IP 地址，以便客户端发起 UDP 连接时能够正确路由回服务端。大多数情况下，这个设置可以正常工作。

    ### 方案二：TProxy 透明代理（适用于全局或网关级 UDP 代理）

    如果你希望无差别地代理设备上的所有 UDP 流量（而不仅仅是特定应用的流量），可以使用 **TProxy（透明代理）** 模式。V2Ray/Xray 支持纯 TPROXY 模式，即 **TCP 和 UDP 流量均通过 TPROXY 机制进行代理**。

    在 Xray 的配置中，你需要将入站协议设置为 `tunnel`（旧称 `dokodemo-door`），并开启特定的网络类型和 TProxy 套接字选项：
    ```json
    {
      "inbounds": [{
        "port": 12345,
        "protocol": "tunnel",
        "settings": {
          "allowedNetwork": "tcp,udp",  // 同时接收 TCP 和 UDP
          "followRedirect": true
        },
        "streamSettings": {
          "sockopt": {
            "tproxy": "tproxy"  // 启用 TProxy
          }
        }
      }]
    }
    ```
    同时，你需要配合系统的 `iptables` 规则，将发往特定目标的 UDP 数据包重定向到 V2Ray 监听的端口上。

    ### 方案三：使用 Tunnel (dokodemo-door) 进行 UDP 端口映射

    如果你的需求是**将某个特定的 UDP 服务**（例如一个游戏服务器）通过代理进行转发，而不是代理所有应用的 UDP 流量，那么使用 `Tunnel`（隧道）入站协议进行端口映射是更轻量的选择。

    `Tunnel` 可以监听一个本地端口，并将所有收到的数据（包括 UDP）转发到指定的远程地址和端口。你只需将 `allowedNetwork` 设置为 `"udp"` 或 `"tcp,udp"` 即可：
    ```json
    {
      "protocol": "tunnel",
      "settings": {
        "allowedNetwork": "udp",  // 或 "tcp,udp"
        "rewriteAddress": "目标服务器地址",
        "rewritePort": 目标端口
      }
    }
    ```
    这种方式适用于点对点的 UDP 流量转发，配置比 TProxy 全局透明代理要简单得多。

    ### 总结

    **HTTP 协议是 V2Ray 中专用于 TCP 的简单代理，不支持 UDP 是其协议本身的固有限制**。要代理 UDP，必须绕过 HTTP 协议，转而使用 **VMess、VLESS、Trojan、Shadowsocks** 等原生支持 UDP 的协议，并根据你的场景选择 **Socks 入站（应用级）、TProxy（网关级）或 Tunnel 端口映射（点对点）** 来实现 UDP 流量的转发。

    # 结论先说

    **标准 HTTP 代理本身不支持 UDP**，所以 `v2ray 出站走http代理` 这个组合，**无法原生代理 UDP 流量（DNS UDP、游戏 UDP 等都会失败）**

    > 
    > HTTP 代理只有 CONNECT，只转发 TCP；RFC9298 MASQUE 的 CONNECT-UDP 属于新实验标准，绝大多数 v2ray、普通 http 代理都不实现，不能拿来日常用NAIST

    > 
    > ✅ 对比：**SOCKS5 原生支持 UDP（UDP ASSOCIATE）**，这是关键区别

    ---

    ## 一、为什么 v2ray + http 代理不能 UDP

    v2ray 的 `http` 出站协议：底层就是普通 http/https 代理，v2ray 会把自己的流量发给上游 http 代理，但是：

    1. 上游 http 代理**没有 UDP 转发能力**
    2. v2ray 配置里就算开了 `udp: true`，UDP 包也没法交给上游 http 代理送出，直接断流

    > 
    > 如果你是这种链路：`本机程序 → v2ray(in:socks5/http) → v2ray(out:http代理) → 外网`
    > 哪怕入站是 socks5 并且勾选 UDP，**UDP 到 http 出站这里就卡住了**

    ---

    ## 二、想要代理 UDP，几种可行方案（按推荐排序）

    ### 方案 1：上游换成 SOCKS5 代理（最简单，优先选）

    链路：`程序 → v2ray → [出站socks5] → 上游socks5代理`
    条件：**上游 SOCKS5 代理本身开启了 UDP ASSOCIATE**（很多商家的 socks5 只做 TCP，要确认）
    v2ray 出站示例片段：

    ```
    {
      "outbounds": [
        {
          "protocol": "socks",
          "settings": {
            "servers": [
              {
                "address": "xxx",
                "port": 1080,
                "users": [{"user":"xx","pass":"xx"}]
              }
            ]
          },
          "streamSettings": {},
          "udp": true
        }
      ]
    }
    ```

    - v2ray 这边必须 `"udp":true`
    - 本地入站用 socks5（不要只用 http 入站）

    ### 方案 2：上游只能给你 HTTP 代理 → UDP over TCP（封装 UDP 进 TCP 隧道，有延迟损耗）

    HTTP 代理只能跑 TCP，只能用工具把 UDP 数据包封装到 TCP 里再走这条隧道：
    常用工具：

    - `socat`：udp<->tcp 桥接
    - `udp2raw` / `gost`：udp over tcp
    - gost 示例：本地监听 UDP，把 UDP 包封装 TCP，发给上游 http 代理

    缺点：UDP 本来优势是低延迟，包套 TCP 后变成可靠传输，**游戏、实时语音体验变差**，适合 DNS 这种低实时性 UDP。

    ### 方案 3：直接用原生支持 UDP 的 v2ray 节点（vmess/vless/trojan 等）

    链路：`程序 → v2ray → 直接vmess/vless服务端`
    v2ray 本身 vmess/vless 等协议原生支持 UDP 转发，**服务端和客户端同时开启 udp 即可**，不需要套一层上游代理，UDP（DNS、游戏）最稳。

    > 
    > 这是 UDP 体验最好的方案，没有额外封装开销

    ### 方案 4：透明代理 / TUN（全局 IP 层代理，全部 TCP+UDP）

    v2ray、sing-box 支持 TUN 模式，接管本机所有 IP 包（TCP/UDP/ICMP），适合路由器、旁路由。
    ⚠️ 前提：出站不能是 http 代理；出站必须是支持 UDP 的 socks5/vless/vmess。

    ---

    ## 三、补充小提示

    1. DNS UDP 是最常见的 UDP 代理场景：如果只是想代理 DNS，可以单独用 TCP DNS（53/TCP）绕开 UDP 限制，但游戏 UDP 不行。
    2. 很多代理客户端面板上有个 “转发 UDP” 开关：**只有出站是 socks5/vless/vmess 才生效，出站 http 代理时这个开关是无效的**。
    3. sing-box（v2ray 替代项目）规则逻辑一致：http 出站依然不支持 UDP。

    如果你需要，我可以给你一份 gost 把 UDP 封装 TCP、走上游 HTTP 代理的完整配置。

    > 
    > ⚠️ 重要提醒：私自搭建、使用翻墙工具是违反国内法规的，以上仅为纯网络原理技术讨论。

* 就算使用 v2ray client 本身启动代理，也需要在 server 端设置`V2RAY_VMESS_AEAD_FORCED=false`环境变量。

* v2ray http 转 socks example

    ```json
    {
      "log": {
        "loglevel": "warning"
      },
      "inbounds": [{
        "port": 1080,
        "listen": "127.0.0.1",
        "protocol": "http"
      }],
      "outbounds": [{
        "protocol": "socks",
        "settings": {
        	"servers": [
    	    {
    	        "address": "127.0.0.1",
    	        "port": 4321
    	    }
    	]
        },
        "tag": "direct"
      }],
      "policy": {
        "levels": {
          "0": {"uplinkOnly": 0}
        }
      }
    }
    ```

* v2ray socks config

    outbound config

    ```json
    {
      "servers": [{
        "address": "127.0.0.1",
        "port": 1234,
        "users": [
          {
            "user": "test user",
            "pass": "test pass",
            "level": 0
          }
        ]
      }]
    }
    ```

    inbound config

    ```json
    {
      "auth": "noauth",
      "accounts": [
        {
          "user": "my-username",
          "pass": "my-password"
        }
      ],
      "udp": false,
      "ip": "127.0.0.1",
      "userLevel": 0
    }
    ```

    `auth: "noauth" | "password"`

    当开启 UDP 时，V2Ray 需要知道本机的 IP 地址。默认值为"127.0.0.1"。

    socks 不对数据加密，因此不适合公网传输。

    这里的 inbound confg 和 outbound config 分别对应 inbound / outbound 中的 setting 项。

* bandwagon KiwiVM address: <https://bwh88.net/clientarea.php>

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
