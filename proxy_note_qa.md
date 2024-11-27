# proxy note qa

[unit]
[u_0]
请写出 v2ray 配置文件的模板。
[u_1]
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

[unit]
[u_0]
请写出 inbounds 的模板。
[u_1]
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