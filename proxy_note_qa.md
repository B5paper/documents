# proxy note qa

[unit]
[idx]
0
[id]
13211585243029108917
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
[idx]
1
[id]
16445054648617368541
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
