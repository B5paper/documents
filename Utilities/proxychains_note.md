# Proxychains Note

## cache

* Proxychains vs Proxychains4 (Proxychains-NG)

    ```
    原始proxychains (2002-2010)
            ↓ (停止维护)
        分支/重写
            ↓
    proxychains-ng (2012-现在)
            ↓
      发行版中的包名：
      - Ubuntu/Debian: proxychains4
      - Arch: proxychains-ng
      - 源代码: 仍叫proxychains-ng
    ```

    对比
    特性	Proxychains (原始/旧版)	Proxychains4 (proxychains-ng)
    维护状态	已停止维护（最后版本2006）	活跃维护（2024年仍在更新）
    项目名称	proxychains	proxychains-ng (Next Generation)
    包名差异	proxychains	proxychains4（二进制名）
    配置文件	/etc/proxychains.conf	/etc/proxychains4.conf
    许可证	GPLv2	GPLv2
    主要开发者	netcreature	rofl0r

    Proxychains-NG的新功能/改进：

    # 1. 更好的DNS处理
    proxychains-ng支持：
    - proxy_dns (通过代理解析DNS)
    - 防止DNS泄露的改进实现
    - 更好的IPv6支持

    # 2. 增强的代理链类型
    proxychains4新增：
    - random_chain (随机代理链)
    - dynamic_chain (动态跳过失效代理)
    - strict_chain (严格链，原始版本也有)

    # 3. 性能优化
    proxychains-ng有：
    - 更快的LD_PRELOAD注入
    - 减少内存占用
    - 更好的多线程支持

    # 4. 兼容性改进
    - 支持更多应用程序
    - 更好的系统调用拦截
    - 修复原始版本的许多bug

    大多数Linux发行版默认安装NG版本：

    ```bash
    # Ubuntu 20.04+
    sudo apt install proxychains4  # 明确安装NG版本

    # 或者
    sudo apt install proxychains   # 可能也是NG，但重命名了

    # Arch Linux
    sudo pacman -S proxychains-ng  # 保持原名
    ```

* proxychains

    Proxychains 是一个强制应用程序通过代理服务器进行网络连接的工具，主要用于：

    * 代理链：支持多级代理跳转，增强匿名性

    * 协议支持：支持 HTTP、SOCKS4、SOCKS5 代理

    * 透明代理：无需修改应用程序代码即可使其通过代理工作

    * 绕过限制：帮助受限环境中的工具（如 nmap、wget、ssh）通过代理访问外部网络

    安装：

    `sudo apt install proxychains4`

    配置文件通常位于：

    * `/etc/proxychains.conf`

    * `/etc/proxychains4.conf`

    * `~/.proxychains/proxychains.conf`

    config example:

    ```conf
    # 代理类型 (http/socks4/socks5)
    [ProxyList]
    socks5 127.0.0.1 1080
    http 192.168.1.100 8080
    socks4 10.0.0.1 9050

    # 代理链模式
    # dynamic_chain: 按顺序使用代理，失败的代理会跳过
    # strict_chain: 严格按顺序使用所有代理
    # random_chain: 随机顺序使用代理
    ```

    usage:

    ```bash
    # 基本语法
    proxychains [命令] [参数]

    # 示例
    proxychains curl https://example.com
    proxychains nmap -sT target.com
    proxychains wget http://example.com/file.zip
    proxychains git clone https://github.com/user/repo.git
    # 通过代理进行端口扫描
    proxychains nmap -sS -Pn target.com
    ```

    options:

    ```bash
    proxychains -f /path/to/custom.conf firefox    # 使用自定义配置文件
    proxychains -q nmap target.com                 # 安静模式（不显示代理信息）
    ```

    注意事项

        DNS 解析：

            默认可能泄露 DNS 请求

            可在配置中启用 proxy_dns 选项

        程序兼容性：

            某些静态链接的程序可能无法正常工作

            GUI 程序可能需要额外配置

        性能影响：

            多级代理会降低网络速度

    验证代理生效:

    ```bash
    # 检查公网 IP 是否改变
    proxychains curl ifconfig.me
    proxychains wget -qO- https://api.ipify.org
    ```

## topics

## Problem shooting

* change the dns server of proxychains

    1. method 1: using environment variables

        ```bash
        export DNS_SERVER=8.8.8.8
        proxychains firefox
        ```

    1. method 2: chaning the bash file that proxychains uses

        file: `/usr/lib/proxychains3/proxyresolv`

        ```bash
        #!/bin/sh
        # This script is called by proxychains to resolve DNS names

        # DNS server used to resolve names
        DNS_SERVER=${PROXYRESOLV_DNS:-4.2.2.2}


        if [ $# = 0 ] ; then
                echo "  usage:"
                echo "          proxyresolv <hostname> "
                exit
        fi


        export LD_PRELOAD=libproxychains.so.3
        dig $1 @$DNS_SERVER +tcp | awk '/A.+[0-9]+\.[0-9]+\.[0-9]/{print $5;}'
        ```

        Reference: <https://blog.carnal0wnage.com/2013/09/changing-proxychains-hardcoded-dns.html>