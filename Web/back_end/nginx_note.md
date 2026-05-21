## cache

* 详细讲讲 nginx 的用法

    # Nginx 全面用法详解
    Nginx 是一款**高性能的 HTTP 和反向代理服务器**，也是一个 IMAP/POP3/SMTP 代理服务器。它以**高并发、低内存占用、稳定性强**著称，广泛用于静态资源服务、反向代理、负载均衡、API 网关等场景。

    ## 一、Nginx 安装
    ### 1. Linux 系统（推荐）
    #### Ubuntu/Debian
    ```bash
    # 更新软件源
    sudo apt update
    # 安装稳定版
    sudo apt install nginx -y
    # 验证安装
    nginx -v
    ```

    #### CentOS/RHEL
    ```bash
    # 安装 EPEL 源（CentOS 7 及以下）
    sudo yum install epel-release -y
    # 安装
    sudo yum install nginx -y
    # 验证
    nginx -v
    ```

    #### 源码编译安装（自定义功能）
    ```bash
    # 安装依赖
    sudo apt install build-essential libpcre3 libpcre3-dev zlib1g zlib1g-dev libssl-dev -y
    # 下载源码（替换为最新版本）
    wget http://nginx.org/download/nginx-1.26.0.tar.gz
    tar -zxvf nginx-1.26.0.tar.gz
    cd nginx-1.26.0
    # 配置编译选项（常用模块）
    ./configure --prefix=/usr/local/nginx \
                --with-http_ssl_module \
                --with-http_v2_module \
                --with-http_gzip_static_module \
                --with-http_stub_status_module
    # 编译安装
    make && sudo make install
    # 添加到系统路径
    echo 'export PATH=$PATH:/usr/local/nginx/sbin' >> ~/.bashrc
    source ~/.bashrc
    ```

    ### 2. Windows 系统
    1.  从 [Nginx 官网](http://nginx.org/en/download.html) 下载 Windows 稳定版
    2.  解压到任意目录（如 `C:\nginx`）
    3.  双击 `nginx.exe` 启动（命令行启动更推荐）

    ## 二、Nginx 基本操作
    ### 1. 系统服务管理（apt/yum 安装）
    ```bash
    # 启动
    sudo systemctl start nginx
    # 停止
    sudo systemctl stop nginx
    # 重启
    sudo systemctl restart nginx
    # 重新加载配置（不中断服务，推荐）
    sudo systemctl reload nginx
    # 查看状态
    sudo systemctl status nginx
    # 设置开机自启
    sudo systemctl enable nginx
    # 禁止开机自启
    sudo systemctl disable nginx
    ```

    ### 2. 直接使用 nginx 命令（源码安装/Windows）
    ```bash
    # 启动
    nginx
    # 快速停止
    nginx -s stop
    # 优雅停止（等待请求处理完成）
    nginx -s quit
    # 重新加载配置
    nginx -s reload
    # 重新打开日志文件
    nginx -s reopen
    # 测试配置文件语法
    nginx -t
    # 查看版本和编译参数
    nginx -v
    nginx -V
    ```

    ## 三、Nginx 配置文件详解
    ### 1. 配置文件位置
    - apt/yum 安装：`/etc/nginx/nginx.conf`（主配置），`/etc/nginx/sites-available/`（站点配置）
    - 源码安装：`/usr/local/nginx/conf/nginx.conf`
    - Windows：`C:\nginx\conf\nginx.conf`

    ### 2. 配置文件结构
    Nginx 配置由**指令**和**指令块**组成，指令以分号结尾，指令块用大括号包裹。
    ```nginx
    # 全局块：配置影响 Nginx 全局的指令
    user nginx;                    # 运行 Nginx 的用户
    worker_processes auto;         # 工作进程数，建议设为 CPU 核心数
    error_log /var/log/nginx/error.log warn;  # 错误日志路径和级别
    pid /var/run/nginx.pid;        # PID 文件路径

    # events 块：配置影响 Nginx 服务器与用户网络连接的属性
    events {
        use epoll;                 # 使用 epoll 事件模型（Linux 推荐）
        worker_connections 1024;   # 每个工作进程的最大连接数
        multi_accept on;           # 允许一个进程同时接受多个新连接
    }

    # http 块：配置 HTTP 服务器的主要参数
    http {
        include       /etc/nginx/mime.types;  # 包含 MIME 类型定义
        default_type  application/octet-stream;  # 默认 MIME 类型

        # 日志格式
        log_format main '$remote_addr - $remote_user [$time_local] "$request" '
                        '$status $body_bytes_sent "$http_referer" '
                        '"$http_user_agent" "$http_x_forwarded_for"';

        access_log /var/log/nginx/access.log main;  # 访问日志路径

        sendfile on;                 # 开启高效文件传输模式
        tcp_nopush on;               # 防止网络阻塞
        tcp_nodelay on;              # 防止网络阻塞

        keepalive_timeout 65;        # 长连接超时时间
        client_max_body_size 20m;    # 客户端请求体最大大小

        gzip on;                     # 开启 Gzip 压缩
        gzip_types text/plain text/css application/json application/javascript text/xml application/xml application/xml+rss text/javascript;

        # 包含所有站点配置文件
        include /etc/nginx/conf.d/*.conf;
        include /etc/nginx/sites-enabled/*;
    }
    ```

    ### 3. server 块（虚拟主机）
    每个 `server` 块代表一个虚拟主机，用于配置不同域名或端口的网站。
    ```nginx
    server {
        listen 80;                   # 监听端口
        server_name example.com www.example.com;  # 域名
        root /var/www/example.com;   # 网站根目录
        index index.html index.htm;  # 默认首页文件

        location / {
            try_files $uri $uri/ =404;  # 尝试查找文件，找不到返回 404
        }

        error_page 404 /404.html;    # 404 错误页面
        error_page 500 502 503 504 /50x.html;  # 5xx 错误页面
    }
    ```

    ### 4. location 块（URL 匹配）
    `location` 块用于匹配特定的 URL 路径，并定义相应的处理规则。
    ```nginx
    # 语法：location [修饰符] 匹配模式 { ... }

    # 1. 精确匹配（=）：只有完全匹配时才生效
    location = /favicon.ico {
        expires 30d;
        access_log off;
    }

    # 2. 前缀匹配（无修饰符）：匹配以指定字符串开头的 URL
    location /static/ {
        root /var/www/example.com;
        expires 7d;
    }

    # 3. 正则匹配（~ 区分大小写，~* 不区分大小写）
    location ~* \.(jpg|jpeg|png|gif|ico|css|js)$ {
        expires 30d;
        access_log off;
    }

    # 4. 优先前缀匹配（^~）：匹配成功后不再进行正则匹配
    location ^~ /api/ {
        proxy_pass http://127.0.0.1:3000;
    }
    ```

    **匹配优先级**：`=` > `^~` > 正则匹配（按顺序）> 普通前缀匹配

    ## 四、Nginx 核心功能实战
    ### 1. 静态资源服务
    Nginx 最基本也是最常用的功能，用于托管 HTML、CSS、JS、图片等静态文件。
    ```nginx
    server {
        listen 80;
        server_name static.example.com;
        root /var/www/static;  # 静态文件根目录

        location / {
            expires 30d;       # 浏览器缓存 30 天
            add_header Cache-Control "public, immutable";
        }

        # 禁止访问 .htaccess 等敏感文件
        location ~ /\.ht {
            deny all;
        }
    }
    ```

    ### 2. 反向代理
    将客户端请求转发到后端服务器，隐藏后端服务地址，提高安全性和性能。
    ```nginx
    server {
        listen 80;
        server_name api.example.com;

        location / {
            # 转发到后端 Node.js 服务
            proxy_pass http://127.0.0.1:3000;

            # 设置请求头，传递客户端真实信息
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;

            # 超时设置
            proxy_connect_timeout 30s;
            proxy_send_timeout 30s;
            proxy_read_timeout 60s;
        }
    }
    ```

    ### 3. 负载均衡
    将请求分发到多个后端服务器，提高系统可用性和处理能力。
    ```nginx
    # 定义上游服务器组
    upstream backend {
        server 192.168.1.10:3000 weight=3;  # weight 权重，值越大分配的请求越多
        server 192.168.1.11:3000;
        server 192.168.1.12:3000 backup;   # backup 备用服务器，其他服务器不可用时启用
        server 192.168.1.13:3000 down;     # down 标记服务器不可用
    }

    server {
        listen 80;
        server_name app.example.com;

        location / {
            proxy_pass http://backend;  # 转发到上游服务器组
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
        }
    }
    ```

    **负载均衡策略**：
    - **轮询（默认）**：请求按顺序依次分配到后端服务器
    - **加权轮询**：根据 `weight` 参数分配请求
    - **IP 哈希（ip_hash）**：根据客户端 IP 哈希结果分配，保证同一客户端始终访问同一服务器
    - **最少连接（least_conn）**：将请求分配到当前连接数最少的服务器
    - **URL 哈希（url_hash）**：根据请求 URL 哈希结果分配（需第三方模块）

    ### 4. HTTPS 配置
    为网站启用 HTTPS 加密，提高数据传输安全性。
    ```nginx
    server {
        listen 443 ssl http2;
        server_name example.com www.example.com;

        # SSL 证书路径
        ssl_certificate /etc/nginx/ssl/example.com.crt;
        ssl_certificate_key /etc/nginx/ssl/example.com.key;

        # SSL 配置优化
        ssl_protocols TLSv1.2 TLSv1.3;  # 只支持安全的 TLS 版本
        ssl_ciphers ECDHE-RSA-AES128-GCM-SHA256:ECDHE:ECDH:AES:HIGH:!NULL:!aNULL:!MD5:!ADH:!RC4;
        ssl_prefer_server_ciphers on;
        ssl_session_cache shared:SSL:10m;
        ssl_session_timeout 10m;

        root /var/www/example.com;
        index index.html;

        location / {
            try_files $uri $uri/ =404;
        }
    }

    # HTTP 重定向到 HTTPS
    server {
        listen 80;
        server_name example.com www.example.com;
        return 301 https://$server_name$request_uri;
    }
    ```

    ### 5. URL 重写（rewrite）
    修改客户端请求的 URL，常用于 URL 美化、重定向、伪静态等场景。
    ```nginx
    server {
        listen 80;
        server_name example.com;
        root /var/www/example.com;

        # 1. 永久重定向（301）：将 www.example.com 重定向到 example.com
        if ($host = www.example.com) {
            return 301 https://example.com$request_uri;
        }

        # 2. 伪静态：将 /article/123 重写为 /article.php?id=123
        rewrite ^/article/(\d+)$ /article.php?id=$1 last;

        # 3. 隐藏文件扩展名：将 /about 重写为 /about.html
        rewrite ^/about$ /about.html last;

        location / {
            try_files $uri $uri/ =404;
        }
    }
    ```

    **rewrite 指令参数**：
    - `last`：停止当前 rewrite 处理，用重写后的 URL 重新匹配 location
    - `break`：停止当前 rewrite 处理，直接使用重写后的 URL
    - `redirect`：临时重定向（302）
    - `permanent`：永久重定向（301）

    ## 五、Nginx 高级功能
    ### 1. 缓存配置
    缓存后端服务器的响应内容，减少后端压力，提高响应速度。
    ```nginx
    # 定义缓存路径和参数
    proxy_cache_path /var/cache/nginx
                     levels=1:2
                     keys_zone=my_cache:10m
                     max_size=1g
                     inactive=60m
                     use_temp_path=off;

    server {
        listen 80;
        server_name cache.example.com;

        location / {
            proxy_pass http://backend;
            proxy_cache my_cache;  # 使用定义的缓存
            proxy_cache_valid 200 302 10m;  # 200 和 302 响应缓存 10 分钟
            proxy_cache_valid 404 1m;       # 404 响应缓存 1 分钟
            proxy_cache_key $host$uri$is_args$args;  # 缓存键
            add_header X-Cache-Status $upstream_cache_status;  # 添加缓存状态头
        }
    }
    ```

    ### 2. 访问控制
    限制特定 IP 或用户访问网站。
    ```nginx
    location /admin/ {
        # 允许指定 IP 访问
        allow 192.168.1.0/24;
        allow 10.0.0.1;
        # 拒绝所有其他 IP
        deny all;

        # 基础认证（用户名密码）
        auth_basic "Admin Area";
        auth_basic_user_file /etc/nginx/.htpasswd;
    }
    ```

    **创建基础认证文件**：
    ```bash
    # 安装 htpasswd 工具
    sudo apt install apache2-utils -y
    # 创建用户和密码文件
    sudo htpasswd -c /etc/nginx/.htpasswd admin
    # 添加更多用户
    sudo htpasswd /etc/nginx/.htpasswd user1
    ```

    ### 3. 日志管理
    Nginx 提供了强大的日志功能，可自定义日志格式和切割日志。
    ```nginx
    # 自定义日志格式
    log_format detailed '$remote_addr [$time_local] "$request" $status $body_bytes_sent '
                        '"$http_referer" "$http_user_agent" "$http_x_forwarded_for" '
                        'rt=$request_time uct="$upstream_connect_time" '
                        'uht="$upstream_header_time" urt="$upstream_response_time"';

    server {
        listen 80;
        server_name example.com;
        access_log /var/log/nginx/example.com.access.log detailed;
        error_log /var/log/nginx/example.com.error.log;
    }
    ```

    **日志切割**：使用 `logrotate` 工具自动切割日志（apt/yum 安装已默认配置）。

    ## 六、常见问题排查
    1. **配置文件语法错误**：运行 `nginx -t` 检查语法
    2. **端口被占用**：使用 `netstat -tulpn | grep :80` 查看占用端口的进程
    3. **403 Forbidden 错误**：检查文件权限和 Nginx 运行用户
    4. **502 Bad Gateway 错误**：后端服务未启动或 proxy_pass 配置错误
    5. **504 Gateway Timeout 错误**：后端服务响应超时，增加 `proxy_read_timeout` 值

    ## 七、最佳实践
    1. **配置文件分离**：将不同站点的配置放在单独的文件中，便于管理
    2. **启用 Gzip 压缩**：减少传输数据量，提高加载速度
    3. **设置浏览器缓存**：对静态资源设置合适的缓存时间
    4. **隐藏 Nginx 版本号**：在 http 块中添加 `server_tokens off;`
    5. **限制请求大小**：设置 `client_max_body_size` 防止大文件攻击
    6. **定期更新 Nginx**：修复安全漏洞，获取新功能
    7. **监控 Nginx 状态**：启用 `stub_status` 模块监控连接数、请求数等指标

    需要我给你一份**生产环境可用的完整 Nginx 配置模板**，包含静态资源、反向代理、HTTPS 和安全加固吗？

