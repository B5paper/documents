# Curl Note

## cache

* 讲讲 curl -I 的作用

    `curl -I` 用于**只获取 HTTP 响应头（headers），不下载响应体（body）**。

    ## 基本含义

    `-I` 等价于 `--head`，它会把请求方法设为 **HEAD**（而不是默认的 GET）。服务器收到 HEAD 请求后，应返回与 GET 相同的响应头，但不返回内容。

    ```bash
    curl -I https://example.com
    ```

    输出类似：

    ```
    HTTP/2 200
    content-type: text/html; charset=UTF-8
    content-length: 1256
    date: ...
    server: ...
    ```

    ## 主要用途

    1. **快速检查 URL 是否可访问**：只看状态码（200/301/404/500 等），不传输整个页面。
    2. **查看重定向**：配合 `-L` 可以看到跳转链；不加 `-L` 则只显示第一跳的 `Location` 头。
       ```bash
       curl -IL https://example.com   # 跟随重定向，逐跳显示响应头
       ```
    3. **检查响应头细节**：`Content-Type`、`Content-Length`、`Cache-Control`、`Set-Cookie`、`Server`、`ETag`、`Last-Modified` 等。
    4. **探测服务器/接口行为**：比如确认是否支持 gzip、CORS 头、安全头（HSTS、CSP）等。
    5. **节省带宽和时间**：对大文件、大页面只取头部，速度更快。

    ## 几个常见搭配

    | 命令 | 作用 |
    |---|---|
    | `curl -I URL` | 只显示响应头 |
    | `curl -IL URL` | 跟随重定向并显示每跳响应头 |
    | `curl -sI URL` | 静默模式（不显示进度条）只显示头 |
    | `curl -I -H "User-Agent: ..." URL` | 带自定义请求头做 HEAD 请求 |
    | `curl -i URL` | 注意：`-i` 是 GET 请求但**包含**响应头 + body，和 `-I` 不同 |

    ## 注意事项

    - **不是所有服务器都正确支持 HEAD**：有些服务器对 HEAD 返回 405，或返回的头与 GET 不一致（比如 `Content-Length` 缺失）。这时可以用 `curl -s -o /dev/null -D - URL` 发 GET 但丢弃 body，只打印头：
      ```bash
      curl -s -o /dev/null -D - https://example.com
      ```
    - `-I` 不会下载 body，所以不能用它验证内容是否正确，只能验证元信息。

    一句话总结：**`curl -I` = 发 HEAD 请求，只看响应头，适合快速探测 URL 状态和头部信息。**

* curl 默认读取的是小写的 http_proxy 和 https_proxy，而不是大写的 HTTP_PROXY

* 发送 post 请求：

`curl --data "post1=value1&post2=value2&etc=valetc" http://host/resource`

    或使用 RESTful API：

    `curl -X POST -d @file http://host/resource`

## topics

