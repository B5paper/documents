* datalad 的代理

    DataLad 底层使用 Git

    HTTP/HTTPS 代理

    ```bash
    # 设置全局代理
    git config --global http.proxy http://proxy.example.com:8080
    git config --global https.proxy https://proxy.example.com:8080

    # 设置特定网站的代理
    git config --global http.https://github.com.proxy http://proxy.example.com:8080

    # 如果需要认证
    git config --global http.proxy http://username:password@proxy.example.com:8080
    ```

    SSH 代理

    ```bash
    # 在 ~/.ssh/config 中添加
    Host github.com
        ProxyCommand nc -X connect -x proxy.example.com:8080 %h %p
    ```

