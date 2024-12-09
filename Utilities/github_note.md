# Github Note

## cache

* github 按 ssh 方式 clone 不会走 git config 设置的代理，会直接连 github.com 的 22 端口

    如果 A 机器通过 B 机器的代理，使用 https 方式进行 git clone 时失败，那么说明 B 机器也无法连接到 github。

    A 机器报错如下：

    ```
    Cloning into 'pynccl'...
    fatal: unable to access 'https://github.com/lancelee82/pynccl.git/': gnutls_handshake() failed: The TLS connection was non-properly terminated.
    ```

* github 的吉祥物是 octocat

* github 的创始人是 Chris Wanstrath

* Pull Request 是 github 引入的一项重要功能

* `@`可以给用户发通知（Notifications）, wiki 可以用来写文档

* GFM: github flavored markdown

* 在 github 里创建新 branch 两种方法

    1. 进入 branch 界面，点 new branch

    2. 在 branch 下拉菜单的搜索框里写 branch name，然后会有 create new branch 的提示

    必须有 repo 的 push 权限才能创建新 branch

* 在 issue 界面里可以为一个 issue 创建一个 branch

## Problems shooting

* Error: `Git pull/push error: RPC failed; result=22, HTTP code = 408`

    原因：http 缓冲区不足

    解决方案：

    `git config http.postBuffer 524288000`

    Ref: <https://stackoverflow.com/questions/22369200/git-pull-push-error-rpc-failed-result-22-http-code-408>