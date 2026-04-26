# aria2 Note

## cache

* 我用 aria2 下载东西时，遇到这个报错：

    ```
    wsdlh@abchhh /d/Datasets
    $ aria2c -c 'https://s3-eu-west-1.amazonaws.com/pfigshare-u-files/18069143/fNIRSdata.zip?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAIYCQYOYV5JSSROOA/20260422/eu-west-1/s3/aws4_request&X-Amz-Date=20260422T140725Z&X-Amz-Expires=10&X-Amz-SignedHeaders=host&X-Amz-Signature=9fb7a7f3cab25632dc2583dc218d02897cb01a5edeaae2ab172711d380a74f68'

    04/22 22:11:18 [NOTICE] Downloading 1 item(s)
    [#97ddd4 0B/0B CN:1 DL:0B]
    04/22 22:11:22 [ERROR] CUID#7 - Download aborted. URI=https://s3-eu-west-1.amazonaws.com/pfigshare-u-files/18069143/fNIRSdata.zip?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAIYCQYOYV5JSSROOA/20260422/eu-west-1/s3/aws4_request&X-Amz-Date=20260422T140725Z&X-Amz-Expires=10&X-Amz-SignedHeaders=host&X-Amz-Signature=9fb7a7f3cab25632dc2583dc218d02897cb01a5edeaae2ab172711d380a74f68
    Exception: [AbstractCommand.cc:351] errorCode=22 URI=https://s3-eu-west-1.amazonaws.com/pfigshare-u-files/18069143/fNIRSdata.zip?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAIYCQYOYV5JSSROOA/20260422/eu-west-1/s3/aws4_request&X-Amz-Date=20260422T140725Z&X-Amz-Expires=10&X-Amz-SignedHeaders=host&X-Amz-Signature=9fb7a7f3cab25632dc2583dc218d02897cb01a5edeaae2ab172711d380a74f68
      -> [HttpSkipResponseCommand.cc:239] errorCode=22 The response status is not successful. status=403

    04/22 22:11:22 [NOTICE] Download GID#97ddd488dae1521b not complete:

    Download Results:
    gid   |stat|avg speed  |path/URI
    ======+====+===========+=======================================================
    97ddd4|ERR |       0B/s|https://s3-eu-west-1.amazonaws.com/pfigshare-u-files/18069143/fNIRSdata.zip?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAIYCQYOYV5JSSROOA/20260422/eu-west-1/s3/aws4_request&X-Amz-Date=20260422T140725Z&X-Amz-Expires=10&X-Amz-SignedHeaders=host&X-Amz-Signature=9fb7a7f3cab25632dc2583dc218d02897cb01a5edeaae2ab172711d380a74f68

    Status Legend:
    (ERR):error occurred.

    aria2 will resume download if the transfer is restarted.
    If there are any errors, then see the log file. See '-l' option in help/man page for details.
    ```

    该如何解决？

    简单直接地说，你遇到的 `errorCode=22` 伴随 `status=403` 报错，是因为你使用的 **URL 下载链接已经失效（过期）了**。

    ---

    **核心原因分析**

    观察你提供的链接，你会发现其中包含一个关键参数：
    `X-Amz-Expires=10`

    这意味着这个由 AWS S3 生成的临时签名链接，**有效期仅仅只有 10 秒钟**。

    * **生成时间**：`X-Amz-Date=20260422T140725Z`（即 14:07:25）。
    * **尝试下载时间**：根据你的日志显示，你是从 `22:11:18` 开始下载的。

    很明显，当你按下回车键时，这个链接已经过期好几个小时了。S3 服务器认为这是一个无效的请求，因此返回了 **403 Forbidden** 错误。

    ---

    **解决方案**

    1. 重新获取链接并立即下载
        
        这种带签名的临时链接通常是点击网页上的“下载”按钮时动态生成的。
        * **操作**：回到下载页面，刷新，重新点击下载按钮。
        * **技巧**：在浏览器弹出下载提示时，迅速复制最新的链接，并立即在终端执行 `aria2c`。

    2. 给链接加上引号（重要细节）
        
        虽然你已经在命令中使用了单引号，但请务必保持这个习惯。因为 URL 中含有 `&` 等特殊字符，如果不加引号，Shell 会将其误认为是后台指令，导致链接断裂。
        > **正确示范：**
        > `aria2c -c 'https://...最新获取的链接...'`

    3. 如果是 Figshare 等平台

        由于你下载的是 `figshare` 的文件，这类科研数据库通常提供更稳定的 API 或永久引用链接（Handle/DOI）。如果需要批量下载，建议查看该数据集是否有对应的命令行工具或 Python SDK，而不是依赖这种极短时效的 S3 预签名链接。

    ---

    **建议**

    如果你经常遇到这类由于网络波动导致 10 秒内没跑起来就报错的情况，建议你在获取链接后**手动将链接中的 `X-Amz-Expires` 值改大一点**（注：这通常不起作用，因为过期时间也是签名计算的一部分，修改参数会导致签名验证失败）。

    所以，**唯一的办法就是：现拿现用，手速要快。**

    你现在的下载任务是通过浏览器直接点击下载比较顺畅，还是必须使用命令行工具？

* aria2 的源代码使用的是 c++ 11，主要用了 class 和智能指针，有时间了学习下

* aria2 文档：<https://aria2.github.io/manual/en/html/index.html>

## note

有些资源只有使用浏览器才能下载，无法用`aria2`下载，此时可以设置 cookie，就可以登录网站下载了。

`aria2c "下载地址" --header="Cookie: v=xxx"`

找 cookie 的方法是，先打开浏览器调试器，然后点击下载链接，看 header 就可以了。

也有文章说，可以使用`--load-cookies`来加载 cookie。

* 如果 aria2c 下载失败，但是 wget 可以下载成功，浏览器也可以下载成功，那么可以试一试给 aria2 加止这个参数`--check-certificate=false`再下载
