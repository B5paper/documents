# aria2 Note

## cache

* aria2 的源代码使用的是 c++ 11，主要用了 class 和智能指针，有时间了学习下

* aria2 文档：<https://aria2.github.io/manual/en/html/index.html>

## note

有些资源只有使用浏览器才能下载，无法用`aria2`下载，此时可以设置 cookie，就可以登录网站下载了。

`aria2c "下载地址" --header="Cookie: v=xxx"`

找 cookie 的方法是，先打开浏览器调试器，然后点击下载链接，看 header 就可以了。

也有文章说，可以使用`--load-cookies`来加载 cookie。

* 如果 aria2c 下载失败，但是 wget 可以下载成功，浏览器也可以下载成功，那么可以试一试给 aria2 加止这个参数`--check-certificate=false`再下载