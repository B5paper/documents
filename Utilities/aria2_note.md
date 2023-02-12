# aria2 Note

有些资源只有使用浏览器才能下载，无法用`aria2`下载，此时可以设置 cookie，就可以登录网站下载了。

`aria2c "下载地址" --header="Cookie: v=xxx"`

找 cookie 的方法是，先打开浏览器调试器，然后点击下载链接，看 header 就可以了。

也有文章说，可以使用`--load-cookies`来加载 cookie。