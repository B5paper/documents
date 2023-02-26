# Linux Note

本篇笔记主要记录一些和 linux 相关的配置细节。

## 解决中文的“门”字显示的问题

在`/etc/fonts/conf.d/64-language-selector-prefer.conf `文件中，把各个字体中`<family>Noto Sans CJK SC</family>`放到其它字体的最前面即可。

