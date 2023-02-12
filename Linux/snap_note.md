# Snap Note

（所有的命令好像不加`sudo`也可以）

查看所有已安装的包：`sudo snap list`

搜索要安装的包：`sudo snap find <text to search>`

安装一个 snap 包：`sudo snap install <snap name>`

更新一个 snap 包：`sudo snap refresh <snap name>`。如果后面不加包的名字就是更新所有包。

把一个包还原到以前安装的版本：`sudo snap revert <snap name>`

删除一个 snap 包：`sudo snap remove <snap name>`

查看历史记录：`snap changes`

Problem shooting:

1. `no override and no default toochain set`

    原因：rust 没有正确安装。

    安装：

    `rustup install stable`

    设置 stable 为默认的版本：

    `rustup default stable`