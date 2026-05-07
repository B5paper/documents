[unit]
[u_0]
`+`命令是什么意思？
[u_1]
移动到下一行的第一个非空白字符，相当于 j + ^ 的组合

[unit]
[u_0]
vim vsplit 如何在右侧新开窗口
[u_1]
`:vsplit`或`:vs`默认会在当前窗口的左侧打开新窗口。

在右侧打开新窗口的几种方法：

1. `:rightbelow vertical split`

    ```vim
    :rightbelow vertical split
    ```

    或简写为：

    ```vim
    :rightbelow vsplit
    ```

    或更短的简写：

    ```vim
    :bo vsplit
    ```

    其中 bo 是 bottom 的缩写，在这里表示"右下方"

2. 使用方向参数

    ```vim
    :vertical split
    ```

    然后按 Ctrl+w L 将新窗口移动到右侧

3. 配置`:vsplit`的默认行为（添加到 `.vimrc`）

    ```vim
    set splitright
    ```

4. 使用快捷键映射

    在 `.vimrc` 中创建自定义快捷键：

    ```vim
    " 使用空格+v在右侧垂直分割
    nnoremap <space>v :rightbelow vnew<CR>
    
    " 或保持原有命令但强制右侧
    nnoremap <leader>v :rightbelow vsplit<CR>
    ```

[unit]
[u_0]
vim 进入子目录导航后的常用命令
[u_1]
* `-` 返回上级目录

* Enter 打开文件或进入子目录

* o 在新窗口中打开

* D 删除文件/目录

* R 重命名

* s 切换排序方式

* q 或 :q 关闭目录窗口

    q 好像不行，必须使用`:q`。

* i 切换显示格式（详细信息/简略信息）

