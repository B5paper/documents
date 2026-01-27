* 问

    ```
    (base) hlc@Ubuntu2204:/tmp$ npm install -g deno
    npm ERR! code EACCES
    npm ERR! syscall mkdir
    npm ERR! path /usr/local/lib/node_modules
    npm ERR! errno -13
    npm ERR! Error: EACCES: permission denied, mkdir '/usr/local/lib/node_modules'
    npm ERR!  [Error: EACCES: permission denied, mkdir '/usr/local/lib/node_modules'] {
    npm ERR!   errno: -13,
    npm ERR!   code: 'EACCES',
    npm ERR!   syscall: 'mkdir',
    npm ERR!   path: '/usr/local/lib/node_modules'
    npm ERR! }
    npm ERR!
    npm ERR! The operation was rejected by your operating system.
    npm ERR! It is likely you do not have the permissions to access this file as the current user
    npm ERR!
    npm ERR! If you believe this might be a permissions issue, please double-check the
    npm ERR! permissions of the file and its containing directories, or try running
    npm ERR! the command again as root/Administrator.

    npm ERR! A complete log of this run can be found in:
    npm ERR!     /home/hlc/.npm/_logs/2025-12-02T15_02_00_671Z-debug-0.log
    ```

    这个通常如何解决？

* 答

    这个错误是因为您没有权限向 /usr/local/lib/node_modules 目录写入文件。有几种解决方法：

    方法1：使用 sudo（最简单）

    ```bash
    sudo npm install -g deno
    ```

    方法2：更改 npm 的全局安装目录权限（推荐）

    ```bash
    # 1. 为当前用户创建全局安装目录
    mkdir -p ~/.npm-global

    # 2. 配置 npm 使用这个目录
    npm config set prefix '~/.npm-global'

    # 3. 将目录添加到 PATH 环境变量
    echo 'export PATH=~/.npm-global/bin:$PATH' >> ~/.bashrc
    source ~/.bashrc

    # 4. 现在可以无权限错误地安装
    npm install -g deno
    ```

    方法3：修复现有目录权限（不推荐，可能有安全风险）

    ```bash
    # 将 /usr/local/lib/node_modules 的所有权更改为当前用户
    sudo chown -R $USER /usr/local/lib/node_modules
    sudo chown -R $USER /usr/local/bin
    ```

    方法4：使用 nvm 管理 Node.js（最佳实践）

    ```bash
    # 1. 安装 nvm
    curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash

    # 2. 重新打开终端或运行
    source ~/.bashrc

    # 3. 安装 Node.js
    nvm install node

    # 4. 现在可以正常安装全局包
    npm install -g deno
    ```

    方法5：使用 Deno 官方的独立安装方法（推荐）

    既然你要安装的是 Deno，可以直接用官方方法：

    ```bash
    # 使用 curl
    curl -fsSL https://deno.land/install.sh | sh

    # 或者使用 Homebrew
    brew install deno
    ```

    推荐方案

    对于 Linux/macOS 系统：

    * 长期方案：使用 nvm（方法4）来管理 Node.js 版本和包

    * 快速方案：使用 sudo（方法1）临时解决

    * 专用方案：直接用 Deno 官方安装脚本（方法5）

    注意：如果选择方法2，安装后可以通过 which deno 检查是否在 PATH 中，并使用 deno --version 验证安装。

