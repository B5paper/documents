# Git Note

## cache

* git submodule sync

    如果`.gitmodules`中的 repo url 路径改变，那么可以使用`git submodule sync`将更改同步到`.git/config`中。

    用法：

    1. 将`.gitmodules`中的所有配置同步到主仓库`.git/config`和子仓库`sub/.git/config`中

        `git submodule sync`

    2. 只同步指定 sub

        `git submodule sync -- <submodule-path>`

        example:

        `git submodule sync -- lib/my-library`

        只修改 my-library 子仓库中的`.git/config`

    注意事项：

    * 如果已经执行过 git submodule update 且本地有 submodule 的 clone，修正 URL 后可能需要删除本地子模块目录重新拉取

    * 如果子模块没有初始化（--init），sync 可能不会生效，因为子模块的 .git 目录还不存在或配置不完整。

    git submodule sync 的作用是同步子模块的远程 URL，具体来说：

    主要功能：

    * 同步 .gitmodules 中的 URL 到 .git/config

        * 当你在 .gitmodules 文件中修改了子模块的 URL 后，git submodule sync 会将这些更改同步到本地的 .git/config 文件中。

    * 更新本地子模块的远程仓库配置

        * 对于已经初始化并拉取过的子模块，它会更新该子模块目录内的 .git/config 中的 remote.origin.url。

    使用场景：

    * 修改了 .gitmodules 中的 URL 后（比如仓库迁移、协议变更等）

    * .git/config 中的 URL 与 .gitmodules 不一致时

    * 重新配置子模块的远程仓库地址

    工作原理：

    ```bash
    # 假设 .gitmodules 中的 URL 已从 old-url 改为 new-url

    # 执行前：
    .gitmodules:          submodule.foo.url = new-url
    .git/config:          submodule.foo.url = old-url   # 旧地址
    foo/.git/config:      [remote "origin"] url = old-url  # 旧地址

    # 执行后：
    git submodule sync

    .git/config:          submodule.foo.url = new-url   # 已更新
    foo/.git/config:      [remote "origin"] url = new-url  # 已更新
    ```

    常用命令组合：

    ```bash
    # 修改 .gitmodules 中的 URL 后
    vim .gitmodules

    # 同步配置
    git submodule sync

    # 更新子模块内容
    git submodule update --init --recursive
    ```

* git 配置命令别名

    ```
    # 设置别名
    git config --global alias.co checkout
    git config --global alias.br branch
    git config --global alias.ci commit
    git config --global alias.st status

    # 复杂别名
    git config --global alias.lg "log --oneline --graph --all"
    git config --global alias.unstage "reset HEAD --"
    git config --global alias.last "log -1 HEAD"
    ```

    example:

    `git st` = `git status`

* `git config --global core.editor "code --wait"`

    将 VS Code 设置为 Git 的全局默认文本编辑器

    * ore.editor：指定 Git 使用的文本编辑器

    * "code --wait"：

        * code：VS Code 的命令行启动命令

        * --wait：重要参数！让 Git 等待 VS Code 关闭后才继续执行

            执行 git commit（不带 -m）时：

            1. Git 会自动打开 VS Code

            2. 你编辑保存提交信息

            3. 关闭 VS Code 窗口后，Git 才会继续执行提交

            4. 如果没有 --wait，Git 会在打开 VS Code 后立即继续执行，导致提交信息为空

    如果你用其他编辑器：

    * Vim: vim

    * Nano: nano

    * Notepad++: "C:/Program Files/Notepad++/notepad++.exe" -multiInst -nosession

* git config 常见用法

    **差异比较配置**

    ```bash
    # 使用 difftool（如 vimdiff, vscode）
    git config --global diff.tool vimdiff
    git config --global difftool.prompt false

    # 更友好的 diff 输出
    git config --global diff.colorMoved zebra
    git config --global diff.algorithm patience  # 更智能的算法
    ```

    **合并与冲突解决**

    ```bash
    # 设置合并工具
    git config --global merge.tool vimdiff

    # 保持合并提交的原始分支信息
    git config --global merge.log true

    # 自动解决某些冲突
    git config --global pull.rebase true  # pull 时使用 rebase
    ```

    **提交模板与钩子**

    ```bash
    # 设置提交信息模板
    git config --global commit.template ~/.gitmessage.txt

    # 设置全局钩子目录
    git config --global core.hooksPath ~/.githooks
    ```

    **性能与行为优化**

    ```bash
    # 提高大仓库性能
    git config --global core.preloadindex true
    git config --global core.fscache true

    # 自动修正拼写错误
    git config --global help.autocorrect 1  # 1秒后自动执行

    # 禁用某些警告
    git config --global advice.detachedHead false

    # 设置默认分支名称
    git config --global init.defaultBranch main
    ```

    **SSH 与代理配置**

    ```bash
    # 指定 SSH 命令
    git config --global core.sshCommand "ssh -i ~/.ssh/id_rsa"

    # 设置 HTTP/HTTPS 代理
    git config --global http.proxy http://proxy.example.com:8080
    git config --global https.proxy https://proxy.example.com:8080
    ```

    **跨平台兼容性**

    ```bash
    # Windows 下处理行尾符
    git config --global core.autocrlf true  # Windows 推荐
    git config --global core.autocrlf input  # Linux/Mac 推荐

    # 文件系统大小写敏感
    git config --global core.ignorecase false
    ```

    **查看与管理配置**

    ```bash
    # 查看所有配置
    git config --list
    git config --list --show-origin  # 显示配置来源

    # 查看特定配置
    git config user.name
    git config --get-all alias.ci

    # 删除配置
    git config --global --unset alias.st
    git config --global --unset-all http.proxy

    # 编辑配置文件
    git config --global --edit
    ```

    **实用组合配置**

    ```bash
    # 开发者常用配置包
    git config --global core.editor "code --wait"
    git config --global color.ui auto
    git config --global alias.lg "log --color --graph --pretty=format:'%Cred%h%Creset -%C(yellow)%d%Creset %s %Cgreen(%cr) %C(bold blue)<%an>%Creset' --abbrev-commit"
    git config --global pull.rebase true
    git config --global fetch.prune true  # 清理已删除的远程分支
    ```

    配置文件位置

    * 全局配置：~/.gitconfig 或 ~/.config/git/config

    * 系统配置：/etc/gitconfig

    * 仓库配置：.git/config

    这些配置可以根据个人工作习惯和团队规范进行调整，显著提升 Git 使用效率。

* git ignore

    1. 配置文件的层级结构

        Git 会从多个位置读取忽略规则，优先级从高到低：

        * .git/info/exclude - 仓库本地的忽略规则

        * 各个目录中的 .gitignore 文件

        * $GIT_DIR/exclude（已废弃）

        * 全局配置 core.excludesFile（默认 ~/.gitignore）

    2. 规则匹配机制

        * 模式匹配：支持简单的通配符模式

            * `*` 匹配任意字符串（不包括路径分隔符）

            * `?` 匹配单个字符

            * `[]` 匹配字符组

            * `**` 匹配任意目录层级

        * 路径处理：

            以 / 开头：相对于 .gitignore 文件所在目录

            以 / 结尾：只匹配目录

            无 /：匹配文件和目录

    3. 实现流程

        ```c
        // 简化的检查逻辑
        if (文件在索引中已存在)
            不被忽略（已跟踪文件优先）
        else
            遍历所有适用的 .gitignore 文件
            如果匹配到忽略规则：
                if (规则以 ! 开头)  // 否定规则
                    取消忽略
                else
                    标记为忽略
        ```

    4. 核心特性

        * 已跟踪文件不受影响：忽略规则只对未跟踪文件有效

        * 目录忽略：忽略目录时，其所有内容都会被忽略

        * 否定规则：用 ! 开头的规则可以覆盖之前的忽略

        * 模式递归：子目录中的 .gitignore 只影响该目录及其子目录

    5. 性能优化

        * 使用缓存机制存储已解析的忽略模式

        * 按需加载 .gitignore 文件

        * 索引状态优先检查

    示例流程

    ```text
    文件: project/src/temp.log
    检查顺序:
    1. 检查是否已在索引 → 否
    2. 检查 project/.gitignore → 无匹配
    3. 检查 project/src/.gitignore → 匹配 "*.log"
    4. 标记为忽略
    ```

* git ignore 与正则表达式的对比

    Git ignore 使用的是自己定义的一套简化模式匹配规则，不是基于正则表达式。比正则表达式简单得多。

    主要区别：

    1. 模式语法不同

        ```bash
        # Git ignore 模式示例
        *.log       # 简单的通配符
        temp/       # 目录匹配
        src/*.js    # 单层目录匹配
        **/test     # 任意层级匹配

        # 正则表达式（复杂且功能不同）
        ^.*\.log$
        temp/.*
        src/[^/]*\.js
        (.*/)?test
        ```

    2. 底层实现

        在 Git 源码中（dir.c、dir.h），主要使用以下结构：

        ```c
        struct path_pattern {
            const char *pattern;      // 原始模式字符串
            int nowildcardlen;        // 非通配符部分长度
            unsigned flags;           // 模式标志
            int patternlen;           // 模式长度
        };
        ```

        匹配逻辑在 fnmatch_icase 等相关函数中实现，但不是使用标准的正则引擎。

    3. 关键差异点

        | 特性 | Git Ignore | 正则表达式 |
        | - | - | - |
        | 通配符 * | 匹配除 / 外的任意字符 | 匹配任意字符 |
        | 通配符 ? | 匹配单个非 / 字符 | 匹配单个任意字符 |
        | [abc] | 字符组，不支持范围缩写 | 支持 [a-z] 等范围 |
        | ** | 特殊：跨目录匹配 | 无此特殊含义 |
        | 转义字符 | 很少，基本不需要转义 | 大量转义需求 |
        | 性能 | 轻量级，优化过的算法 | 相对较重 |

    4. 实际的匹配函数

        Git 使用自定义的 wildmatch 函数族，这是专门为文件名匹配设计的算法：

        ```c
        // 简化的匹配逻辑示意
        static int wildmatch(const char *pattern, const char *text,
                             unsigned int flags)
        {
            // 特殊的通配符处理逻辑
            // 针对 ** 的特殊处理
            // 针对路径分隔符的特殊处理
        }
        ```

    5. 为什么不用正则？

        性能：文件匹配需要高性能，正则表达式相对较重

        简单性：用户需要简单直观的语法

        可预测性：简化模式更容易理解和调试

        历史兼容：Git 早期版本的设计选择

    6. 特殊情况处理

        ```bash
        # Git ignore 的特殊语义
        *.py[co]    # 匹配 .pyc 或 .pyo（不是字符组范围！）
        # 这实际上匹配字面字符 [、c、o，而不是正则的字符组
        ```

    总结

    Git ignore 使用的是 专门设计的简化通配符系统，类似于 shell 的 glob 模式但有一些扩展（特别是 **）。这种设计在表达能力、性能和易用性之间取得了良好平衡，比正则表达式更适合文件忽略的场景。

* 使用 git archive + 手动创建新仓库

    ```bash
    # 1. 为每个版本下载压缩包
    mkdir my-repo && cd my-repo
    wget https://github.com/user/repo/archive/refs/tags/v1.0.tar.gz
    wget https://github.com/user/repo/archive/refs/tags/v2.0.tar.gz
    wget https://github.com/user/repo/archive/refs/tags/v3.0.tar.gz

    # 2. 创建新的git仓库
    git init
    tar -xzf v1.0.tar.gz --strip-components=1
    git add . && git commit -m "v1.0"
    rm -rf *  # 清空目录

    tar -xzf v2.0.tar.gz --strip-components=1
    git add . && git commit -m "v2.0"
    rm -rf *

    tar -xzf v3.0.tar.gz --strip-components=1
    git add . && git commit -m "v3.0"

    # 现在你有一个只有3个提交的新仓库
    ```

* git commit 后发现 user 和 email 写错了，该如何补救

    * 只修改最近一次提交的作者信息

        `git commit --amend --author="正确的姓名 <正确的邮箱>"`

    * 修改最近一次提交的作者信息

        `git commit --amend --reset-author`

        这会使用你在 git config 中配置的用户名和邮箱。

    * 修改多个提交的作者信息

        ```bash
        # 修改最近3次提交
        git rebase -i HEAD~3 --exec "git commit --amend --reset-author --no-edit"
        ```

        或者使用更强大的方法：

        ```bash
        # 交互式 rebase，标记要修改的提交
        git rebase -i HEAD~3
        # 在编辑器中，将需要修改的提交前的 pick 改为 edit
        # 然后对每个标记为 edit 的提交执行：
        git commit --amend --author="正确的姓名 <正确的邮箱>"
        git rebase --continue
        ```

        如果需要修改整个仓库的历史记录，可以使用 git filter-branch：

        ```bash
        git filter-branch --env-filter '
        OLD_EMAIL="旧的邮箱"
        CORRECT_NAME="正确的姓名"
        CORRECT_EMAIL="正确的邮箱"
        if [ "$GIT_COMMITTER_EMAIL" = "$OLD_EMAIL" ]
        then
            export GIT_COMMITTER_NAME="$CORRECT_NAME"
            export GIT_COMMITTER_EMAIL="$CORRECT_EMAIL"
        fi
        if [ "$GIT_AUTHOR_EMAIL" = "$OLD_EMAIL" ]
        then
            export GIT_AUTHOR_NAME="$CORRECT_NAME"
            export GIT_AUTHOR_EMAIL="$CORRECT_EMAIL"
        fi
        ' --tag-name-filter cat -- --branches --tags
        ```

    * 如果已经推送到远程仓库，修改历史记录后需要使用 git push --force（谨慎使用）

* `git cherry-pick`

    用于将指定的提交应用到当前分支。

    * 主要作用

        * 选择性地复制单个或多个提交到当前分支

        * 常用于修复bug、功能迁移，或者从其他分支提取特定改动

        * 不合并整个分支，只引入特定的提交

    * 基本用法

        ```bash
        # 1. 基本用法 - 应用单个提交
        git cherry-pick <commit-hash>

        # 2. 应用多个提交
        git cherry-pick <commit1> <commit2> <commit3>

        # 3. 应用连续的提交范围（左开右闭）
        git cherry-pick <start-commit>..<end-commit>

        # 4. 应用连续的提交范围（包含起始提交）
        git cherry-pick <start-commit>^..<end-commit>
        ```

    * 常用选项

        ```bash
        # 编辑提交信息
        git cherry-pick -e <commit>

        # 不自动提交，只更新工作区
        git cherry-pick -n <commit>

        # 解决冲突后继续
        git cherry-pick --continue

        # 放弃当前cherry-pick操作
        git cherry-pick --abort

        # 跳过当前提交
        git cherry-pick --skip
        ```

    * 工作流程示例

        ```bash
        # 1. 切换到目标分支
        git checkout main

        # 2. 从开发分支选择特定提交
        git cherry-pick abc123

        # 3. 如果有冲突，解决后继续
        # 解决冲突后...
        git add .
        git cherry-pick --continue
        ```

    * 典型应用场景

        * 修复bug：将修复提交从开发分支应用到生产分支

        * 功能移植：只移植某个功能相关的提交

        * 代码审查：只接受部分提交改动

        * 分支维护：在不同版本分支间同步特定修复

    * 注意事项

        * 每个`cherry-pick`都会创建新的提交（即使内容相同，提交ID也不同）

        * 可能产生冲突，需要手动解决

        * 顺序依赖的提交需要按顺序`cherry-pick`

        * 不适合大量提交的迁移（此时应考虑 merge 或 rebase）

    与 git merge 和 git rebase 不同，cherry-pick 提供了更精细的提交选择控制，让你能够精确地选择需要的改动应用到当前分支。

    example:

    ```
    (base) hlc@hlc-VirtualBox:~/Documents/Projects/git_test/repo-2$ git init
    Initialized empty Git repository in /home/hlc/Documents/Projects/git_test/repo-2/.git/
    (base) hlc@hlc-VirtualBox:~/Documents/Projects/git_test/repo-2$ git remote add origin ../repo-server/
    (base) hlc@hlc-VirtualBox:~/Documents/Projects/git_test/repo-2$ git checkout -b main
    Switched to a new branch 'main'
    (base) hlc@hlc-VirtualBox:~/Documents/Projects/git_test/repo-2$ git cherry-pick bdb04eac91dcc38477bd235ba6e1e8860e94c928 a7f9daf03e71290e94861ab5d3df42d05c5721f4
    fatal: bad object bdb04eac91dcc38477bd235ba6e1e8860e94c928
    (base) hlc@hlc-VirtualBox:~/Documents/Projects/git_test/repo-2$ git fetch
    remote: Enumerating objects: 9, done.
    remote: Counting objects: 100% (9/9), done.
    remote: Compressing objects: 100% (7/7), done.
    remote: Total 9 (delta 2), reused 0 (delta 0), pack-reused 0
    Unpacking objects: 100% (9/9), 727 bytes | 181.00 KiB/s, done.
    From ../repo-server
     * [new branch]      master     -> origin/master
     * [new tag]         v1.0       -> v1.0
     * [new tag]         v2.0       -> v2.0
    (base) hlc@hlc-VirtualBox:~/Documents/Projects/git_test/repo-2$ git cherry-pick bdb04eac91dcc38477bd235ba6e1e8860e94c928 a7f9daf03e71290e94861ab5d3df42d05c5721f4
    [main e88ad40] commit 1
     Date: Wed Dec 24 14:45:00 2025 +0800
     1 file changed, 0 insertions(+), 0 deletions(-)
     create mode 100644 file_1.txt
    [main 34dadd6] commit 3
     Date: Thu Dec 25 10:01:21 2025 +0800
     1 file changed, 0 insertions(+), 0 deletions(-)
     create mode 100644 file_3.txt
    (base) hlc@hlc-VirtualBox:~/Documents/Projects/git_test/repo-2$ git log --graph
    * commit 34dadd6973f7726b39a7a20975f520ed17041f5a (HEAD -> main)
    | Author: Liucheng Hu <lchu@siorigin.com>
    | Date:   Thu Dec 25 10:01:21 2025 +0800
    | 
    |     commit 3
    | 
    * commit e88ad401688c2033fa983b8507943661687a6b28
      Author: Liucheng Hu <lchu@siorigin.com>
      Date:   Wed Dec 24 14:45:00 2025 +0800
      
          commit 1 

    (base) hlc@hlc-VirtualBox:~/Documents/Projects/git_test/repo-2$ ls
    file_1.txt  file_3.txt
    ```

    注：

    1. 可以看到，必须 fetch 后才能 cherry-pick。

    1. cherry-pick 后，除了 commit hash 值和原版 commit 不同，剩下的都相同。

    1. 可以看到 cherry-pick 不是基于快照的，而是基于 diff 的。因为 file_2.txt 不在其中

* git adverse

    If you want to create a new branch to retain commits you create, you may
    do so (now or later) by using -c with the switch command. Example:

    git switch -c <new-branch-name>

    Or undo this operation with:

    git switch -


    (base) hlc@Ubuntu2204:~/Documents/Projects/git_test/a$ git push
    fatal: The current branch b_2 has no upstream branch.
    To push the current branch and set the remote as upstream, use

        git push --set-upstream origin b_2

* 如果遇到问题

    * 问题2：本地已存在同名分支

        ```bash
        # 如果本地已有 branch_local，想重新关联
        git checkout branch_local
        git branch --set-upstream-to=origin/branch_B
        git pull
        ```

    * 问题3：想要删除旧的本地分支重新开始

        ```bash
        # 切换到其他分支
        git checkout branch_A

        # 删除本地分支
        git branch -D branch_local

        # 重新创建并追踪
        git checkout -b branch_local origin/branch_B
        ```

* 验证操作结果

    ```bash
    # 查看所有分支及追踪关系
    git branch -avv

    # 查看当前分支追踪的远程分支
    git rev-parse --abbrev-ref @{upstream}

    # 查看远程分支的最近提交
    git log --oneline origin/branch_B
    ```

* 查看当前的追踪关系

    ```bash
    # 查看所有分支的追踪情况
    git branch -vv

    # 查看特定分支的追踪信息
    git rev-parse --abbrev-ref <branch_name>@{upstream}

    # 或简写
    git rev-parse --abbrev-ref @{u}
    ```

* 如果推送被拒绝的情况

    如果远程已存在同名分支，需要强制推送（谨慎使用）：

    ```bash
    git push -f origin <branch_name>
    ```

    或者先拉取远程分支再推送：

    ```bash
    # 如果远程已有同名分支，先拉取
    git pull origin <branch_name>
    # 解决可能的冲突后
    git push origin <branch_name>
    ```

* 查看远程 repo 信息

    ```bash
    # 查看所有远程分支
    git branch -r

    # 查看所有分支（本地和远程）
    git branch -a

    # 查看远程仓库信息
    git remote show origin
    ```

* git clone 时指定 tab

    ```bash
    git clone -b <tag_name> <repository_url>
    ```

    example:

    ```bash
    git clone -b v1.0.0 https://github.com/user/repo.git
    ```

    这里发生的是：

    * Git 找到标签 v1.0.0 对应的提

    * 将代码克隆到本地

    * 直接检出到那个提交，而不是检出到一个分支

    * 你处于"分离头指针（Detached HEAD）"状态

* 从远程仓库只拉取三个游离 commit

    ```bash
    git init
    git remote add origin <repo-url>

    git fetch --depth 1 origin tag v1.0
    git fetch --depth 1 origin tag v2.0
    git fetch --depth 1 origin tag v3.0

    git log --all

    git checkout v1.0  # 查看v1.0
    git checkout v2.0  # 查看v2.0
    git checkout v3.0  # 查看v3.0
    ```

* git branch 不同个数的`-v`的效果

    ```
    (base) hlc@hlc-VirtualBox:~/Documents/Projects/git_test/my-repo$ git branch
    * master
    (base) hlc@hlc-VirtualBox:~/Documents/Projects/git_test/my-repo$ git branch -v
    * master 0cbd77a commit 2
    (base) hlc@hlc-VirtualBox:~/Documents/Projects/git_test/my-repo$ git branch -vv
    * master 0cbd77a [origin/master] commit 2
    ```

* `git log--decorate`

    在提交历史中显示引用信息，让分支、标签等指针更直观地展示。

    主要作用：

    1. 显示引用位置

        在每个提交旁显示它所在的分支、标签

        用不同颜色区分不同类型的引用

    2. 查看分支拓扑关系

        清楚看到哪些提交属于哪个分支

        了解分支的合并点和起点

    显示的内容：

        HEAD - 当前检出的位置

        分支名 - 本地分支（如 main, feature/login）

        远程分支 - 远程跟踪分支（如 origin/main）

        标签 - 版本标签（如 tag: v1.0）

    在较新的 Git 版本中（2.13+），--decorate 通常是默认启用的，可以通过以下配置查看：

    ```bash
    git config --get log.decorate  # 查看当前设置
    ```

    如果想永久启用，可以设置：

    ```bash
    git config --global log.decorate auto
    ```

    实际价值：

    通过 --decorate，你可以一目了然地：

        知道当前在哪个分支（HEAD 指向）

        看到哪些提交已经推送到了远程

        识别重要的版本标签

        理解分支的合并和分离状态

* `git fetch`与`git fetch --all`的区别

    1. git fetch（默认行为）

        ```bash
        # 只获取当前分支配置的远程仓库
        git fetch
        ```

        * 默认只获取当前分支追踪的远程仓库（通常是 origin）

        * 如果当前分支配置了 upstream，则从对应的远程仓库获取

        * 如果当前分支没有配置 upstream，则从 origin 获取

        * 只更新远程跟踪分支（如 origin/main），不会修改你的本地分支

    2. `git fetch --all`
    
        ```bash
        # 获取所有已配置的远程仓库
        git fetch --all
        ```

        * 获取所有配置的远程仓库（origin、upstream 等）

        * 适合多个远程仓库的场景

        * 一次性更新所有远程跟踪分支

* 常用 fetch 选项

    ```bash
    # 获取所有标签
    git fetch --tags

    # 获取特定远程
    git fetch origin

    # 获取特定分支
    git fetch origin main

    # 清除已不存在的远程分支的本地引用
    git fetch --prune
    ```

* `git merge v1.0 v2.0 v3.0`

    将三个分支（标签/分支）合并到当前分支。

    具体作用：

    1. 多分支合并：一次性将 v1.0、v2.0、v3.0 三个引用（可以是标签或分支名）的代码合并到当前所在分支

    2. 创建合并提交：

        * Git 会找出当前分支与这三个分支的共同祖先

        * 计算四路合并结果

        * 生成一个新的合并提交，这个提交会有多个父提交

    实际效果相当于：
    
    ```bash
    # 分步执行的效果类似：
    git merge v1.0
    git merge v2.0  
    git merge v3.0
    ```

    但一次性合并更高效，且只会创建一个合并提交。

    这是一个相对少用但强大的功能，适用于需要一次性集成多个来源更改的场景。

* `git fetch` 的默认行为是更新所有远程分支的信息，而不是只更新当前分支。

    指定更新特定分支:

    ```bash
    # 只更新特定分支
    git fetch origin main      # 只更新 origin/main
    git fetch origin main:foo  # 更新到特定本地分支
    ```

* 查看远程分支信息

    ```bash
    git branch -r              # 显示远程跟踪分支
    git log origin/main        # 查看远程 main 分支
    git log origin/develop     # 查看远程 develop 分支
    ```

* `git log --all`

    显示所有分支的提交历史，而不仅仅是当前分支。

    主要功能：

    * 显示所有分支的提交 - 包括本地分支和远程跟踪分支

    * 展示完整的项目历史 - 而不仅仅是当前分支的线性历史

    常用组合：

    ```bash
    # 以图形化方式显示所有分支历史
    git log --all --oneline --graph

    # 显示所有分支的历史，包含统计信息
    git log --all --stat

    # 查看所有分支中某个文件的修改历史
    git log --all -- path/to/file
    ```

    与其他选项的对比：

    | 命令 | 作用 |
    | - | - |
    | `git log` | 仅当前分支的历史  |
    | `git log --all` | 所有分支的历史 |
    | `git log --branches` | 所有本地分支的历史 |
    | `git log --remotes` | 所有远程分支的历史 |

* `git fetch origin tag v1.0`：只拉取 v1.0 tag

    ```
    (base) hlc@hlc-VirtualBox:~/Documents/Projects/git_test/repo-2$ git remote add origin ../repo-server/
    (base) hlc@hlc-VirtualBox:~/Documents/Projects/git_test/repo-2$ git fetch origin tag v1.0
    remote: Enumerating objects: 3, done.
    remote: Counting objects: 100% (3/3), done.
    remote: Total 3 (delta 0), reused 0 (delta 0), pack-reused 0
    Unpacking objects: 100% (3/3), 189 bytes | 189.00 KiB/s, done.
    From ../repo-server
     * [new tag]         v1.0       -> v1.0
    ```

    如果 v2.0 依赖 v1.0，那么会下载依赖项：

    ```
    (base) hlc@hlc-VirtualBox:~/Documents/Projects/git_test/repo-2$ git fetch origin tag v2.0
    remote: Enumerating objects: 5, done.
    remote: Counting objects: 100% (5/5), done.
    remote: Compressing objects: 100% (3/3), done.
    remote: Total 5 (delta 0), reused 0 (delta 0), pack-reused 0
    Unpacking objects: 100% (5/5), 394 bytes | 394.00 KiB/s, done.
    From ../repo-server
     * [new tag]         v2.0       -> v2.0
     * [new tag]         v1.0       -> v1.0
    ```

    如果只想下载一个 commit，那么可以使用`--depth`：

    ```
    (base) hlc@hlc-VirtualBox:~/Documents/Projects/git_test/repo-2$ git fetch --depth 1 origin tag v2.0
    remote: Enumerating objects: 3, done.
    remote: Counting objects: 100% (3/3), done.
    remote: Compressing objects: 100% (2/2), done.
    remote: Total 3 (delta 0), reused 0 (delta 0), pack-reused 0
    Unpacking objects: 100% (3/3), 226 bytes | 226.00 KiB/s, done.
    From ../repo-server
     * [new tag]         v2.0       -> v2.0
    ```

    相关命令：

    * `git fetch origin --tags`：获取所有标签

    * `git pull origin tag v1.0`：获取并尝试合并标签（通常不推荐）

* `git remote`只显示 remote 的 name

    `git remote -v`会显示 remote 的 name 和对应的 fetch 和 push 的 url。

* `git fetch origin`：获取 origin 的所有分支和标签

    ```
    (base) hlc@hlc-VirtualBox:~/Documents/Projects/git_test/repo-2$ git init 
    Initialized empty Git repository in /home/hlc/Documents/Projects/git_test/repo-2/.git/
    (base) hlc@hlc-VirtualBox:~/Documents/Projects/git_test/repo-2$ git remote add origin ../repo-server/
    (base) hlc@hlc-VirtualBox:~/Documents/Projects/git_test/repo-2$ git fetch origin
    remote: Enumerating objects: 5, done.
    remote: Counting objects: 100% (5/5), done.
    remote: Compressing objects: 100% (3/3), done.
    remote: Total 5 (delta 0), reused 0 (delta 0), pack-reused 0
    Unpacking objects: 100% (5/5), 394 bytes | 394.00 KiB/s, done.
    From ../repo-server
     * [new branch]      master     -> origin/master
     * [new tag]         v2.0       -> v2.0
     * [new tag]         v1.0       -> v1.0
    ```

    注：

    1. `git fetch origin`只获取`origin`的 branch 和 tag，其他的 remote 不被 fetch

* `git remote show origin`: 查看远程分支信息

* 比较版本差异：

    ```bash
    # 比较 v1.0 和 v2.0 的差异
    git diff v1.0..v2.0

    # 查看某个文件的变化
    git diff v1.0..v3.0 -- README.md

    # 查看提交历史
    git log v1.0..v3.0 --oneline
    ```

* 精确拉取三个提交

    ```bash
    # 1. 创建空仓库
    mkdir repo && cd repo
    git init

    # 2. 添加远程
    git remote add origin https://github.com/user/repo.git

    # 3. 只拉取这三个标签对应的提交
    git fetch --depth 1 origin tag v1.0
    git fetch --depth 1 origin tag v2.0
    git fetch --depth 1 origin tag v3.0

    # 4. 创建分支并包含这三个提交
    git checkout -b my-versions v3.0
    git merge v2.0  # 这可能会快进，因为v2.0是v3.0的祖先
    git merge v1.0  # 同样会快进
    ```

    验证它们在同一个分支上:

    ```bash
    # 查看标签的提交关系
    git log --oneline --graph --decorate v1.0 v2.0 v3.0

    # 查看某个标签在哪些分支上
    git branch --contains v1.0
    git branch --contains v2.0
    git branch --contains v3.0

    # 查看标签详情
    git show --name-only v1.0
    ```

    注：

    1. `git fetch --depth 1 origin tag v3.0`这种用法没问题，可以只拉取一个 commit

    1. `git merge v1.0`这个不能正常 work，这会引入一个 merge commit。

    1. 整个过程中，commit v1.0 ~ v3.0 都是孤立 commit，无线性关系

* 只关心 v1.0, v2.0, v3.0 这三个 commit，对其他的 commit 毫不关心

    * 单仓库多分支切换（如果需要git功能）

        (未测试)

        ```bash
        # 1. 先浅克隆一个版本
        git clone --depth 1 --branch v1.0 https://github.com/user/repo.git
        cd repo

        # 2. 添加其他版本作为远程分支（但只拉取特定提交）
        git remote set-branches origin v2.0 v3.0
        git fetch --depth 1 origin v2.0 v3.0

        # 3. 现在可以切换到不同版本
        git checkout v2.0  # 分离头指针状态
        git checkout v3.0  # 分离头指针状态
        ```

    * 使用 sparse-checkout（Git 2.25+）

        （未测试）

        ```bash
        # 1. 创建空仓库
        mkdir repo-all-versions
        cd repo-all-versions
        git init

        # 2. 添加远程
        git remote add origin https://github.com/user/repo.git

        # 3. 启用 sparse-checkout
        git sparse-checkout init --cone

        # 4. 只拉取特定标签
        for tag in v1.0 v2.0 v3.0; do
            git fetch --depth 1 origin tag $tag
            git checkout tags/$tag -b $tag
        done
        ```

* wget 下载 git repo

    ```bash
    # 方式2： 如果你连 .git 都不想要
    cd myproject/vendor
    wget https://github.com/thirdparty/lib/archive/refs/tags/v1.0.tar.gz
    tar -xzf v1.0.tar.gz
    mv lib-1.0 lib
    ```

    ```bash
    # 直接下载解压到 vendor 目录
    cd vendor
    curl -L <url>/archive/refs/tags/v1.0.tar.gz | tar -xz --strip-components=1
    ```

* git archive（只下载文件，无git历史）

    ```bash
    # 直接下载tar包（最干净，没有.git文件夹）
    curl -L https://github.com/user/repo/archive/refs/tags/v1.0.tar.gz | tar -xz
    # 或者
    wget https://github.com/user/repo/archive/refs/tags/v1.0.tar.gz
    tar -xzf v1.0.tar.gz
    ```

* git submodule 常用命令

    状态检查：git submodule status

    初始化：git submodule init

    更新所有：git submodule update --init --recursive

    删除子模块：

    * git submodule deinit -f libs/mylib

    * rm -rf .git/modules/libs/mylib

    * git rm -f libs/mylib

    注意事项

    * 版本固定：父仓库记录的是子模块的提交哈希，不是分支

    * 递归子模块：使用 --recursive 选项处理嵌套子模块

    * 工作流程：在子模块目录中的修改需要单独提交和推送

    * 团队协作：所有成员都需要初始化子模块

    * 分离头指针：子模块总是检出特定提交，不在任何分支上

    * 双重配置：.gitmodules（共享）和 .git/config（本地）

    * 独立仓库：每个子模块都是完整的Git仓库

    子模块适用于将第三方库或共享组件作为依赖管理，但复杂度较高，对于简单依赖可考虑 Git subtree 或包管理器。

* git submodule 缩写

    ```bash
    # 等价于 init + update
    git submodule update --init

    # 递归初始化并更新所有子模块
    git submodule update --init --recursive

    # 克隆时直接初始化并更新
    git clone --recurse-submodules <repo-url>
    ```

* git submodule 实测

    * 在仓库 A 中`git submodule add <sub-repo-B> <dst-dir>`时，会自动创建`<dst-dir>`目录，并拉取`<sub-repo-B>`的代码

        此时 A 中`.git/config`会发生改变，添加 submodule 相关信息。

    * A 中添加完 submodule 后，执行提交时，只会提交 A 中：

        1. 新文件`.gitmodules`

        2. 新建的空目录`<dst-dir>`

    * 将 A clone 到 C 时，C 中会有：

        1. 文件`.submodule`

        2. 空目录`<dst-dir>`

        注意，此时 C 的`.git/config`中并没有 submodule 相关的信息。

    * 在 C 中执行`git submodule init`后，`.gitmodules`中的数据会被写入到`.git/config`中

    * 在 C 中执行`git submodule update`后，会正式开始拉取 B 的内容，并 checkout 到指定的 commit

        注意：
        
        * 此时 B 是 head detected 状态

        * repo C 中，submodule B 的 commit 会被暂时锁定在 A add B 时的那个 commit
        
            就算此时 repo B 进行了提交，在 C 中执行`git submodule update`或重新 clone C 也不会改变 B 的 commit。

        * 只有 repo A 改变了 submodule B 的 commit，在 C 中执行`git submodule update`才会同步更新 B 的 commit

        * 如果 C 想脱离 A 的控制，独立更新 B 的 commit，需要进入`<sub-B-dir>`中执行`git pull`。

* git submodule status

    你会看到类似：

    ```
     e3a1c9b8c9f2e6f4a8d9c2e3b9f5a1d2c3e4f5 repo_B (heads/main)
    ```

    如果前面是空格（不是`-`），说明子模块已 checkout 成功。

* git 中 head detected 状态

    当你在分离头指针状态时：

    * HEAD 直接指向一个具体的提交，而不是指向一个分支

    * 你不在任何分支上，就像一个"游离"的状态

    * 新建的提交不会自动保存到任何分支

    这种状态的特性:

    * 优点：

        * 可以查看历史版本的代码

        * 可以基于特定版本创建新分支

    * 缺点：

        * 新建提交会丢失：

            ```bash
            # 如果修改并提交
            git add .
            git commit -m "修改"
            # 这个提交没有分支指向它！
            ```

        * 切换分支会丢失工作：

            ```bash
            git checkout main
            # 警告：你刚刚的提交可能会被垃圾回收
            ```

    使用场景：

    * 临时查看历史版本某个特定 tag （commit）的内容

        此时不能提交修改。

    * 从某个 tag 出发，重新构建 branch

        创建新 branch 后，可以提交修改。
    
    如何脱离这个状态：

    ```bash
    # 克隆默认分支
    git clone https://github.com/user/repo.git

    # 切换到标签
    git checkout v1.0.0
    # 此时是分离头指针状态

    # 如果需要修改，创建分支
    git checkout -b fix-v1.0.0

    # 现在可以安全地修改和提交
    git add .
    git commit -m "基于v1.0.0的新功能"
    ```

* 删除远程分支

    ```bash
    # 删除远程分支
    git push origin --delete <branch_name>

    # 或简写
    git push origin :<branch_name>
    ```

* 删除旧的追踪关系

    ```bash
    git branch --unset-upstream
    ```

* 快速检查远程分支是否存在

    ```bash
    # 在操作前可以先检查远程分支
    git ls-remote --heads origin | grep branch_B

    # 或查看远程分支详情
    git remote show origin
    ```

* 验证你是否在分离头指针状态

    ```bash
    # 克隆后查看状态
    git status
    # 输出：HEAD detached at v1.0.0

    # 查看HEAD指向
    git log --oneline -1
    cat .git/HEAD
    ```

* 按需拉取指定数量的 commit

    只拉取一个 commit

    ```bash
    # 只拉取最新一层的提交
    git clone --depth 1 -b v1.0 https://github.com/user/repo.git
    ```

    可以拉取更多历史（如果需要）：

    ```bash
    git fetch --deepen 10  # 再拉取10个历史提交
    ```

* working directory 中的每个文件都有两种状态：tracked 或 untracked

    * Tracked files are files that were in the last snapshort; they can be unmodified, modified, or staged.
    
    * Untracked files are everything else - any files in your working directory that were not in your last snapshot and are not in your staging area.

    [如果一个 staged 文件又被修改了，会成为什么状态呢？答：这个文件会同时变成 staged 和 unstaged 两种状态，当然，文件内容是不同的。可以再 add 一次以合并两种状态。]

* git 中文件的三种状态：`committed`, `modified`, `staged`

    * committed means that the data is safely stored in your local database

    * modified means that you have changed the file but have not committed it to your database yet

    * staged means that you have marked a modified file in its current version to go into your next commit snapshot

* git project 的三个 sections

    * the Git directory

        The Git directory is where Git stores the metadata and object database for your project.

    * the working directory

        The working directory is a single checkout of one version of the project.

    * the staging area

        The staging area is a file, generally contained in your Git directory, that stores information about what will go into your next commit.

* git subtree

    将外部仓库合并到主项目的子目录中，成为项目的一部分。

    usage:

    ```bash
    # 添加远程仓库
    git remote add <远程名> <repository-url>

    # 添加子树（将外部仓库合并到指定目录）
    git subtree add --prefix=<本地目录> <远程名或URL> <分支> --squash

    # 拉取更新
    git subtree pull --prefix=<目录> <远程名> <分支> --squash

    # 推送修改回子项目
    git subtree push --prefix=<目录> <远程名> <分支>
    ```

    特点

    * 代码合并：外部代码成为主项目的一部分

    * 单仓库管理：所有代码在一个仓库中，无需额外初始化

    * 操作复杂：更新和推送命令较长

    * 历史合并：可选择是否保留子项目完整历史

* 如果 git repo 的 remote 是 ssh 开头的地址，那么即使 submodule 中的 url 是 http 开头的地址，在 git submodule update 时也会使用 ssh config 中的代理。

    （存疑）

* Git pull 输出详细信息方法

    ```bash
    git pull -v
    # 或者
    git pull --verbose
    ```

    设置 git 配置使其默认显示详细信息:

    ```bash
    # 设置 pull 默认显示详细信息
    git config --global pull.verbose true

    # 或者设置 fetch 显示详细信息
    git config --global fetch.verbose true
    ```

    使用 `GIT_TRACE` 环境变量:

    ```bash
    # 显示详细的执行过程
    GIT_TRACE=1 git pull

    # 显示更详细的网络通信信息
    GIT_TRACE_PACKET=1 git pull

    # 同时启用多种跟踪
    GIT_TRACE=1 GIT_TRACE_PACKET=1 git pull
    ```

* git merge 在使用 Fast-forward 时，看不出来是一个 merge 操作

    如果是处理冲突的 commit，则会显示一条 merge commit 的提示

    > Merge: aea1ecc ebdd241

    如果想将这个 merge commit 变成一个正常的 commit，那么有下面几种方法：

    1. 使用`git rebase -i <commit_id_before_merge>`，merge commit 选 pick （如果选 squash 会报错）。需要重新处理 confilit。

        假如现在有：

        ```
        commit 3
        commit 2
        commit 1
        ```

        我们想把 commit 2 和 3 合成一个 commit，那么在 rebase 时需要

        `git rebase -i <commit_1_id>`

        此时 todo list 会显示为：

        ```
        pick commit 2
        pick commit 3
        ```

        我们需要把 commit 3 的 pick 改为 squash (s)，保存退出即可。

        squash 与 fixup 的区别为：

        `squash`：合并到前一个 commit，并保留提交信息。

        `fixup`：合并到前一个 commit，但丢弃提交信息。

    1. 如果最新的一个 commit 是 merge commit，那么可以用`git reset <commit_before_merge>`将 commit 回退到 merge 的上一个 commit，同时又保留了 working directory 的更改，此时只需要再`git add .`，`git commit`就可以了。

    注：

    * `git rebase --rebase-merges <commit_id>`不可以以 rebase 的方式处理 merge commit.

* `git revert`可以以提交 commit 的形式向前回退一个 commit。

    `git revert HEAD`，必须要加上`HEAD`，否则无法 work。

    `git revert <commit-id>`可以 revert 到指定 commit 的前一个 commit。

    一个 revert 提交的 commit 也可以被 revert。

* `git rebase`非交互模式，当遇到文件冲突时，不会让用户去处理 conflict，把不同的 commit 合并成一个，创建一个 merge commit，而是先把 upstream 的 commit 全都照搬过来，然后再把 local 的 commit 叠加到上面

    在非交互模式下，常用的语法为`git rebase <upstream>/<remote_branch> [<local_branch>]`。

    首先要保证有一个有效的 remote:

    `git remote add <new_name> <remote_path/url>`

    然后拉取一下信息，不然找不到 main branch：`git fetch`

    然后设置当前 branch 的 upstream:

    `git branch --set-upstream-to=origin/main`

    最后就可以直接运行`git rebase`了。

    以后每次需要`git rebase`前，都要先`git fetch`一次，拿到 upstream 的信息。

* git reset note

    * `git reset`与`git checkout`相似
    
        `git checkout`只移动 head ref，不移动 branch ref，执行完后会出现 detach 状态。

        `git reset`会同时移动 head ref 和 branch ref。

        两者不同之处如下图所示：

        <img width=700 src='../../Reference_resources/ref_8/pic_0.jpg'>

        (不清楚 branch ref 是什么意思)

    * `git reset`有三种模式，`--mixed`,`--soft`和`--hard`

        其中，`--soft`表示只改变 commit history，不改变 staging area (staging index) 和 working directory.

        `--mixed`表示同时改变 commit history 和 staging area，不改变 working directory。

        由于 staging area 被改变，所以有可能有些文件被变成 untracked 状态。

        `--hard`表示同时改变这三者。

    * 如果不指定模式和 commit id，`git reset`会默认加上`git reset --mixed HEAD`

        由此可以推断，`git reset`直接执行，表示丢弃 staging area 中的所有内容，同时不改变 working area 中的内容。

        `git reset --soft` will do nothing。

        `git reset ＜file＞`: Remove the specified file from the staging area, but leave the working directory unchanged. This unstages a file without overwriting any changes.

    * 可以使用`git ls-file -s`列出 staging area 中的一些文件

        `-s`表示`--staged`，可以打印出文件的 hash value。

        如果不写`-s`，那么只输出文件的路径，不输出 hash 值。

    * `git reset --hard HEAD~2`: The git reset HEAD~2 command moves the current branch backward by two commits, effectively removing the two snapshots we just created from the project history.

        这样可以 removing 一些 commits。

    * 由于 git reset 可能会删除一些 commit，而这些 commits 可能被别人引用，所以最好不要在 public repo 上执行这个。如果有回滚 commit 的需求，可以使用`git revert`。

* git 启动 interactive rebase mode

    首先使用`git rebase --interactive HEAD~N`，或者`git rebase -i HEAD~N`进入交互式 rebase 模式。

    这表示从 HEAD commit 开始算起，将最近的 N 个 commit 合并成一个。

    进入交互模式后，将需要 squash 的 commit 改成这个样式：

    ```
    pick d94e78 Prepare the workbench for feature Z     --- older commit
    s 4e9baa Cool implementation 
    s afb581 Fix this and that  
    s 643d0e Code cleanup
    s 87871a I'm ready! 
    s 0c3317 Whoops, not yet... 
    s 871adf OK, feature Z is fully implemented      --- newer commit
    ```

    保存后退出，commit 会自动合并，然后提示是否修改 comment，可以改可以不改。再保存退出，就完成了。

* 查看 git repo 是超前还是落后

	* `git status -sb`

		执行之前先执行`git fetch`

	* 方法二：

		1. Do a fetch: git fetch.
		2. Get how many commits current branch is behind: behind_count = $(git rev-list --count HEAD..@{u}).
		3. Get how many commits current branch is ahead: ahead_count = $(git rev-list --count @{u}..HEAD). (It assumes that where you fetch from is where you push to, see push.default configuration option).
		4. If both behind_count and ahead_count are 0, then current branch is up to date.
		5. If behind_count is 0 and ahead_count is greater than 0, then current branch is ahead.
		6. If behind_count is greater than 0 and ahead_count is 0, then current branch is behind.
		7. If both behind_count and ahead_count are greater than 0, then current branch is diverged.

		Explanation:

    	* `git rev-list` list all commits of giving commits range. --count option output how many commits would have been listed, and suppress all other output.
    	* `HEAD` names current branch.
    	* `@{u}` refers to the local upstream of current branch (configured with branch.<name>.remote and branch.<name>.merge). There is also @{push}, it is usually points to the same as @{u}.
    	* `<rev1>..<rev2>` specifies commits range that include commits that are reachable from but exclude those that are reachable from . When either or is omitted, it defaults to HEAD.

	* 方法三

		You can do this with a combination of git merge-base and git rev-parse. If git merge-base <branch> <remote branch> returns the same as git rev-parse <remote branch>, then your local branch is ahead. If it returns the same as git rev-parse <branch>, then your local branch is behind. If merge-base returns a different answer than either rev-parse, then the branches have diverged and you'll need to do a merge.

		It would be best to do a git fetch before checking the branches, though, otherwise your determination of whether or not you need to pull will be out of date. You'll also want to verify that each branch you check has a remote tracking branch. You can use git for-each-ref --format='%(upstream:short)' refs/heads/<branch> to do that. That command will return the remote tracking branch of <branch> or the empty string if it doesn't have one. Somewhere on SO there's a different version which will return an error if the branch doesn't haven't a remote tracking branch, which may be more useful for your purpose.

## topics

### log

* 可以使用`git log`查看提交历史。`git log -p`可以查看每次提交修改的内容。`git log -p -2`可以只查看最后两次 commit 的内容。`git log --stat`可以查看每次提交中每个文件修改了多少行。

* `git log --pretty=oneline`可以以单行形式只显示 sha-1 码和 comments 信息。`oneline`还可以替换成`short`，`full`以及`fuller`。还可以使用`format`设置自定义的格式：`git log --pretty=format:"%h - %an, %ar : %s"`。

    输出：

    ```
    ca82a6d - Scott Chacon, 6 years ago : changed the version number
    085bb3b - Scott Chacon, 6 years ago : removed unnecessary test
    a11bef0 - Scott Chacon, 6 years ago : first commit
    ```

    `format`的格式参考如下：

    | Option | Description of Output |
    | - | - |
    | `%H` | Commit hash |
    | `%h` | Abbreviated commit hash |
    | `%T` | Tree hash |
    | `%t` | Abbreviated tree hash |
    | `%P` | Parent hashes |
    | `%p` | Abbreviated parent hashes |
    | `%an` | Author name |
    | `%ae` | Author email |
    | `%ad` | Author date (format respects the `--date=option`) |
    | `%ar` | Author date, relative |
    | `%cn` | Committer name |
    | `%ce` | Committer email |
    | `%cd` | Committer date |
    | `%cr` | Committer date, relative |
    | `%s` | Subject |

* `oneline`和`format`通常和`log --graph`合起来用，得到 branch 和 merge 历史。`git log --pretty=format:"%h %s" --graph`

* 有关`git log`的常用参数：

    | Option | Description |
    | - | - |
    | `-p` | Show the patch introduced with each commit. |
    | `--stat` | Show statistics for files modified in each commit. |
    | `--shortstat` | Display only the changed/insertions/deletions line from the `--stat` command. |
    | `--name-only` | Show the list of files modified after the commit information. |
    | `--name-status` | Show the list of files affected with added/modified/deleted information as well. |
    | `--abbrev-commit` | Show only the first few characters of the SHA-A checksum instead of all 40. |
    | `--relative-date` | Display the date in a relative format (for example, "2 weeks ago") instead of using the full date format. |
    | `--graph` | Display an ASCII graph of the branch and merge history beside the log output. |
    | `--pretty` | Show commits in an alternate format. Options include oneline, short, full, fuller and format (where you specify your own format). |

* `git log -<n>`可以显示最后`n`次 commit 的信息，但是这个通常不常用，因为`git log`每次都只输出一页。

* `git log`还常和`--since`和`--until`合起来用：

    ```bash
    git log --since=2.weeks
    git log --since=2008-01-15
    git log --since="2 years 1 day 3 minutes ago"
    ```

* `git log --all`：查看所有 branch 的记录

### merge

* 使用 commit 1 merge commit 2，如果 commit 1 领先 commit 2，那么 commit 1 没有变化

    如果 commit 1 和 commit 2 是 diverge 状态，并且修改的是同一行，或者相邻的几行，那么在执行`git merge`时会显示冲突（conflict）状态。

    如果两个 branch 是 diverge 状态，并且在相邻较远的两段代码上有不同，那么会不会有 conflict 状态？

* git merge 会保存 branch 的所有 commit history

* `git merge origin/master`可以 merge remote branch

* `git merge <from_branch> [<to_branch>]`

    default merge to current branch.

    执行 merge 操作时，`<to_branch>`必须存在。

    `git merge <branch_name>`：把`<branch_name>`分支 merge 到当前分支。即对比`<branch_name>`分支最新的一次 commit 与当前分支的最新 commit，如果这两个 commit 在同一条线上，那么直接使用 fast-forward，改变当前 branch 的指针。如果这两个 commit 不在同一条线上，那么对当前 branch 创建一个新的 merge commit，并要求你手动处理 conflict。

    注意，fast-forward 不会创建新 commit，而 conflict 会创建新 commit。

    `git merge origin/master`

### rebase

* `git rebase -i`比较像从某个 commit 开始，将各个 commit 重新提交一遍。如果每次 commit 都有冲突，那么就需要一直处理冲突。如果不想每次都处理，想只保留最后一次 commit 的结果，或许可以用到`skip`选项。

### submodle

* `git submodule init`

    作用：初始化本地配置文件，建立子模块映射关系

    具体操作：

    1. 读取 .gitmodules 文件
        
        ```conf
        # .gitmodules 示例
        [submodule "libs/mylib"]
            path = libs/mylib
            url = https://github.com/example/lib.git
        ```

    2. 在 .git/config 中添加对应配置

        ```conf
        # 添加后 .git/config 内容
        [submodule "libs/mylib"]
            url = https://github.com/example/lib.git
            active = true
        ```

    3. 检查子模块目录是否存在

        * 如果目录不存在，仅配置，不克隆代码

        * 标记子模块为"active"状态

    注意：init 只是配置，不下载代码！

* `git submodule update`

    作用：检出子模块的指定版本代码

    具体操作：

    1. 读取父仓库记录的特定提交

        ```bash
        # 父仓库中记录的子模块状态（git ls-tree HEAD）
        160000 commit abc123...  libs/mylib
        # abc123 是子模块的特定提交哈希
        ```

    2. 克隆或更新子模块仓库

        * 如果子模块目录是空的：执行 git clone

        * 如果已存在：执行 git fetch + git checkout

    3. 检出指定提交（分离头指针状态）
    
        ```bash
        # 进入子模块目录
        cd libs/mylib
        # 检出父仓库记录的特定提交（不是分支！）
        git checkout abc123def456...
        ```

    4. 递归处理嵌套子模块（使用 --recursive 时）

* git submodule

    将外部仓库作为子模块链接到主项目中，保持独立版本控制。

    usage:

    * 添加子模块

        ```bash
        # 添加子模块
        git submodule add <repository-url> <local-path>
        ```
        
        example:

        ```bash
        git submodule add https://github.com/example/lib.git libs/mylib
        ```

        这会在当前仓库中添加 .gitmodules 文件和子模块目录。

    * 克隆包含子模块的项目

        ```bash
        # 克隆包含子模块的项目
        git clone <主项目仓库 url>
        git submodule init
        git submodule update
        ```

        ```bash
        # 或克隆时直接拉取子模块
        git clone --recursive <主项目仓库 url>
        ```

    * 更新子模块

        ```bash
        # 更新到主仓库记录的版本
        git submodule update

        # 更新到远程最新版本（进入子模块目录）
        cd libs/mylib
        git pull origin main

        # 提交主项目中子模块的引用更新
        cd ../..
        git add libs/mylib
        git commit -m "更新子模块"
        ```

    特点

    * 独立仓库：子模块是独立的 Git 仓库

    * 指针引用：主项目只记录子模块的 commit hash

    * 需要显式初始化更新：克隆后需额外操作获取子模块内容

    * 分离的版本控制：子模块和主项目分别维护历史

### clone

* 如果不指定分支，默认克隆的是远程仓库的默认分支（通常是 main 或 master）

* `git clone -b <tag_name> <repo_url>`

    这个命令会：

    1. 拉取整个仓库的所有历史记录（所有分支、所有标签的所有提交）

    2. 然后 checkout 到 v1.0 标签对应的提交

    3. 你处于分离头指针状态

    即使你只看 v1.0 的代码，本地仍然有全部历史。

* git clone 时，指定 branch

    有以下几种方式：

    * 使用 -b 或 --branch 参数（最常用）

        ```bash
        git clone -b <branch_name> <repository_url>
        ```

        example:

        ```bash
        git clone -b develop https://github.com/user/repo.git
        ```

        这个命令等价于执行了以下操作：

        1. 将远程仓库的**所有数据**拉取到本地

            即`git fetch`

        2. 在本地创建一个 branch `<branch_name>`，内容与远程 branch `<branch_name>`相同，名字也相同

            `git branch <branch_name> origin/<branch_name>`

        3. 切换到刚创建的 branch

            `git switch <branch_name>`

        4. 设置 tracking 关系

            `git branch --set-upstream-to=origin/<branch_name>`

    * 使用`-b`结合`--single-branch`

        只拉取**指定分支**内容，不拉取其他分支。

        `git clone -b <branch> --single-branch <repo_url>`

### branch

* delete a remote branch

    `git push origin --delete <remote_branch>`

    给 remote 发送一个 delete signal，从而让远程删除 branch

* 本地创建完 branch 后，把它推向 remote repo:

    ```bash
    git checkout <local_branch_name>
    git push -u <remote_name> <remote_brnahc_name>
    ```

    比如：

    ```bash
    git checkout hlc_my_branch
    git push -u origin hlc_my_branch
    ```

* 在本地新建了一个 branch，但是远程仓库没有这个 branch，如何把这个 branch 推到远程仓库

    * 推送，并建立追踪关系（最常用）

        ```bash
        git push -u origin <remote_branch_name>
        # 等价于
        # git push --set-upstream origin <remote_branch_name>
        ```

    * 方法二：只推送，不建立追踪

        ```bash
        git push origin <branch_name>
        ```

        这种方式要求本地的 branch 和远程的 branch 的名称相同。如果不同会报错。

        如果一定要求本地 branch 和远程 branch 的名称不同，那么必须用方法三。

    * 方法三：推送并重命名远程分支名

        ```bash
        git push origin <local_branch>:<remote_branch>
        ```

* 修改本地 branch 追踪的远程 branch

    * 使用 git branch --set-upstream-to（推荐）

        ```bash
        git branch --set-upstream-to=origin/<remote_branch> [local_branch]
        ```

    * 使用 git push -u 重新建立追踪

        ```bash
        git push -u origin <remote_branch>
        ```

    * 使用 git config

        ```bash
        # 设置追踪关系
        git config branch.<local_branch>.remote origin
        git config branch.<local_branch>.merge refs/heads/<remote_branch>
        ```

    注：

    1. 如果 remote 名不叫`origin`，需要改成对应的 remote name，比如`upstream`

* 本地的 branch 为 branch_A，远程仓库中别人新建了一个 branch_B，如何在本地新建 branch_local，并将 branch_B 的内容拉取下来，并建立跟踪关系

    * 方法1：直接创建本地分支并追踪远程分支（最推荐）

        ```bash
        # 1. 先获取远程最新的分支信息
        git fetch origin

        # 2. 创建本地分支并直接追踪远程分支
        git checkout -b branch_local origin/branch_B

        # 或简写为
        git checkout --track origin/branch_B
        ```

    * 方法2：先创建再设置追踪

        ```bash
        # 1. 获取远程分支信息
        git fetch origin

        # 2. 创建本地分支但不切换
        git branch branch_local origin/branch_B

        # 3. 切换到新分支
        git checkout branch_local
        ```

    * 方法3：使用 pull 的方式

        ```bash
        # 1. 先创建本地分支（基于当前分支）
        git checkout -b branch_local

        # 2. 设置追踪关系
        git branch --set-upstream-to=origin/branch_B

        # 3. 拉取远程内容
        git pull
        ```

        注:

        1. 感觉这种方法不对，因为可能有 merge 冲突

* `git branch -m <branch>`: Rename the current branch to `＜branch＞`

* `git branch -a`: List all remote branches. 

### 常见场景

* add a new remote repo and push the local branch to remote

    ```bash
    git remote add <new_remote_name> <repo_url>
    git push <new-remote-repo> <local_branch_name>
    ```

### proxy

* git clone 支持`http_proxy`和`https_proxy`环境变量

* 如果使用 http/https 在 github 上 clone repo，那么设置`http.proxy`, `https.proxy`就足够。但是如果使用 ssh 在 github 上 clone repo，那么就需要配置 ssh 的代理

    `~/.ssh/config`:

    ```conf
    Host github.com
        Hostname github.com
        ServerAliveInterval 55
        ForwardAgent yes
        ProxyCommand /usr/bin/corkscrew <replace_with_your_company_proxy_server> <3128> %h %p
    ```

    这个方法目前未验证。

    ref: <https://gist.github.com/coin8086/7228b177221f6db913933021ac33bb92>

### stash

* 多次 git stash 所发生的

    当连续执行多次 git stash 后，stash 栈中会按顺序存储多次暂存的记录（最新的是 stash@{0}，之前的是 stash@{1}、stash@{2} 等）。

    stash 栈的变化：

    * `git stash apply stash@{1}`：仅应用改动，stash@{1} 仍保留在栈中。

    * `git stash pop stash@{1}`：应用改动并删除 stash@{1}，后续的 stash@{0} 会重新编号为 stash@{0}。

    由于 git stash 后，工作目录总是会会恢复最后一次 commit 的内容，所以多次 stash 的操作互相独立，五不干扰，没有依赖关系。

    如果执行多次 git stash pop，那么 stash 的内容会互相叠加，如果有冲突，需要用户处理冲突。

* git stash 只暂存两个地方的文件

    1. 已经使用 git add 添加过的文件

    2. 在 working directory 中，并且之前有 tracing 的文件

    如果一个文件是新创建的，并且没有使用 git add 添加到 staging area，那么 git stash 不会暂存这个文件。

    （如果一个文件既被 git add 添加到了 staging are，又被在 working directory 中做了修改，那么 git stash 存储后，在 git stash pop 时会恢复两个区域的记录吗，还是只恢复一个？）

* git stash show 与 git stash list

    `git stash show`只显示最后一次 stash 的文件修改增删行数信息。`git stash list`会列出所有 stash 的修改记录。

* git stash 可临时保存工作区和暂存区中的更改，不产生 commit

    `git stash`等价于`git stash push -m "可选说明文字"  # 推荐添加备注方便识别`

    查看 stash 记录：`git stash list`

    恢复最近一次的 stash（并保留 stash 记录）: `git stash apply`

    恢复指定某条记录：`git stash apply stash@{n}  # n 为 stash 编号`

    恢复并删除对应的 stash 记录: `git stash pop stash@{n}  # 默认弹出最近的（stash@{0}）
    
    只保存工作目录的修改（不包含已暂存的文件）：`git stash push --keep-index`

    包含未跟踪的文件（如新创建的文件）：`git stash -u  # 或 --include-untracked`

    删除某条 stash 记录：
    
    `git stash drop stash@{n}  # 删除指定记录`

    `git stash clear           # 清空所有 stash`

    查看 stash 的改动内容：`git stash show stash@{n} -p  # -p 显示详细差异`

* `git stash`常用场景

    * 切换分支时保存未提交的修改：

        ```bash
        git stash
        git checkout other-branch
        # 处理其他任务后回到原分支
        git checkout original-branch
        git stash pop
        ```

## notes

* git 其实是用了很多的磁盘空间来实现更灵活的版本管理

Some materials to learn:

* <https://www.atlassian.com/git/tutorials/learn-undoing-changes-with-bitbucket>

* <https://www.atlassian.com/git/tutorials/export-git-archive>

* <https://www.atlassian.com/git/tutorials/setting-up-a-repository>

* <https://support.atlassian.com/bitbucket-cloud/docs/use-pull-requests-for-code-review/#Workwithpullrequests-Mergestrategies>

* <https://phoenixnap.com/kb/git-push-tag>

* This seems to be a Git book: <https://www.gitkraken.com/learn/git/problems/git-push-to-remote-branch>

* <https://www.atlassian.com/git/tutorials/setting-up-a-repository>

* <https://git-scm.com/book/en/v2/Git-Tools-Rewriting-History>

* background and basic concepts

    * CVCs: Centralized Version Control Systems

        * CVS
        * Subversion
        * Perforce

    * DVCSs: Distributed Version Control Systems

        * Git
        * Mercurial
        * Bazaar
        * Darcs

    其他的版本控制系统机制使用的是存储文件的改变，而 git 存储的是快照（snapshots）。如果文件没有改变，git 会在当前版本中指向之前的 reference，否则的话会创建个新快照。

## Configs

### cache

* 显示当前的所有配置：`git config --list --show-origin`

    `--show-origin`表示显示来源的配置文件。

* 可以通过`git config <key>`显示一个指定 key 的值：`git config user.name`

proxy:

```bash
git config --global http.proxy http://proxyUsername:proxyPassword@proxy.server.com:port
git config --global https.proxy http://proxyUsername:proxyPassword@proxy.server.com:port
```

修改默认编辑器：`git config --global core.editor emacs`

* 清除 git proxy

    ```bash
    git config --global --unset http.proxy
    git config --global --unset https.proxy
    ```

### note

git 的配置文件有三个位置：

* `/etc/gitconfig`

    针对所有用户的设置。可以使用`git config --system`来修改这个文件。

* `~/.gitconfig`或`~/.config/git/config`

    针对指定用户的设置。可以使用`git config --global`来修改这个文件。

* `.git/config`

    项目中的配置文件。只指定当前项目的配置。可以直接使用`git config`配置这个文件。

适用范围小的配置文件会覆盖适用范围大的配置文件。

对于 windows 系统来说，全局的配置文件在`c:\users\$user`文件夹下，系统级的配置文件在`c:\programdata\git\config`下。此时只能使用管理员权限通过`git config -f <file>`修改系统级的配置文件。

使用 git 首先需要配置用户名和邮箱：

`git config --global user.name "John Doe"`

`git config --global user.email johndoe@example.com`

配置默认的文本编辑器：`git config --global core.editor emacs`

在 windows 上可以这样配置：

`git config --global core.editor "'c:/program files/notepad++/notepad++.exe' -multiInst  -nosession"`

列出当前的设置：

`git config --list`

如果一个值被定义了多次，git 会选取最后出现的那个为准。

检查特定值：

`git config user.name`

查看帮助文档：

```bash
git help <verb>
git <verb> --help
man git-<verb>
```

例如：

```bash
git help config
```

### .gitignore rules

可以在项目目录下创建一个`.gitignore`来忽视一些文件。文件规则：

* Blank lines or lines starting with `#` are ignored.

* Standard glob patterns work.

    An asterisk (`*`) matches zero or more characters;

    `[abc]` matches any character inside the brackets;

    a question mark (`?`) matches a single character;

    brackets enclosing characters separated by a hyphen (`[0-9]`) matches any character between them.

    You can also use two asterisks to match nested directories: `a/**/z` would match `a/z`, `a/b/z`, `a/b/c/z`, and so on.

* You can start patterns with a forward slash (/) to avoid recursivity.

* You can end patterns with a forward slash (/) to specify a directory.

* You can negate a pattern by starting it with an exclamation point (!)

Example 1:

```gitignore
*.[oa]
*~
```

Example 2:

```gitignore
# no .a files
*.a

# but do track lib.a, even though you're ignoring .a files above
!lib.a

# only ignore the TODO file in the current director, not subdir/TODO
/TODO

# ignore all files in the build/ directory
build/

# ignore doc/notes.txt, but not doc/server/arch.txt
doc/*.txt

# ignore all .pdf files in the doc/ directory
doc/**/*.pdf
```

[如果直接输入`TODO`的话，会递归地匹配所有的 TODO 文件吗？]

## Basic operations

* initialize a repository

    初始化一个仓库：

    ```bash
    git init
    ```

* clone a repository

    ```bash
    git clone [url]
    ```

    比如：

    ```bash
    git clone https://github.com/libgit2/libgit2
    ```

    这个命令将创建一个`libgit2`文件夹，并将所有内容复制到这个文件夹内。如果想指定文件夹，可以使用

    ```bash
    git clone https://github.com/libgit2/libgit2 mylibgit
    ```

    常用的 transfer protocols:

    * `https://`

    * `git://`

    * ssh: `user@server:path/to/repo.git`

* check status of the repo

    查看文件的状态：

    `git status`

    将 untracked 文件放到 staged area 里：

    ```bash
    git add README
    ```

    Short status: `git status -s`或`git status --short`

    输出：

    ```
    M  README
    MM  Rakefile
    A   lib/git.rb
    M   lib/simplegit.rb
    ??  LICENSE.txt
    ```

    其中`??`代表 new files that aren't tracked, `A`代表 new files that have been added to the staging area, `M`代表 modified files。

    状态一共有两栏，左侧一栏表示 staging area，右侧一栏表示 working tree。

* add files to staging area

    `git add (files)`

    如果`add`添加的是一个文件夹，那么会递归地添加文件夹中的所有文件。

    `-n`, `--dry-run`: Don’t actually add the file(s), just show if they exist and/or will be ignored.

    example:

    `git add Documentation/\*.txt`: Adds content from all `*.txt` files under Documentation directory and its subdirectories. The asterisk `*` is used to escape from shell.

    `git add git-*.sh`: Add content from all `git-*.sh` files in current directory, not its subdirectories.

## Other operations

* `git diff`

    To see what you've changed but not yet staged.

    `git diff --staged` 或 `git diff --cached`: to see what you've staged that will go into your next commit.

    [`git diff`比较的是 working directory 中文件和 last commited 文件的差异，还是 working directory 中的文件和 staged area 中的文件？]

    `git diff`好像只能比较已经 tracked 的文件。如果一个文件是新创建的，既没有在 working tree 也没有在 staging area，那么使用`git diff`就不会有任何输出。如果一个文件已经被`git add xxx`添加过，出现在 staging area 中，那么使用`git diff --staged`就可以看到 diff 信息。

    [有关 diff 的输出格式，有时间了看下]

    `git diff --stat`可以直观地查看哪个文件修改了多少：

    ```
    (base) PS D:\Documents\documents> git diff --stat
    Utilities/git_note.md | 64 +++++++++++++++++++++++++++++++++++++++++++++++++++
    1 file changed, 64 insertions(+)
    ```

* `git commit`

    `git commit`选择的 editor 与 shell 的`$EDITOR`环境变量有关。也可以用`git config --global core.editor`设置。

    `git commit`会显示出`git status`的信息，如果想要更详细的信息，可以使用`git commit -v`。如果想一行提交，可以使用`git commit -m "comments"`。

    如果想跳过`git add`阶段，可以使用`git commit -a -m "comments"`。[书上说 -a 的作用是 make Git automatically stage every file that is already tracked before doing the commit. 那么没有被标记 tracked 的文件会不会被 commit 呢？]

* `git rm`

    `git rm`可以删除 staging area 和 working directory 中的文件。删除操作会被提交到 staging area，以后就不再 track 这个文件了。[如果一个文件已经被 committed，然后它又被修改了，那么在删除的时候似乎需要加上`-f`选项]

    如果只想让 git 不再追踪某个文件，从 staging area 里删除，但又不想让它在硬盘上删除，可以使用：`git rm --cached README`。

    `git rm`接受的参数可以是 files, directories, and file-glob patterns。比如`git rm log/\*.log`（这里的反斜杠`\`用于区别 git 和 shell 的 string expansion）。再比如`git rm \*~`可以移除所有以`~`结尾的文件。

[如果我们直接用`mv`命令删除一个文件，然后再`git add .`，`git commit`，会发生什么呢？]

想要重命名一个文件可以用`git mv file_from file_to`，它等价于下面三个命令的组合：

```bash
mv README.md README
git rm README.md
git add README
```

`git reflog`：查看包括`reset --hard`之类的修改记录

## 一些操作

查看一个 tag 所在的 branch: `git branch -a --contains <tag>`

* Pull a specific commit

    不能直接 pull specific commit，但是可以先与远程仓库同步，然后再使用`git-checkout COMMIT_ID`切换到指定的 commit。

    只合并一个指定的　commit 到主分支：

    ```bash
    git fetch origin
    git checkout master
    git cherry-pick abc123
    ```

    合并所有的 commit 到主分支：

    ```bash
    git checkout master
    git merge abc123
    ```

其他的各种 git 相关的技巧：<https://unfuddle.com/stack/tips-tricks/git-pull-specific-commit/>，有时间了看看

* git commit

    * `git commit --author="Liucheng Hu"`

        不知道为啥，`git config --global user.name "Liucheng Hu"`设置完后，`git commit`时的作者并不是这个。`git commmit`时必须单独指定`--author`才行。

    * `git commit -s`

        Signed off. 签名。不清楚这个有啥用。在 github 上会显示 signed by xxx。
        
    * `git commit --amend`

        修正最后一次 commit 的内容，可以用这个操作补上`--author`信息或者`-s`签名。

* git branch

    `git branch <new-branch-name>`：创建一个新 branch

    本地把 branch 创建好后，不需要 commit，只需要`git push [<remote_name>] [<local_branch>]`就可以了。

    `git branch -r`: list remote tracking branches

    local branch 创建好后，与远程仓库中的 branch 并没有建立联系。如果想要建立联系，可以使用`git push --set-upstream origin test`。

    删除一个本地的 branch: `git branch -d <local_branch_name>`（只有当这个 branch 被 merge 过之后，才可以用`-d`删除。如果 branch 没有被 merge，那么可以用`-D`强制删除）

    直接从 github 里删除 branch: `git push origin --delete <remote_brance_name>`

    `git clone xxx`一个新仓库里，似乎只会 clone main branch，其他的 branch 不会 clone。如何能把其他的 branch clone 下来呢？

    `git branch -vv`：查看 branch 更详细的信息，包括对应的 remote branch 信息

* git remote

    remote 仓库在 git 中是以配置的形式存在的。`git remote`也是用来设置配置。

    Examples:

    * `git remote`: list existing remotes

        输出：

        ```
        origin
        ```

        表示目前只有`origin`这一个有关远程仓库的配置。

    * `git remove -v`：显示每个 remote 配置的详细信息

    * `git remote add <shortname> <url>`: 添加一个 remote 配置

    * `git remote show <remote>`：联网拉取 remote 的信息。主要是远程 branch 和 local branch 的对应关系以及状态等。

    * `git remote rename <old_name> <new_name>`

    * `git remote rm <remote>`

* git push

    `git push -u <remote_name> <local_branch_name>`: Push the local repo branch under `<local_branch_name>` to the remote repo at `<remote_name>`.

    不清楚加`-u`和不加`-u`有啥区别。

    `git push <remote_name> <local_branch>`: 把本地的`<local_branch>` push 到`<remote>`。

    `git push [-f] [--set-upstream] [remote_name [local_branch][:remote_branch]]`

    `git push --set-upstream`与`git push -u`等价。

    create a new branch: `git checkout -b branch_name`

    list local branch the corredponding remote branch: `git branch -vv`

    如果不清楚 remote branch 的情况，可以直接使用`git push -u origin HEAD`。这样就等同于`git push --set-upstream origin/HEAD HEAD`

* git rebase

    1. 找到最后一个没有问题的 commit，并执行`git rebase`：

        `git rebase -i <commit_sha>`

        接下来`git`会从`<commit_sha>`（不包含`<commit_sha>`）开始，遍历（walk through）接下来每一个 commit，然后你可以做一些修改，并使用`git commit --amend`提交。

        这时候会进入一个文件编辑器，一般是`nano`。第一个 commit 我们保持`pick`，剩下的 commit 我们可以把每个 commit 前面的`pick`改成`s`，表示`squash`，即虽然采取这次 commit 的修改内容，但是把它整合到其它的 commit 里。

        每次遇到 conflict 的时候，需要我们手动去 merge，然后再按照`git status`里的提示继续就可以了，一般是执行`git rebase --continue`。

    `git rebase <branch_name>`会找到当前分支和`<branch_name>`分支的公共 commit 节点，然后从这里开始，将`<branch_name>`分支的 commits replay 到当前分支。

    如下图所示：

    <div text-align:center>
    <img width=400 src='./pic_1.png' />
    </div>

    ref: 
    
    1. <https://medium.com/mindorks/understanding-git-merge-git-rebase-88e2afd42671>，一个讲`git merge`和`git rebase`的博文，挺好的，很详细。
    
    1. <https://www.atlassian.com/git/tutorials/rewriting-history/git-rebase>

    显然我们最好不要把其他 branch rebase 到`main`分支上，如果这样的话，那么`main`分支就会增加许多 commits。如杲我们此时把`main`分支 push 到 remote 仓库上，那么其他的开发者在 sync 的时候就会平白无故在 main 分支中冒出来很多 commits，感到疑惑。

    如果是要使用交互模式（interactive）模式的 rebase，那么会打开一个编辑器，要求对每一次 commit 进行处理。常见的命令如下所示：

    ```
    pick 2231360 some old commit
    pick ee2adc2 Adds new feature


    # Rebase 2cf755d..ee2adc2 onto 2cf755d (9 commands)
    #
    # Commands:
    # p, pick = use commit
    # r, reword = use commit, but edit the commit message
    # e, edit = use commit, but stop for amending
    # s, squash = use commit, but meld into previous commit
    # f, fixup = like "squash", but discard this commit's log message
    # x, exec = run command (the rest of the line) using shell
    # d, drop = remove commit
    ```

    其中`p`和`s`用得比较多。`p`表示采取这个 commit，`s`表示采取这个 commit，但把这个 commit 合并到上一个 comit 里。

* pull request

    所谓的 pr，实际上 merge 的是指定的两个 branch。

* 有关 branch，commit

    我们可以在其他的 branch 上开发，在其他 branch 上随便 commit，最后在 merge 到 main branch 上时，只需要用 squash 把多个 commit 合并成一个就可以了。

    最后把我们的 main branch merge 到别人的仓库里。

* 所谓的 fast-forward，指的是 main branch HEAD 所在的 commit 节点，可以顺着其他 commit 节点，找到其他 branch 的 HEAD 所在的 commit。

* git tag

    Ref:

    * <https://m.php.cn/tool/git/484989.html>

        简单讲了一些 git tag 的用法。

        问题：如果我们在不同的 branch 中打 tag，那么在 github 中跳转到对应的 tag 时，会自动切换不同的 branch 吗？

* git diff

    `git diff`会对比 working directory 和 staging area 中的改动。如果 staged area 中没有内容，那么会直接对比 working directory 和 last commit 中的变化。

    `git diff --staged`或`git diff --cached`会对比 staged area 和 last commit 中的改动。

* `git rm`

    `git rm`相当于我们先`rm <file_name>`，然后再`git add <file_name>`。

    如果一个文件已经在 staged area 中，那么再把它使用`git rm <file_name>`删除，此时 git 会报错，要求加参数`--cached`或`-f`。

    `git rm --cached <file_name>`表示将 staged area 中的对应文件改变为删除状态，而不改变 working directory 中的文件。

    `git rm -f <file_name>`表示将 staged area 中的对应文件改变为删除状态，同时删除 working directory 中的文件。

* `git mv`

    如果我们直接使用`mv <file_name_1> <file_name_2>`，然后再`git add .`，那么最后使用`git status`看到的是两个操作：

    1. 删除`<file_name_1>`
    2. 新添加文件`<file_name_2>`

    但是我们想要的操作仅仅是对文件重命名而已。此时可以使用`git mv <file_name_1> <file_name_2>`，效果如下：

    ```
    (base) hlc@hlc-NUC11PAHi7:~/Documents/Projects/git_test/helloworld$ git status
    On branch main
    Your branch is ahead of 'origin/main' by 8 commits.
    (use "git push" to publish your local commits)

    Changes to be committed:
    (use "git restore --staged <file>..." to unstage)
        renamed:    main_1.txt -> main_3.txt
    ```

* `git log`

    `git log -p -<N>`可以显示最近`N`次 commit 改动的详细内容。

    `git log --stat`可以简单看看每个文件改了多少行。

* `git reset`

    `git reset HEAD <file>...`可以将某个文件的状态从 staged area 设置回到 unstaged 状态。

    好像也可以使用`git restore --staged <file>`完成相同的功能。

    It has three primary forms of invocation. These forms correspond to command line arguments --soft, --mixed, --hard. The three arguments each correspond to Git's three internal state management mechanism's, The Commit Tree (HEAD), The Staging Index, and The Working Directory.

    `git reset`：等价于` git reset --mixed HEAD`。如果当前的 HEAD 指向分支的 ref，那么不改变 workding directory 中的内容，并且清空 staging area 中的内容。

    `git reset <commit>`：清空 staging area 中的内容，改变当前的 HEAD，但是不会改变 workding directory 中的内容。

* `git checkout`

    `git checkout -- <file>...`可以将某个已经修改过，但是没有 staged 的文件，恢复到 last commit 的内容。

    >  Git just replaced that file with the last staged or committed version

    好像也可以使用`git restore <file>`完成相同的功能。

    这两个命令只适用于已经 tracked 的文件，如果一个文件是新创建的，那么

    `git checkout -b <branch-name>`：如果分支不存在，则创建分支并切换

    `git checkout <commit>`的作用似乎和`git reset <commit>`的作用完全相同。

    如果进入了 detach 模式，可以使用`git checkout <branch_name>`使得 HEAD 重新指到 branch name 上。比如`git checkout main`。如果进入了 detach 模式后，提交了 commit，并且想保留更改，那么可以使用`git branch <new-branch-name>`创建一个新的 branch，保留提交记录。

    看了看网上的主流建议，是说`git reset`专门用于指定 HEAD 移动到哪个 commit 上，`git checkout`专门用于切换分支。

    `git reset <commit>`和`git checkout <commit>`都不会对 working directory 中的文件有影响。

    Ref: <https://www.geeksforgeeks.org/git-difference-between-git-revert-checkout-and-reset/>

    有时间了看看 ref。

    设置当前 branch 的 remote branch: `git checkout --track origin/dev`

* `git restore`

* `git fetch`

    `git fetch <remote>`从指定的 remote config 中更新内容。

    `git fetch [remote_name] [branch_name]`

* `git tag`

    * git 没有办法单独 pull 某个指定的 commit 或 tag，只能 git clone 完后，执行`git checkout <tag_name>`，或`git checkout <commit_id>`切换到指定的 commit

* search code in a specific branch

    <https://stackoverflow.com/questions/31891733/searching-code-in-a-specific-github-branch>

* `git revert <commit_1> [commit_2] ...`

    取消指定的 commit，并将`revert`作为一个新 commit。

    A revert operation will take the specified commit, inverse the changes from that commit, and create a new "revert commit". The ref pointers are then updated to point at the new revert commit making it the tip of the branch.

    `git revert HEAD`

    如果`git revert`一个中间的 commit 会发生什么？

    假如一个文件有三行：

    ```
    first commit
    second commit
    third commit
    ```

    如果我们执行`git revert <second_commit>`，那么会发生冲突 confict：

    对于 revert 而言，它想把第二行变成它的 parent 的 commit，即 empty：

    ```
    first commit
    ```

    而对于 third commit 而言，它想把第二行变成 third commit 的内容，即：

    ```
    first commit
    second commit
    third commit
    ```

    因此会发生冲突。

    common options:

    * `-e`, `--edit`: This is a default option and doesn't need to be specified. This option will open the configured system editor and prompts you to edit the commit message prior to committing the revert。 默认缺省参数，打开编译器写 commit comment。

    * `--no-edit`：不打开 editor

    * `-n`, `--no-commit`: Passing this option will prevent git revert from creating a new commit that inverses the target commit. Instead of creating the new commit this option will add the inverse changes to the Staging Index and Working Directory. 只改变文件内容，不提交 commit

## Commit

* git remove untracked files

    `git clean -f` removes untracked files within the directory whre you call it only.

## Branch

### cache

`git branch -D <branch>`: Force delete the specified branch

* `git branch -d <branch>`删除一个 branch

    This is a “safe” operation in that Git prevents you from deleting the branch if it has unmerged changes.

* `git branch <new_branch_name>`创建一个新 branch，但不切换到新创建的 branch

* `git branch`等价于`git branch --list`

* git 先`get fetch <remote_name>:<remote_branch>`，再`git checkout <remote_branch>`，就可以自动把 remote branch 同步到本地一个新的 branch 了。

* 假如 HEAD 在 branch test 上，那么无法用`git branch -d test`删除 test branch

    即，无法删除当前所在分支。

* 将 remote branch fork 到 local branch

    假设 remote 的 name 是`origin`，那么

    ```bash
    git fetch origin
    git branch <local_branch_name> origin/<remote_branch_name>
    git checkout <local_branch_name>
    ```

    对于已经存在的 local branch，可以使用`git pull`只拉取指定的 branch：

    ```bash
    git pull {repo} {remotebranchname}:{localbranchname}

    git pull origin xyz:xyz
    ```

    其他介绍的方法并不是很优雅，比如`git pull origin branch_2`，其实是先执行`git fetch`，再执行`git merge origin/branch_2`，这样是把`origin/branch_2` merge 到`branch_1`上。

    如果使用`git branch branch_2`，再执行`git pull origin branch_2`，那么相当于从`branch_1` fork 出了一份`branch_2`，然后再 merge remote branch_2。本质上相当于将 branch_2 的内容强行 merge 到 branch_1 上。这样也不太优雅。

* how to reset the remote branch to a specific commit in git

    ```bash
    git reset --hard <commit-hash>
    git push -f origin master
    ```

### note

`git branch --set-upstream-to=origin/main`可以设置当前 branch 对应的 remote branch。

当 remote 的 branch 不存在时，这个命令无法使用。必须先使用`git push -u`将 local 的 branch push 到 remote branch。

## Remote

### cache

* git remote 采用 ssh 协议时的一个 example

    `ssh://hlc@<ip>:<port>/home/hlc/Documents/Projects/my_proj`

    注意`<port>`和路径之间是没有`:`的。

    如果不写 port 的话，写法就是`ssh://hlc@<ip>/path/to/my_project`，同样也没有`:`。

## Rebase

* 使用`git rebase`合并多个 commit

	```bash
	# 从HEAD版本开始往过去数3个版本
	$ git rebase -i HEAD~3

	# 从指定版本开始交互式合并（不包含此版本）
	$ git rebase -i [commitid]
	```

	说明：

	* `-i（--interactive）`：弹出交互式的界面进行编辑合并

	* `[commitid]`：要合并多个版本之前的版本号，注意：[commitid] 本身不参与合并

	指令解释（交互编辑时使用）：

    p, pick = use commit
    r, reword = use commit, but edit the commit message
    e, edit = use commit, but stop for amending
    s, squash = use commit, but meld into previous commit
    f, fixup = like "squash", but discard this commit's log message
    x, exec = run command (the rest of the line) using shell
    d, drop = remove commit

	合并完成后，推送远程：

	```bash
	$ git push --force origin master
	```

	冲突解决
	
	在 git rebase 过程中，可能会存在冲突，此时就需要解决冲突。

	```bash
	# 查看冲突
	$ git status

	# 解决冲突之后，本地提交
	$ git add .

	# rebase 继续
	$ git rebase --continue
	```

## Topics

### merge, branch

* git merge two branches

	将 master branch merge 到 development branch:

	```bash
	git checkout development
	git merge master
	```

	or

	```bash
	git checkout development
	git rebase master
	```

### remote, branch

* 列出 git repo 中所有的 remote branches

    * `git branch -r`，或者`git branch --remote
    
    * `git ls-remote`

## git remote

`git remote`只涉及到 url，不涉及到下面的 branch。

## git pull

* `git pull`会 pull 所有设置 upstream 的 branch。（那么`git pull --all`是干嘛用的？）

* `git pull --rebase`会把远程的仓库强行覆盖本地的 branch。有时间了研究下`git pull`的三种模式（fast-forward, merge, rebase）


## Miscellaneous

* `git status`显示中文目录为`\xxx\xxx\xxx/`的形式

    解决方案：`git config --global core.quotepath false`

* 代理

    `git config --global http.proxy http://127.0.0.1:10809`

* how does a PR work

    Ref: <https://zellwk.com/blog/edit-pull-request/>

## Problem shooting

* `git The requested URL returned error: 403`

    This is commonly because that you don't have enough access privileges. Please check if the token is correct or expired.

* `The process '/usr/bin/git' failed with exit code 128`

    Check if the URL is correct. Sometimes some tags in the URL doesn't exist at all.
