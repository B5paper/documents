# Git Note

## Background and basic concepts

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

git 中的文件有三种状态：`committed`, `modified`, `staged`。

* committed means that the data is safely stored in your local database

* modified means that you have changed the file but have not committed it to your database yet

* staged means that you have marked a modified file in its current version to go into your next commit snapshot

git project 的三个 sections:

* the Git directory

    The Git directory is where Git stores the metadata and object database for your project.

* the working directory

    The working directory is a single checkout of one version of the project.

* the staging area

    The staging area is a file, generally contained in your Git directory, that stores information about what will go into your next commit.

working directory 中的每个文件都有两种状态：tracked 或 untracked。Tracked files are files that were in the last snapshort; they can be unmodified, modified, or staged. Untracked files are everything else - any files in your working directory that were not in your last snapshot and are not in your staging area.

[如果一个 staged 文件又被修改了，会成为什么状态呢？答：这个文件会同时变成 staged 和 unstaged 两种状态，当然，文件内容是不同的。可以再 add 一次以合并两种状态。]

## Configs

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

## .gitignore rules

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

## Other operations

* `git diff`

    To see what you've changed but not yet staged.

    `git diff --staged` 或 `git diff --cached`: to see what you've staged that will go into your next commit.

    [`git diff`比较的是 working directory 中文件和 last commited 文件的差异，还是 working directory 中的文件和 staged area 中的文件？]

    `git diff`好像只能比较已经 tracked 的文件。如果一个文件是新创建的，既没有在 working tree 也没有在 staging area，那么使用`git diff`就不会有任何输出。如果一个文件已经被`git add xxx`添加过，出现在 staging area 中，那么使用`git diff --staged`就可以看到 diff 信息。

    [有关 diff 的输出格式，有时间了看下]

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

可以使用`git log`查看提交历史。`git log -p`可以查看每次提交修改的内容。`git log -p -2`可以只查看最后两次 commit 的内容。`git log --stat`可以查看每次提交中每个文件修改了多少行。

`git log --pretty=oneline`可以以单行形式只显示 sha-1 码和 comments 信息。`oneline`还可以替换成`short`，`full`以及`fuller`。还可以使用`format`设置自定义的格式：`git log --pretty=format:"%h - %an, %ar : %s"`。

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

`oneline`和`format`通常和`log --graph`合起来用，得到 branch 和 merge 历史。`git log --pretty=format:"%h %s" --graph`

有关`git log`的常用参数：

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

`git log -<n>`可以显示最后`n`次 commit 的信息，但是这个通常不常用，因为`git log`每次都只输出一页。

`git log`还常和`--since`和`--until`合起来用：

```bash
git log --since=2.weeks
git log --since=2008-01-15
git log --since="2 years 1 day 3 minutes ago"
```

## config

显示当前的所有配置：`git config --list --show-origin`

`--show-origin`表示显示来源的配置文件。

还可以通过`git config <key>`显示一个指定 key 的值：`git config user.name`

proxy:

```bash
git config --global http.proxy http://proxyUsername:proxyPassword@proxy.server.com:port
git config --global https.proxy http://proxyUsername:proxyPassword@proxy.server.com:port
```

修改默认编辑器：`git config --global core.editor emacs`

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

    删除一个本地的 branch: `git branch -d <local_branch_name>`

    直接从 github 里删除 branch: `git push origin --delete <remote_brance_name>`

    `git clone xxx`一个新仓库里，似乎只会 clone main branch，其他的 branch 不会 clone。如何能把其他的 branch clone 下来呢？

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

    `git push <remote> <branch>`: 把 local 的`<branch>` push 到`<remote>`。

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

* `git fetch`只更新当前 branch 的信息，不更新其他 branch 的信息。

* `git merge <from_branch> [<to_branch>]`

    default merge to current branch.

    执行 merge 操作时，`<to_branch>`必须存在。

    `git merge <branch_name>`：把`<branch_name>`分支 merge 到当前分支。即对比`<branch_name>`分支最新的一次 commit 与当前分支的最新 commit，如果这两个 commit 在同一条线上，那么直接使用 fast-forward，改变当前 branch 的指针。如果这两个 commit 不在同一条线上，那么对当前 branch 创建一个新的 merge commit，并要求你手动处理 conflict。

    注意，fast-forward 不会创建新 commit，而 conflict 会创建新 commit。

* `git pull`会 pull 所有设置 upstream 的 branch。（那么`git pull --all`是干嘛用的？）

* 所谓的 fast-forward，指的是 main branch HEAD 所在的 commit 节点，可以顺着其他 commit 节点，找到其他 branch 的 HEAD 所在的 commit。

* `git pull --rebase`会把远程的仓库强行覆盖本地的 branch。有时间了研究下`git pull`的三种模式（fast-forward, merge, rebase）

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

* `git checkout`

    `git checkout -- <file>...`可以将某个已经修改过，但是没有 staged 的文件，恢复到 last commit 的内容。

    >  Git just replaced that file with the last staged or committed version

    好像也可以使用`git restore <file>`完成相同的功能。

    这两个命令只适用于已经 tracked 的文件，如果一个文件是新创建的，那么

* `git restore`

* `git fetch`

    `git fetch <remote>`从指定的 remote config 中更新内容。

* `git tag`


## Miscellaneous

* `git status`显示中文目录为`\xxx\xxx\xxx/`的形式

    解决方案：`git config --global core.quotepath false`

* 代理

    `git config --global http.proxy http://127.0.0.1:10809`