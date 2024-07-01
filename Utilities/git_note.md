# Git Note

## cache

* 列出 git repo 中所有的 remote branches

    * `git branch -r`，或者`git branch --remote
    
    * `git ls-remote`

* git 中 checkout new branch 是什么意思？为什么说`git branch <branch>`只创建新 branch，不 check out new branch？

    猜测：chekcout 指的很可能是切换 branch

* git 其实是用了很多的磁盘空间来实现更灵活的版本管理

* git branch

    `git brahc`等价于`git branch --list`

    `git branch <branch>`创建一个新 branch

    `git branch -d <branch>`删除一个 branche

    This is a “safe” operation in that Git prevents you from deleting the branch if it has unmerged changes.

    `git branch -D <branch>`: Force delete the specified branch

    `git branch -m <branch>`: Rename the current branch to `＜branch＞`

    `git branch -a`: List all remote branches. 

    add a new remote branch and push the local branch to remote branch:

    ```bash
    git remote add <new-remote-repo> https://bitbucket.com/user/repo.git
    git push <new-remote-repo> crazy-experiment~
    ```

    delete a remote branch:

    `git push origin --delete crazy-experiment`

    or

    `git push origin :crazy-experiment`

    这一个比较令人困惑，其实他是给 remote 发送一个 delete signal，从而让远程删除 branch

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

## notes

Some materials to learn:

* <https://www.atlassian.com/git/tutorials/learn-undoing-changes-with-bitbucket>

* <https://www.atlassian.com/git/tutorials/export-git-archive>

* <https://www.atlassian.com/git/tutorials/setting-up-a-repository>

* <https://support.atlassian.com/bitbucket-cloud/docs/use-pull-requests-for-code-review/#Workwithpullrequests-Mergestrategies>

* <https://phoenixnap.com/kb/git-push-tag>

* This seems to be a Git book: <https://www.gitkraken.com/learn/git/problems/git-push-to-remote-branch>

* <https://www.atlassian.com/git/tutorials/setting-up-a-repository>

* <https://git-scm.com/book/en/v2/Git-Tools-Rewriting-History>

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

`git log --all`：查看所有 branch 的记录

`git reflog`：查看包括`reset --hard`之类的修改记录

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

    `git push <remote> <branch>`: 把 local 的`<branch>` push 到`<remote>`。

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

* `git fetch`只更新当前 branch 的信息，不更新其他 branch 的信息。

* `git merge <from_branch> [<to_branch>]`

    default merge to current branch.

    执行 merge 操作时，`<to_branch>`必须存在。

    `git merge <branch_name>`：把`<branch_name>`分支 merge 到当前分支。即对比`<branch_name>`分支最新的一次 commit 与当前分支的最新 commit，如果这两个 commit 在同一条线上，那么直接使用 fast-forward，改变当前 branch 的指针。如果这两个 commit 不在同一条线上，那么对当前 branch 创建一个新的 merge commit，并要求你手动处理 conflict。

    注意，fast-forward 不会创建新 commit，而 conflict 会创建新 commit。

    `git merge origin/master`

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

## git branch

`git branch --set-upstream-to=origin/main`可以设置当前 branch 对应的 remote branch。

当 remote 的 branch 不存在时，这个命令无法使用。必须先使用`git push -u`将 local 的 branch push 到 remote branch。

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
