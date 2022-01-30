# Git Note

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

初始化一个仓库：

```bash
git init
```

clone 一个仓库：

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

working directory 中的每个文件都有两种状态：tracked 或 untracked。Tracked files are files that were in the last snapshort; they can be unmodified, modified, or staged. Untracked files are everything else - any files in your working directory that were not in your last snapshot and are not in your staging area.

[如果一个 staged 文件又被修改了，会成为什么状态呢？答：这个文件会同时变成 staged 和 unstaged 两种状态，当然，文件内容是不同的。可以再 add 一次以合并两种状态。]

查看文件的状态：

`git status`

将 untracked 文件放到 staged area 里：

```bash
git add README
```

syntax:

`git add (files)`

如果`add`添加的是一个文件夹，那么会递归地添加文件夹中的所有文件。

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

可以在项目目录下创建一个`.gitignore`来忽视一些文件。文件规则：

* Blank lines or lines starting with `#` are ignored.

* Standard glob patterns work.

    An asterisk (`*`) matckes zero or more characters;

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

`git diff`: to see what you've changed but not yet staged.

`git diff --staged` 或 `git diff --cached`: to see what you've staged that will go into your next commit.

[`git diff`比较的是 working directory 中文件和 last commited 文件的差异，还是 working directory 中的文件和 staged area 中的文件？]

`git commit`选择的 editor 与 shell 的`$EDITOR`环境变量有关。也可以用`git config --global core.editor`设置。

`git commit`会显示出`git status`的信息，如果想要更详细的信息，可以使用`git commit -v`。如果想一行提交，可以使用`git commit -m "comments"`。

如果想跳过`git add`阶段，可以使用`git commit -a -m "comments"`。[书上说 -a 的作用是 make Git automatically stage every file that is already tracked before doing the commit. 那么没有被标记 tracked 的文件会不会被 commit 呢？]

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

## Miscellaneous

* `git status`显示中文目录为`\xxx\xxx\xxx/`的形式

    解决方案：`git config --global core.quotepath false`