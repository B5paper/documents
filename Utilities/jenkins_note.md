# Jenkins Note

Jenkins 是 Java 写的一个用于 CI 程序，使用 web UI 作为前端。

官网：<https://www.jenkins.io/>

安装：

官网提供了使用 apt 源和 docker 镜像两种方式。最好别用 docker 的，因为 jenkins 在编译项目的时候可能会用到很多依赖库，如果使用 docker 镜像的话，这些依赖都需要在容器中安装一遍；如果选择挂载，那么 host 上`/usr/lib`等地方装的库该怎么挂载到 container 里呢，就算挂载进去还得避免与 container 里的库冲突，如果额外指定路径，还需要在编译或运行程序的时候设置额外的路径，很麻烦，这个问题几乎没法解决。综上，还不如直接用 host 上的库。

在官网下载页面，有各个系统的对应版本，点进去后（<https://pkg.jenkins.io/debian-stable/>），有详细的添加 apt 源的命令和安装教程。直接照着执行就可以了。

安装结束后，第一次使用时，admin 会有一个自动生成的随机 token 作为密码，这个 token 的路径在 terminal 里会给出。初次设置完用户名和密码后，这个 token 会被自动删除。

Jenkins 会创建一个`jenkins`用户，其 home 目录默认设置在`/var/lib/jenkins`。我们可以以`root`身份执行`su jenkins`使用`jenkins`用户登录 shell。

Jenkins 使用的代理为`jenkins` linux 用户的环境变量。我们可以在`/var/lib/jenkins/.bashrc`中设置。

Jenkins 会监听 8080 端口，直接在浏览器地址栏输入`localhost:8080`或者`<addr>:8080`就可以进入登陆页面了。初次进入后，会让安装一些插件，选推荐安装的就行。记得在安装前把代理配置好。

## Usage

创建一个新项目：

点左侧的 New Item，输入项目名称，项目类型选 Freestyle project。

在 Configuration 页面，General 部分，把 Github project 勾上，填项目对应的 github 地址。（这个好像仅仅是描述项目用的，用于跳转到 github 项目对应页面）

在 Source Code Management （SCM）部分，勾选 Git，填项目地址，修改 branch 为`*/main`（默认是`*/master`，这个显然是不对的）。这个比较重要，用于后面的自动构建。

在 Build Triggers 部分，勾选 GitHub hook trigger for GITScm polling。这样可以 enable Github webhook。后面需要在 github 中做对应的配置配合这个才行。

接着在 Build Steps 部分，写需要执行的 command。目前我写的是

```bash
#!/usr/bin/bash
source ${HOME}/.bashrc
make
```

不清楚 jenkins 用的是 bash 还是 sh，也不清楚会不会自动`source .bashrc`。理论上 jenkins 用的应该是 sh，会自动加载`$HOME/.bashrc`中的内容。

填完这些保存就可以了。接下来进入 github 项目页面， Settings -> Webhooks -> Add webhook。在 Payload URL 中填`http://address:8080/github-webhook/`，Content type 选`applicaiont/json`，然后点击下面的 Add webhook 确认就可以了。

github 这边配置好后，每次执行 commit 操作，github 会向 jenkins 服务器发送一个消息。此时 jenkins 会自动执行 build steps 脚本中的内容。

