# Github Note

## cache

* 在 github 里创建新 branch 两种方法

    1. 进入 branch 界面，点 new branch

    2. 在 branch 下拉菜单的搜索框里写 branch name，然后会有 create new branch 的提示

    必须有 repo 的 push 权限才能创建新 branch

* 在 issue 界面里可以为一个 issue 创建一个 branch

## Problems shooting

* Error: `Git pull/push error: RPC failed; result=22, HTTP code = 408`

    原因：http 缓冲区不足

    解决方案：

    `git config http.postBuffer 524288000`

    Ref: <https://stackoverflow.com/questions/22369200/git-pull-push-error-rpc-failed-result-22-http-code-408>