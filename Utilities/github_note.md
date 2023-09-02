# Github Note

## Problems shooting

* Error: `Git pull/push error: RPC failed; result=22, HTTP code = 408`

    原因：http 缓冲区不足

    解决方案：

    `git config http.postBuffer 524288000`

    Ref: <https://stackoverflow.com/questions/22369200/git-pull-push-error-rpc-failed-result-22-http-code-408>