# Github Actions

Github 提供了机器可以让我们跑 CI。只需要在项目里创建一个`.github/workflows`文件夹，然后写`xxx.yml`配置文件就可以了。

Your workflow contains one or more **jobs** which can run in sequential order or in parallel. Each job will run inside its own virtual machine **runner**, or inside a container, and has one or more steps that either run a script that you define or run an **action**.

每一个 job 都在一个独立的 runner 上运行。

A workflow is a configurable automated process that will run one or more jobs. Workflows are defined by a YAML file checked in to your repository.

## workflows

常用字段：

* `name`：workflow 的名称，显示在左侧导航栏。这个名称是静态的。

* `run-name`：每次执行 CI 时显示的名称。这个字符串可以动态生成。

    Example: `run-name: Deploy to ${{ inputs.deploy_target }} by @${{ github.actor }}`

* `on`：定义 event trigger。常用的有`push`。

    Example:

    可以简单点：

    ```yaml
    on: push
    ```

    使用多个 events:

    ```yaml
    on: [push, fork]
    ```

    使用 filters:

    ```yaml
    on:
        push:
            branches:
            - 'main'
            - 'releases/**'
    ```

* `jobs`：定义一个或多个 jobs

    Example:

    ```yaml
    jobs:
        my_job_1:
            runs-on: ubuntu-latest
            steps:
                - uses: actions/checkout@v3  # 用于同步代码的
                - run: echo hello, world!

        my_job_2:
            runs-on: ubuntu-latest
            steps:
                - run: make

    ```

* `env`：定义对所有 jobs 都生效的环境变量。

    Example:

    ```yaml
    env:
        SERVER: production
    ```

## 环境变量的使用

Ref: <https://blog.csdn.net/frank_haha/article/details/127158562>

github 的环境变量都是加密的，主要用于 github actions。

github secrets 分成三类，environment secrets 优先级最高，然后是 repository, 最后是 orgnization。

对于 environment 级别，首先需要创建个类似 namespace 的东西，在里面再创建一些环境变量的键值对。然后按照如下方式使用环境变量：

```yaml
runs-on: ubuntu-latest
environment:
    name: <created_namespace>

steps:
    password: ${{ secrets.PYPI_PASSWORD }}
```

如果是 repository 级别的 secrets 环境变量，则不需要指定 namespace。

## Problem shooting

1. 在执行`./run.sh`时，需要设置`http_proxy`和`https_proxy`环境变量。

    这两个环境变量可以直接设置成`socks5://address:port`，使用 socks 代理。
