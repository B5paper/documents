# datalad note

## cache

* DataLad

    定位：数据管理和分发平台

    核心功能：

    * 基于 Git 和 git-annex 的数据版本控制

    * 管理大型数据集（TB/PB级别）

    * 支持数据共享和协作

    * 数据溯源和可重复性

    * 分层存储（只下载需要的数据）

    * 常用于科学研究和数据科学

    **datalad 如何配置代理**

    为 DataLad 配置代理需要根据其底层工具链（主要是Git、git-annex）进行分层设置，因为DataLad本身不直接处理网络传输。下面我为你梳理了清晰的配置流程和验证方法。

    🗺️ 配置思路概览

    DataLad的通信主要依赖两个组件，需要分别配置：

    | 组件 | 作用 | 是否需要配置代理 |
    | :--- | :--- | :--- |
    | **Git** | 下载数据集元数据、目录结构等小文件。 | **是**，这是基础。 |
    | **git-annex** | 下载实际的大文件内容。 | **是**，这是关键。 |

    ⚙️ 具体配置步骤

    1. 配置 Git 代理

        这是所有操作的基础。在命令行中执行以下命令（替换为你自己的代理地址和端口）：
        ```bash
        # 设置 HTTP/HTTPS 代理
        git config --global http.proxy http://your-proxy-address:port
        git config --global https.proxy https://your-proxy-address:port

        # 如果不需要代理了，可以这样取消
        git config --global --unset http.proxy
        git config --global --unset https.proxy
        ```
        **提示**：如果你的代理需要认证，格式为 `http://用户名:密码@代理地址:端口`。出于安全考虑，建议使用无密码的代理或配置环境变量。

    2. 配置 git-annex 代理

        这是下载大文件内容的关键。git-annex 的 HTTP/HTTPS 传输会尝试**复用 Git 的代理设置**。如果复用失败，你可以显式地通过环境变量来设置：

        ```bash
        # 在 Linux/macOS 的终端或 Windows 的 Git Bash 中
        export http_proxy=http://your-proxy-address:port
        export https_proxy=http://your-proxy-address:port  # 注意：很多HTTPS代理也用http://开头

        # Windows的命令提示符(cmd)中
        set http_proxy=http://your-proxy-address:port
        set https_proxy=http://your-proxy-address:port
        ```
        **注意**：环境变量是临时生效的。要永久生效，需要将 `export` 或 `set` 命令添加到你的 shell 配置文件（如 `~/.bashrc` 或 `~/.zshrc`）或系统环境变量中。

    3. 针对特殊存储后端的代理

        如果你的数据集存储在 **Amazon S3、Google Cloud Storage** 等“特殊远程”上，这些后端的传输库可能有自己的代理配置，需要查阅其官方文档。

        ✅ 验证配置是否生效

        执行一次下载操作是测试代理是否生效的最佳方法。

        1.  **克隆一个已知的公共数据集**：

            ```bash
            datalad clone ///openneuro/ds000001  # 这是一个神经科学测试数据集
            ```

        2.  在克隆过程中，观察命令行输出或使用网络监控工具，查看流量是否经过了你配置的代理服务器。

        3.  **调试**：如果失败，可以尝试增加 git-annex 的日志级别来获取更多信息：

            ```bash
            git annex web
            ```

            然后访问输出的本地网址，在“转移”页面查看详细的下载日志。

    🔧 高级情况处理

    *   **SSH协议的数据源**：如果数据源使用 `git@github.com:` 或 `ssh://` 这类地址，Git的HTTP代理对其无效。此时需要配置 **SSH 通过代理连接**，即在你的 `~/.ssh/config` 文件中为相应主机添加 `ProxyCommand` 配置。

    *   **公司或机构内部代理**：很多机构会使用自动配置脚本（PAC）。你可以将PAC脚本地址设置给Git：

        ```bash
        git config --global http.proxy http://your-proxy:port
        # 或者指向PAC文件
        git config --global http.proxyautoconfigurl http://your-company.com/proxy.pac
        ```

    💡 总结与建议

    配置的核心是 **Git代理 + git-annex环境变量**。最推荐的做法是：**同时设置 Git 的全局代理配置和系统的 `http_proxy` 环境变量**，这样可以最大程度地覆盖所有传输场景。

* DataLad使用指南与最佳实践

    我来详细介绍 DataLad 的使用方法。DataLad 是一个分布式数据管理系统，基于 Git 和 Git-annex，专门用于管理大型数据集。

    一、安装 DataLad

    Linux/Mac

    ```bash
    # 使用 pip
    pip install datalad

    # 或使用 conda
    conda install -c conda-forge datalad
    ```

    Windows

    建议使用 Windows Subsystem for Linux (WSL)

    二、基础使用流程

    1. 创建数据仓库

        ```bash
        # 创建新的数据仓库
        datalad create mydataset
        cd mydataset

        # 从现有仓库克隆
        datalad clone <url> [目标目录]
        ```

    2. 添加数据文件

        ```bash
        # 添加小文件（直接存储在 Git 中）
        datalad save -m "添加小文件" small_file.txt

        # 添加大文件（存储在 Git-annex 中）
        datalad save --to-git -m "添加大文件" large_file.nii.gz

        # 添加整个目录
        datalad save -m "添加数据目录" data/
        ```

    3. 数据获取和更新

        ```bash
        # 获取元数据（不下载实际文件）
        datalad get file.txt

        # 下载所有数据
        datalad get .

        # 更新仓库
        datalad update --merge
        ```

    4. 发布和共享

        ```bash
        # 发布到 GitHub/GitLab 等
        datalad create-sibling-github --github-organization orgname --dataset . mydataset

        # 或发布到 SSH 服务器
        datalad create-sibling --name myserver --url ssh://server/path dataset

        # 推送更改
        datalad push --to origin
        ```

    三、核心功能示例

    1. 版本控制数据集

        ```bash
        # 创建包含数据描述的仓库
        datalad create -c text2git myneurodata
        cd myneurodata

        # 添加 README
        echo "# 神经影像数据集" > README.md
        datalad save -m "添加 README" README.md

        # 添加数据文件
        cp /path/to/data/*.nii.gz .
        datalad save -m "添加原始数据" *.nii.gz
        ```

    2. 使用数据集

        ```bash
        # 克隆数据集（只获取元数据）
        datalad clone https://github.com/dataset/example cloned_dataset
        cd cloned_dataset

        # 按需获取文件
        datalad get sub-01/ses-01/anat/sub-01_T1w.nii.gz

        # 列出所有文件状态
        datalad status

        # 查看文件历史
        git log --oneline sub-01/ses-01/anat/sub-01_T1w.nii.gz
        ```

    3. 数据处理管道

        ```bash
        # 创建可重复的分析
        datalad run \
            -m "运行 fMRI 预处理" \
            -i inputs/raw_data.nii.gz \
            -o outputs/processed_data.nii.gz \
            "python preprocess.py inputs/raw_data.nii.gz outputs/processed_data.nii.gz"
        ```

    4. 数据搜索

        ```bash
        # 搜索 DataLad 数据集
        datalad search "fMRI"

        # 在本地仓库中搜索
        datalad find --type file "*.nii.gz"
        ```

    四、实用命令参考

    仓库管理

    ```bash
    # 查看仓库信息
    datalad wtf

    # 列出子数据集
    datalad subdatasets

    # 添加远程仓库
    datalad siblings add --name backup --url ssh://backup/path

    # 移除文件
    datalad remove file.txt
    ```

    数据操作

    ```bash
    # 批量下载
    datalad get sub-*/anat/*.nii.gz

    # 删除本地文件内容（保留元数据）
    datalad drop file.nii.gz

    # 验证数据完整性
    datalad fsck

    # 导出数据为普通文件
    datalad export-archive --compression=gzip data.zip
    ```

    五、配置建议

    1. 创建配置文件 ~/.gitconfig

        ```gitconfig
        [datalad "dataset"]
            # 设置默认文本文件类型
            textcompletion = all
        ```

    2. 设置 Git 大文件存储

        ```bash
        git config --global user.name "Your Name"
        git config --global user.email "your@email.com"
        ```

    六、最佳实践

    * 结构化组织数据

        ```text
        dataset/
        ├── code/          # 分析代码
        ├── inputs/        # 原始数据
        ├── outputs/       # 处理结果
        ├── derivatives/   # 派生数据
        └── README.md      # 文档
        ```

    * 使用有意义的提交信息

        ```bash
        datalad save -m "添加: 受试者01的T1加权像 [sub-01_T1w.nii.gz]"
        ```

    * 定期更新子数据集

        ```bash
        datalad update --dataset . --recursive
        ```

    * 使用容器确保可重复性

        ```bash
        datalad containers-add myanalysis --url docker://image:tag
        datalad containers-run myanalysis command
        ```

    七、故障排除

    ```bash
    # 查看详细日志
    datalad -l debug <command>

    # 修复损坏的仓库
    datalad rev-save

    # 重置到之前的状态
    datalad unlock --recursive
    git reset --hard HEAD~1

    # 获取帮助
    datalad --help
    datalad <command> --help
    ```

    八、学习资源

    * 官方文档：http://handbook.datalad.org

    * 教程：https://datalad.org/tutorials/

    * 示例数据集：https://github.com/datalad-datasets

    DataLad 特别适合管理科研数据、大型文件集合和需要版本控制的二进制文件。刚开始可以从小型项目入手，逐步掌握其分布式数据管理的强大功能。

* datalad 的代理

    DataLad 底层使用 Git

    HTTP/HTTPS 代理

    ```bash
    # 设置全局代理
    git config --global http.proxy http://proxy.example.com:8080
    git config --global https.proxy https://proxy.example.com:8080

    # 设置特定网站的代理
    git config --global http.https://github.com.proxy http://proxy.example.com:8080

    # 如果需要认证
    git config --global http.proxy http://username:password@proxy.example.com:8080
    ```

    SSH 代理

    ```bash
    # 在 ~/.ssh/config 中添加
    Host github.com
        ProxyCommand nc -X connect -x proxy.example.com:8080 %h %p
    ```

## topics
