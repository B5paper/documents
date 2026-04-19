## cache

* 简述 svn 如何安装和使用

    以下是 SVN（Subversion）的安装与基本使用步骤简述：

    一、安装

    * Windows

        - 下载安装包：https://www.visualsvn.com/downloads/
        - 安装 TortoiseSVN（图形化工具）或命令行版 SlikSVN
        - 安装后重启资源管理器即可

    * Linux (Ubuntu/Debian)

        ```bash
        sudo apt update
        sudo apt install subversion
        ```

    * Linux (CentOS/RHEL)

        ```bash
        sudo yum install subversion
        ```

    * macOS

        ```bash
        brew install subversion
        ```

    二、基本使用

    1. 检出（Checkout）仓库

        ```bash
        svn checkout http://svn服务器地址/项目目录
        # 或简写
        svn co http://svn服务器地址/项目目录
        ```

    2. 添加新文件

        ```bash
        svn add 文件名
        ```

    3. 提交修改

        ```bash
        svn commit -m "提交说明"
        # 或简写
        svn ci -m "提交说明"
        ```

    4. 更新本地代码

        ```bash
        svn update
        # 或简写
        svn up
        ```

    5. 查看状态

        ```bash
        svn status
        # 或简写
        svn st
        ```

    6. 查看日志

        ```bash
        svn log
        ```

    7. 撤销本地修改

        ```bash
        # 撤销未提交的文件修改
        svn revert 文件名

        # 撤销整个目录
        svn revert -R 目录名
        ```

    8. 解决冲突

        ```bash
        # 编辑冲突文件后标记为已解决
        svn resolve --accept working 文件名
        ```

    三、典型工作流程

    ```bash
    # 1. 第一次获取代码
    svn checkout http://svn.example.com/myproject

    # 2. 进入目录开始工作
    cd myproject

    # 3. 修改代码后查看变动
    svn status

    # 4. 添加新文件
    svn add newfile.txt

    # 5. 提交前先更新（避免冲突）
    svn update

    # 6. 提交代码
    svn commit -m "修复了登录bug"
    ```

    四、常用命令速查

    | 命令 | 说明 |
    |------|------|
    | `svn checkout` | 检出代码 |
    | `svn update` | 更新代码 |
    | `svn commit` | 提交代码 |
    | `svn add` | 添加新文件 |
    | `svn delete` | 删除文件 |
    | `svn move` | 移动/重命名 |
    | `svn copy` | 复制文件/分支 |
    | `svn log` | 查看历史 |
    | `svn diff` | 查看差异 |
    | `svn info` | 查看仓库信息 |

    > **提示**：SVN 是集中式版本控制，需要确保能访问中央仓库；建议新手先使用 `svn help` 查看完整命令帮助。

* 我看到 svn 工程目录下有这几个文件夹：branches  tags  trunk，代表什么意思？

    这是 SVN 标准的**项目布局**（也称为主干/分支/标签布局），每个文件夹有特定用途：

    1. trunk（主干）

        - **作用**：存放主开发线代码，即最新、最活跃的开发版本
        - **用途**：日常开发都在这里进行，是项目的核心代码库
        - **类比**：像河流的主干道

    2. branches（分支）

        - **作用**：存放从主干分离出来的分支代码
        - **用途**：
          - 开发新功能（功能分支）
          - 修复重大 bug（修复分支）
          - 版本发布前的稳定化工作（发布分支）
          - 实验性开发
        - **类比**：从主干分出来的支流

    3. tags（标签）

        - **作用**：存放里程碑版本（只读快照）
        - **用途**：
          - 记录正式发布的版本（如 v1.0、v2.0）
          - 记录重要的开发阶段节点
          - 作为备份点，便于回溯
        - **类比**：给河流某一段拍个照片，永久保存

    **典型工作流程示例**

    ```bash
    # 1. 日常开发在 trunk
    svn checkout http://svn.example/project/trunk

    # 2. 开发新功能时创建分支
    svn copy http://svn.example/project/trunk \
             http://svn.example/project/branches/feature-login \
             -m "创建登录功能分支"

    # 3. 功能完成后合并回 trunk
    cd trunk
    svn merge http://svn.example/project/branches/feature-login
    svn commit -m "合并登录功能"

    # 4. 发布 v1.0 时打标签
    svn copy http://svn.example/project/trunk \
             http://svn.example/project/tags/release-1.0 \
             -m "发布 1.0 版本"
    ```

    **重要区别**

    | 特性 | trunk | branches | tags |
    |------|-------|----------|------|
    | **是否修改** | 经常修改 | 经常修改 | 一般只读 |
    | **生命周期** | 永久存在 | 临时（完成后可删除） | 永久保存 |
    | **用途** | 日常开发 | 并行开发/隔离变更 | 记录历史版本 |
    | **创建方式** | 项目初始化时 | svn copy | svn copy |

    **最佳实践**

    1. **trunk 保持稳定**：确保 trunk 随时可以编译运行
    2. **分支及时清理**：合并完成后的分支及时删除
    3. **标签永不修改**：tags 一旦创建就不再修改（就像照片）
    4. **分支命名规范**：`feature/xxx`、`bugfix/xxx`、`release/xxx`

    这种布局不是强制的，但已成为 SVN 社区的事实标准，几乎所有 SVN 项目都采用这种结构。

