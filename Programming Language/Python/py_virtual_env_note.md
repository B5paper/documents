# Python virtual env 虚拟环境

## cache

* pyenv 安装 python 3.12.12 时，需要额外安装一些 libxxx-dev 的库。pyenv 安装的 python 都是现场编译的吗？

    可能用到的依赖：

    ```bash
    sudo apt update
    sudo apt install -y build-essential zlib1g-dev libncurses5-dev \
    libgdbm-dev libnss3-dev libssl-dev libreadline-dev \
    libffi-dev libsqlite3-dev libbz2-dev liblzma-dev
    ```

* pyenv 安装后的提示

    ```
    WARNING: seems you still have not added 'pyenv' to the load path.

    # Load pyenv automatically by appending
    # the following to 
    # ~/.bash_profile if it exists, otherwise ~/.profile (for login shells)
    # and ~/.bashrc (for interactive shells) :

    export PYENV_ROOT="$HOME/.pyenv"
    [[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"
    eval "$(pyenv init - bash)"

    # Restart your shell for the changes to take effect.

    # Load pyenv-virtualenv automatically by adding
    # the following to ~/.bashrc:

    eval "$(pyenv virtualenv-init -)"

    ```

* Python创建虚拟环境方法总结

    在Python中创建虚拟环境有多种方法，以下是几种常用的方式：

    1. 使用 venv（Python 3.3+ 推荐）

        ```python
        # 在命令行中执行，不是在Python脚本中
        python -m venv myenv
        ```

        激活虚拟环境：

        * Windows:

            ```cmd
            myenv\Scripts\activate
            ```

        * macOS/Linux:

            ```bash
            source myenv/bin/activate
            ```

        停用虚拟环境：

        ```bash
        deactivate
        ```

    2. 使用 virtualenv

        首先安装 virtualenv：

        ```bash
        pip install virtualenv
        ```

        创建虚拟环境：

        ```bash
        virtualenv myenv
        ```

        激活方式与 venv 相同。

    3. 使用 conda（如果你安装了Anaconda/Miniconda）

        ```bash
        conda create --name myenv python=3.9
        ```

        激活：

        ```bash
        conda activate myenv
        ```

        停用：

        ```bash
        conda deactivate
        ```

    4. 使用 pipenv

        安装 pipenv：

        ```bash
        pip install pipenv
        ```

        创建虚拟环境：

        ```bash
        pipenv install
        ```

        激活：

        ```bash
        pipenv shell
        ```

    5. 使用 poetry

        安装 poetry：

        ```bash
        pip install poetry
        ```

        创建新项目（包含虚拟环境）：

        ```bash
        poetry new myproject
        cd myproject
        poetry install
        ```

        激活：

        ```bash
        poetry shell
        ```

    实际使用示例

    ```bash
    # 1. 创建虚拟环境
    python -m venv myproject_env

    # 2. 激活虚拟环境
    # Windows:
    myproject_env\Scripts\activate
    # macOS/Linux:
    source myproject_env/bin/activate

    # 3. 安装包
    pip install requests pandas numpy

    # 4. 查看已安装的包
    pip list

    # 5. 导出依赖
    pip freeze > requirements.txt

    # 6. 停用虚拟环境
    deactivate
    ```

    **在脚本中管理虚拟环境**

    虽然通常虚拟环境在命令行中管理，但你也可以在Python脚本中检查：

    ```python
    import sys
    import os

    def check_venv():
        # 检查是否在虚拟环境中
        if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
            print("在虚拟环境中运行")
            print(f"Python路径: {sys.executable}")
        else:
            print("不在虚拟环境中运行")

    check_venv()
    ```

    **最佳实践建议**

    * 为每个项目创建独立的虚拟环境

    * 在虚拟环境中安装项目依赖

    * 使用 requirements.txt 记录依赖

    * 不要将虚拟环境文件夹提交到版本控制

    推荐使用 venv，因为它是Python标准库的一部分，不需要额外安装。

* 简述 python如何创建虚拟python环境

    在 Python 中创建虚拟环境主要有以下几种方法：

    1. 使用 venv（Python 3.3+ 内置）

        ```bash
        # 创建虚拟环境
        python -m venv myenv

        # 激活虚拟环境
        # Windows:
        myenv\Scripts\activate
        # Linux/Mac:
        source myenv/bin/activate

        # 停用虚拟环境
        deactivate
        ```

    2. 使用 virtualenv（需安装）

        ```bash
        # 安装 virtualenv
        pip install virtualenv

        # 创建虚拟环境
        virtualenv myenv

        # 指定 Python 版本
        virtualenv -p python3.9 myenv
        ```

    3. 使用 conda（Anaconda/Miniconda）

        ```bash
        # 创建虚拟环境
        conda create --name myenv python=3.9

        # 激活环境
        conda activate myenv

        # 停用环境
        conda deactivate
        ```

    4. 使用 pipenv

        ```bash
        # 安装 pipenv
        pip install pipenv

        # 创建虚拟环境并安装包
        pipenv install requests

        # 进入虚拟环境
        pipenv shell
        ```

    最佳实践建议：

    * 推荐使用 venv：Python 3.3+ 自带，无需额外安装

    * requirements.txt 管理依赖：

        ```bash
        # 导出依赖
        pip freeze > requirements.txt

        # 安装依赖
        pip install -r requirements.txt
        ```

    * .gitignore 中添加虚拟环境目录：

        ```text
        myenv/
        venv/
        .venv/
        env/
        ```

    虚拟环境的主要作用是隔离项目依赖，避免不同项目之间的包版本冲突。

* venv 创建的虚拟环境会继承创建时使用的 Python 解释器版本, 无法在创建后更改 Python 版本：一旦创建，Python 版本就固定了

* 使用 virtualenv -p python3.9 myenv 指定 python 版本时，如果版本不存在，那么会自动安装吗

    不会，virtualenv 不会自动安装 Python 版本。

    virtualenv -p 的工作原理

    * 只接受已安装的 Python 解释器

    * 需要你提前安装好指定版本的 Python

    * 如果版本不存在，会报错

    实际示例

    ```bash
    # 假设系统没有安装 python3.9
    virtualenv -p python3.9 myenv

    # 报错信息：
    RuntimeError: failed to find interpreter for Builtin discover of python_spec='python3.9'

    # 或者：
    The executable python3.9 does not exist
    ```

    正确的工作流程

    1. 先检查 Python 是否已安装

        ```bash
        # 检查 Python 3.9 是否存在
        python3.9 --version
        # 或者
        which python3.9

        # 列出所有已安装的 Python 版本
        ls /usr/bin/python*
        # Windows: 检查 C:\Python39\ 等目录
        ```

    2. 如果没有，先安装 Python

        Ubuntu/Debian

        ```bash
        sudo apt update
        sudo apt install python3.9 python3.9-venv
        ```

        macOS (Homebrew)

        ```bash
        # 安装指定版本
        brew install python@3.9

        # 查看安装路径
        brew list python@3.9
        ```

        Windows

        * 从官网下载 Python 3.9 安装包

        * 或使用 Chocolatey: choco install python39

    3. 指定完整路径创建虚拟环境

        ```bash
        # 找到 Python 3.9 的完整路径
        which python3.9
        # 输出可能是: /usr/bin/python3.9 或 /opt/homebrew/bin/python3.9

        # 使用完整路径
        virtualenv -p /usr/bin/python3.9 myenv

        # 或者使用 which 命令
        virtualenv -p $(which python3.9) myenv
        ```

    自动管理 Python 版本的解决方案

    1. 使用 pyenv（自动安装 Python 版本）

        ```bash
        # 安装 pyenv
        curl https://pyenv.run | bash

        # 查看可安装的 Python 版本
        pyenv install --list

        # 自动下载安装 Python 3.9
        pyenv install 3.9.0

        # 创建虚拟环境（pyenv 管理版本，virtualenv/venv 管理环境）
        pyenv local 3.9.0
        python -m venv myenv
        ```

    2. 使用 conda（最方便）

        ```bash
        # conda 会自动下载安装指定版本的 Python
        conda create -n myenv python=3.9

        # 甚至指定小版本
        conda create -n myenv python=3.9.13
        ```

    3. 使用 asdf

        ```bash
        # 安装 Python 插件
        asdf plugin-add python

        # 安装指定版本
        asdf install python 3.9.0

        # 设置为当前目录使用的版本
        asdf local python 3.9.0
        ```

    最佳实践总结

    ```bash
    # 方案1：手动管理（适合固定环境）
    # 1. 先安装 Python 3.9
    # 2. 再创建虚拟环境
    virtualenv -p python3.9 myenv

    # 方案2：使用 pyenv（推荐，自动化）
    # 1. 安装 pyenv
    # 2. 让 pyenv 安装和管理 Python 版本
    pyenv install 3.9.0
    pyenv local 3.9.0
    python -m venv myenv

    # 方案3：使用 conda（数据科学/机器学习项目）
    # 1. 安装 Anaconda/Miniconda
    # 2. 一条命令搞定
    conda create -n myenv python=3.9
    ```

    关键点：virtualenv 只是一个虚拟环境创建工具，不负责 Python 解释器的安装。你需要先确保所需版本的 Python 已经安装在系统中。

* venv

    venv 是 Python 内置的虚拟环境工具，用于创建独立的 Python 运行环境。

    doc: <https://docs.python.org/3/library/venv.html>

    **基本用法**

    1. 创建虚拟环境

        ```bash
        # 在当前目录创建名为 my_env 的虚拟环境
        python -m venv venv

        # 指定 Python 版本创建
        python3.9 -m venv my_env

        # 创建包含系统site-packages的虚拟环境
        python -m venv --system-site-packages my_env
        ```

        注：

        1. venv 会在当前目录下创建一个叫`my_env`的文件夹，并把所有虚拟环境相关的文件都放到这个文件夹下

        1. venv 不能安装新的 python 版本，只能用现有 python 版本。

    2. 激活虚拟环境

        Windows:

        ```bash
        my_env\Scripts\activate
        ```

        Linux/Mac:
        
        ```bash
        source my_env/bin/activate
        ```

        激活后，终端提示符会显示环境名：`(my_env) $`

    3. 停用虚拟环境

        ```bash
        deactivate
        ```

    **常用命令**

    * 查看当前环境

        ```bash
        # 查看Python解释器路径
        which python  # Linux/Mac
        where python  # Windows

        # 查看已安装包
        pip list
        ```

    * 管理依赖

        ```bash
        # 安装包
        pip install package_name

        # 从requirements.txt安装
        pip install -r requirements.txt

        # 生成requirements.txt
        pip freeze > requirements.txt
        ```

    * 删除虚拟环境

        ```bash
        # 只需删除对应的文件夹即可
        rm -rf venv/  # Linux/Mac
        rmdir /s venv # Windows
        ```

    **在IDE中使用**

    * VSCode: 选择解释器路径（Ctrl+Shift+P → "Python: Select Interpreter"）

    * PyCharm: File → Settings → Project → Python Interpreter

    * Jupyter: !python -m ipykernel install --user --name=venv

    **注意事项**

    * 每个项目使用独立环境，避免包冲突

    * 不要将 venv 目录提交到版本控制（添加到.gitignore）

    * 虚拟环境不包含Python解释器副本，只是软链接

    * 可同时激活多个虚拟环境（通过设置环境变量）

    **快捷命令**

    ```bash
    # 创建并激活虚拟环境（Linux/Mac）
    python -m venv venv && source venv/bin/activate

    # 创建带常用工具的虚拟环境
    python -m venv venv --prompt="myenv" && source venv/bin/activate
    pip install ipython black isort flake8
    ```

    这就是 venv 的基本用法，简单高效地管理 Python 项目依赖！

    venv 的核心机制: 轻量级链接（不是完整安装）
    
    ```bash
    # pyvenv.cfg 文件内容示例
    home = /usr/bin              # 指向基础解释器
    include-system-site-packages = false
    version = 3.9.7
    ```

    * Python 解释器：软链接到系统 Python（Unix）或复制可执行文件（Windows）

    * site-packages：完全独立的空目录

    * 标准库：通过路径重定向复用基础 Python 的

* pyenv

    Pyenv 是一个 Python 版本管理工具，支持安装多个 Python 版本并在不同版本间切换，适合开发测试不同环境下的 Python 项目。

    github repo: <https://github.com/pyenv/pyenv>

    一、安装 Pyenv

    使用官方提供的安装脚本（支持 Linux/macOS）：

    ```bash
    curl https://pyenv.run | bash
    ```

    或使用：

    ```bash
    curl -L https://github.com/pyenv/pyenv-installer/raw/master/bin/pyenv-installer | bash
    ```

    二、配置 Shell 环境

    将以下内容添加到 Shell 配置文件（如 ~/.bashrc、~/.zshrc）：

    ```bash
    export PATH="$HOME/.pyenv/bin:$PATH"
    eval "$(pyenv init --path)"
    eval "$(pyenv virtualenv-init -)"  # 如使用 virtualenv 插件
    ```

    重启终端或执行：

    ```bash
    source ~/.bashrc
    ```

    三、基本使用命令

    1. 查看可用版本
    
        ```bash
        pyenv install --list          # 列出所有可安装版本
        pyenv versions                # 查看已安装版本（当前使用的版本前有 * 标记）
        pyenv version                 # 查看当前使用的版本
        ```

        注：

        1. 这里的 version 指的不是 pyenv 版本，是 python 的版本

    2. 安装 Python 版本

        ```bash
        pyenv install 3.9.7          # 安装指定版本
        pyenv install 3.10.2 -v      # -v 显示详细安装日志
        ```

    3. 切换 Python 版本

        ```bash
        pyenv global 3.9.7           # 全局切换
        pyenv local 3.8.12           # 在当前目录切换（生成 .python-version 文件）
        pyenv shell 3.7.11           # 当前 Shell 会话临时切换
        ```

    4. 卸载版本

        ```bash
        pyenv uninstall 3.8.0        # 卸载指定版本
        ```

    5. 设置镜像加速安装（适用于国内）

        ```bash
        # 在 ~/.bashrc 中添加
        export PYTHON_BUILD_MIRROR_URL="https://mirrors.aliyun.com/python/"
        ```

    四、虚拟环境管理（pyenv-virtualenv 插件）

    创建虚拟环境：

    ```bash
    pyenv virtualenv 3.9.7 myenv-3.9.7
    ```

    激活/切换虚拟环境：

    ```bash
    pyenv activate myenv-3.9.7
    pyenv deactivate
    ```

    删除虚拟环境：

    ```bash
    pyenv virtualenv-delete myenv-3.9.7
    ```

    五、常用场景示例

    为不同项目设置独立版本：

    ```bash
    cd ~/projectA
    pyenv local 3.8.0   # 该项目目录下使用 Python 3.8.0
    cd ~/projectB
    pyenv local 3.10.2  # 该项目目录下使用 Python 3.10.2
    ```

    六、注意事项

    * 安装 Python 前需确保系统有编译依赖（如 zlib、readline 等）。

    * 通过 pyenv local 设置的版本优先级高于 global。

    * 可配合 pipenv 或 poetry 进行更精细的依赖管理。

    总结：Pyenv 通过简单的命令实现了多版本 Python 的隔离管理，是 Python 开发环境配置的实用工具。

* pyenv 会把 python 安装在哪里

    Pyenv 将 Python 版本安装在以下目录中：

    一、安装位置

    1. 主目录（默认）

        ```bash
        # Linux/macOS:
        ~/.pyenv/versions/

        # 示例路径：
        ~/.pyenv/versions/3.9.7/
        ~/.pyenv/versions/3.10.2/
        ~/.pyenv/versions/3.11.0/
        ```

    2. 结构示例

        ```text
        ~/.pyenv/
        ├── versions/           # 所有安装的 Python 版本
        │   ├── 3.9.7/
        │   │   ├── bin/
        │   │   ├── lib/
        │   │   └── include/
        │   ├── 3.10.2/
        │   └── myenv-3.9.7/   # 虚拟环境也在此目录
        ├── shims/             # 垫片文件（pyenv 管理）
        ├── cache/             # 下载的源码包缓存
        └── plugins/           # 插件目录
        ```

    二、验证安装位置

    1. 查看具体路径

        ```bash
        # 查看 Python 解释器路径
        pyenv which python
        # 输出示例：/home/username/.pyenv/versions/3.9.7/bin/python

        # 查看 pip 路径
        pyenv which pip
        ```

    2. 查看版本目录

        ```bash
        # 列出所有已安装版本及其路径
        pyenv versions --path
        # 或
        ls -la ~/.pyenv/versions/
        ```

    三、虚拟环境位置

    1. 虚拟环境也存储在 versions 目录

        ```bash
        # 创建虚拟环境
        pyenv virtualenv 3.9.7 myproject-env

        # 虚拟环境路径
        ~/.pyenv/versions/myproject-env/
        ```

    四、工作原理

    * 垫片（Shims）机制

        * shims 目录：~/.pyenv/shims/

        * 作用：pyenv 在 PATH 最前面插入 shims 目录

        * 执行流程：
        
            ```text
            执行 python 命令 → shims/python → pyenv → 选择对应版本 → 执行真实解释器
            ```

    查看 PATH 验证

    ```bash
    echo $PATH
    # 输出示例：/home/username/.pyenv/shims:/usr/local/bin:/usr/bin:...
    ```

    五、与系统 Python 的关系

    1. 系统 Python（不受 pyenv 管理）

        ```bash
        # 系统默认位置
        /usr/bin/python3    # Linux
        /usr/local/bin/python3  # macOS (Homebrew)
        ```

    2. 隔离性

        pyenv 安装的 Python 与系统 Python 完全隔离

        每个版本独立安装，不共享包

        切换版本不会影响系统 Python

    六、配置自定义安装位置（不推荐）

    环境变量控制

    ```bash
    # 设置安装前缀（编译时）
    export PYENV_ROOT=/custom/path
    # 之后安装的版本会存放到 /custom/path/versions/
    ```

    七、清理与维护

    1. 删除版本

        ```bash
        # 删除指定版本（会从 ~/.pyenv/versions/ 移除）
        pyenv uninstall 3.8.0

        # 手动删除（效果相同）
        rm -rf ~/.pyenv/versions/3.8.0
        ```

    2. 查看磁盘占用

        ```bash
        du -sh ~/.pyenv/versions/
        ```

    总结

    * 主要安装位置：~/.pyenv/versions/

    * 虚拟环境位置：同目录下，以环境名命名

    * 完全用户级：不需要 sudo，不污染系统目录

    * 版本独立：每个 Python 版本都有完整独立的文件结构

    这样设计的优势是安全、可隔离、便于管理多个版本，且不会影响系统自带的 Python。

    **venv 与 pyenv 的关键区别**
    
    方面	venv	pyenv
    Python版本	不安装新版本，复用现有的 Python	可以安装多个 Python 版本
    site-packages	每个虚拟环境独立一份	每个 Python 版本全局共享一份
    存储位置	项目目录内（可自定义）	~/.pyenv/versions/
    激活方式	source myenv/bin/activate	pyenv local/global/shell

## note
