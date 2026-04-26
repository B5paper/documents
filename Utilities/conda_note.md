# Conda

搜索某个包：`conda search -f <package_name>`

列出已经安装的所有的包：`conda list`

列出所有可安装的包：`conda search`

安装一个特定版本的包：`conda install pandas=1.0.2`

1. python 3.4 安装不上

    似乎是因为较老的包都放到了 conda-forge channel 里。

    Ref: <https://stackoverflow.com/questions/56850972/why-conda-cannot-create-environment-with-python-3-4-installed-in-it>

## cache

* conda 如何删除一个环境

    删除 Conda 环境的操作非常直接，但在 Cygwin 环境下建议先 **deactivate**（退出）当前环境，以确保没有任何进程正在占用该文件夹。

    **标准删除命令**

    使用 `--name` (或 `-n`) 参数指定环境名称：

    ```bash
    conda remove --name 环境名称 --all
    ```

    或者使用路径（如果你知道环境的具体安装路径）：

    ```bash
    conda remove --prefix /path/to/env --all
    ```

    ---

    **操作建议与注意事项**

    * **确认环境列表**：在删除之前，建议先运行以下命令确认环境的准确名称：
        ```bash
        conda env list
        ```
    * **退出当前环境**：如果你正处于要删除的环境中，请先退出：
        ```bash
        conda deactivate
        ```
    * **参数解释**：
        * `--name`：指定要操作的环境名。
        * `--all`：这个参数至关重要，它告诉 Conda 删除该环境下的**所有**包及该环境目录本身。

    ---

    **Cygwin 特有的“小烦恼”**

    在 Cygwin 中删除环境时，偶尔会遇到 **Permission Denied**（拒绝访问）的问题。这通常是因为：
    1.  某个 Cygwin 进程（如 Python 解释器）仍在后台运行。
    2.  Windows 的文件权限与 Cygwin 的模拟权限发生了冲突。

    **解决办法**：
    如果命令报错，尝试关闭所有 Cygwin 终端，新开一个窗口，直接运行删除命令。如果依然失败，可以在 Windows 的资源管理器中手动删除该环境对应的文件夹。

    你需要删除的是一个用来测试的临时环境，还是打算彻底清理不再使用的重型项目环境？

* conda 如何 reset base 环境

    由于 `base` 环境是 Conda 的根环境，它**不能**像普通环境那样直接用 `conda remove --all` 删除。重置 `base` 环境通常意味着将其恢复到最初安装时的状态。

    在 Cygwin 中操作时，请务必确保你没有在其他窗口运行属于 `base` 环境的 Python 进程。

    ---

    **方法一：使用 Conda Revision（最优雅）**

    Conda 会记录环境变更的历史，你可以利用这个功能“回滚”到最初状态。

    1.  **查看历史版本：**
        ```bash
        conda list --revisions
        ```
        你会看到编号为 `0` 的初始安装记录。

    2.  **恢复到初始状态：**
        ```bash
        conda install --revision 0
        ```
        > **注意**：这会卸载你后来安装的所有包，并将自带包恢复到初始版本。

    ---

    **方法二：强制克隆覆盖（最彻底）**

    如果你觉得 `base` 环境已经彻底“坏了”，连 `revision` 都无法修复，可以使用以下暴力手段。

    1.  **从基础克隆：**
        你可以通过安装包重新同步 `base`。但由于 `base` 不能自杀，一种常见的变通方法是运行：
        ```bash
        # 强制重新安装所有基础包
        conda install --force-reinstall anaconda  # 如果是 Anaconda
        # 或者
        conda install --force-reinstall miniconda # 如果是 Miniconda
        ```

    2.  **清理孤儿包：**
        重置后，清理不再需要的缓存和残留文件：
        ```bash
        conda clean --all
        ```

    ---

    **方法三：终极手段（手动清理）**

    如果上述命令在 Cygwin 中报错（常见于 Windows 路径锁定），最干净的方法是：
    1.  导出你需要的其他环境：`conda env export > env.yaml`。
    2.  **直接卸载并重装** Miniconda/Anaconda。
        * 在 Windows 控制面板卸载。
        * 手动删除 `C:\Users\用户名\anaconda3` 文件夹。
        * 重新安装。

    ---

    **避坑小建议**

    在 Cygwin/Bash 中使用 Conda，为了避免再次把 `base` 搞乱，强烈建议遵循**“不动 base”**原则：

    > **最佳实践：** 始终创建一个新环境来工作。
    > ```bash
    > conda create -n my_env python=3.10
    > conda activate my_env
    > ```
    > 这样即使 `my_env` 崩了，你只需要 `conda remove -n my_env --all` 即可，完全不伤元气。

    你现在是因为 `base` 环境出现了依赖冲突报错，还是单纯想清理一下空间？

