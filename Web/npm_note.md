## cache

* 详细讲讲 npm 安装包到不同位置的模式

    **npm 包安装模式全解析：本地 vs 全局 + 依赖类型详解**

    npm 提供了**两种核心安装位置模式**（本地/全局）和**五种依赖类型**，它们决定了包的存储位置、使用范围和生命周期。理解这些模式是正确管理 Node.js 项目依赖的基础。

    一、核心原则（npm 官方推荐）

    > **Install it locally if you're going to `require()` it. Install it globally if you're going to run it on the command line.**
    > 如果你要在代码中 `require()` 它，就本地安装；如果你要在命令行运行它，就全局安装。

    二、本地安装（Local Installation）

    **默认模式**，也是绝大多数情况下应该使用的模式。

    1. 基本信息

        - **安装命令**：`npm install <package-name>`（npm 5+ 自动保存到 `dependencies`）
        - **存储位置**：当前项目根目录下的 `./node_modules` 文件夹
        - **可执行文件位置**：`./node_modules/.bin`（包提供的 CLI 命令会放在这里）

    2. 可执行文件的三种使用方式

        1. **通过 npx 临时执行**（推荐）：
           ```bash
           npx webpack --version  # 自动查找本地 .bin 目录下的命令
           ```

        2. **通过 package.json scripts 执行**：
           ```json
           {
             "scripts": {
               "build": "webpack --mode production"
             }
           }
           ```
           运行 `npm run build` 时，npm 会自动将 `./node_modules/.bin` 加入 PATH 环境变量。

        3. **直接调用完整路径**（不推荐）：
           ```bash
           ./node_modules/.bin/webpack --version
           ```

    3. 本地安装的两大细分类型

        npm 将本地依赖分为**生产依赖**和**开发依赖**，它们的安装行为和生命周期完全不同。

        | 依赖类型 | 安装命令 | 记录位置 | 适用场景 | 生产环境是否安装 |
        |---------|---------|---------|---------|----------------|
        | 生产依赖<br>`dependencies` | `npm install <pkg>`<br>`npm install <pkg> --save-prod` | `package.json` 的 `dependencies` 字段 | 项目运行时必需的库<br>如：react、express、lodash | ✅ 是（默认） |
        | 开发依赖<br>`devDependencies` | `npm install <pkg> --save-dev`<br>`npm install <pkg> -D` | `package.json` 的 `devDependencies` 字段 | 仅开发/构建/测试需要的工具<br>如：webpack、jest、eslint | ❌ 否（使用 `--production` 标志时） |

        **生产环境安装命令**：
        ```bash
        npm install --production  # 只安装 dependencies，跳过 devDependencies
        npm ci --production       # 基于 package-lock.json 精确安装生产依赖
        ```

    三、全局安装（Global Installation）

    将包安装到系统级目录，所有项目共享同一个版本。

    1. 基本信息

        - **安装命令**：`npm install -g <package-name>`（`-g` 是 `--global` 的简写）
        - **不记录**在任何项目的 `package.json` 中
        - **核心用途**：安装需要在任何目录下都能执行的**命令行工具**

    2. 全局安装的默认路径（按系统/安装方式）

        npm 使用 `prefix` 配置项确定全局安装的根目录，可通过 `npm prefix -g` 查看当前值。

        | 环境 | 全局前缀（prefix） | 包代码位置 | 可执行文件位置 |
        |-----|-------------------|-----------|---------------|
        | Windows（独立安装） | `%APPDATA%\npm` | `%APPDATA%\npm\node_modules` | `%APPDATA%\npm`（已加入系统 PATH） |
        | macOS/Linux（独立安装） | `/usr/local` | `/usr/local/lib/node_modules` | `/usr/local/bin`（已加入系统 PATH） |
        | macOS/Linux（使用 nvm） | `~/.nvm/versions/node/<node-version>` | `~/.nvm/versions/node/<version>/lib/node_modules` | `~/.nvm/versions/node/<version>/bin` |

    3. 权限问题与解决方案

        在 macOS/Linux 系统中，向 `/usr/local` 安装全局包通常需要 `sudo` 权限，这可能导致文件权限问题。**推荐解决方案**：
        1. 使用 Node.js 版本管理器（如 nvm、n），它会将 Node.js 和 npm 安装在用户目录下，无需 sudo
        2. 手动修改 npm 全局前缀到用户目录：
           ```bash
           # 创建自定义全局目录
           mkdir ~/.npm-global

           # 配置 npm 使用该目录作为全局前缀
           npm config set prefix ~/.npm-global

           # 将该目录的 bin 文件夹加入系统 PATH（添加到 ~/.bashrc 或 ~/.zshrc）
           export PATH=~/.npm-global/bin:$PATH
           ```

    4. 全局安装的适用场景与禁忌

        ✅ **应该全局安装的包**：
        - npm 本身：`npm install -g npm@latest`
        - 项目脚手架工具：`create-react-app`、`vue-cli`、`vite`
        - 通用命令行工具：`nodemon`、`pm2`、`eslint`（全局用于快速检查单个文件）

        ❌ **绝对不要全局安装的包**：
        - 任何需要在代码中 `require()` 的库（如 react、lodash）
        - 项目特定的构建工具或依赖

        **最佳实践**：优先使用 `npx` 临时执行全局工具，避免全局安装过时版本：
        ```bash
        npx create-react-app my-app  # 临时安装最新版 create-react-app 并执行
        ```

    四、本地 vs 全局：核心区别对比表

    | 对比维度 | 本地安装 | 全局安装 |
    |---------|---------|---------|
    | 安装命令 | `npm install <pkg>` | `npm install -g <pkg>` |
    | 存储位置 | 项目内 `./node_modules` | 系统级全局目录 |
    | 影响范围 | 仅当前项目可用 | 系统所有项目和终端可用 |
    | 版本隔离 | 每个项目可以有自己的版本 | 所有项目共享同一个版本 |
    | 记录位置 | `package.json` 的 dependencies/devDependencies | 不记录在任何项目文件中 |
    | 可移植性 | 克隆项目后 `npm install` 即可恢复所有依赖 | 需要手动安装全局依赖 |
    | 适用场景 | 项目运行或开发所需的所有库和工具 | 跨项目通用的命令行工具 |

    五、其他依赖类型的安装行为

    除了基础的生产和开发依赖，npm 还支持三种特殊依赖类型，它们的安装行为各有不同。

    1. 对等依赖（peerDependencies）

        **定义**：声明"我的包需要宿主项目提供某个版本的依赖"，不自动安装（npm 7+ 会尝试自动安装）。

        **核心作用**：解决插件与宿主框架的版本兼容问题，避免重复安装和实例冲突。

        **适用场景**：组件库、插件系统
        - 一个 Vue 3 组件库需要宿主项目安装 `vue@^3.0.0`
        - 一个 Babel 插件需要宿主项目安装 `@babel/core@^7.0.0`

        **示例**：
        ```json
        {
          "name": "my-vue-components",
          "peerDependencies": {
            "vue": "^3.0.0",
            "vue-router": "^4.0.0"
          },
          "peerDependenciesMeta": {
            "vue-router": {
              "optional": true  // 将 vue-router 标记为可选对等依赖
            }
          }
        }
        ```

        **npm 版本行为差异**：
        - npm 6 及以下：仅在版本不兼容时发出警告，不自动安装
        - npm 7+：默认自动安装 peerDependencies，版本冲突时会抛出错误

    2. 可选依赖（optionalDependencies）

        **定义**：即使安装失败也不会导致整个 `npm install` 过程失败的依赖。

        **适用场景**：跨平台依赖、增强性功能依赖
        - `fsevents`：仅在 macOS 系统上可用的文件系统监控库
        - 某些可选的性能优化库

        **安装命令**：`npm install <pkg> --save-optional`

        **特点**：
        - 安装失败时 npm 会跳过并继续安装其他依赖
        - 代码中需要使用 `try-catch` 包裹对可选依赖的引用
        - 会覆盖 `dependencies` 中的同名依赖，不要同时在两个地方声明

    3. 捆绑依赖（bundledDependencies / bundleDependencies）

        **定义**：发布 npm 包时，会将这些依赖一起打包到最终的 tarball 中。

        **适用场景**：
        - 修复第三方依赖的 bug 但无法提交 PR
        - 依赖未发布到 npm 公共仓库
        - 确保依赖的精确版本不会被意外更新

        **示例**：
        ```json
        {
          "bundledDependencies": [
            "my-private-dep",
            "fixed-third-party-dep"
          ]
        }
        ```

    六、npm 依赖安装的底层机制：扁平化（Hoisting）

    npm 3+ 引入了**依赖扁平化**机制，以解决早期版本中依赖嵌套过深的问题。

    **工作原理**

    - npm 会尽可能将所有依赖提升到项目根目录的 `node_modules` 中
    - 当多个包依赖同一个库的**兼容版本**时，只安装一份在根目录
    - 当出现**版本冲突**时（如 A 依赖 lodash@4，B 依赖 lodash@3），冲突的版本会嵌套安装在各自的 `node_modules` 中

    **示例依赖树**：

    ```
    node_modules/
    ├── lodash@4.17.21  # 根项目和 package-a 共享
    └── package-a/
        └── node_modules/
            └── lodash@3.10.1  # 版本冲突，嵌套安装
    ```

    七、常用相关命令

    **查看已安装的包**

    ```bash
    # 查看本地安装的包（仅显示顶层）
    npm list --depth=0

    # 查看全局安装的包（仅显示顶层）
    npm list -g --depth=0

    # 查看某个特定包的安装位置
    npm ls <package-name>
    npm ls -g <package-name>
    ```

    **卸载包**

    ```bash
    # 卸载本地包（同时从 package.json 移除）
    npm uninstall <package-name>

    # 卸载全局包
    npm uninstall -g <package-name>
    ```

    **查看和配置 npm 路径**

    ```bash
    # 查看本地项目根目录
    npm prefix

    # 查看全局前缀
    npm prefix -g

    # 查看 npm 缓存位置
    npm config get cache
    ```

    八、最佳实践与常见误区

    1. **优先本地安装**：99% 的依赖都应该本地安装，确保项目的可移植性和版本隔离
    2. **最小化全局安装**：只安装必须全局使用的命令行工具，优先使用 `npx`
    3. **正确区分依赖类型**：严格将开发工具放在 `devDependencies`，运行时依赖放在 `dependencies`
    4. **不要提交 node_modules 到版本控制**：通过 `package.json` 和 `package-lock.json` 管理依赖
    5. **使用 package-lock.json**：确保团队所有成员和生产环境安装完全相同的依赖版本
    6. **避免 sudo npm install -g**：使用 nvm 或自定义全局前缀解决权限问题

    需要我补充一个**常见包安装模式速查表**，或者讲解一下 `npm link` 本地开发调试模式的用法吗？

