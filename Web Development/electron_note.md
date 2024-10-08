# Electron Note

## cache

* electron create hello world program steps

    1. 安装 nvm, nodejs, npm

        <https://nodejs.org/en/download/package-manager>

        照着安装步骤执行即可。这一步可能需要用到代理。似乎有的代理配置只认`HTTP_PROXY`, `HTTPS_PROXY`，不认`http_proxy`, `https_proxy`。

    2. 创建工程文件夹，安装 electron

        ref: <https://www.electronjs.org/docs/latest/tutorial/tutorial-first-app>

        ```bash
        mkdir my-electron-app && cd my-electron-app
        npm init
        npm install electron --save-dev
        ```

        配置 electron:

        `package.json`:

        ```json
        {
            "name": "my-electron-app",
            "version": "1.0.0",
            "description": "Hello World!",
            "main": "main.js",
            "scripts": {
                "start": "electron .",
                "test": "echo \"Error: no test specified\" && exit 1"
            },
            "author": "Jane Doe",
            "license": "MIT",
            "devDependencies": {
                "electron": "23.1.3"
            }
        }
        ```

    3. 创建`main.js`文件

        `main.js`:

        ```js
        console.log('Hello from Electron 👋')
        ```

    4. 启动程序

        `npm run start`

        output:

        ```
        (base) hlc@hlc-VirtualBox:~/Documents/Projects/my-electron-app$ npm run start

        > my-electron-app@1.0.0 start
        > electron .

        Hello from Electron 👋

        ```

* 使用`apt install nodejs`安装的 nodejs 版本太旧，无法成功安装 electron

## note

(empty)