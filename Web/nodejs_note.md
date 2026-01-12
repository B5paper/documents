# NodeJS Note

official site: <https://nodejs.dev/>

## cache

* node.js

    官网：<nodejs.org>

    安装 Node.js

    * 访问 nodejs.org 下载 LTS 版本

    * 验证安装：终端输入 node --version 和 npm --version

    基础学习

    1. JavaScript 基础

        掌握 ES6+ 语法（箭头函数、Promise、async/await）

        了解模块化（CommonJS 模块）

    2. 核心模块学习

        ```js
        // 示例：文件操作
        const fs = require('fs');
        const path = require('path');

        // 示例：HTTP 服务器
        const http = require('http');
        const server = http.createServer((req, res) => {
            res.end('Hello Node.js');
        });
        ```

    第一个项目

    * 初始化项目
    
        ```bash
        mkdir my-node-app
        cd my-node-app
        npm init -y
        ```

    * 创建入口文件

        ```javascript
        // index.js
        console.log('Node.js 运行成功！');
        ```

    * 运行程序

        ```bash
        node index.js
        ```

    关键概念掌握
        
    1. 包管理

        使用 npm 或 yarn 管理依赖

        了解 package.json 结构

        学习常用命令：install, run, update

    2. 异步编程

        回调函数 → Promise → async/await 演进

        事件循环机制理解

    3. 模块系统

        CommonJS（require/exports）

        ES Modules（import/export）

    实战练习

    项目建议：

        命令行工具：文件批量重命名

        Web 服务器：静态文件服务器

        API 服务：RESTful API

        数据抓取：简单的爬虫程序

    示例：简单 HTTP 服务器

    ```javascript
    const http = require('http');
    const port = 3000;

    const server = http.createServer((req, res) => {
        res.writeHead(200, { 'Content-Type': 'text/html' });
        res.end('<h1>Hello Node.js!</h1>');
    });

    server.listen(port, () => {
        console.log(`服务器运行在 http://localhost:${port}`);
    });
    ```

    进阶方向

    学习框架

        Express.js：最流行的 Web 框架

        Koa：更轻量、现代的框架

        NestJS：企业级框架

    工具掌握

        Nodemon：开发热重载

        ESLint：代码检查

        Jest/Mocha：单元测试

    学习资源

    官方文档

        Node.js 官方文档

            <https://nodejs.org/docs/>

        npm 文档

            <https://docs.npmjs.com/>

    推荐教程

        Node.js 官方入门教程

        freeCodeCamp Node.js 课程

        《Node.js 实战》

        《深入浅出 Node.js》

    最佳实践

        错误处理：始终处理异步错误

        代码结构：保持模块化

        环境配置：使用 dotenv 管理环境变量

        日志记录：使用 Winston 或 Morgan

        安全性：定期更新依赖，处理敏感数据

* Node.js 搭建简易 HTTP Server

    1. 基本 HTTP Server

        ```javascript
        // server.js
        const http = require('http');

        const server = http.createServer((req, res) => {
        // 设置响应头
        res.writeHead(200, {
            'Content-Type': 'text/plain; charset=utf-8',
            'Access-Control-Allow-Origin': '*' // 允许跨域
        });
        
        // 根据请求路径返回不同内容
        if (req.url === '/api/data') {
            res.end(JSON.stringify({ message: 'Hello from Node.js', data: [1, 2, 3] }));
        } else {
            res.end('Node.js Server is running!');
        }
        });

        const PORT = 3000;
        server.listen(PORT, () => {
        console.log(`Server running at http://localhost:${PORT}`);
        });
        ```

    2. 更完整的示例（支持路由和静态文件）

        ```javascript
        // server.js
        const http = require('http');
        const fs = require('fs');
        const path = require('path');

        const server = http.createServer((req, res) => {
        console.log(`${req.method} ${req.url}`);
        
        // 处理 CORS 预检请求
        if (req.method === 'OPTIONS') {
            res.writeHead(204, {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'GET, POST, PUT, DELETE',
            'Access-Control-Allow-Headers': 'Content-Type'
            });
            return res.end();
        }
        
        // 路由处理
        if (req.url === '/api/users') {
            if (req.method === 'GET') {
            res.writeHead(200, {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            });
            res.end(JSON.stringify([
                { id: 1, name: 'Alice' },
                { id: 2, name: 'Bob' }
            ]));
            } else if (req.method === 'POST') {
            let body = '';
            req.on('data', chunk => body += chunk);
            req.on('end', () => {
                res.writeHead(201, {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
                });
                res.end(JSON.stringify({ 
                message: 'User created', 
                data: JSON.parse(body) 
                }));
            });
            }
        } else if (req.url === '/') {
            // 返回 HTML 文件
            fs.readFile(path.join(__dirname, 'index.html'), (err, content) => {
            res.writeHead(200, { 'Content-Type': 'text/html' });
            res.end(content);
            });
        } else {
            res.writeHead(404);
            res.end('Not Found');
        }
        });

        server.listen(3000, () => {
        console.log('Server running on port 3000');
        });
        ```

* 前端与 Node.js Server 通信方式

    1. 使用 fetch API

        ```html
        <!-- index.html -->
        <!DOCTYPE html>
        <html>
        <head>
            <title>与Node.js通信示例</title>
        </head>
        <body>
            <div id="app">
                <button onclick="getData()">获取数据</button>
                <button onclick="postData()">发送数据</button>
                <div id="result"></div>
            </div>

            <script>
                // GET 请求示例
                async function getData() {
                    try {
                        const response = await fetch('http://localhost:3000/api/users');
                        const data = await response.json();
                        document.getElementById('result').innerHTML = 
                            JSON.stringify(data, null, 2);
                    } catch (error) {
                        console.error('Error:', error);
                    }
                }

                // POST 请求示例
                async function postData() {
                    try {
                        const response = await fetch('http://localhost:3000/api/users', {
                            method: 'POST',
                            headers: {
                                'Content-Type': 'application/json'
                            },
                            body: JSON.stringify({
                                name: 'Charlie',
                                age: 25
                            })
                        });
                        const data = await response.json();
                        document.getElementById('result').innerHTML = 
                            JSON.stringify(data, null, 2);
                    } catch (error) {
                        console.error('Error:', error);
                    }
                }

                // 实时通信示例（EventSource）
                function setupSSE() {
                    const eventSource = new EventSource('http://localhost:3000/api/events');
                    
                    eventSource.onmessage = (event) => {
                        console.log('实时数据:', event.data);
                    };
                    
                    eventSource.onerror = (error) => {
                        console.error('EventSource 错误:', error);
                    };
                }
            </script>
        </body>
        </html>
        ```

    2. 使用 axios（推荐）
    html

    <script src="https://cdn.jsdelivr.net/npm/axios/dist/axios.min.js"></script>
    <script>
        // 配置 axios
        axios.defaults.baseURL = 'http://localhost:3000';
        
        // 请求拦截器
        axios.interceptors.request.use(config => {
            config.headers['Authorization'] = 'Bearer token123';
            return config;
        });
        
        // 响应拦截器
        axios.interceptors.response.use(
            response => response.data,
            error => {
                console.error('请求失败:', error);
                return Promise.reject(error);
            }
        );
        
        // 使用示例
        async function fetchData() {
            try {
                const data = await axios.get('/api/users');
                console.log('用户数据:', data);
                
                // POST 请求
                const result = await axios.post('/api/users', {
                    name: 'David',
                    email: 'david@example.com'
                });
                console.log('创建结果:', result);
            } catch (error) {
                console.error('请求错误:', error);
            }
        }
    </script>

    3. WebSocket 实时通信
    javascript

    // server.js - WebSocket 示例
    const WebSocket = require('ws');
    const wss = new WebSocket.Server({ port: 8080 });

    wss.on('connection', (ws) => {
        console.log('客户端已连接');
        
        // 发送欢迎消息
        ws.send(JSON.stringify({ type: 'welcome', message: 'Connected!' }));
        
        // 接收客户端消息
        ws.on('message', (message) => {
            console.log('收到消息:', message);
            
            // 广播给所有客户端
            wss.clients.forEach(client => {
                if (client.readyState === WebSocket.OPEN) {
                    client.send(JSON.stringify({
                        type: 'message',
                        data: message,
                        timestamp: new Date().toISOString()
                    }));
                }
            });
        });
        
        // 定时发送数据
        const interval = setInterval(() => {
            ws.send(JSON.stringify({
                type: 'update',
                data: { value: Math.random() },
                timestamp: new Date().toISOString()
            }));
        }, 5000);
        
        ws.on('close', () => {
            clearInterval(interval);
            console.log('客户端断开连接');
        });
    });

    运行步骤

        创建项目：

    bash

    mkdir my-server
    cd my-server
    npm init -y

        创建文件：

    text

    my-server/
    ├── server.js
    ├── index.html
    └── package.json

        运行服务器：

    bash

    node server.js

        访问应用：

        浏览器打开：http://localhost:3000

        测试API：http://localhost:3000/api/users

    关键要点

        通信方式：

            RESTful API：最常用的通信方式

            WebSocket：实时双向通信

            Server-Sent Events：服务器推送

            长轮询：兼容性好的实时方案

        注意事项：

            设置正确的 Content-Type 响应头

            处理跨域问题（CORS）

            错误处理和异常捕获

            请求验证和安全性

            性能优化（压缩、缓存等）

        实际开发建议：

            使用 Express.js 等框架简化开发

            使用中间件处理通用逻辑

            实现路由分层和模块化

            添加日志记录和监控

            配置环境变量管理敏感信息

    这样搭建的 HTTP Server 可以满足基本的前后端通信需求，对于更复杂的应用建议使用成熟的框架如 Express、Koa 等。

## topics

## Introduction

hello world program:

(`myfirst.js`)

```js
var http = require('http');

http.createServer(function (req, res) {
  res.writeHead(200, {'Content-Type': 'text/html'});
  res.end('Hello World!');
}).listen(8080);
```

运行：`node myfirst.js`

include a module: `var http = require('http');`

**create your own modules**

Use the `exports` keyword to make properties and methods available outside the module file.

(`myfirstmodule.js`)

```
exports.myDateTime = function() {
    return Date();
};
```

include your own module

```js
var http = require('http');
var dt = require('./myfirstmodule');
http.createServer(function(req, res) {
    res.writeHead(200, {'Content-Type': 'text/html'});
    res.write('The date and time are currently: ' + dt.myDateTime());
    res.end();
}).listen(8080);
```

## npm

NPM is a package manager for `Node.js` packages.

Official site: <https://www.npmjs.com>

Download a package called `upper-case`:

`npm install upper-case`

NPM creates a folder named `node_modules`, where the package will be placed. All pakcages you install in the future will be placed in this folder.

**Package 与 Module 的区别**

package 指的是下面几种情况：

* a) A folder containing a program described by a `package.json` file.

* b) A gzipped tarball containing (a)

* c) A URL that resolves to (b)

* d) A `<name>@<version>` that is published on the registry with (c)

* e) A `<name>@<tag>` that points to (d)

* f) A `<name>` that has a `latest` tag satisfying (e)

* g) A `git` url that, when cloned, results in (a)

module 指的是下面两种情况：

* A folder with a `package.json` file containing a `"main"` field

* A JavaScript file.

通常 module 文件夹会放到`node_modules`目录下，并且可以被 Node.js `require()`函数加载。

更新 npm：`npm install npm@latest -g`

npm 的 packages 按作用域可以分成两种：

* unscoped

    这种是公开的（public），可以直接在`package.json`中使用`package-name`进行引用。

* scoped

    这种默认是私有的（private），如果想公开，必须在发布时使用参数指定。公开的 package 可以通过`@username/package-name`引用到。

## built-in modules

### http

```js
var http = require('http');

//create a server object:
http.createServer(function (req, res) {
  res.write('Hello World!'); //write a response to the client
  res.end(); //end the response
}).listen(8080); //the server object listens on port 8080 
```

### fs

* `fs.readFile()`

    ```js
    var fs = require('fs');

    fs.readFile('demofile.txt', 'utf8', function(err, data) {
        if (err) throw err;
        console.log(data);
    });
    ```

* `fs.appendFile()`

    ```js
    var fs = require('fs');

    fs.appendFile('mynewfile1.txt', 'Hello content!', function (err) {
    if (err) throw err;
    console.log('Saved!');
    }); 
    ```

* `fs.open()`

    ```js
    var fs = require('fs');

    fs.open('mynewfile2.txt', 'w', function (err, file) {
    if (err) throw err;
    console.log('Saved!');
    }); 
    ```

* `fs.writeFile()`

    ```js
    var fs = require('fs');

    fs.writeFile('mynewfile3.txt', 'Hello content!', function (err) {
    if (err) throw err;
    console.log('Saved!');
    }); 
    ```

* `fs.unlink()`

    Delete a file with the File System module.

    ```js
    var fs = require('fs');

    fs.unlink('mynewfile2.txt', function (err) {
    if (err) throw err;
    console.log('File deleted!');
    }); 
    ```

* `fs.rename()`

    ```js
    var fs = require('fs');

    fs.rename('mynewfile1.txt', 'myrenamedfile.txt', function (err) {
    if (err) throw err;
    console.log('File Renamed!');
    }); 
    ```

### url

* `url.parse()`

    This method will return a URL object with each part of the address as properties:

    ```js
    var url = require('url');
    var adr = 'http://localhost:8080/default.htm?year=2017&month=february';
    var q = url.parse(adr, true);

    console.log(q.host); //returns 'localhost:8080'
    console.log(q.pathname); //returns '/default.htm'
    console.log(q.search); //returns '?year=2017&month=february'

    var qdata = q.query; //returns an object: { year: 2017, month: 'february' }
    console.log(qdata.month); //returns 'february'
    ```

### events

```js
var fs = require('fs');
var rs = fs.createReadStream('./demofile.txt');
rs.on('open', function () {
  console.log('The file is open');
}); 
```

All event properties and methods are an instance of an `EventEmitter` object.

```js
var events = require('events');
var eventEmitter = new events.EventEmitter();
```

You can assign event handlers to your own events with the EventEmitter object. In the example below we have created a function that will be executed when a `scream` event is fired. To fire an event, use the `emit()` method.

```js
var events = require('events');
var eventEmitter = new events.EventEmitter();

// Create an evenet handler:
var myEventHandler = function() {
    console.log('I hear a scream!');
}

// Assign the event handler to an event
eventEmitter.on('scream', myEventHandler);

// Fire the 'scream' event
eventEmitter.emit('scream');
```

### upload files

To upload files, we need to install `formidable`:

`npm install formidable`

```js
var http = require('http');
var formidable = require('formidable');
var fs = require('fs');

http.createServer(function (req, res) {
  if (req.url == '/fileupload') {
    var form = new formidable.IncomingForm();
    form.parse(req, function (err, fields, files) {
      var oldpath = files.filetoupload.path;
      var newpath = 'C:/Users/Your Name/' + files.filetoupload.name;
      fs.rename(oldpath, newpath, function (err) {
        if (err) throw err;
        res.write('File uploaded and moved!');
        res.end();
      });
 });
  } else {
    res.writeHead(200, {'Content-Type': 'text/html'});
    res.write('<form action="fileupload" method="post" enctype="multipart/form-data">');
    res.write('<input type="file" name="filetoupload"><br>');
    res.write('<input type="submit">');
    res.write('</form>');
    return res.end();
  }
}).listen(8080); 
```

