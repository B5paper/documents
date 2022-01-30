# NodeJS Note

official site: <https://nodejs.dev/>

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

