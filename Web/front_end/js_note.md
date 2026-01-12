# JavaScript Note

## 引入 js 脚本的方式

1. 内部脚本

    js 脚本可以被放在`<head>`或`<body>`中。把脚本置于 `<body>`元素的底部，可改善显示速度，因为脚本编译会拖慢显示。

    ```html
    <script>
    document.getElementById("demo").innerHTML = "我的第一段 JavaScript";
    </script>
    ```

1. 外部脚本

    外部文件：`myScript.js`

    ```js
    function myFunction() {
        document.getElementById("demo").innerHTML = "段落被更改。";
    }
    ```

    html 文件：

    ```html
    <head>
    <script src="myScript.js"></script>
    </head>
    ```

1. 外部引用

    ```html
    <script src="https://www.w3school.com.cn/js/myScript1.js"></script>
    ```

1. 内嵌的简单 js

    ```html
    <button onclick="document.getElementById('myImage').src='/i/eg_bulboff.gif'">关灯</button>
    ```

## 输出

1. `window.alert()`
1. `document.write()`（在 html 文档完全加载后再调用会删除当前页面的所有内容）
1. `innerHTML`
1. `console.log()`

## 变量

使用`var`声明变量：

```js
var x, y;
var x = 3;
var s = 'nihao';
var a = 1, b = 2, c = 3;
```

只声明，不赋值的变量的值为`undefined`。

重复声明某个变量，将不会丢失它的值。

**数据类型**

```js
var length = 7;  // 数字 
var lastName = "Gates";   // 字符串
var cars = ['Porsche', 'Volvo', 'BMW'];  // 数组
var x = {firstName: 'Bill', lastName: 'Gates'};  // 对象

typeof length  // "number"
typeof lastName  // "string"
typeof cars  // "object"
typeof x  // "object"
typeof true  // "boolean"
typeof function myfunc() {}  // "function"

var person;
typeof person  // "undefined", person 的值和类型都是 undefined
person = null;
typeof null  // "object"
```

可以通过设置`null`来清空对象。

```js
null === undefined  // false
null == undefined  // true
```

对象与对象无法比较：

```js
var x = new String('bill');
var y = new String('bill');

x == y  // false
x === y  // false
```

## 数组

在指定位置插入元素：

```js
// 原来的数组
var array = ["one", "two", "four"];
// splice(position, numberOfItemsToRemove, item)
// 拼接函数(索引位置, 要删除元素的数量, 元素)
array.splice(2, 0, "three");
 
array;  // 现在数组是这个样子 ["one", "two", "three", "four"]
```

查找元素的索引：`arr.indexOf(element, index)`。如果找不到，返回`-1`。

## 字符串

1. 可以是单引号，也可以是双引号
1. 使用`+`拼接字符串

字符串长度：

```js
var txt = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
var sln = txt.length;
```

转义字符：

`\'`, `\"`, `\\`, `\b`, `\f`, `\n`, `\r`, `\t`, `\v`

查找子字符串

```js
var str = "The full name of China is the People's Republic of China.";
var pos = str.indexOf("China");
var last_pos = str.lastIndexOf("China");
```

`search()`没有位置参数，但是可以使用正则表达式。

常用字符串函数：

| Function | Description |
| - | - |
| `indexOf(str, pos)` | 查找字符串第一次出现的位置 |
| `lastIndexOf(str, pos)` | |
| `search(str)` | |
| `str.slice(start, end)` | |
| `str.substring(start, end)` | |
| `str.substr(start, length)` | |
| `str.replace()` | |
| `str.toUpperCase()` | |
| `str.toLowerCase()` | |
| `str.concat()` | |
| `str.trim()` | |
| `str.charAt()` | |
| `str.charCodeAt()` | |
| `[]` | 利用下标返回只读的字符，若越界，返回 undefined |
| `str.split()` |

## 运算符

对数字和字符串相加，得到的是字符串。

| 运算符 | 描述 |
|-|-|
| `**` | 取幂 |
| `==` | 等于 |
| `===` | 等值等型 |
| `!=` | 不相等 |
| `!==` | 不等值或不等型 |
| `?` | 三元运算符 |
| `&&` | 逻辑与 |
| `||` | 逻辑或 |
| `!` | 逻辑非 |
| `typeof` | 返回变量的类型 |
| `instanceof` | 如果对象是类型的实例，则返回`true` |
| `&` | 按位与 |
| `|` | 按位或 |
| `~` | 按位非 |
| `^` | 异或 |
| `<<` | 零填充左位移 |
| `>>` | 有符号右位移 |
| `>>>` | 零填充右位移 |

## 函数

```js
var x = myFunction(7, 8);

// 函数提升 hoisting
function myFunction(a, b) {
    return a * b;
}

// 匿名函数
var x = function(a, b) {return a * b;}
var z = x(4, 3);

// Function() 构造器，尽量不要使用
var myFunction = new Function("a", "b", "return a * b");
var x = myFunction(4, 3);
```

使用表达式定义的函数不会被提升。

```js
function toCelsius(fahrenheit) {
    return (5/9) * (fahrenheit-32);
}

document.getElementById("demo").innerHTML = toCelsius(77);
```

自调用函数：

```js
// 一个匿名的自调用函数
(function () {
    var x = "hello";
})();
```

函数被调用时收到的参数数目：

```js
function myFunction(a, b) {
    return arguments.length;
}
```

`arguments`是一个数组，包含函数调用时的所有参数：

```js
x = findMax(1, 123, 500, 115, 44, 88);

function findMax() {
    var i;
    var max = -Infinity;
    for (i = 0; i < arguments.length; i++) {
        if (arguments[i] > max) {
            max = arguments[i];
        }
    }
    return max;
}
```

`toString()`方法以字符串返回函数：

```js
function myFunction(a, b) {
    return a * b;
}

var txt = myFunction.toString();
```

箭头函数：

```js
// ES5
var x = function(x, y) {
  return x * y;
}

// ES6
const x = (x, y) => x * y;
```

箭头函数没有自己的 this。它们不适合定义对象方法。

箭头函数未被提升。它们必须在使用前进行定义。

使用 const 比使用 var 更安全，因为函数表达式始终是常量值。

如果函数是单个语句，则只能省略 return 关键字和大括号。因此，保留它们可能是一个好习惯：

```js
const x = (x, y) => { return x * y };
```

如果调用参数时省略了部分参数，则丢失的部分被设置为`undefined`。

```js
function myFunction(x, y) {
    if (y === undefined) {
        y = 0;
    }
}
```

参数的传递方式：对象通过引用传递，其它通过值传递。

使用函数构造器来调用函数：

```js
// 这是函数构造器：
function myFunction(arg1, arg2) {
    this.firstName = arg1;
    this.lastName  = arg2;
}

// 创建了一个新对象：
var x = new myFunction("Bill", "Gates");
x.firstName;                             // 会返回 "Bill"
```

构造器内的 this 关键词没有值。

this 的值会成为调用函数时创建的新对象。

**`call()`**

使某个成员方法作用于指定的对象上，相当于改变了`this`的指向：

```js
var person = {
    fullName: function() {
        return this.firstName + " " + this.lastName;
    }
}
var person1 = {
    firstName:"Bill",
    lastName: "Gates",
}
var person2 = {
    firstName:"Steve",
    lastName: "Jobs",
}
person.fullName.call(person1);  // 将返回 "Bill Gates"
```

带参数：

```js
var person = {
  fullName: function(city, country) {
    return this.firstName + " " + this.lastName + "," + city + "," + country;
  }
}
var person1 = {
  firstName:"Bill",
  lastName: "Gates"
}
person.fullName.call(person1, "Seattle", "USA");
```

**`apply`**

## 对象

```js
var person = {
  firstName: "Bill",
  lastName : "Gates",
  id       : 678,
  fullName : function() {
    return this.firstName + " " + this.lastName;
  }
};
```

访问对象属性：

```js
// method 1
objectName.propertyName

// method 2
objectName["propertyName"]
```

为对象添加/删除属性：

```js
person.nationality = "English";
delete person.nationality;
```

（`delete`不应用于预定义的 js 对象属性，这样做会使应用程序崩溃）

对象构造器：

```js
function Person(first, last, age, eye) {
    this.firstName = first;
    this.lastName = last;
    this.age = age;
    this.eyeColor = eye;
}

var myFather = new Person("Bill", "Gates", 62, "blue");
var myMother = new Person("Steve", "Jobs", 56, "green");
```

js 内置的对象构造器：

```js
var x1 = new Object();    // 一个新的 Object 对象
var x2 = new String();    // 一个新的 String 对象
var x3 = new Number();    // 一个新的 Number 对象
var x4 = new Boolean();   // 一个新的 Boolean 对象
var x5 = new Array();     // 一个新的 Array 对象
var x6 = new RegExp();    // 一个新的 RegExp 对象
var x7 = new Function();  // 一个新的 Function 对象
var x8 = new Date();      // 一个新的 Date 对象

var x1 = {};            // 新对象
var x2 = "";            // 新的原始字符串
var x3 = 0;             // 新的原始数值
var x4 = false;         // 新的原始逻辑值
var x5 = [];            // 新的数组对象
var x6 = /()/           // 新的正则表达式对象
var x7 = function(){};  // 新的函数对象
```

`Math()`对象不再此列。`Math`是全局对象。`new`关键词不可用于 Math。

getter 和 setter：

```js
var person = {
    firstName: "Bill",
    lastName: "Gates",
    language: "en",
    get lang() {
        return this.language;
    }
    set lang(lang) {
        this.language = lang;
    }
};

person.lang = "en";
document.getElementById("demo").innerHTML = person.lang;
```

使用`prototype`动态地向对象构造器添加新属性：

```js
function Person(first, last, age, eyecolor) {
    this.firstName = first;
    this.lastName = last;
    this.age = age;
    this.eyeColor = eyecolor;
}
Person.prototype.nationality = "English";
Person.prototype.name = function() {
    return this.firstName + " " + this.lastName;
};
```

ES5 新的对象方法：

```js
// 添加或更改对象属性
Object.defineProperty(object, property, descriptor)

// 添加或更改多个对象属性
Object.defineProperties(object, descriptors)

// 访问属性
Object.getOwnPropertyDescriptor(object, property)

// 以数组返回所有属性
Object.getOwnPropertyNames(object)

// 以数组返回所有可枚举的属性
Object.keys(object)

// 访问原型
Object.getPrototypeOf(object)

// 阻止向对象添加属性
Object.preventExtensions(object)

// 如果可将属性添加到对象，则返回 true
Object.isExtensible(object)

// 防止更改对象属性（而不是值）
Object.seal(object)

// 如果对象被密封，则返回 true
Object.isSealed(object)

// 防止对对象进行任何更改
Object.freeze(object)

// 如果对象被冻结，则返回 true
Object.isFrozen(object)
```

Examples:

```js
// 定义对象
var obj = {counter : 0};

// 定义 setters
Object.defineProperty(obj, "reset", {
  get : function () {this.counter = 0;}
});
Object.defineProperty(obj, "increment", {
  get : function () {this.counter++;}
});
Object.defineProperty(obj, "decrement", {
  get : function () {this.counter--;}
});
Object.defineProperty(obj, "add", {
  set : function (value) {this.counter += value;}
});
Object.defineProperty(obj, "subtract", {
  set : function (value) {this.counter -= value;}
});

Object.defineProperty(person, "language", {value : "ZH"});

// 操作计数器：
obj.reset;
obj.add = 5;
obj.subtract = 1;
obj.increment;
obj.decrement;
```

元数据：

ES5 允许更改以下属性元数据：

```js
writable : true      // 属性值可修改
enumerable : true    // 属性可枚举
configurable : true  // 属性可重新配置
writable : false     // 属性值不可修改
enumerable : false   // 属性不可枚举
configurable : false // 属性不可重新配置

// 定义 getter
get: function() { return language }
// 定义 setter
set: function(value) { language = value }

// 使语言为只读：
Object.defineProperty(person, "language", {writable:false});

// 使语言不可枚举：
Object.defineProperty(person, "language", {enumerable:false});
```

## 逻辑控制

* `for`

    ```js
    for (i = 0; i < cars.length; i++) { 
        text += cars[i] + "<br>";
    }

    for (i = 0, len = cars.length, text = ""; i < len; i++) { 
        text += cars[i] + "<br>";
    }
    ```

* `for/in`

    ```js
    var person = {fname:"Bill", lname:"Gates", age:62}; 

    var text = "";
    var x;
    for (x in person) {
        text += person[x];
    }
    ```

* `while`

    ```js
    while (i < 10) {
        text += "数字是 " + i;
        i++;
    }
    ```

* `do ... while`

    ```js
    do {
        text += "The number is " + i;
        i++;
    }
    while (i < 10);
    ```

## 表单

```html
<form name="myForm" action="/action_page_post.php" onsubmit="return validateForm()" method="post">
姓名：<input type="text" name="fname">
<input type="submit" value="Submit">
</form>

<script>
function validateForm() {
    var x = document.forms["myForm"]["fname"].value;
    if (x == "") {
        alert("必须填写姓名");
        return false;
    }
}
</script>
```

## DOM

### DOM Document

**查找 HTML 元素**

| 方法 | 描述 |
| - | - |
| `document.getElementById(id)` | 通过元素 id 来查找元素 |
| `document.getElementsByTagName(name)` | 通过标签名来查找元素 |
| `document.getElementsByClassName(name)` | 通过类名来查找元素 |

**改变 HTML 元素**

| 方法 | 描述 |
| - | - |
| `element.innerHTML = new html content` | 改变元素的 inner HTML |
| `element.attribute = new value` | 改变 HTML 元素的属性值 |
| `element.setAttribute(attribute, value)` | 改变 HTML 元素的属性值 |
| `element.style.property = new style` | 改变 HTML 元素的样式 |

**添加和删除元素**

| 方法 | 描述 |
| - | - |
| `document.createElement(element)` | 创建 HTML 元素 |
| `document.removeChild(element)` | 删除 HTML 元素 |
| `document.appendChild(element)` | 添加 HTML 元素 |
| `document.replaceChild(element)` | 替换 HTML 元素 |
| `document.write(text)` | 写入 HTML 输出流 |

**添加事件处理程序**

| 方法 | 描述 |
| - | - |
| `document.getElementById(id).onclick = function(){code}` | 向 onclick 事件添加事件处理程序 |

**查找 HTML 对象**

| 属性 | 描述 |
| - | - |
| `document.anchors` | 返回拥有 name 属性的所有`<a>`元素。|
| `document.baseURI` | 返回文档的绝对基准 URI |
| `document.body` | 返回`<body>`元素 |
| `document.cookie` | 返回文档的 cookie |
| `document.doctype` | 返回文档的 doctype |
| `document.documentElement` | 返回`<html>`元素 |
| `document.documentMode` | 返回浏览器使用的模式 |
| `document.documentURI` | 返回文档的 URI |
| `document.domain` | 返回文档服务器的域名 |
| `document.embeds` | 返回所有`<embed>`元素 |
| `document.forms` | 返回所有`<form>`元素 |
| `document.head` | 返回`<head>`元素 |
| `document.images` | 返回所有`<img>`元素 |
| `document.implementation` | 返回 DOM 实现 |
| `document.inputEncoding` | 返回文档的编码（字符集） |
| `document.lastModified` | 返回文档更新的日期和时间 |
| `document.links` | 返回拥有 href 属性的所有`<area>`和`<a>`元素 |
| `document.readyState` | 返回文档的（加载）状态 |
| `document.referrer` | 返回引用的 URI（链接文档） |
| `document.scripts` | 返回所有`<script>`元素 |
| `document.strictErrorChecking` | 返回是否强制执行错误检查 |
| `document.title` | 返回`<title>`元素 |
| `document.URL` | 返回文档的完整 URL |

### DOM Element

### DOM Attribute

### DOM Event

## AJAX

AJAX = Asynchronous JavaScript And XML

`XMLHttpRequest` object methods:

* `new XMLHttpRequest()`: Creates a new `XMLHttpRequest` object
* `abort()`: Cancels the current request
* `getAllResponseHeaders()`: Returns header information
* `getResponseHeader(string)`: Returns specific header information
* `open(method, url, async, user, psw)`: Specifies the request

    * `method`: the request type `GET` or `POST`
    * `url`: the file location, e.g. `"demo_get.asp?fname=Henry&lname=Fold"`
    * `async`: `true` (asynchronous) or `false` (synchronous)
    * `user`: optional user name
    * `psw`: optional password

* `send()`: Sends the request to the server. Used for `GET` requests.

* `send(string)`: Sends the request to the server. Used for `POST` requests.

* `setRequestHeader()`: Adds a label/value pair to the header to be sent.

Properties:

* `response`：返回什么类型取决于`XMLRequest.responsyType`。

* `onload`: Defines a function to be called when the request is recived (loaded)

* `onreadystatechange`: Defines a function to be called when the readyState property changes

* `readyState`: Holds the status of the XMLHttpRequest

    * `0`: request not initialized
    * `1`: server connection established
    * `2`: request received
    * `3`: processing request
    * `4`: request finished and response is ready

* `responseText`: Returns the response data as a string

* `responseXML`: Returns the respose data as XML data

* `status`: Returns the status-number of a request

    * `200`: OK
    * `403`: Forbidden
    * `404`: Not Found

* `statusText`: Return the status-text (e.g. "OK" or "Not Found")

* `responseType`

    * `""`: An empty `responseType` string is the same as `"test"`, the default type.

    * `"arraybuffer"`: a `ArrayBuffer` conotaining binary data.

    * `"blob"`：a `Blob` object containing the binary data. 

    * `"document"`: an HTML `Document` or xml `XMLDocument`.

    * `"json"`: a `JSON` object.

    * `"text"`

Example 1:

```js
const xhttp = new XMLHttpRequest();
xhttp.onload = function() {
    document.getElementById("demo").innerHTML = this.responseText;
}
xhttp.open('GET', 'ajax_info.txt', true);
xhttp.send();
```

Example 2:

```js
xhttp.open("POST", "demo_post.asp");
xhttp.send();

xhttp.open("POST", "ajax_test.asp");
xhttp.setRequestHeader("Content-type", "application/x-www-form-urlencoded");
xhttp.send("fname=Henry&lname=Ford");
```

## 开启摄像头并截图

```js
navigator.mediaDevices.getUserMedia({video: {width: 300, height: 300}, audio: false}).then(mediaStream => {
    document.getElementById('video').srcObject = mediaStream;
    document.getElementById('video').play();
})

// canvas 绘制
document.getElementById('take').onclick = () => {
    let ctx = document.getElementById('canvas').getContext('2d');
    ctx.drawImage(document.getElementById('video'), 0, 0, 300, 300);

    // 得到图片数据
    let img = document.getElementById('canvas').toDataURL();

    // 在某个 img tag 里再画出来：

    document.getElementById('imgTag').src = img;
}
```

从视频流中获取一张静止的图片，得到大小信息：

```js
var track = mediaStream.getVideoTracks()[0];
var imageCapture = new ImageCapture(track);
var photoSettings = imageCapture.getPhotoSettings();

oCanvas.width = photoSettings.imageWidth;
oCanvas.height = photoSettings.imageHeight;
```

点击按钮保存图片：

```js
document.getElementById('btn').onclick = function() {
    ctx.drawImage(video, 0, 0);
    var base64Img = canvas.toDataURL();
    var oA = document.createElement('a');
    oA.href = base64Img;
    oA.download = '截图.png'; // 下载的文件名可以此处修改
    oA.click();
};
```

## Miscellaneous

1. 注释

    1. `//`
    1. `/*  */`

1. js 中标识符首字符必须是字母，下划线或美元符号（`$`）。

1. js 对大小写敏感。

1. 长代码可以直接换行，但是不能通过`\`换行。长字符串可以通过反斜杠`\`换行，也可以通过`+`拼接字符串。

1. Hoisting

    提升（Hoisting）是 JavaScript 将声明移至顶部的默认行为。在 JavaScript 中，可以在声明变量之前使用它。

    用`let`或`const`声明的变量和常量不会被提升！

1. 变量作用域

    * 全局作用域

        ```js
        var carName = "porsche";

        // 此处的代码可以使用 carName

        function myFunction() {
        // 此处的代码也可以使用 carName
        }
        ```

    * 函数作用域

        ```js
        // 此处的代码不可以使用 carName

        function myFunction() {
        var carName = "porsche";
        // code here CAN use carName
        }

        // 此处的代码不可以使用 carName
        ```

    * 块作用域（block scope）

        1. 通过`var`关键词声明的变量没有块作用域。

            ```js
            { 
                var x = 10; 
            }
            // 此处可以使用 x
            ```

        1. 可以使用`let`关键词声明拥有块作用域的变量。

            ```js
            { 
                let x = 10;
            }
            // 此处不可以使用 x
            ```

1. `this`

    如果一个函数是一个成员方法，那么`this`指向对象；如果一个函数是一个全局函数，那么`this`指的就是全局对象，在浏览器中它是`window`，在网页中它是`document`。

1. 四舍五入：`Number.parseFloat(x).toFixed(2);`（对小数字符串进行解析，四舍五入保留两位小数）

1. Cross-Origin Requests

    当在使用`XMLHttpRequest`访问非本网站的 api 时，会遇到这个错误。此时 GET 请求需加上`&origin=*`。
    
1. 获得维基百科的页面/图片：<https://github.com/mudroljub/wikipedia-api-docs>

    另外一些有用的资源：<https://stackoverflow.com/questions/7185288/how-can-i-get-wikipedia-content-using-wikipedias-api>

    <https://www.jianshu.com/p/c05ee1927c85>
