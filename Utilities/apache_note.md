# Apache Note

列出当前加载的 modules：`apachectl -t -D DUMP_MODULES`

log 文件：`/var/log/apache2/error.log`，`/var/log/apache2/access.log`

## cgi

### 与 cgi 相关的背景知识

Apache 是一个 web server，同类型的 web server 还有 nginx 等。这些 web server 只能处理静态页面的跳转，当前端提交一个表单（form）或发送请求（ajax 等）时，web server 无法处理。为了处理表单请求，就引入了 cgi 的概念。

cgi 是一种协议，协议指定了 cgi 程序通过标准输入输出 + 环境变量与 web server 交互，所以 cgi 程序与语言无关。

下面是调用一个 cgi 程序的详细过程：

1. 在 html 页面里，将`<form>`的`action`指向一个 cgi 程序。填写表单并提交。
1. web server 接收到 http 请求，进行部分解析，并将解析结果放到环境变量里，此时调用 cgi 程序，将 http 请求的剩下部分从`cin`输入到程序里。
1. cgi 程序执行，读取标准输入和环境变量，并对相应内容进行处理。
1. cgi 程序按照 MIME 消息格式，从`cout`输出一个 html 页面。web server 将输出再打包成 http 报文发送给前端，前端将 html 页面显示出来。

每个 cgi 程序只能处理一个用户请求，处理完后就退出了。

常见的环境变量：

* 与请求相关的环境变量：

    | Name | Description |
    | - | - |
    | `REQUEST_METHOD` | 服务器与 cgi 程序之间的信息传输方式，比如`POST` |
    | `QUERY_STRING` | 采用 get 时所传输的信息，比如`name=hgp&id=1` |
    | `CONTENT_LENGTH` | stdio 中的有效信息长度 |
    | `CONTENT_TYPE` | 指示所传来的信息的 MIME 类型。比如`application/x-www-form-urlencoded` |
    | `CONTENT_FILE` | 使用 windows httpd / wincgi 标准时，用来传送数据的文件名 |
    | `PATH_INFO` | 路径信息 |
    | `PATH_TRANSLATED` | cgi 程序的完整路径名 |
    | `SCRIPT_NAME` | 所调用的 cgi 程序的名字 |

* 与服务器相关的环境变量

    | Name | Description |
    | - | - |
    | `GATEWAY_INTERFACE` | 服务器所实现的 cgi 版本 |
    | `SERVER_NAME` | 服务器的 ip 或名字 |
    | `SERVER_PORT` | 主机的端口号 |
    | `SERVER_SOFTWARE` | 调用 cgi 程序的 http 服务器的名称和版本号 |

* 与客户端相关的环境变量

    | Name | Description |
    | - | - |
    | `REMOTE_ADDR` | 客户机的主机名 |
    | `REMOTE_HOST` | 客户机的 ip 地址 |
    | `ACCEPT` | 列出能被请求接受的应答方式，比如`image/gif;image/jpeg` |
    | `ACCEPT_ENCODING` | 列出客户机支持的编码方式 |
    | `ACCEPT_LANGUAGE` | 表明客户机可接受语言的 ISO 代码 |
    | `AUTORIZATION` | 表明被证实了的用户 |
    | `FORM` | 列出客户机的 email 地址 |
    | `IF_MODIFIED_SINGCE` | 当用 get 方式请求并且只有当文档比指定日期更早时才返回数据 |
    | `PRAGMA` | 设定将来要用到的服务器代理 |
    | `REFFERER` | 指出连接到当前文档的文档的 URL |
    | `USER_AGENT` | 客户端浏览器的信息 | 

似乎不同的服务器对应着不同的环境变量，除了上面的那些，这里还有一些常见的，如有需要，可作参考：<https://digital.com/web-hosting/cgi-access/variables/>

## 执行 cgi 程序

加载`mod_cgi`模块：`sudo a2enmod cgi`。这个命令会把`mods-availables`中的`cgi.load`软链接到`mods-enabled`中。

卸载`mod_cgi`模块：`sudo a2dismod cgi`。会删除`mods-enabled`中的`cgi.load`软链接。

无论是加载还是卸载，都需要重启`apache`服务器：`service apache2 restart`。

`apache2.conf`的配置：

```conf
<Directory "xxx">
Options +ExecCGI
AddHandler cgi-script .cgi .py
</Directory>
```

这样的话，`xxx`目录下的`.cgi`文件和`.py`文件都能被执行了。注意这些文件对于`webmaster`用户需要有可执行权限。

我们写个有固定输出的 c++ cgi 程序，然后放到这个文件夹下：

```c++
#include <iostream>
using namespace std;

int main()
{
    cout << "Content-type: text/html\n\n";  // 这里必须有两个`\n`，不然会报错
    cout << "<body>\n";
    cout << "<h2>This is my first cgi program!</h2>\n";
    cout << "</body>\n";
    return 0;
}
```

编译：`g++ main.cpp -o a.cgi`

html 页面的表单可以简单一写：

```html
<body>
<form action='./a.cgi' method='post'>
<input type='submit' value='submit'>
</form>
</body>
```

此时点击`submit`按钮后，即可自动显示`a.cgi`程序输出的内容。

一个比较清晰的官方配置教程：<https://httpd.apache.org/docs/current/howto/cgi.html>


## cgicc 库

一个详细的 c++ cgi 教程，使用 gnu cgicc 库：<https://www.tutorialspoint.com/cplusplus/cpp_web_programming.htm>

cgi 程序使用标准输入输出和 web server 进行交互。我们可以按照 get 方法或 post 方法对表单的编码格式进行自行解码，也可以使用现成的库方便我们解码。`cgicc`就是一个 GNU 工程中的一个库，它更新得也挺频繁，最后一次更新是 2020 年。

Official site: <https://www.gnu.org/software/cgicc/>

安装：`sudo apt install libcgicc-dev`

假如某个 html 表单长这样：

```html
<!doctype html>
<html>
<head>
<title>CGICC library test</title>
</head>

<body>
    <form action=./a.cgi method='post'>
        <label for='my_name'>Input your name: </label>
        <input type='text' name='my_name'>
        <br />
        <label for='my_age'>Input your age: </label>
        <input type='text' name='my_age'>
        <br />
        <input type='submit' value='Submit'>
    </form>
</body>
</html>
```

我们想通过 cgi 程序回显它的输入，那么可以这样写代码：

```c++
#include <iostream>
#include <cstdlib>
#include "cgicc/Cgicc.h"
#include "cgicc/HTTPHTMLHeader.h"
#include "cgicc/HTMLClasses.h"
using namespace std;
using namespace cgicc;

int main()
{
    Cgicc cgi;
    vector<FormEntry> elements = cgi.getElements();
    cout << HTTPHTMLHeader() << endl;  // 等价于 Content-Type: text/html\n\n
    cout << html() << endl;  // 相当于 <html>
    cout << head(title("CGICC library test")) << endl;
    cout << body() << endl;
    cout << h2("Now show the values of the form:") << endl;
    cout << br() << endl;
    for (auto &elm: elements)
    {
        cout << "<p>" << elm.getName() << ": " << elm.getValue() << "</p>" << endl;
    }
    cout << body() << endl;
    cout << html() << endl;  // 相当于 </html>
    return 0;
}
```

编译：`g++ main.cpp -lcgicc -o a.cgi`

## 其它一些 cgi 的库

* libcgi: <http://libcgi.sourceforge.net/>, <http://sunhe.jinr.ru/docs/libcgi.html>
* uncgi: <http://www.midwinter.com/~koreth/uncgi.html>
* cgi 库合集：<https://www.lemoda.net/c/cgi-libraries/>
* <https://github.com/boutell/cgic>

## fastcgi

因为 cgi 每次都要打开一个新的进程，所以效率低下。fast cgi 采用守护进程的方式，web server 将网页表单提交信息作为队列，通过 unix domain socket, named pipe 或 tcp 传递给这个守护进程，然后守护进程分别进行处理。

apache 使用`mod_fcgid`实现 fastcgi 的消息转发功能：<https://httpd.apache.org/mod_fcgid/mod/mod_fcgid.html>

安装：`sudo apt install libapache2-mod-fcgid`

启动：`sudo a2enmod fcgid`

停止：`sudo a2dismod fcgid`

配置：

在`apache2.conf`中，使用`AddHandler`命令增加对 fastcgi 程序的支持：

```conf
<Directory "xxx">
Options +ExecCGI
AddHandler cgi-script .cgi .py
AddHandler fcgid-script .fcgi
</Directory>
```

最后重启 apache2：`service apache2 restart`。

在上传文件时，`fcgid`默认的最大限制是 128K。不过这个值可以修改。

## libfcgi

在开发 fast cgi 程序时，我们使用一个 GNU 支持的库 libfcgi。因为它太稳定了，所以早就停止了维护，官网<fastcgi.com>也于 2016 年关闭，目前将所有的东西都归档到了 github 上：

官网：<https://fastcgi-archives.github.io/>

安装：`sudo apt install libfcgi-dev`

我们使用 c 写一个 fastcgi 程序（注意，必须是纯 c，用了 iostream 之类后，会出现 Internal Error）：

```C++
#include <cstdlib>
#include <unistd.h>
#include "fcgi_stdio.h"

int main()
{
    int len;
    char *content, *p;
    while (FCGI_Accept() >= 0)
    {
        len = atoi(getenv("CONTENT_LENGTH"));
        content = (char*)malloc(len+1);
        p = content;
        for (int i = 0; i < len; ++i)
            *p++ = getchar();
        *p = '\0';
        
        printf("Content-type: text/html\r\n\r\n");
        printf("<title>FastCGI Test</title>\n");
        printf("<h1>Here is the content for the form:</h1>\n");
        printf("<p>%s</p>\n", content);
        printf("<p>length: %d</p>\n", len);
        printf("<p>Process PID: %d</p>\n", getpid());
        
        free(content);
    }
    return 0;
}
```

编译：`gcc main.c -lfcgi -o a.fcgi`

然后就可以产生类似 cgi 的效果。查看后台：`ps -ax | grep a.fcig`，当网页关闭后，这个进程也依然在后台执行，等待数据接入，说明 fast cgi 程序执行成功。

c++ 的写法要涉及到 sreambuf，似乎比较复杂，以后有空了再写。

这个库的文档很少很少，如果有需要，可以阅读它的源代码和 examples 进行学习。

这个库不对表单的内容进行解析，需要我们手动解析，很麻烦。或许换个库会好些。

**多线程 fcgi**

`fcgi_stdio.h`对`cout`，`cin`，`printf`，`FILE`等 stdio 的函数或流对象或类型做了替换，使之可以绑定到当前的 request 上，从而对当前的 request 像标准输入输出那样编程。但是对于多线程问题，显然会出现多个 request，我们只绑定一个输入输出是不行的。所以需要对不同的 request 指定不同的输入输出。此时不能再`#include "fcgi_stdio.h`这个头文件，只需要`#include "fcgiapp.h`即可。

对于 C 程序，只需要调用 fcgi 库提供的输入输出函数，指明输入输出的 request 即可：

```c++
#include <fcgiapp.h>
#include <thread>
#include <cstdlib>

void thread_func()
{
    FCGX_Request request;
    FCGX_InitRequest(&request, 0, 0);
    FCGX_Accept_r(&request);

    int len = atoi(FCGX_GetParam("CONTENT_LENGTH", request.envp));
    char *content = (char*) malloc(len);
    FCGX_GetStr(content, len, request.in);

    FCGX_FPrintF(request.out, "Content-Type: text/html\n\n");
    FCGX_FPrintF(request.out, "<title>FastCGI Test</title>\n");
    FCGX_FPrintF(request.out, "<h2>The form content is:</h2>\n");
    FCGX_FPrintF(request.out, "<p>%s<p>\n", content);

    FCGX_Finish_r(&request);
    free(content);
}

int main()
{
    FCGX_Init();
    thread thd(thread_func);  // start another single thread
    thd.join();
    return 0;
}
```

对于 C++ 程序，需要将 iostream 重新定义缓冲区，让它输入输出到指定的 request 的流缓冲区：

```c++
#include <iostream>
#include <fcgio.h>  // this header is necessary for fcgi_streambuf
#include <fcgiapp.h>
#include <thread>
#include <string>
#include <cstdlib>

void thread_func()
{
    FCGX_Request request;
    FCGX_InitRequest(&request, 0, 0);
    FCGX_Accept_r(&request);

    const int IN_BUFFER_SIZE = 16;
    char in_buf[IN_BUFFER_SIZE];
    fcgi_streambuf in_buf(request.in, in_buf, IN_BUFFER_SIZE);  // additional buffer is necessary
    fcgi_streambuf out_buf(request.out), err_buf(request.err);
    istream cin(&in_buf);
    ostream cout(&out_buf), cerr(&err_buf);

    string content(istream_iterator<char>(cin), istream_iterator<char>());  // 保留换行和分隔符

    cout << "Content-Type: text/html\n\n";
    cout << "<title>FastCGI Test</title>\n";
    cout << "<h2>The form content is:</h2>\n";
    cout << "<p>" << content << "</p>" << endl;

    FCGX_Finish_r(&request);
}

int main()
{
    FCGX_Init();
    thread thd(thread_func);
    thd.join();
    return 0;
}
```

Problem Shooting:

1. `Error undefined reference to symbol 'FCGX_InitRequest'`

    `FCGX_InitRequest()`在`libfcgi`中实现，因此需要在编译时加上两个库：`-lfcgi -lfcgi++`。

1. 在对`cin`进行重绑定 streambuf 后，不能直接使用`cin >> content_str;`读取输入的数据。需要设定一个 buffer。

    因为 libfcgi 为了提高效率，设定`request.in`，`request.out`，`request.err`的缓冲区都为 0。对于输出还好，但是对于输入，就会漏掉很多内容。我们需要自己设定一个额外的缓冲区，这样才能完整地接收所有输入。

    另外`cin`对换行、特殊符号之类的也不友好，不能这样写：

    ```c++
    string content, buffer;
    while (content.size() < content_len)
    {
        cin >> buffer;  // 这样会漏掉空格和换行符
        content.append(buffer);
    }
    ```

    如果不使用额外的 buffer，可以这样写：

    ```c++
    // ......

    fcgi_streambuf in_buf(request.in), out_buf(request.out), err_buf(request.err);
    istream cin(&in_buf);
    ostream cout(&out_buf), cerr(&err_buf);

    int len = atoi(FCGX_GetParam("CONTENT_LENGTH", request.envp));
    string content;
    content.resize(len);
    cin.read(&len[0], len);  // 或者 cin.read(const_cast<char*>content.data(), len);

    // ......
    ```


## MIME

用户提交的表单数据经过标准输入输出送至 cgi/fastcgi 程序后，并不是可以直接拿来用的，这些数据是经过浏览器按照 MIME 格式进行编码的，并且 apache 服务器做了一部分的解码。

在`<form>`标签里，`enctype`属性指定了使用什么样的 MIME 格式进行编码。默认值为`application/x-www-form-urlencoded`，表示将空格转换成`+`号，将特殊符号转换成 ASCII HEX 值。还可取值`multipart/form-data`，表示不对空格或特殊字符进行编码，但是会使用`boundary`以及`header`等格式来区分不同消息；`text/plain`表示将空格转换成`+`，不对其它字符进行编码。

如果用户提交的数据有文件，那么我们必须使用`multipart/form-data`进行编码。这种 MIME 消息比较复杂，它可以有复杂的树状结构，因此我们直接使用`mimetic`这个库进行解码。（**请不要使用 gmime 这个库，资料很少，极其难用，从来没解析成功过**）

`multipart/form-data`的 MIME 消息格式：

```
Content-Type: multipart/form-data; boundary=aBoundaryString
(other headers associated with the multipart document as a whole)

--aBoundaryString
Content-Disposition: form-data; name="myFile"; filename="img.jpg"
Content-Type: image/jpeg

(data)
--aBoundaryString
Content-Disposition: form-data; name="myField"

(data)
--aBoundaryString
(more subparts)
--aBoundaryString--

```

更多标准可以参考文档：<https://developer.mozilla.org/en-US/docs/Web/HTTP/Basics_of_HTTP/MIME_types>

下面是几个 MIME 消息例子：

Example 1:

```
From: Some One <someone@example.com>
MIME-Version: 1.0
Content-Type: multipart/mixed;
        boundary="XXXXboundary text"

This is a multipart message in MIME format.

--XXXXboundary text
Content-Type: text/plain

this is the body text

--XXXXboundary text
Content-Type: text/plain;
Content-Disposition: attachment;
        filename="test.txt"

this is the attachment text

--XXXXboundary text--
```

来源：<https://developer.mozilla.org/en-US/docs/Web/HTTP/Basics_of_HTTP/MIME_types>

Example 2:

```
MIME-Version: 1.0
 X-Mailer: MailBee.NET 8.0.4.428
 Subject: This is the subject of a sample message
 To: user@example.com
 Content-Type: multipart/alternative;
 boundary="XXXXboundary text"

--XXXXboundary text
 Content-Type: text/plain;
 charset="utf-8"
 Content-Transfer-Encoding: quoted-printable

This is the body text of a sample message.

--XXXXboundary text
 Content-Type: text/html;
 charset="utf-8"
 Content-Transfer-Encoding: quoted-printable

<pre>This is the body text of a sample message.</pre>
--XXXXboundary text--
```

```
MIME-Version: 1.0
X-Mailer: MailBee.NET 8.0.4.428
Subject: test subject
To: kevinm@datamotion.com
Content-Type: multipart/mixed;
       boundary="XXXXboundary text"
 
--XXXXboundary text
Content-Type: multipart/alternative;
       boundary="XXXXboundary text"
 
--XXXXboundary text
Content-Type: text/plain;
       charset="utf-8"
Content-Transfer-Encoding: quoted-printable
 
This is the body text of a sample message.
--XXXXboundary text
Content-Type: text/html;
       charset="utf-8"
Content-Transfer-Encoding: quoted-printable
<pre>This is the body text of a sample message.</pre>

--XXXXboundary text
Content-Type: text/plain;
name="log_attachment.txt"
Content-Disposition: attachment;
filename="log_attachment.txt"
Content-Transfer-Encoding: base64

TUlNRS1WZXJzaW9uOiAxLjANClgtTWFpbGVyOiBNYWlsQmVlLk5FVCA4LjAuNC40MjgNClN1Ympl
Y3Q6IHRlc3Qgc3ViamVjdA0KVG86IGtldmlubUBkYXRhbW90aW9uLmNvbQ0KQ29udGVudC1UeXBl
OiBtdWx0aXBhcnQvYWx0ZXJuYXRpdmU7DQoJYm91bmRhcnk9Ii0tLS09X05leHRQYXJ0XzAwMF9B
RTZCXzcyNUUwOUFGLjg4QjdGOTM0Ig0KDQoNCi0tLS0tLT1fTmV4dFBhcnRfMDAwX0FFNkJfNzI1
RTA5QUYuODhCN0Y5MzQNCkNvbnRlbnQtVHlwZTogdGV4dC9wbGFpbjsNCgljaGFyc2V0PSJ1dGYt
OCINCkNvbnRlbnQtVHJhbnNmZXItRW5jb2Rpbmc6IHF1b3RlZC1wcmludGFibGUNCg0KdGVzdCBi
b2R5DQotLS0tLS09X05leHRQYXJ0XzAwMF9BRTZCXzcyNUUwOUFGLjg4QjdGOTM0DQpDb250ZW50
LVR5cGU6IHRleHQvaHRtbDsNCgljaGFyc2V0PSJ1dGYtOCINCkNvbnRlbnQtVHJhbnNmZXItRW5j
b2Rpbmc6IHF1b3RlZC1wcmludGFibGUNCg0KPHByZT50ZXN0IGJvZHk8L3ByZT4NCi0tLS0tLT1f
TmV4dFBhcnRfMDAwX0FFNkJfNzI1RTA5QUYuODhCN0Y5MzQtLQ0K

--XXXXboundary text--
ENTER YOUR CODE HERE
```

来源：<https://kb.datamotion.com/?ht_kb=what-does-a-sample-mime-message-look-like>

有几个地方需要格外注意：
a
1. `content-type`是一个 header，MIME 消息第一个 header 后面跟着的是一个`boundary`，需要注意的是，`boundary`要么不换行，要么换行后加上缩进：

    不换行：

    ```
    Content-Type: multipart/form-data; boundary=xxxxxxthisisaboundaryxxxx
    ```

    换行加缩进：

    ```
    Content-Type: multipart/form-data;
        boundary=xxxxxxthisisaboundaryxxxx
    ```

    缩进加 4 个空格还是 8 个空格好像不影响。具体没研究过，但是一定要加缩进，不然会识别不了内容。

1. header 与 body 间需要多加一行，但 body 和下面的 boundary 不用多加换行

    ```
    Content-Type: multipart/form-data;
        boundary=-----thisisaboundary------

    This is the body content.
    -------thisisaboundary------
    ```

1. 实际使用 boundary 时，需要使用定义 boundary 时的字符串前加两个`-`，在 MIME 消息的最后一条 boundary 后面，也需要加上两个`-`。

解释 mime 消息的一些库：

1. <https://github.com/iafonov/multipart-parser-c>

## mimetic 库

官方网站：<http://www.codesink.org/mimetic_mime_library.html>

可以用源码直接编译安装：`./configure`，`make`，`sudo make install`

也可以直接用 apt 安装：`sudo apt install libmimetic-dev`。

下面是一个解析 MIME 消息文件的例子：

```c++
#include <iostream>
#include <mimetic/mimetic.h>
#include <string>

using namespace std;
using namespace mimetic;

void dfs(MimeEntity *part, int indent)
{
    string indent_str;
    for (int i = 0; i < indent * 4; ++i)
        indent_str.push_back(' ');
    
    cout << indent_str << "Content-Type: " << part->header().contentType().str() << endl;
    cout << indent_str << "Content-Disposition: " << part->header().contentDisposition().str() << endl;
    cout << indent_str << "Content-Transfer-Encoding: " << part->header().contentTransferEncoding().str() << endl;
    cout << indent_str << part->body() << endl;
    cout << endl;

    MimeEntityList &parts = part->body().parts();
    for (auto &p: parts)
    {
        dfs(p, indent+1);
    } 
}

int main(int argc, char** argv)
{
    if (argc != 2)
    {
        cout << "Wrong program arguments." << endl;
        return -1;
    }

    ifstream fs(argv[1]);
    if (!fs.is_open())
    {
        cout << "Fail to open file." << endl;
        return -1;
    }

    ios_base::sync_with_stdio(false);
    MimeEntity me(fs);

    dfs(&me, 0);
    return 0;
}
```

编译：`g++ source.cpp -lmimetic -o parse`

运行：`parse mime_message.txt`

如果 MIME 消息的内容是经过 base64 编码的，mimetic 还提供了 base64 解码器来解码，非常方便。

在实际结合 apache 的`mod_fcgid`进行 MIME 消息解码时，apache 会解析一部分消息，从而使我们获取的消息不全，无法正常解码。我们需要先把消息补全，再交给 mimetic 进行解码：

```c++
#include <iostream>
#include <fcgio.h>  // this header is necessary for fcgi_streambuf
#include <fcgiapp.h>
#include <thread>
#include <string>
#include <cstdlib>
#include <mimetic>

void thread_func()
{
    FCGX_Request request;
    FCGX_InitRequest(&request, 0, 0);
    FCGX_Accept_r(&request);

    const int IN_BUFFER_SIZE = 16;
    char in_buf[IN_BUFFER_SIZE];
    fcgi_streambuf in_buf(request.in, in_buf, IN_BUFFER_SIZE);  // additional buffer is necessary
    fcgi_streambuf out_buf(request.out), err_buf(request.err);
    istream cin(&in_buf);
    ostream cout(&out_buf), cerr(&err_buf);

    string content(istream_iterator<char>(cin), istream_iterator<char>());  // 保留换行和分隔符
    content = string("Content-Type: ") + FCGX_GetParam("CONTENT_TYPE", request.envp) + "\n\n" + content;

    MimeEntity me;
    me.load(content.begini(), content.end());

    // do some thing with mime

    cout << "Content-Type: text/html\n\n";
    cout << "<title>FastCGI Test</title>\n";
    cout << "<h2>The form content is:</h2>\n";
    cout << "<p>" << content << "</p>" << endl;

    FCGX_Finish_r(&request);
}

int main()
{
    FCGX_Init();
    thread thd(thread_func);
    thd.join();
    return 0;
}
```
