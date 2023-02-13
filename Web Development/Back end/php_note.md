# PHP Note

hello, world:

```php
<!DOCTYPE html>
<html>
<body>

<?php
echo "我的第一段 PHP 脚本！";
?>

</body>
</html>
```

## 变量

```php
<?php
$x=5;
$y=6;
$z=$x+$y;
echo $z;
?>
```

PHP 变量可用于保存值（x=5）和表达式（z=x+y）。

变量的作用域：

* local（局部）

    在函数内部声明，只能在函数内部进行访问。

* global（全局）

    在函数之外声明，只能在函数以外进行访问。

* static（静态）

    static 变量与 c++ 差不多：

    ```php
    <?php

    function myTest() {
    static $x=0;
    echo $x;
    $x++;
    }

    myTest();
    myTest();
    myTest();

    ?>
    ```

Examples：

```php
<?php
$x=5; // 全局作用域

function myTest() {
  $y=10; // 局部作用域
  echo "<p>测试函数内部的变量：</p>";
  echo "变量 x 是：$x";
  echo "<br>";
  echo "变量 y 是：$y";
} 

myTest();

echo "<p>测试函数之外的变量：</p>";
echo "变量 x 是：$x";
echo "<br>";
echo "变量 y 是：$y";
?>
```

`global`关键词用于在函数内访问全局变量：

```php
<?php
$x=5;
$y=10;

function myTest() {
  global $x,$y;
  $y=$x+$y;
}

myTest();
echo $y; // 输出 15
?>
```

PHP 在名为`$GLOBALS[index]`的数组中存储了所有的全局变量。下标存有变量名。这个数组在函数内也可以访问，并能够用于直接更新全局变量：

```php
<?php
$x=5;
$y=10;

function myTest() {
  $GLOBALS['y']=$GLOBALS['x']+$GLOBALS['y'];
} 

myTest();
echo $y; // 输出 15
?>
```

**超全局变量**

超全局变量在一个脚本的全部作用域中都可用，在函数中无需执行`global $variable;`就可以访问它们。

* `$GLOBALS`

    `$GLOBALS`这种全局变量用于在 PHP 脚本中的任意位置访问全局变量（从函数或方法中均可）。

    PHP 在名为`$GLOBALS[index]`的数组中存储了所有全局变量。变量的名字就是数组的键。

    ```php
    <?php 
    $x = 75; 
    $y = 25;
    
    function addition() { 
    $GLOBALS['z'] = $GLOBALS['x'] + $GLOBALS['y']; 
    }
    
    addition(); 
    echo $z; 
    ?>
    ```

* `_SERVER`

    `$_SERVER`保存关于报头、路径和脚本位置的信息。

    一些常用的元素：

    |元素|描述|
    |-|-|
    | `$_SERVER['PHP_SELF']` | 	返回当前执行脚本的文件名。 |
    | `$_SERVER['GATEWAY_INTERFACE']` | 返回服务器使用的 CGI 规范的版本。 |
    | `$_SERVER['SERVER_ADDR']` | 返回当前运行脚本所在的服务器的 IP 地址。 |
    | `$_SERVER['SERVER_NAME']` | 返回当前运行脚本所在的服务器的主机名（比如 www.w3school.com.cn）。 |
    | `$_SERVER['SERVER_SOFTWARE']` | 返回服务器标识字符串（比如 Apache/2.2.24）。 |
    | `$_SERVER['SERVER_PROTOCOL']` | 返回请求页面时通信协议的名称和版本（例如，“HTTP/1.0”）。 |
    | `$_SERVER['REQUEST_METHOD']` | 返回访问页面使用的请求方法（例如 POST）。 |
    | `$_SERVER['REQUEST_TIME']` | 返回请求开始时的时间戳（例如 1577687494）。 |
    | `$_SERVER['QUERY_STRING']` | 	返回查询字符串，如果是通过查询字符串访问此页面。 |
    | `$_SERVER['HTTP_ACCEPT']` | 返回来自当前请求的请求头。 |
    | `$_SERVER['HTTP_ACCEPT_CHARSET']` | 返回来自当前请求的 Accept_Charset 头（ 例如 utf-8,ISO-8859-1） |
    | `$_SERVER['HTTP_HOST']` | 返回来自当前请求的 Host 头。 |
    | `$_SERVER['HTTP_REFERER']` | 返回当前页面的完整 URL（不可靠，因为不是所有用户代理都支持）。 |
    | `$_SERVER['HTTPS']` | 是否通过安全 HTTP 协议查询脚本。|
    | `$_SERVER['REMOTE_ADDR']` | 返回浏览当前页面的用户的 IP 地址。 |
    | `$_SERVER['REMOTE_HOST']` | 返回浏览当前页面的用户的主机名。 |
    | `$_SERVER['REMOTE_PORT']` | 返回用户机器上连接到 Web 服务器所使用的端口号。 |
    | `$_SERVER['SCRIPT_FILENAME']` | 返回当前执行脚本的绝对路径。 |
    | `$_SERVER['SERVER_ADMIN']` | 该值指明了 Apache 服务器配置文件中的 SERVER_ADMIN 参数。 |
    | `$_SERVER['SERVER_PORT']` | Web 服务器使用的端口。默认值为 “80”。|
    | `$_SERVER['SERVER_SIGNATURE']` | 返回服务器版本和虚拟主机名。 |
    | `$_SERVER['PATH_TRANSLATED']` | 当前脚本所在文件系统（非文档根目录）的基本路径。 |
    | `$_SERVER['SCRIPT_NAME']` | 返回当前脚本的路径。 |
    | `$_SERVER['SCRIPT_URI']` | 返回当前页面的 URI。 |

* `_REQUEST`

    `$_REQUEST`用于收集 HTML 表单提交的数据。

    下面是一个指定文件本身来处理表单数据的例子：

    ```php
    <html>
    <body>

    <form method="post" action="<?php echo $_SERVER['PHP_SELF'];?>">
    Name: <input type="text" name="fname">
    <input type="submit">
    </form>

    <?php 
    $name = $_REQUEST['fname']; 
    echo $name; 
    ?>

    </body>
    </html>
    ```

* `_POST`

    PHP `$_POST`广泛用于收集提交`method="post"`的 HTML 表单后的表单数据。`$_POST`也常用于传递变量。

    ```php
    <html>
    <body>

    <form method="post" action="<?php echo $_SERVER['PHP_SELF'];?>">
    Name: <input type="text" name="fname">
    <input type="submit">
    </form>

    <?php 
    $name = $_POST['fname'];
    echo $name; 
    ?>

    </body>
    </html>
    ```

* `_GET`

    `$_GET`可用于收集提交 HTML 表单 (method="get") 之后的表单数据：

    html:

    ```html
    <html>
    <body>

    <a href="test_get.php?subject=PHP&web=W3school.com.cn">测试 $GET</a>

    </body>
    </html>
    ```

    php:

    ```php
    <html>
    <body>

    <?php 
    echo "在 " . $_GET['web'] . " 学习 " . $_GET['subject'];
    ?>

    </body>
    </html>
    ```
    
    
    `$_GET`也可以收集 URL 中的发送的数据。

* `_FILES`
* `_ENV`
* `_COOKIE`
* `_SESSION`

## Echo / Print

* echo - 能够输出一个以上的字符串

    ```php
    <?php
    echo "<h2>PHP is fun!</h2>";
    echo "Hello world!<br>";
    echo "I'm about to learn PHP!<br>";
    echo "This", " string", " was", " made", " with multiple parameters.";
    ?>
    ```

    ```php
    <?php
    $txt1="Learn PHP";
    $txt2="W3School.com.cn";
    $cars=array("Volvo","BMW","SAAB");

    echo $txt1;
    echo "<br>";
    echo "Study PHP at $txt2";
    echo "My car is a {$cars[0]}";
    ?>
    ```

* print - 只能输出一个字符串，并始终返回 1

    ```php
    <?php
    $txt1="Learn PHP";
    $txt2="W3School.com.cn";
    $cars=array("Volvo","BMW","SAAB");

    print $txt1;
    print "<br>";
    print "Study PHP at $txt2";
    print "My car is a {$cars[0]}";
    ?>
    ```

## php 数据类型

1. 字符串

    使用单引号或双引号括起来。

    ```php
    <?php 
    $x = "Hello world!";
    echo $x;
    echo "<br>"; 
    $x = 'Hello world!';
    echo $x;
    ?>
    ```

1. 整数

    可以用三种格式规定整数：十进制、十六进制（前缀是 0x）或八进制（前缀是 0）。

    `var_dump()`会返回变量的数据类型和值：

    ```php
    <?php 
    $x = 5985;
    var_dump($x);
    echo "<br>"; 
    $x = -345; // 负数
    var_dump($x);
    echo "<br>"; 
    $x = 0x8C; // 十六进制数
    var_dump($x);
    echo "<br>";
    $x = 047; // 八进制数
    var_dump($x);
    ?>
    ```

* 逻辑

    ```php
    $x=true;
    $y=false;
    ```

* 数组

    数组在一个变量中存储多个值。

    ```php
    <?php 
    $cars=array("Volvo","BMW","SAAB");
    var_dump($cars);
    ?>
    ```

    在 PHP 中，有三种数组类型：

    1. 索引数组 - 带有数字索引的数组

        * 自动分配索引（索引从 0 开始）

            ```php
            $cars=array("porsche","BMW","Volvo");
            ```

        * 手动分配索引

            ```php
            $cars[0]="porsche";
            $cars[1]="BMW";
            $cars[2]="Volvo";
            ```

        Example:

        ```php
        <?php
        $cars=array("porsche","BMW","Volvo");
        echo "I like " . $cars[0] . ", " . $cars[1] . " and " . $cars[2] . ".";
        ?>
        ```

        `count()`用于返回数组的长度：

        ```php
        <?php
        $cars=array("porsche","BMW","Volvo");
        echo count($cars);
        ?>
        ```

        遍历索引数组：

        ```php
        <?php
        $cars=array("porsche","BMW","Volvo");
        $arrlength=count($cars);

        for($x=0;$x<$arrlength;$x++) {
        echo $cars[$x];
        echo "<br>";
        }
        ?>
        ```

    1. 关联数组 - 带有指定键的数组

        与 Python 里的字典差不多。

        ```php
        $age=array("Bill"=>"35","Steve"=>"37","Elon"=>"43");
        ```

        或者：

        ```php
        $age['Bill']="63";
        $age['Steve']="56";
        $age['Elon']="47";
        ```

        使用时指定键即可：

        ```php
        <?php
        $age=array("Bill"=>"63","Steve"=>"56","Elon"=>"47");
        echo "Elon is " . $age['Elon'] . " years old.";
        ?>
        ```

        遍历关联数组：

        ```php
        <?php
        $age=array("Bill"=>"63","Steve"=>"56","Elon"=>"47");

        foreach($age as $x=>$x_value) {
        echo "Key=" . $x . ", Value=" . $x_value;
        echo "<br>";
        }
        ?>
        ```

    1. 多维数组 - 包含一个或多个数组的数组

    **数组排序**



* 对象

    对象使用`class`来声明对象的类。

    ```php
    <?php
    class Car
    {
    var $color;
    function Car($color="green") {
        $this->color = $color;
    }
    function what_color() {
        return $this->color;
    }
    }
    ?>
    ```

* NULL 值

    `null`表示变量无值。可以通过把值设置为`null`，将变量清空。

    ```php
    <?php
    $x="Hello world!";
    $x=null;
    var_dump($x);
    ?>
    ```

## 字符串函数

## 常量

## 运算符

## 流程控制

* `if`

    ```php
    if (条件) {
        当条件为 true 时执行的代码;
    }
    ```

    Example:

    ```php
    <?php
    $t=date("H");

    if ($t<"20") {
    echo "Have a good day!";
    }
    ?>
    ```

* `if ... else`

    ```php
    if (条件) {
        条件为 true 时执行的代码;
    } else {
        条件为 false 时执行的代码;
    }
    ```

    Example:

    ```php
    <?php
    $t=date("H");

    if ($t<"20") {
    echo "Have a good day!";
    } else {
    echo "Have a good night!";
    }
    ?>
    ```

* `if ... elseif ... else`

    ```php
    if (条件) {
        条件为 true 时执行的代码;
    } elseif (condition) {
        条件为 true 时执行的代码;
    } else {
        条件为 false 时执行的代码;
    }
    ```

* `switch`

    ```php
    switch (expression)
    {
    case label1:
        expression = label1 时执行的代码 ;
        break;  
    case label2:
        expression = label2 时执行的代码 ;
        break;
    default:
        表达式的值不等于 label1 及 label2 时执行的代码;
    }
    ```

    Example:

    ```php
    <?php
    $favfruit="orange";

    switch ($favfruit) {
    case "apple":
        echo "Your favorite fruit is apple!";
        break;
    case "banana":
        echo "Your favorite fruit is banana!";
        break;
    case "orange":
        echo "Your favorite fruit is orange!";
        break;
    default:
        echo "Your favorite fruit is neither apple, banana, or orange!";
    }
    ?>
    ```

* `while`

    ```php
    while (条件为真) {
        要执行的代码;
    }
    ```

    Example:

    ```php
    <?php 
    $x=1; 

    while($x<=5) {
        echo "这个数字是：$x <br>";
        $x++;
    } 
    ?>
    ```

* `do ... while`

    ```php
    do {
    要执行的代码;
    } while (条件为真);
    ```

    Example:

    ```php
    <?php 
    $x=1; 

    do {
        echo "这个数字是：$x <br>";
        $x++;
    } while ($x<=5);
    ?>
    ```

* `for`

    ```php
    for (init counter; test counter; increment counter) {
        code to be executed;
    }
    ```

    Example:

    ```php
    <?php 
    for ($x=0; $x<=10; $x++) {
        echo "数字是：$x <br>";
    } 
    ?>
    ```

* `foreach`

    foreach 循环只适用于数组，并用于遍历数组中的每个键/值对。

    ```php
    foreach ($array as $value) {
    code to be executed;
    }
    ```

    Example:

    ```php
    <?php 
    $colors = array("red","green","blue","yellow"); 

    foreach ($colors as $value) {
    echo "$value <br>";
    }
    ?>
    ```

## 函数

```php
function functionName() {
  被执行的代码;
}
```

Example:

```php
<?php
function sayHi() {
  echo "Hello world!";
}

sayhi(); // 调用函数
?>
```

给函数传递参数：

```php
<?php
function familyName($fname,$year) {
  echo "$fname Zhang. Born in $year <br>";
}

familyName("Li","1975");
familyName("Hong","1978");
familyName("Tao","1983");
?>
```

参数的默认值：

```php
<?php
function setHeight($minheight=50) {
  echo "The height is : $minheight <br>";
}

setHeight(350);
setHeight(); // 将使用默认值 50
setHeight(135);
setHeight(80);
?>
```

函数的返回值：

```php
<?php
function sum($x,$y) {
  $z=$x+$y;
  return $z;
}

echo "5 + 10 = " . sum(5,10) . "<br>";
echo "7 + 13 = " . sum(7,13) . "<br>";
echo "2 + 4 = " . sum(2,4);
?>
```



## Miscellaneous

1. php 脚本以`<?php`开头，以`?>`结尾，默认扩展名是`.php`。

1. PHP 语句以分号结尾（;）。PHP 代码块的关闭标签也会自动表明分号（因此在 PHP 代码块的最后一行不必使用分号）。

1. 注释：

    ```php
    <!DOCTYPE html>
    <html>
    <body>

    <?php
    // 这是单行注释

    # 这也是单行注释

    /*
    这是多行注释块
    它横跨了
    多行
    */
    ?>

    </body>
    </html>
    ```

1. php 中，用户定义的函数、类和关键词，对大小写不敏感。而变量对大小写敏感。

1. php 调用外部程序

    1. `shell_exec`

        ```php
        <?php 

        $command = escapeshellcmd('/usr/custom/test.py');
        $output = shell_exec($command);
        echo $output;

        ?>
        ```

    1. `passthru`

        Execute an external program and display raw output.

        ```php
        passthru('/usr/bin/python2.7 /srv/http/assets/py/switch.py arg1 arg2');
        ```

    1. `exec`

        Execute an external program.

    1. `system`

        Execute an external program and display the output.
