# Java Note

运行 java 程序需要 JRE (java runtime environment)，编译 java 程序需要 JDK。这两者都可以在 java 官网下载。

Java 官网：<https://www.oracle.com/java/>

（好像得注册账号才能下载）

Java 有三个版本：

1. Jave SE：标准版
1. Java EE：企业版
1. Java ME：微型版

通常使用标准版。JDK 里已经包含 JRE 了。

ubuntu 安装 java: <https://www.cnblogs.com/ziyue7575/p/13898610.html>。（我记得我当时装的时候，直接使用 openjdk 就可以了）

编译：`javac <ClassName>.java`

运行：`java <ClassName>`

`classpath`会指定类加载器的搜索路径。如果指定了`classpath`后，那么类加载器就不会从当前路径搜索了。

注释：

1. 单行注释：`//`
1. 多行注释

    ```java
    /*
    comments
    */
    ```

hello world 程序：

```java
public class Test {
    public static void main(String[] args) {
        System.out.println("Hello world!");
    }
}
```

一个`.java`文件内可以定义多个类，但是`public class`类名必须和文件名保持一致。

文档注释：

```java
/**
* comments
*/
```

标识符可以由字母、数字、美元符号（`$`）和下划线（`_`）组成，但不能以数字开头。

## 变量类型

java 中的变量必须赋值后才能访问。

变量不能重复声明。

* 整型

    * `byte`，单个字节，有符号

    * `short`，两个字节，有符号

    * `int`，4 个字节，有符号

        整数默认的字面量为`int`型。`xxxL`或`xxxl`表示 long 类型。当字面量超出 int 的范围时，会编译阶段报错。

        `0xx`代表八进制，`0bxx`二进制，`0xxx`十六进制。

    * `long`，8 个字节，有符号

* 浮点型

    * `float`

        可以在字面量后加上`f`或`F`，将其作为`float`类型。

    * `double`

        java 中小数字面量的默认类型值为`double`。双精度浮点数可以`d`或`D`结尾，也可以省略。

* 字符型`char`，2 个字节，无符号

    因为两个字节，所以 char 类型可以直接支持汉字。

    常用的转义字符：`\t`，`\n`，`\'`，`\u`

* 布尔型`boolean`

类型转换：小容量可以直接转换成大容量，大容量转换成小容量会在编译阶段报错，但是可以强制类型转换，比如：

```java
long a = 1;
int b = (int) a;
```

另外，如果数值没有超过`byte`的表示范围，那么可以直接赋值给`byte`类型。`short`和`char`同理。

`char`，`short`，`byte`做混合运算时，会先转换成`int`，然后再运算。

多种类型混合运算时，最终的结果类型为最大容量类型。

* `null`常量只有一个值`null`，表示对象的引用为空。

## 表达式

一个单独的表达式，比如`123;`，在 java 里无法通过编译。

## 运算符

java 中有`++`和`--`，规则和 c++ 一样。

关系运算符：`<`，`>`，`==`，`<=`，`>=`，`!=`

逻辑运算符：`&`，`|`，`!`，`&&`，`||`

逻辑运算符要求两边都是布尔类型。

赋值运算符：`=`，`+=`，`-=`，`*=`，`/=`，`%=`。

```java
byte a = 1;
a += 1;  // 等价于 a = (byte) (a + 1);
a = a + 1;  // 编译器会报错
```

三目运算符`a ? b : c`，语法与 c++ 相同。

字符串拼接：`+`，当左操作数和右操作数分别为数字和字符串时，会将数字转换成字符串再拼接。

```java
int i = 1;
i = i++;
```

在 java 中和 c++ 中，`i`的值分别是多少？

## 控制语句

* `if...else...`

    ```java
    if (cond) {
        // do something
    } else {
        // do something else
    }

    if (cond1) {
        // ...
    } else if (cond2) {
        // ...
    } else {
        // ...
    }
    ```

* `switch`

    语法与 c 一模一样。

    ```java
    switch () {
        case v1:
            // ...
            break;
        case v2:
            // ...
            break;
        default:
            // ...
    }
    ```

    switch 支持`String`和`int`两种类型。如果是其它整数类型，会先转换成`int`类型。

    另一本书里写，switch 支持 byte, short, char, int, enum, String。

* `while`

    ```java
    while (condition) {
        // do something
    }

    do {
        // do something
    } while (condition);  // 别忘了分号
    ```

* `for`

    ```java
    for (int i = 0; i < n; ++i) {
        // do something
    }
    ```

## 方法

方法的重载与返回值类型无关，它需要满足两个条件：

1. 方法名相同
1. 参数个数或参数类型不同

## 数组

```java
int[] x = new int[10];

// 或者
int[] x;
x = new int[10];

int[] a = {1, 2, 3};
int[] a2 = new int[4];
int[] a3 = {};

myMethod(new int[]{1, 2, 3});  // 匿名数组

int[][] a = {{1, 2}, {3, 5, 3}, {4}};
```

可以用`length`属性获取数组的长度。数组索引从 0 开始。

基类数组中可以存储子类对象。

拷贝：

```java
System.arraycopy();
```


标准库中的`Arrays`类：

```java
import java.util.Arrays;

public class Test {
    public static void main(String[] args) {
        int[] arr = {3, 5, 1, 4, 2};
        int[] arr_copy = Arrays.copyOfRange(arr, 1, 7);  // 复制数组
        Arrays.sort(arr);  // 排序
        Arrays.fill(arr, 8);  // 填充数组
        System.out.println("转换数组为字符串： " + Arrays.toString(arr));
    }
}
```

`Arrays`工具类的一些方法：

* `static void sort(int[] a)`

* `static int binarySearch(Object[] a, Object key)`

    使用二分搜索法搜索指定数组，以获得指定对象。

* `static int[] copyOfRange(int[] original, int from, int to)`

    将指定数组的指定范围复制到一个新数组

* `static void fill(Object[] a, Object val)`

* `static String toString(int[] arr)`

## 类

```java
public class A {
    int val_a;
    String str;
}

A a = new A();
```

`public`类默认的属性为公有，可以对成员变量使用`private`修饰将其作为私有的。

```java
public class Test {
    public static void main(String[] args) {
        A a = new A();
    }
}

class A {
    int val;

    public int getVal() {
        return val;
    }

    public void setVal(int newVal) {
        val = newVal;
    }
}
```

静态代码块，在类加载时执行：

```java
class A {
    static {
        // do something
    }

    public static void main(String[] args) {
        // do otherthing
    }
}
```

静态变量的声明定义和静态代码块的执行按顺序执行。

静态代码块通常用于初始化静态成员变量。

实例语句块：

```java
class A {
    {
        // do something
    }

    public A() {

    }
}
```

在调用构造方法（创建对象）前，先调用实例语句块。

java 中的类没有先后顺序，不需要声明。

`this()`本身也可以作为构造函数。其作为构造函数时，只能出现在其余构造参数的第一行，且只能出现一次。`super()`同理。`super`和`this`都不能用在静态方法中。

子类的构造方法在第一行会默认隐式调用`super()`。因此要保证父类的无参数构造方法存在。

`super`的机制与`this`不同，因此`super`不能单独使用。

静态方法只能调用静态方法和访问静态成员变量。

Java 只允许单继承：

```java
class A {

}

class B extends A {

}
```

`Object`类是所有类的基类。

java 类中的成员变量可以直接赋值：

```java
class A {
    int a = 1;
}
```

覆盖：

1. 要求两个类有继承关系
1. 两个方法必须有相同的返回值，方法名，参数列表
1. 子类方法的访问权限不能更低，但可以更高
1. 重写之后的方法不能比之前的方法抛出更多的异常，但可以更少。

注意：

1. 私有方法无法覆盖
1. 构造方法不能被继承，也无法被覆盖
1. 静态方法覆盖没有意义

如果子类与父类中的函数的名称相同，参数列表相同，但返回值不同，那么编译时会报错，说无法覆盖。

`static`只能修饰成员变量，不能修饰局部变量，否则编译器会报错。

`final`:

* `final`修饰的类不能被继承
* `final`修饰的方法不能被子类重写
* `final`修饰的变量（成员变量和局部变量）是常量，并且只能赋值一次
* `final`修饰的成员变量必须手动初始化

    `static final`经常连用，他们修饰的变量称为常量，通常用大写字母加下划线命名。常量和静态变量都存储在方法区。

```java
final class A {}
public final void print() {}
final String NAME = "XXX";
```

抽象类：

```java
abstract class Animal {
    abstract int call();
}
```

如果一个类中定义了抽象方法，则该类必须定义为抽象类。`final`和`abstract`无法组合。

一个抽象类可以不包含任何抽象方法，只需要用`abstract`修饰即可。

抽象类无法被实例化。

如果子类不是抽象类，且未覆盖父类中的抽象方法，那么会报错。

接口：只存在常量和抽象方法的抽象类称为接口。

```java
interface Animal {
    String ANIMAL_ACTION = "动物的行为动作";
    void call();  // 不能有方法体
}

class Cow implements Animal {
    public void call() {
        System.out.println();
    }
}
```

接口中的全局常量默认使用`public static final`修饰。抽象方法默认使用`public abstract`修饰。实现一个接口类时，可用`implements`关键字。

（为什么不能直接用`extends`继承，而非要用`implements`？）

一个类实现一个接口，必须实现接口中所有的方法。

允许接口的多重继承：

```java
class A extends B, C {

}
```

一个类可以实现多个接口。如果要调用不同接口中的方法，只需要强制转型就可以。

同时存在继承和实现：`class A extends ... implements ...`

父类型的引用允许指向子类型的对象：

```java
Animal a2 = new Cat();
Animal a3 = new Bird();
```

java 中允许向上转型，也允许向下转型，但要求两个类有继承关系。

多态：编译和运行时是不同的状态。

```java
obj instanceof SomeClass  // true of false
```

对类型进行向下转换时，可以使用`instanceof`判断类型。

私有方法无法被覆盖。

覆盖时返回值类型可以向下转换，但是不可以向上转换。（没啥用）

java 中比较内置类型时，使用`==`；比较引用对象时，需要重写`equals()`方法。

常见的几个需要重写的方法：

```java
protected Object clone()
int hashCode()
String toString()
protected void finalize()
protected Object clone()
```

将指针置为空指针，即可减少对象的引用计数，之后会被垃圾回收器自动销毁：

```java
A obj = new A();
obj = null;
```

建议启动垃圾回收器：

```java
System.gc();
```

### 匿名内部类

```java
interface A {
    public void some_method();
}

new A() {  // 匿名内部类，不建议使用
    public void some_method() {
        System.out.println("hehe")
    }
}  // 所谓的匿名类，其实就是一个对象
```

### 工具类

**`String`**

```java
String str1 = "abc";
String str2 = new String();
String str3 = new String("abc");
```

垃圾回收器不会释放字面量。

常用方法：

* `int indexOf(int ch)`

* `int indexOf(String str)`

* `char charAt(int index)`

* `boolean endsWith(String suffix)`

* `int length()`

* `boolean equals(Object anObject)`

* `boolean isEmpty()`

* `boolean startsWith(String prefix)`

* `boolean contains(CharSequence cs)`

    判断字符串中是否包含指定的字符序列。

* `String toLowerCase()`

* `String toUpperCase()`

* `char[] toCharArray()`

    将字符串转换为一个字符数组

* `String replace(CharSequence oldstr, CharSequence newstr)`

* `String[] split(String regex)`

* `String substring(int beginIndex)`

* `String substring(int beginInex, int endIndex)`

* `String trim()`

    返回一个新字符串，它去除了原字符串首尾的空格。

`String`字符串是常量，一旦创建，其内容和长度不可改变。

**StringBuffer**

```java
StringBuffer sb = new StringBuffer("abcdef");
```

`StringBuffer`是线程安全的，`StringBuilder`是非线程安全的。

常用方法

* `StringBuffer append(char c)`

* `StringBuffer append(String s)`

* `StringBuffer insert(int offset, String str)`

* `StringBuffer delete(int start, int end)`

* `StringBuffer deleteCharAt(int index)`

* `StringBuffer replace(int start, int end, String s)`

* `void setCharAt(int index, char ch)`

* `StringBuffer reverse()`

* `String toString()`

注意这些方法都是 in-place 的。

**集合类**

集合类分为`Collection`和`Map`两种。

`Collection`包括`List`和`Set`两种。`List`又包括`ArrayList`，`LinkedList`，`Vector`三种。`Set`包括`HashSet`和`TreeSet`两种。`LinkedHashSet`是`HashSet`的子类。

`Collectioin`接口：

* `boolean add(Object o)`

* `boolean addAll(Collection c)`

* `void clear()`

* `boolean remove(Ojbect o)`

* `boolean removeAll(Collection o)`

* `boolean isEmpty()`

* `boolean contains(Object o)`

* `boolean containsAll(Collection c)`

* `Iterator iterator()`

* `int size()`

`List`接口：

* `void add(int index, Object element)`

* `boolean addAll(int index, Collection c)`

* `Object get(int index)`

* `Object remove(int index)`

* `Object set(int index, Object element)`

* `int indexOf(Object o)`

* `int lastIndexOf(Object o)`

* `List subList(int fromIndex, int toIndex)`

```java
// 指定类型
ArrayList<String> arr = new ArrayList<String>();
```

`HashSet`:

```java
import java.util.HashSet;

public static void main(String[] args) {
    HashSet s = new HashSet();
    s.add(123);
    s.add("abc");
    Iterator it = s.iterator();
    while (it.hasNext()) {
        Object obj = it.next();
        System.out.println(obj);
    }
}
```

`Map`接口：

* `void add(int index, Object element)`

* `boolean addAll(int index, Collectioin c)`

* `Object get(int index)`

* `Object remove(int index)`

* `Object set(int index, Object element)`

* `int indexOf(Object o)`

* `int lastIndexOf(Object o)`

* `List subList(int fromIndex, int toIndex)`

```java
public class Test {
    public static void main(String[] args) throws Exception {
        HashMap m = new HashMap();
        m.put("1", "nihao");
        m.put(2, "hello");
        Set ks = m.keySet();
        Iterator it = ks.iterator();
        while (it.hasNext()) {
            Object key = it.next();
            Object val = m.get(key);
            System.out.println(key + ", " + val);
        }
    }
}
```

```java
HashMap m = new HashMap();
m.put(1, "nihao");
Set entrySet = m.entrySet();
Iterator it = entrySet.iterator();
while (it.hasNext()) {
    Map.Entry entry = (Map.Entry)it.next();
    Object key = entry.getKey();
    Object value = entry.getValue();
}
```

`Properties`继承自`HashTable`，用于存储字符串的匹配：

```java
public class Test {
    public static void main(String[] args) throws Exception {
        Properties p = new Properties();
        p.setProperty("hello", "123");
        Set<String> names = p.stringPropertyNames();
        for (String key: names) {
            String val = p.getProperty(key);
            System.out.println(key + ", " + val);
        }
    }
}
```

**Iterator**

```java
import java.util.Iterator;
import java.util.ArrayList;

public class Test {
    public static void main(String[] args) {
        ArrayList list = new ArrayList();
        list.add("nihao");
        list.add(123);
        Iterator it = list.iterator();
        while (it.hasNext()) {
            Object obj = it.next();
            System.out.println(obj);
        }
        for (Object obj: list) {
            System.out.println(obj);
        }
    }
}
```


**数组**

java 中的数组是引用，存储在堆中，长度不可变，数组中的元素的内存地址连续。

`length`属性可获取元素个数。



**包装类**

将内置类型转换成引用类型，可以使用包装类：

`char` -> `Character`, `long` -> `Long`, `byte` -> `Byte`, `float` -> `Float`, `int` -> `Integer`, `double` -> `Double`, `short` -> `Short`, `boolean` -> `Boolean`

## IO

字节流：`java.io.InputStream`，`java.io.OutputStream`

字符流：`java.io.Reader`，`java.io.Writer`

`InputStream`类：

* `int read()`

    从流中读取一个字节。

* `int read(byte[] b)`

    读取指定数量的字节，并存储在缓冲区中

* `int read(byte[] b, int off, int len)`

    读取若干字节，存储到数组`b`的指定位置。

* `void close()`

`InputStream`类：

* `void write(int b)`

* `void write(byte[] b)`

* `void write(byte[] b, int off, int len)`

    从数组`b`的指定位置处将数据写入到输出流中。

* `void flush()`

* `void close()`

文件流：

* `FileInputStream`, `FileOutputStream`

    ```java
    import java.lang.String;
    import java.io.*;

    public class Test {
        public static void main(String[] args) throws Exception {
            FileOutputStream out = null;
            try {
                out = new FileOutputStream("d:/nihao.txt");
                String s = "hehehehe";
                byte[] arr = s.getBytes();
                for (int i = 0; i < arr.length; i++) {
                    out.write(arr);
                    out.write('\n');
                }
            } catch (IOException e) {
                e.printStackTrace();
            } finally {
                if (out != null) {
                    out.close();
                }
            }
        }
    }
    ```

按字节复制文件：

```java
import java.lang.String;
import java.io.*;

public class Test {
    public static void main(String[] args) throws Exception {  // 这个 throw Exception 必须写
        FileInputStream in = new FileInputStream("d:/nihao.jpg");
        FileOutputStream out = new FileOutputStream("d:/nihao2.jpg");
        int len;
        long startTime = System.currentTimeMillis();
        while ((len = in.read()) != -1) {
            out.write(len);
        }
        long endTime = System.currentTimeMillis();
        System.out.println("time consumption: " + (endTime - startTime) + "ms.");
        in.close();
        out.close();
    }
}
```

使用缓冲区复制文件：

```java
import java.lang.String;
import java.io.*;

public class Test {
    public static void main(String[] args) throws Exception {
        FileInputStream in = new FileInputStream("d:/nihao.jpg");
        FileOutputStream out = new FileOutputStream("d:/nihao2.jpg");
        int len;
        byte[] buffer = new byte[1024];
        long startTime = System.currentTimeMillis();
        while ((len = in.read(buffer)) != -1) {
            out.write(buffer, 0, len);
        }
        long endTime = System.currentTimeMillis();
        System.out.println("time consumption: " + (endTime - startTime) + "ms.");
        in.close();
        out.close();
    }
}
```

使用字节缓冲流，这个稍慢一点：

```java
import java.lang.String;
import java.io.*;

public class Test {
    public static void main(String[] args) throws Exception {
        FileInputStream in = new FileInputStream("d:/nihao.jpg");
        FileOutputStream out = new FileOutputStream("d:/nihao2.jpg");
        BufferedInputStream bufferIn = new BufferedInputStream(in);
        BufferedOutputStream bufferOut = new BufferedOutputStream(out);
        long startTime = System.currentTimeMillis();
        int length;
        while ((length = bufferIn.read()) != -1) {
            bufferOut.write(length);
        }
        long endTime = System.currentTimeMillis();
        System.out.println("time consumption: " + (endTime - startTime) + " ms.");
        bufferIn.close();
        bufferOut.close();
    }
}
```

字符流：

* `Reader()`, `Writer()`

* `FileReader()`, `FileWriter()`

这些类处理的不是`byte`数据，而是可以直接处理`String`数据，稍微方便一些。

字符缓冲流：

* `BufferedReader`, `BufferedWriter`

`BufferedReader`有一个`readLine()`方法，每次可以读一行，返回一个字符串。由于这两个类使用了缓冲区，所以只有当缓冲区满，或者调用`close()`时才会刷新缓冲区中的内容。

**File**

`java.io.File`类用于处理文件和目录。

* `File(String pathname)`
* `File(String parent, String child)`
* `File(File parent, String child)`

方法：

* `boolean exists()`
* `boolean delete()`
* `boolean createNewFile()`
* `String getName()`
* `String getPath()`
* `String isDirectory()`
* `long lastModified()`
* `long length()`
* `String[] list()`
* `File[] listFiles()`

## 线程

有两种方法创建子线程：

1. 继承`Thread`

2. 实现`Runnable`接口，然后使用`public Thread(Runnable target)`或`public Thread(Runnable target, String name)`将其作为成员。

通常推荐使用第二种方法，这种有两个优点：

1. 适合多个程序代码相同的线程处理同一资源的情况。

1. 可以避免由于Java的单继承特性带来的局限。

线程的五种状态：新建，就绪，运行，阻塞，死亡

线程不安全的一段代码：

```java
import java.lang.String;

public class Test {
    public static void main(String[] args) throws Exception {
        Ticket ticket = new Ticket();
        Thread t1 = new Thread(ticket, "window 1");
        Thread t2 = new Thread(ticket, "window 2");
        Thread t3 = new Thread(ticket, "window 3");
        Thread t4 = new Thread(ticket, "window 4");
        t1.start();
        t2.start();
        t3.start();
        t4.start();
    }
}

class Ticket implements Runnable {
    private int tickets = 100;
    public void run() {
        while (true) {
            try {
                if (tickets > 0) {
                    Thread.sleep(10);
                    String name = Thread.currentThread().getName();
                    System.out.println(name + " ticket " + tickets--);
                } else {
                    break;
                }
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }
    }
}
```

可以用`synchronized`修饰同步代码块：

```java
import java.lang.String;

public class Test {
    public static void main(String[] args) throws Exception {
        Ticket ticket = new Ticket();
        Thread t1 = new Thread(ticket, "window 1");
        Thread t2 = new Thread(ticket, "window 2");
        Thread t3 = new Thread(ticket, "window 3");
        Thread t4 = new Thread(ticket, "window 4");
        t1.start();
        t2.start();
        t3.start();
        t4.start();
    }
}

class Ticket implements Runnable {
    private int tickets = 100;
    public void run() {
        while (true) {
            synchronized (this) {
                try {
                    if (tickets > 0) {
                        Thread.sleep(10);
                        String name = Thread.currentThread().getName();
                        System.out.println(name + " ticket " + tickets--);
                    } else {
                        break;
                    }
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
            }
        }
    }
}
```

也可以用`synchronized`修饰方法，以获得同步的性能：

```java
权限修饰符 synchronized void myfunc(param) {
    // codes
}
```

被`synchronized`修饰的方法某一时刻只允许一个线程访问：

```java
import java.lang.String;

public class Test {
    public static void main(String[] args) throws Exception {
        Ticket ticket = new Ticket();
        Thread t1 = new Thread(ticket, "window 1");
        Thread t2 = new Thread(ticket, "window 2");
        Thread t3 = new Thread(ticket, "window 3");
        Thread t4 = new Thread(ticket, "window 4");
        t1.start();
        t2.start();
        t3.start();
        t4.start();
    }
}

class Ticket implements Runnable {
    private int tickets = 100;
    public void run() {
        while (true) {
            saleTicket();
            if (tickets <= 0) {
                break;
            }
        }
    }

    private synchronized void saleTicket() {
        if (tickets > 0) {
            try {
                Thread.sleep(10);
                String name = Thread.currentThread().getName();
                System.out.println(name + " ticket " + tickets--);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }
    }
}
```

## 异常

两种异常类型：

* `Error`: 通常是由系统产生
* `Exception`：通常是由程序产生

```java
try {
    // some code
} catch (Exception e) {
    // process e
    System.out.println(e.getMessage());
} finally {
    // ...
}
```

`finally`中的语句不会被`return`影响，在执行完`catch`后会继续执行。

抛出异常的方法和调用其的父方法都需要加上`Exception`：

```java
public class Test {
    public static void main(String[] args) throws Exception {
        div(1, 2);
    }

    public static int div(int x, int y) throws Exception {
        return x / y;
    }
}
```

## 访问控制

级别由小到大：

* `private`

    只能被该类中的方法访问，不能被其它类直接访问。

* `default`

    默认级别。可以被本类包中的所有类访问。

* `protected`

    可被同一包下的其他类访问，也能被不同包下的子类访问。

* `public`

    可被所有类访问。

## I/O

从标准输入读取内容。

```java
import java.util.Scanner;

public class Test {
    public static void main(String[] argv) {
        java.util.Scanner s = new Scanner(System.in);
        int i = s.nextInt();
        System.out.println(i);
        String hello = s.next();
        System.out.println(hello);
    }
}
```

可用`nextDouble()`获取标准输入的 double 值。

## 包

`package`只能出现在代码文件第一行。

常用的格式：

`package` + 公司域名倒序 + 项目名 + 模块名 + 功能名。此时类的名称发生变化。 

运行时使用`java com.pro.mod.func`

编译到当前目录：`javac -d . xxx.java`

`import xxx.*`

`java.lang`中的模块会自动导入。

## Miscellaneous

1. 函数按值传递参数，赋值按照引用传递。

函数参数是按值传递还是按引用传递？如果按值传递，那么在函数内赋值是按引用还是按值？

1. 小技巧，防止出现空指针异常

    ```java
    "123".equals(myString)
    ```

1. 成员变量直接初始化是在构造方法执行前还是后？

1. 需要继续学习的东西：

    1. tomcat：web 服务器，是一个 servlet 容器，用于动态加载 servlet 应用。

    1. servlet

    1. JSP (Java Server Pages)

    1. mysql 和 jdbc, c3p0 数据库连接池

    1. Struts2 框架

    1. Hibernate 框架（对 JDBC 的封装）

    1. Spring 框架

    1. cglib

    1. log4j

