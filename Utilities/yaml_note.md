# YAML Note

Some learning materials that shoud be processed:

1. <https://www.jianshu.com/p/d777ae6b69f3>

1. <https://spacelift.io/blog/yaml>

1. <https://www.cloudbees.com/blog/yaml-tutorial-everything-you-need-get-started>

yaml 是 Yet Another Markup Language 的缩写。

与 yaml 相关的库：<https://yaml.org/>

Ref:

* <https://www.cloudbees.com/blog/yaml-tutorial-everything-you-need-get-started>

  一个 tutorial，写得挺好的。

yaml 文件以三条短线作为开始：

```yaml
---
xxxxx
xxxx
xxxxxxx
```

## 字典与数组

### 字典类型 dictionary (or map)

字典是一种无序映射：

```yaml
---
foo: bar  # string -> string
an_int: 3  # string -> integer
an_float:  3.14  # string -> float
```

上面的例子中`string`类型的对象被映射到了各种不同类型。

字典中的每一个项称为一个键值对（key-value pair）。

dictionary 类型有时也被称为 map。

### 数组类型 lists (or arrays)

可以在元素前加一个短斜线`-`表示有序列表。

```yaml
---
- val_1
- val_2
- val_3
```

因为是有序列表，所以这三个元素的顺序不会乱。

### yaml 与 json 的转换

yaml 与 json 可以互相转换，从而可以验证一些 yaml 语法是否是自己想要的效果。

一个在线转换的网站：<https://onlineyamltools.com/convert-yaml-to-json>

Example 1:

yaml 文件：

```yaml
foo: bar
an_int: 3
an_float: 3.14
```

对应的 json 文件：

```json
{
  "foo": "bar",
  "an_int": 3,
  "an_float": 3.14
}
```

Example 2:

yaml 文件：

```yaml
- val_1
- val_2
- val_3
```

对应的 json 文件：

```json
[
  "val_1",
  "val_2",
  "val_3"
]
```


### 嵌套类型

* 字典类型的值可以是字典

  ```yaml
  key_1:
    subkey_1: subval_1
    subkey_2: subval_2
    subkey_3: subval_3
  ```

  对应的 json 文件：

  ```json
  {
    "key_1": {
      "subkey_1": "subval_1",
      "subkey_2": "subval_2",
      "subkey_3": "subval_3"
    }
  }
  ```

* 字典类型的值也可以是列表（maps of lists）

  ```yaml
  key_1:
    - val_1
    - val_2
  ```

  对应的 json 文件：

  ```json
  {
    "key_1": [
      "val_1",
      "val_2"
    ]
  }
  ```

* 列表类型的值可以是字典

  ```yaml
  - key_1: val_1
    key_2: val_2
    key_3: val_3
  - key_4: val_4
    key_5: val_5
  ```

  对应的 json 文件：

  ```json
  [
    {
      "key_1": "val_1",
      "key_2": "val_2",
      "key_3": "val_3"
    },
    {
      "key_4": "val_4",
      "key_5": "val_5"
    }
  ]
  ```

* 列表类型的值也可以是列表

  ```yaml
  - - a
    - b
    - c
  ```

  对应的 json 文件：

  ```json
  [
    [
      "a",
      "b",
      "c"
    ]
  ]
  ```

  这种写法似乎不多见。

**缩进**

yaml 使用缩进表示从属关系。缩进可以是一个空格，也可以是多个。


我们可以通过 yaml 和 json 的互相转换来理解 yaml。

map 嵌套 map:

```yaml
key_1:
  subkey_1: val_1
  subkey_2: val_2
  subkey_3: val_3
```

有序列表：

```yaml
kay_1:
  - val_1
  - val_2
  - val_3
```

对应的 json：

```json
{
  "kay_1": [
    "val_1",
    "val_2",
    "val_3"
  ]
}
```

有序列表里也可以有 map：

```yaml
key_1:
  - val_1
  - subkey_1: subval_1
    subkey_2: subval_2
  - val_3
```

对应的 json 内容：

```json
{
  "key_1": [
    "val_1",
    {
      "subkey_1": "subval_1",
      "subkey_2": "subval_2"
    },
    "val_3"
  ]
}
```

## yaml 中的字符串

yaml 中的字符串可以用引号括起来，也可以不用：

```yaml
str_1: string_1
str_2: This is also a string.
str_3: "double quotes string"
str_4: 'single quotes string'
```

对应的 json 文件：

```json
{
  "str_1": "string_1",
  "str_2": "This is also a string.",
  "str_3": "double quotes string",
  "str_4": "single quotes string"
}
```

如果一串字符序列不再表示原本字符串的意思，而是表示控制字符，换页，制表等意思，那么这一串字符就叫作 escape character。

比如`\n`是一个反斜杠和一个字符`n`，作为 escape character 时，表示换行。

yaml 中对字符串不加单引号或双引号时，不对 escape character 处理。加双引号时，处理 escape character；加单引号时，同样不处理 escape character。

Example:

```yaml
str_1: haha\nhehe
str_2: "haha\nhehe"
str_3: 'haha\nhehe'
```

对应的 json 文件：

```json
{
  "str_1": "haha\\nhehe",
  "str_2": "haha\nhehe",
  "str_3": "haha\\nhehe"
}
```

如果字符串太长，一行写不下，我们可以用`>`把它拆分成多行来写：

```yaml
my_str: >
  hello,
  world
```

对应的 json 文件：

```json
{
  "my_str": "hello, world\n"
}
```

可以看到每换一行，就会在这一行单词的后面加一个空格。

如果不想使用`\n`代表换行，那么可以使用`|`写成所见即所得格式：

```yaml
my_str: |
  hello,
  world!
```

对应的 json 文件：

```json
{
  "my_str": "hello,\nworld!\n"
}
```

## Reference

### 基本类型

* 字符串

* Nulls

    空值在 yaml 中是一种类型。可以使用短横线（tilde, `-`）表示，也可以用`null`表示。

    ```yaml
    ---
    foo: ~
    bar: null
    ```

    对应的 json 文件：

    ```json
    {
        "foo": null,
        "bar": null
    }
    ```

* Booleans

    在 yaml 中使用`True`和`False`来表示布尔值。

    ```yaml
    ---
    a: True
    b: true
    c: False
    d: false
    e: On
    f: on
    g: Yes
    h: yes
    ```

    对应的 json 文件：

    ```json
    {
        "a": true,
        "b": true,
        "c": false,
        "d": false,
        "e": "On",
        "f": "on",
        "g": "Yes",
        "h": "yes"
    }
    ```

* array 既可以使用`-`写成多行的形式，也可以使用方括号`[]`简写

    ```yaml
    ---
    items: [ 1, 2, 3, 4, 5 ]
    names: [ "one", "two", "three", "four" ]
    ```

    对应的 json 文件：

    ```json
    {
        "items": [
            1,
            2,
            3,
            4,
            5
        ],
        "names": [
            "one",
            "two",
            "three",
            "four"
        ]
    }
    ```

* 字典也可以写成单行的形式

    ```yaml
    ---
    foo: { thing1: huey, thing2: louie, thing3: dewey }
    ```

    对应的 json 文件：

    ```json
    {
        "foo": {
            "thing1": "huey",
            "thing2": "louie",
            "thing3": "dewey"
        }
    }
    ```

* numeric types 数值类型

    ```yaml
    ---
    foo: 12345
    bar: 0x12d4  # hex
    plop: 023332  # octal

    foo: 1230.15
    bar: 1230150.0

    foo: .inf
    bar: -.Inf
    plop: .NAN
    ```

* `>+`, `>-`, `|+`, `|-`

    `>+`会在字符串末尾增加一个`\n`，`>-`则不会。`>`默认情况下是`>+`。

    ```yaml
    ---
    my str: >+
        hello
        world

    my_str_2: >-
        hello
        world
    ```

    对应的 json 文件：

    ```json
    {
        "my str": "hello world\n\n",
        "my_str_2": "hello world"
    }
    ```

    `|`同理：

    ```yaml
    my str: |+
        hello
        world

    my_str_2: |-
        hello
        world
    ```

    对应的 json 文件：

    ```json
    {
        "my str": "hello\nworld\n\n",
        "my_str_2": "hello\nworld"
    }
    ```

* 字符串中的单引号和双引号