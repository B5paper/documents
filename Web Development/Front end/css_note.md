# CSS Note

CSS stands for Cascading Style Sheets.

## Selectors

Syntax:

`selector {declaration; declaration}`

例子：

`h1 {color:blue; font-size:12px;}`

或者换行写：

```css
p {
    color: red;
    text-align: center;
}
```

有五种 selector：

1. simple selector (select elements based on name, id, class)

1. combinator selectors (select elements based on a specific relationship between them)

1. pseudo-class selectors (select elements based on a certain state)

1. pseudo-elements selectors (select and style a part of an element)

1. attribute selectors (select elements based on an attribute or attribute value)

下面是一些例子：

* `.class`

* `.class1.class2`：选择 class 同时包含`class1`和`class2`的元素。

* `.class1 .class2`：在`class1`元素中，选择`class2`的元素。

* `#id`

    ```css
    #para1 {
        color: red
    }
    ```

* `*`：选择所有元素

* `element`：选择 tag 名称

* `element1,element2`：选择多个 tab 名称

    ```css
    h1, h2, p {
        text-align: center;
        color: red;
    }
    ```

* `element1 element2`：在`element1`中选择`element2`

* `element.class`：element 和 class 的结合。

* `element>element`：例子：`div > p`。Selects all `<p>` elements where the parent is a `<div>` element.

* `element+element`：例子：`div + p`。Selects the first `<p>` element that is placed immediately after `<div>` elements.

* `element1~element2`：例子：`p ~ ul`。Selects every `<ul>` element that is preceded by a `<p>` element.

* `[attribute]`：选择含有`attribute`的元素。

* `[attribute=value]`

* `[attribute~=value]`：选择`attribute`包含`value`单词的元素。

* `[attribute|=value]`：选择以`value`开始的元素。

* `[attribute^=value]`：选择以`value`开始的元素。

* `[attribute$=value]`：选择以`value`结尾的元素。

* `[attribute*=value]`：选择包含子字符串`value`的元素。

* `:active`

    `a:active`：selects the active link

* `::after`

    `P::after`：Insert something after the content of each `<p>` element。

* `::before`

    `p::before`：Insert something before the content of each `<p>` element.

* `:checked`

    `input:checked`：Selects every checked `<input>` element.

* `:default`

    `input:default`: Selects the default `<input>` element

* `:disabled`

* `:empty`

    `p:empty`：Selects every `<p>` element that has no children (including text nodes)

* `:enabled`

* `:first-child`

    `p:first-child`: Selects every `<p>` element that is the first child of its parent.

* `::first-letter`

    `p::first-letter`: Selects the first letter of every `<p>` element.

* `::first-line`

* `:first-of-type`

    `p:first-of-type`: Selects every `<p>` element that is the first `<p>` element of its parent

* `:focus`

* `:fullscreen`

    Selects the element that is in full-screen mode.

* `:hover`

* `:in-range`

    `input:in-range`: Selects input elements with a value within a specified range

* `:indeterminate`

    `input:indeterminate`: Selects input elements that are in an indeterminate state

* `:invalid`

* `:lang(language)`

    `p:lang(it)`: Selects every `<p>` element with a lang attribute equal to `"it"` (Italian)

* `:last-child`

    `p:last-child`: Selects every `<p>` element that is the last child of its parent.

* `:last-of-type`

* `:link`

    `a:link`: Selects all unvisited links

* `::marker`

    `::marker`: Selects the markers of list items.

* `:not(selector)`

    `:not(p)`: 选择不是`<p>`的元素

* `:nth-child(n)`

    `p:nth-child(2)`

* `:nth-last-child(n)`

* `:nth-last-of-type(n)`

* `:nth-of-type(n)`

* `:only-of-type`

    `p:only-of-type`: Select every `<p>` element that is the only `<p>` element of its parent.

* `:only-child`

* `:optional`

    `input:optional`: Selects input elements with no `"required"` attribute.

* `:out-of-range`

    `input:out-of-range`: Selects input elements with a value outside a specified range.

* `::placeholder`

    `input::placeholder`: Selects input elements with the `"placeholder"` attribute specified.

* `:read-only`

* `:read-write`

* `:required`

* `:root`

    `:root`: Selects the document's root element.

* `::selection`

    `::selection`: Selects the portion of an element that is selected by a user.

* `:target`

    `#news:target`: Selects the current activate `#news` element (clicked on a URL containing that anchor name)

* `:valid`

* `:visited`

    `a:visited`: Selects all visited links

## invoke a css

external:

```html
<!DOCTYPE html>
<html>
<head>
<link rel="stylesheet" href="mystyle.css">
</head>
<body>

<h1>This is a heading</h1>
<p>This is a paragraph.</p>

</body>
</html>
```

internal:

```html
<!DOCTYPE html>
<html>
<head>
<style>
body {
  background-color: linen;
}

h1 {
  color: maroon;
  margin-left: 40px;
}
</style>
</head>
<body>

<h1>This is a heading</h1>
<p>This is a paragraph.</p>

</body>
</html>
```

inline:

```html
<!DOCTYPE html>
<html>
<body>

<h1 style="color:blue;text-align:center;">This is a heading</h1>
<p style="color:red;">This is a paragraph.</p>

</body>
</html>
```

If some properties have been defined for the same selector (element) in different style sheets, the value from the last read style sheet will be used. 

比如这种情况，外部的设置会覆盖内部的：

```html
<head>
<style>
h1 {
  color: orange;
}
</style>
<link rel="stylesheet" type="text/css" href="mystyle.css">
</head>
```

而如果`<link>`写在`<style>`上面，那么`<style>`就会覆盖`<link>`。

All the styles in a page will "cascade" into a new "virtual" style sheet by the following rules, where number one has the highest priority:

1. Inline style (inside an HTML element)
2. External and internal style sheets (in the head section)
3. Browser default

## Miscellaneous

1. `calc()`可以做简单的运算

    Examples:

    ```css
    .p {
        width: calc(100px + 100px);
        width: calc(100% - 30px);
        width: calc(2em * 5);
        width: calc(var(--variable-width) + 20px);
    }
    ```