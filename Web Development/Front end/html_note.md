# HTML Note

Reference: <https://developer.mozilla.org/en-US/docs/Web/HTML>

```html
<!DOCTYPE html>
<html>
<head>
<title>Page Title</title>
</head>
<body>

<h1>This is a Heading</h1>
<p>This is a paragraph.</p>

</body>
</html>
```

## 常用元素一览

1. `<!DOCTYPE html>`

    在 html 文件的最顶端，用于帮助浏览器正常显示页面。

1. `<h1></h1> ... <h6></h6>`

    用于定义标题。

1. `<p></p>`

    用于定义段落。

1. `<a></a>`

    用于定义超链接。

    Example:

    ```html
    <a href="https://www.w3schools.com">This is a link</a>
    ```

1. `<img>`

    显示图片。

    ```html
    <img src="w3schools.jpg" alt="W3Schools.com" width="104" height="142">
    ```

## DOM

### Document

除了其它的`Document`对象外，`document`也是一个`Document`对象。

**Properties**

* `children`

**Methods**

* `childElementCount`

## form

## Canvas

```html
<canvas id='myCanvas' width='200' height='100'></canvas>

<script>
    var c = document.getElementById('myCanvas');
    var ctx = c.getContext('2d');
    
    // draw a line
    ctx.moveTo(0, 0);
    ctx.lineTo(200, 100);
    ctx.stroke();

    // draw a circle
    ctx.beginPath();
    ctx.arc(95, 50, 40, 0, 2 * Math.PI);
    ctx.stroke();

    // draw a text
    ctx.font = '30px Arial';
    ctx.fillText('Hello, world', 10, 50);
    ctx.strokeText('Hello, world!', 10, 50);

    // draw image
    var img = document.getElementById('scream');
    ctx.drawImage(img, 10, 10);
</script>
```

## miscellaneous

1. html 对大小写不敏感