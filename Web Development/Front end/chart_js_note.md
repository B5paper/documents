# Chart.js Note

Official site: <https://www.chartjs.org/>

## Introduction

首先需要使用 html 指定一个 canvas 来画图：

```html
<div>
  <canvas id="myChart"></canvas>
</div>
```

然后画一个简单的：

```js
var ctx = document.getElementById('myChart');
var myChart = new Chart(ctx, {
    type: 'bar',
    data: {
        labels: ['Red', 'Blue', 'Yellow', 'Green', 'Purple', 'Orange'],
        datasets: [{
            label: '# of Votes',
            data: [12, 19, 3, 5, 2, 3],
            backgroundColor: [
                'rgba(255, 99, 132, 0.2)',
                'rgba(54, 162, 235, 0.2)',
                'rgba(255, 206, 86, 0.2)',
                'rgba(75, 192, 192, 0.2)',
                'rgba(153, 102, 255, 0.2)',
                'rgba(255, 159, 64, 0.2)'
            ],
            borderColor: [
                'rgba(255, 99, 132, 1)',
                'rgba(54, 162, 235, 1)',
                'rgba(255, 206, 86, 1)',
                'rgba(75, 192, 192, 1)',
                'rgba(153, 102, 255, 1)',
                'rgba(255, 159, 64, 1)'
            ],
            borderWidth: 1
        }]
    },
    options: {
        scales: {
            y: {
                beginAtZero: true
            }
        }
    }
});
```

其中`type`指定了所画的图表的类型，`data`指定了所用的数据，`options`指定了细节。其中`options`里有个`plugins`指定了 title，legend 之类的东西，还挺重要的。这些东西可以参考文档。

`data`的输入有几种不同的方式，都可以在文件中查到。

总得来说，我觉得这个库挺难用的，如非必要，就别用了。

primitive:

```js
data: [20, 10]
labels: ['a', 'b']
```

object:

```js
data: [{x: 10, y: 20}, {x: 15, y: null}, {x: 20, y: 10}]

data: [{x:'2016-12-25', y:20}, {x:'2016-12-26', y:10}]

data: [{x:'Sales', y:20}, {x:'Revenue', y:10}]
```

这种情况下可以禁止解析：`parsing: false`，此时数据必须是已经排好序的。

```js
type: 'bar',
data: {
    datasets: [{
        data: [{id: 'Sales', nested: {value: 1500}}, {id: 'Purchases', nested: {value: 500}}]
    }]
},
options: {
    parsing: {
        xAxisKey: 'id',
        yAxisKey: 'nested.value'
    }
}
```

```js
data: {
    January: 10,
    February: 20
}
```