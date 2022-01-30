# VUE Note

Official Site: <https://cn.vuejs.org/>

## Introduction

* vue 可以使用模板语法来渲染 dom 元素

    ```html
    <div id="app">
    {{ message }}
    </div>
    ```

    ```js
    var app = new Vue({
    el: '#app',
    data: {
        message: 'Hello Vue!'
    }
    })
    ```

* 绑定元素的 attribute

    ```html
    <div id="app-2">
    <span v-bind:title="message">
        鼠标悬停几秒钟查看此处动态绑定的提示信息！
    </span>
    </div>
    ```

    ```js
    var app2 = new Vue({
    el: '#app-2',
    data: {
        message: '页面加载于 ' + new Date().toLocaleString()
    }
    })
    ```

* 控制一个元素是否显示

    ```html
    <div id="app-3">
    <p v-if="seen">现在你看到我了</p>
    </div>
    ```

    ```js
    var app3 = new Vue({
    el: '#app-3',
    data: {
        seen: true
    }
    })
    ```

* 绑定数组的数组来显示一个数据列表

    ```html
    <div id="app-4">
    <ol>
        <li v-for="todo in todos">
        {{ todo.text }}
        </li>
    </ol>
    </div>
    ```

    ```js
    var app4 = new Vue({
    el: '#app-4',
    data: {
        todos: [
        { text: '学习 JavaScript' },
        { text: '学习 Vue' },
        { text: '整个牛项目' }
        ]
    }
    })
    ```

* 监听事件

    ```html
    <div id="app-5">
    <p>{{ message }}</p>
    <button v-on:click="reverseMessage">反转消息</button>
    </div>
    ```

    ```js
    var app5 = new Vue({
    el: '#app-5',
    data: {
        message: 'Hello Vue.js!'
    },
    methods: {
        reverseMessage: function () {
        this.message = this.message.split('').reverse().join('')
        }
    }
    })
    ```

* 表单输入和数据状态间的双向绑定：

    ```html
    <div id="app-6">
    <p>{{ message }}</p>
    <input v-model="message">
    </div>
    ```

    ```js
    var app6 = new Vue({
    el: '#app-6',
    data: {
        message: 'Hello Vue!'
    }
    })
    ```

* 注册组件

    ```js
    // 定义名为 todo-item 的新组件
    Vue.component('todo-item', {
    template: '<li>这是个待办项</li>'
    })

    var app = new Vue(...)
    ```

    构建组件模板：

    ```html
    <ol>
    <!-- 创建一个 todo-item 组件的实例 -->
    <todo-item></todo-item>
    </ol>
    ```

    从父作用域将数据传到子组件：

    ```js
    Vue.component('todo-item', {
    // todo-item 组件现在接受一个
    // "prop"，类似于一个自定义 attribute。
    // 这个 prop 名为 todo。
    props: ['todo'],
    template: '<li>{{ todo.text }}</li>'
    })
    ```

    使用`v-bind`指令将待办项传到循环输出的每个组件中：

    ```js
    <div id="app-7">
    <ol>
        <!--
        现在我们为每个 todo-item 提供 todo 对象
        todo 对象是变量，即其内容可以是动态的。
        我们也需要为每个组件提供一个“key”，稍后再
        作详细解释。
        -->
        <todo-item
        v-for="item in groceryList"
        v-bind:todo="item"
        v-bind:key="item.id"
        ></todo-item>
    </ol>
    </div>
    ```

    一个假想的组件应用模板：

    ```html
    <div id="app">
    <app-nav></app-nav>
    <app-view>
        <app-sidebar></app-sidebar>
        <app-content></app-content>
    </app-view>
    </div>
    ```

## vue 实例

一个新的 vue 实例：

```js
var vm = new Vue({

})
```

数据与属性进行绑定：

```js
// 我们的数据对象
var data = { a: 1 }

// 该对象被加入到一个 Vue 实例中
var vm = new Vue({
  data: data
})

// 获得这个实例上的 property
// 返回源数据中对应的字段
vm.a == data.a // => true

// 设置 property 也会影响到原始数据
vm.a = 2
data.a // => 2

// ……反之亦然
data.a = 3
vm.a // => 3
```

注意只有初始化时的属性才是可以绑定的，后续添加的属性是无法绑定的。

`freeze()`可以阻止追踪变化：

```js
var obj = {
  foo: 'bar'
}

Object.freeze(obj)

new Vue({
  el: '#app',
  data: obj
})
```

这样 html 页面将不会再更新：

```html
<div id="app">
  <p>{{ foo }}</p>
  <!-- 这里的 `foo` 不会更新！ -->
  <button v-on:click="foo = 'baz'">Change it</button>
</div>
```

vue 还暴露了一些特殊的属性或方法，以`$`作为前缀，方便调用：

```js
var data = { a: 1 }
var vm = new Vue({
  el: '#example',
  data: data
})

vm.$data === data // => true
vm.$el === document.getElementById('example') // => true

// $watch 是一个实例方法
vm.$watch('a', function (newValue, oldValue) {
  // 这个回调将在 `vm.a` 改变后调用
})
```

生命周期勾子：

```js
new Vue({
  data: {
    a: 1
  },
  created: function () {
    // `this` 指向 vm 实例
    console.log('a is: ' + this.a)
  }
})
// => "a is: 1"
```

（不要在 property 或回调函数上使用箭头函数）

## 模板语法

插值：

```html
<span>Message: {{ msg }}</span>
<span v-once>这个将不会改变: {{ msg }}</span>
```

输出原始 html：

```html
<p>Using mustaches: {{ rawHtml }}</p>
<p>Using v-html directive: <span v-html="rawHtml"></span></p>
```

对属性进行绑定：

```html
<div v-bind:id="dynamicId"></div>
<button v-bind:disabled="isButtonDisabled">Button</button>
```

对于布尔 attribute (它们只要存在就意味着值为`true`)

vue 还支持简单的表达式：

```html
{{ number + 1 }}

{{ ok ? 'YES' : 'NO' }}

{{ message.split('').reverse().join('') }}

<div v-bind:id="'list-' + id"></div>
```

指令与指令的参数：

```html
<a v-bind:href="url">...</a>
<p v-if="seen">现在你看到我了</p>
<a v-on:click="doSomething">...</a>
<a v-bind:[attributeName]="url"> ... </a>
```

方括号表示动态参数，`attributeName`会被作为一个 JavaScript 表达式进行动态求值，求得的值将会作为最终的参数来使用。

修饰符：

```html
<form v-on:submit.prevent="onSubmit">...</form>
```

修饰符 (modifier) 是以半角句号`.`指明的特殊后缀，用于指出一个指令应该以特殊方式绑定。例如，`.prevent`修饰符告诉`v-on`指令对于触发的事件调用 event.`preventDefault()`。

`v-bind`和`v-on`的缩写：

```html
<!-- 完整语法 -->
<a v-bind:href="url">...</a>

<!-- 缩写 -->
<a :href="url">...</a>

<!-- 动态参数的缩写 (2.6.0+) -->
<a :[key]="url"> ... </a>
```

```html
<!-- 完整语法 -->
<a v-on:click="doSomething">...</a>

<!-- 缩写 -->
<a @click="doSomething">...</a>

<!-- 动态参数的缩写 (2.6.0+) -->
<a @[event]="doSomething"> ... </a>
```

## 计算属性和侦听器

computed property example:

```html
<div id="example">
  <p>Original message: "{{ message }}"</p>
  <p>Computed reversed message: "{{ reversedMessage }}"</p>
</div>
```

```js
var vm = new Vue({
  el: '#example',
  data: {
    message: 'Hello'
  },
  computed: {
    // 计算属性的 getter
    reversedMessage: function () {
      // `this` 指向 vm 实例
      return this.message.split('').reverse().join('')
    }
  }
})
```

计算属性的 setter：

```js
// ...
computed: {
  fullName: {
    // getter
    get: function () {
      return this.firstName + ' ' + this.lastName
    },
    // setter
    set: function (newValue) {
      var names = newValue.split(' ')
      this.firstName = names[0]
      this.lastName = names[names.length - 1]
    }
  }
}
// ...
```

侦听属性：

```html
<div id="watch-example">
  <p>
    Ask a yes/no question:
    <input v-model="question">
  </p>
  <p>{{ answer }}</p>
</div>
```

```js
<!-- 因为 AJAX 库和通用工具的生态已经相当丰富，Vue 核心代码没有重复 -->
<!-- 提供这些功能以保持精简。这也可以让你自由选择自己更熟悉的工具。 -->
<script src="https://cdn.jsdelivr.net/npm/axios@0.12.0/dist/axios.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/lodash@4.13.1/lodash.min.js"></script>
<script>
var watchExampleVM = new Vue({
  el: '#watch-example',
  data: {
    question: '',
    answer: 'I cannot give you an answer until you ask a question!'
  },
  watch: {
    // 如果 `question` 发生改变，这个函数就会运行
    question: function (newQuestion, oldQuestion) {
      this.answer = 'Waiting for you to stop typing...'
      this.debouncedGetAnswer()
    }
  },
  created: function () {
    // `_.debounce` 是一个通过 Lodash 限制操作频率的函数。
    // 在这个例子中，我们希望限制访问 yesno.wtf/api 的频率
    // AJAX 请求直到用户输入完毕才会发出。想要了解更多关于
    // `_.debounce` 函数 (及其近亲 `_.throttle`) 的知识，
    // 请参考：https://lodash.com/docs#debounce
    this.debouncedGetAnswer = _.debounce(this.getAnswer, 500)
  },
  methods: {
    getAnswer: function () {
      if (this.question.indexOf('?') === -1) {
        this.answer = 'Questions usually contain a question mark. ;-)'
        return
      }
      this.answer = 'Thinking...'
      var vm = this
      axios.get('https://yesno.wtf/api')
        .then(function (response) {
          vm.answer = _.capitalize(response.data.answer)
        })
        .catch(function (error) {
          vm.answer = 'Error! Could not reach the API. ' + error
        })
    }
  }
})
</script>
```

## Class 与 Style 绑定

我们可以传给`v-bind:class`一个对象，以动态地切换 class：

```html
<div
  class="static"
  v-bind:class="{ active: isActive, 'text-danger': hasError }"
></div>
```

```js
data: {
  isActive: true,
  hasError: false
}
```

结果渲染为：

```html
<div class="static active"></div>
```

绑定的数据对象不必内联定义在模板里，甚至可以成为一个 computed 属性：

```html
<div v-bind:class="classObject"></div>
```

```js
data: {
  isActive: true,
  error: null
},
computed: {
  classObject: function () {
    return {
      active: this.isActive && !this.error,
      'text-danger': this.error && this.error.type === 'fatal'
    }
  }
}
```

我们可以把一个数组传给`v-bind:class`，以应用一个 class 列表：

```html
<div v-bind:class="[activeClass, errorClass]"></div>
```

```js
data: {
  activeClass: 'active',
  errorClass: 'text-danger'
}
```

最终会被渲染为：

```html
<div class="active text-danger"></div>
```

也可以使用三元表达式：

```html
<div v-bind:class="[isActive ? activeClass : '', errorClass]"></div>
```

在数组中也可以使用对象语法：

```html
<div v-bind:class="[{ active: isActive }, errorClass]"></div>
```

当在一个自定义组件上使用 class property 时，这些 class 将被添加到该组件的根元素上面。这个元素上已经存在的 class 不会被覆盖。

我们可以用对象样式或数组样式绑定内联语法：

```html
<div v-bind:style="{ color: activeColor, fontSize: fontSize + 'px' }"></div>
```

```js
data: {
  activeColor: 'red',
  fontSize: 30
}
```

这样也是可以的：

```html
<div v-bind:style="styleObject"></div>
```

```js
data: {
  styleObject: {
    color: 'red',
    fontSize: '13px'
  }
}
```

还可以使用数组语法，将多个样式对象绑定到一个元素上：

```js
<div v-bind:style="[baseStyles, overridingStyles]"></div>
```

## 条件渲染

Examples:

```html
<h1 v-if="awesome">Vue is awesome!</h1>
```

```html
<h1 v-if="awesome">Vue is awesome!</h1>
<h1 v-else>Oh no 😢</h1>
```

vue 会复用已经渲染好的元素，如果想让元素独立地渲染，需要加上`key`属性：

```html
<template v-if="loginType === 'username'">
  <label>Username</label>
  <input placeholder="Enter your username" key="username-input">
</template>
<template v-else>
  <label>Email</label>
  <input placeholder="Enter your email address" key="email-input">
</template>
```

`v-show`也可以条件渲染，不过只是决定显示或不显示元素，并不会销毁元素。

Example:

```html
<h1 v-show="ok">Hello!</h1>
```

`v-show`不支持`<template>`，也不支持`<v-else>`。

## 列表渲染

使用`v-for`可以将一个数组渲染成列表之类的：

```html
<ul id="example-1">
  <li v-for="item in items" :key="item.message">
    {{ item.message }}
  </li>
</ul>
```

```js
var example1 = new Vue({
  el: '#example-1',
  data: {
    items: [
      { message: 'Foo' },
      { message: 'Bar' }
    ]
  }
})
```

`v-for`可以访问父作用域的 property，还支持当前项的索引：

```html
<ul id="example-2">
  <li v-for="(item, index) in items">
    {{ parentMessage }} - {{ index }} - {{ item.message }}
  </li>
</ul>
```

```js
var example2 = new Vue({
  el: '#example-2',
  data: {
    parentMessage: 'Parent',
    items: [
      { message: 'Foo' },
      { message: 'Bar' }
    ]
  }
})
```

还可以遍历一个对象的 property：

```html
<ul id="v-for-object" class="demo">
  <li v-for="value in object">
    {{ value }}
  </li>
</ul>
```

```js
new Vue({
  el: '#v-for-object',
  data: {
    object: {
      title: 'How to do lists in Vue',
      author: 'Jane Doe',
      publishedAt: '2016-04-10'
    }
  }
})
```

也可以提供第二个参数为 property 名称（键名）：

```html
<div v-for="(value, name) in object">
  {{ name }}: {{ value }}
</div>
```

还可以使用第 3 个参数作为索引：

```html
<div v-for="(value, name, index) in object">
  {{ index }}. {{ name }}: {{ value }}
</div>
```

为了保证 vue 每次都能按顺序遍历，可以为每项提供一个`key` attribute：

```html
<div v-for="item in items" v-bind:key="item.id">
  <!-- 内容 -->
</div>
```

vue 还为数组提供了一些方法，它们会触发视图更新：

* `push()`
* `pop()`
* `shift()`
* `unshift()`
* `splice()`
* `sort()`
* `reverse()`

我们还可以用`filter()`，`concat()`，`slice()`等方法，用新数组替换旧数组：

```js
example1.items = example1.items.filter(function (item) {
  return item.message.match(/Foo/)
})
```

如果我们想显示一个数组经过过滤或排序后的版本，那么可以使用计算属性：

```html
<li v-for="n in evenNumbers">{{ n }}</li>
```

```js
data: {
  numbers: [ 1, 2, 3, 4, 5 ]
},
computed: {
  evenNumbers: function () {
    return this.numbers.filter(function (number) {
      return number % 2 === 0
    })
  }
}
```

如果无法使用计算属性，那么可以使用 method：

```html
<ul v-for="set in sets">
  <li v-for="n in even(set)">{{ n }}</li>
</ul>
```

```js
data: {
  sets: [[ 1, 2, 3, 4, 5 ], [6, 7, 8, 9, 10]]
},
methods: {
  even: function (numbers) {
    return numbers.filter(function (number) {
      return number % 2 === 0
    })
  }
}
```

可以使用 range 型的`v-for`，索引从 1 开始：

```html
<div>
  <span v-for="n in 10">{{ n }} </span>
</div>
```

在`<template>`中使用`v-for`可以循环渲染一段包含多个元素的内容：

```html
<ul>
  <template v-for="item in items">
    <li>{{ item.msg }}</li>
    <li class="divider" role="presentation"></li>
  </template>
</ul>
```

## 事件处理

通常使用`v-on`来处理事件：

```html
<div id="example-2">
  <!-- `greet` 是在下面定义的方法名 -->
  <button v-on:click="greet">Greet</button>
</div>
```

```js
var example2 = new Vue({
  el: '#example-2',
  data: {
    name: 'Vue.js'
  },
  // 在 `methods` 对象中定义方法
  methods: {
    greet: function (event) {
      // `this` 在方法里指向当前 Vue 实例
      alert('Hello ' + this.name + '!')
      // `event` 是原生 DOM 事件
      if (event) {
        alert(event.target.tagName)
      }
    }
  }
})

// 也可以用 JavaScript 直接调用方法
example2.greet() // => 'Hello Vue.js!'
```

还可以在内联 js 语句中调用方法：

```html
<div id="example-3">
  <button v-on:click="say('hi')">Say hi</button>
  <button v-on:click="say('what')">Say what</button>
</div>
```

```js
new Vue({
  el: '#example-3',
  methods: {
    say: function (message) {
      alert(message)
    }
  }
})
```

可以使用特殊变量`$event`访问到原始的 DOM 事件：

```html
<button v-on:click="warn('Form cannot be submitted yet.', $event)">
  Submit
</button>
```

```js
// ...
methods: {
  warn: function (message, event) {
    // 现在我们可以访问原生事件对象
    if (event) {
      event.preventDefault()
    }
    alert(message)
  }
}
```

事件修饰符：

* `.stop`
* `.prevent`
* `.capture`
* `.self`
* `.once`
* `.passive`

```html
<!-- 阻止单击事件继续传播 -->
<a v-on:click.stop="doThis"></a>

<!-- 提交事件不再重载页面 -->
<form v-on:submit.prevent="onSubmit"></form>

<!-- 修饰符可以串联 -->
<a v-on:click.stop.prevent="doThat"></a>

<!-- 只有修饰符 -->
<form v-on:submit.prevent></form>

<!-- 添加事件监听器时使用事件捕获模式 -->
<!-- 即内部元素触发的事件先在此处理，然后才交由内部元素进行处理 -->
<div v-on:click.capture="doThis">...</div>

<!-- 只当在 event.target 是当前元素自身时触发处理函数 -->
<!-- 即事件不是从内部元素触发的 -->
<div v-on:click.self="doThat">...</div>
```

按键修饰符：

```html
<!-- 只有在 `key` 是 `Enter` 时调用 `vm.submit()` -->
<input v-on:keyup.enter="submit">

<input v-on:keyup.page-down="onPageDown">
```

你可以直接将`KeyboardEvent.key`暴露的任意有效按键名转换为 kebab-case 来作为修饰符。

在上述示例中，处理函数只会在`event.key`等于 PageDown 时被调用。

vue 还提供了绝大多数常用按键码的别名：

* `.enter`
* `.tab`
* `.delete`（捕获删除和退格键）
* `.esc`
* `.space`
* `.up`
* `.down`
* `.left`
* `.right`

系统修饰键：

* `.ctrl`
* `.alt`
* `.shift`
* `.meta`

`.excat`修饰符允许你控制由精确的系统修饰符组合触发的事件。

```html
<!-- 即使 Alt 或 Shift 被一同按下时也会触发 -->
<button v-on:click.ctrl="onClick">A</button>

<!-- 有且只有 Ctrl 被按下的时候才触发 -->
<button v-on:click.ctrl.exact="onCtrlClick">A</button>

<!-- 没有任何系统修饰符被按下的时候才触发 -->
<button v-on:click.exact="onClick">A</button>
```

鼠标按钮修饰符：

* `.left`
* `.right`
* `.middle`

## 表单输入绑定

`v-model`是语法糖，它负责监听用户的输入事件以更新数据，并对一些极端场景进行一些特殊处理。


* `text` 和 `textarea` 元素使用 `value` property 和 `input` 事件；
* `checkbox` 和 `radio` 使用 `checked` property 和 `change` 事件；
* select 字段将 `value` 作为 prop 并将 `change` 作为事件。

Examples:

```html
<input v-model="message" placeholder="edit me">
<p>Message is: {{ message }}</p>


<span>Multiline message is:</span>
<p style="white-space: pre-line;">{{ message }}</p>
<br>
<textarea v-model="message" placeholder="add multiple lines"></textarea>
```

单个复选框，绑定到布尔值：

```html
<input type="checkbox" id="checkbox" v-model="checked">
<label for="checkbox">{{ checked }}</label>
```

多个复选框，绑定到同一个数组：

```html
<input type="checkbox" id="jack" value="Jack" v-model="checkedNames">
<label for="jack">Jack</label>
<input type="checkbox" id="john" value="John" v-model="checkedNames">
<label for="john">John</label>
<input type="checkbox" id="mike" value="Mike" v-model="checkedNames">
<label for="mike">Mike</label>
<br>
<span>Checked names: {{ checkedNames }}</span>
```

```js
new Vue({
  el: '...',
  data: {
    checkedNames: []
  }
})
```

单选按钮：

```html
<div id="example-4">
  <input type="radio" id="one" value="One" v-model="picked">
  <label for="one">One</label>
  <br>
  <input type="radio" id="two" value="Two" v-model="picked">
  <label for="two">Two</label>
  <br>
  <span>Picked: {{ picked }}</span>
</div>
```

```js
new Vue({
  el: '#example-4',
  data: {
    picked: ''
  }
})
```

下拉选择：

```html
<div id="example-5">
  <select v-model="selected">
    <option disabled value="">请选择</option>
    <option>A</option>
    <option>B</option>
    <option>C</option>
  </select>
  <span>Selected: {{ selected }}</span>
</div>
```

```js
new Vue({
  el: '...',
  data: {
    selected: ''
  }
})
```

多选框：

```html
<div id="example-6">
  <select v-model="selected" multiple style="width: 50px;">
    <option>A</option>
    <option>B</option>
    <option>C</option>
  </select>
  <br>
  <span>Selected: {{ selected }}</span>
</div>
```

```js
new Vue({
  el: '#example-6',
  data: {
    selected: []
  }
})
```

用`v-for`渲染的动态选项：

```html
<select v-model="selected">
  <option v-for="option in options" v-bind:value="option.value">
    {{ option.text }}
  </option>
</select>
<span>Selected: {{ selected }}</span>
```

```js
new Vue({
  el: '...',
  data: {
    selected: 'A',
    options: [
      { text: 'One', value: 'A' },
      { text: 'Two', value: 'B' },
      { text: 'Three', value: 'C' }
    ]
  }
})
```

值绑定：

静态绑定：

```html
<!-- 当选中时，`picked` 为字符串 "a" -->
<input type="radio" v-model="picked" value="a">

<!-- `toggle` 为 true 或 false -->
<input type="checkbox" v-model="toggle">

<!-- 当选中第一个选项时，`selected` 为字符串 "abc" -->
<select v-model="selected">
  <option value="abc">ABC</option>
</select>
```

动态绑定（值是 vue 实例的一个动态 property）：

复选框：

```html
<input
  type="checkbox"
  v-model="toggle"
  true-value="yes"
  false-value="no"
>
```

```js
// 当选中时
vm.toggle === 'yes'
// 当没有选中时
vm.toggle === 'no'
```

单选按钮：

```html
<input type="radio" v-model="pick" v-bind:value="a">
```

```js
// 当选中时
vm.pick === vm.a
```

选择框的选项：

```html
<select v-model="selected">
    <!-- 内联对象字面量 -->
  <option v-bind:value="{ number: 123 }">123</option>
</select>
```

```js
// 当选中时
typeof vm.selected // => 'object'
vm.selected.number // => 123
```

修饰符：

* `.lazy`

    在默认情况下，`v-model`在每次`input`事件触发后将输入框的值与数据进行同步 (除了上述输入法组合文字时)。你可以添加`lazy`修饰符，从而转为在`change`事件_之后_进行同步：

    ```html
    <!-- 在“change”时而非“input”时更新 -->
    <input v-model.lazy="msg">
    ```

* `.number`

    如果想自动将用户的输入值转为数值类型，可以给`v-model`添加`number`修饰符：

    ```html
    <input v-model.number="age" type="number">
    ```

    这通常很有用，因为即使在`type="number"`时，HTML 输入元素的值也总会返回字符串。如果这个值无法被`parseFloat()`解析，则会返回原始的值。

* `.trim`

    如果要自动过滤用户输入的首尾空白字符，可以给`v-model`添加`trim`修饰符：

    ```html
    <input v-model.trim="msg">
    ```

## 组件基础

组件是可复用的 vue 实例：

```js
// 定义一个名为 button-counter 的新组件
Vue.component('button-counter', {
  data: function () {
    return {
      count: 0
    }
  },
  template: '<button v-on:click="count++">You clicked me {{ count }} times.</button>'
})
```

```html
<div id="components-demo">
  <button-counter></button-counter>
</div>
```

```js
new Vue({ el: '#components-demo' })
```

如果调用多次组件，那么每用一次组件，就会有一个它的新实例被创建：

```html
<div id="components-demo">
  <button-counter></button-counter>
  <button-counter></button-counter>
  <button-counter></button-counter>
</div>
```

一个组件的`data`必须是一个函数，因为每个实例都不相同：

```js
data: function () {
  return {
    count: 0
  }
}
```

全局注册：

```js
Vue.component('my-component-name', {
  // ... options ...
})
```

我们可以在 html 中给组件传递 property：

```js
Vue.component('blog-post', {
  props: ['title'],
  template: '<h3>{{ title }}</h3>'
})
```

```html
<blog-post title="My journey with Vue"></blog-post>
<blog-post title="Blogging with Vue"></blog-post>
<blog-post title="Why Vue is so fun"></blog-post>
```

也可以利用根 vue 组件中的数据动态地传递 prop：

```js
new Vue({
  el: '#blog-post-demo',
  data: {
    posts: [
      { id: 1, title: 'My journey with Vue' },
      { id: 2, title: 'Blogging with Vue' },
      { id: 3, title: 'Why Vue is so fun' }
    ]
  }
})
```

```html
<blog-post
  v-for="post in posts"
  v-bind:key="post.id"
  v-bind:title="post.title"
></blog-post>
```

如果我们的页面中有多个内容，一个一个传参会比较复杂：

```html
<blog-post
  v-for="post in posts"
  v-bind:key="post.id"
  v-bind:title="post.title"
  v-bind:content="post.content"
  v-bind:publishedAt="post.publishedAt"
  v-bind:comments="post.comments"
></blog-post>
```

我们可以使用传递对象+使用模板字符串的方式：

```html
<blog-post
  v-for="post in posts"
  v-bind:key="post.id"
  v-bind:post="post"
></blog-post>
```

```js
Vue.component('blog-post', {
  props: ['post'],
  template: `
    <div class="blog-post">
      <h3>{{ post.title }}</h3>
      <div v-html="post.content"></div>
    </div>
  `
})
```

如果想让父组件监听子组件事件，可以使用子组件调用`$emit()`发送一个 event，父组件修改对应的 property 值，然后值的变化最终反映在子组件中：

```js
new Vue({
  el: '#blog-posts-events-demo',
  data: {
    posts: [/* ... */],
    postFontSize: 1
  }
})
```

```html
<div id="blog-posts-events-demo">
  <div :style="{ fontSize: postFontSize + 'em' }">
    <blog-post
      v-for="post in posts"
      v-bind:key="post.id"
      v-bind:post="post"
    ></blog-post>
  </div>
</div>
```

```js
Vue.component('blog-post', {
  props: ['post'],
  template: `
    <div class="blog-post">
      <h3>{{ post.title }}</h3>
      <button>
        Enlarge text
      </button>
      <div v-html="post.content"></div>
    </div>
  `
})
```

```html
<blog-post
  ...
  v-on:enlarge-text="postFontSize += 0.1"
></blog-post>
```

```html
<button v-on:click="$emit('enlarge-text')">
  Enlarge text
</button>
```

如果想通过 html 元素直接向组件传递参数，可以使用 slot 机制：

```html
<alert-box>
  Something bad happened.
</alert-box>
```

```js
Vue.component('alert-box', {
  template: `
    <div class="demo-alert-box">
      <strong>Error!</strong>
      <slot></slot>
    </div>
  `
})
```

最终渲染出的结果长这样：

<strong>Error!</strong> Something bad happened.

动态组件：

```html
<!-- 组件会在 `currentTabComponent` 改变时改变 -->
<component v-bind:is="currentTabComponent"></component>
```

**动态添加/删除组件**

似乎最好用`v-for`配合`data`中的某个数组实现。

1. 一些特殊元素

    * `this.$emit`：消息的发送者（元素）

    * `this.$parent`：父元素

    * `this.$children`：所有子元素的数组

    * `this.$ref`：通过`ref`属性来定位。

        ```html
        <child @change='change' ref='myChild'></child>
        ```

        ```js
        this.$refs.myChild
        ```