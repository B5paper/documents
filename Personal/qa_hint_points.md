* flex 程序中，初始化的调用是`yylex();`，不是`yyflex();`

* 在 glfw get key 之前，需要`glfwPollEvents();`

* 在进入 while 循环前，必须执行`glEnableVertexAttribArray(0);`

* opengl routine 遗忘点

    1. 先 init glfw，再 init glew

    2. bind buffer 时，先有`GL_ARRAY_BUFFER`，再有`GL_ELEMENT_ARRAY_BUFFER`.

    3. enable 的是 vertex attrib array, vertex 在位置 1，attrib 在位置 2，array 在位置 3

    4. 解释 array buffer 时，使用的是 vertex attrib pointer，vertex 在位置 1，attrib 在位置 2，pointer 在位置 3

    5. 记得读取 shader 文件中的内容

    6. `glDrawElements()`中填的是使用的 index 的数量，如果画三角形，那么必须为 3 的倍数

* flex 程序，先是`%{ %}`，再`%% %%`

* `yywrap()`, `yylex()`中间没有下划线

* 获取 key 按键时，函数叫 glfw get key, 不叫 glfw key press

* bind buffer 时，target 是`GL_ARRAY_BUFFER`，不是`GL_VERTEX_ARRAY`。

* `glDrawArrays()`完后，还需要`glfwSwapBuffers()`才能显示内容。

* `alloc_chrdev_region()`时，第二个参数是 start，第三个参数才是 num dev。

* flex 程序在 init 时，是`yylex();`，不是`yyflex();`

* 在`\n {return 0;}`规则时，是`return 0;`，不是`return;`

* 重新进入已经 stop 的容器，使用的是`docker start`，不是直接`docker -ia`。