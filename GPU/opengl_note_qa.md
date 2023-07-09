# OpenGL Note QA

[unit]
[u_0]
请使用 glfw3 和 glew 创建一个空白窗口。
[u_1]
`main.c`:

```c
#include <GL/glew.h>
#include <GLFW/glfw3.h>  // glfw3.h 一定要写在 glew.h 的后面才行，不然会编译报错

int main()
{
    glfwInit();
    GLFWwindow *window = glfwCreateWindow(1024, 768, "opengl qa test", NULL, NULL);
    glfwMakeContextCurrent(window);
    glewInit();  // glewInit() 必须写在这里。如果写在 glfwInit(); 的下一行，那么会在运行时报错。
    glfwSetInputMode(window, GLFW_STICKY_KEYS, GL_TRUE);
    do {
        glClear(GL_COLOR_BUFFER_BIT);
        glfwSwapBuffers(window);
        glfwPollEvents();
    } while (glfwGetKey(window, GLFW_KEY_ESCAPE) != GLFW_PRESS &&
        glfwWindowShouldClose(window) == 0);
    return 0;
}
```

编译：

```bash
gcc main.c -lOpenGL -lglfw -lGLEW -o main
```

运行：

```bash
./main
```

效果：

出现一个黑色窗口，按 Esc 键退出。


[unit]
[u_0]
请不使用 shader 画出一个三角形。
[u_1]
```c
#include <stdio.h>
#include <stdlib.h>
#include <GL/glew.h>
#include <GLFW/glfw3.h>

int main()
{
    glfwInit();
    GLFWwindow* window = glfwCreateWindow(1024, 768, "opengl qa test", NULL, NULL);
    glfwMakeContextCurrent(window);
    glewInit();
    glfwSetInputMode(window, GLFW_STICKY_KEYS, GL_TRUE);
    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    static const GLfloat g_vertex_buffer_data[] = {
        -1.0f, -1.0f, 0.0f,
        1.0f, -1.0f, 0.0f,
        0.0f,  1.0f, 0.0f,
    };
    GLuint vertexbuffer;
    glGenBuffers(1, &vertexbuffer);
    glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer);
    glBufferData(GL_ARRAY_BUFFER, sizeof(g_vertex_buffer_data), g_vertex_buffer_data, GL_STATIC_DRAW);
    do {
        glClear(GL_COLOR_BUFFER_BIT);
		glEnableVertexAttribArray(0);
        glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer);
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, NULL);
        glDrawArrays(GL_TRIANGLES, 0, 3);
        glDisableVertexAttribArray(0);
        glfwSwapBuffers(window);
        glfwPollEvents();
    } while (glfwGetKey(window, GLFW_KEY_ESCAPE) != GLFW_PRESS &&
        glfwWindowShouldClose(window) == 0);

	glDeleteBuffers(1, &vertexbuffer);
    glfwTerminate();
    return 0;
}
```

编译：

```bash
gcc main.c -lGLEW -lglfw -lGL -o main
```

运行：

```bash
./main
```

[unit]
[u_0]
请使用 shader 画出一个三角形。
[u_1]
(empty)

