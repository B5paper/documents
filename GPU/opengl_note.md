# OpenGL Note

Ref: <www.opengl-tutorial.org>

## Quick start

需要安装的包：

```bash
apt install cmake make g++ libx11-dev libxi-dev libgl1-mesa-dev libglu1-mesa-dev libxrandr-dev libxext-dev libxcursor-dev libxinerama-dev libxi-dev
apt install libglew-dev libglfw3-dev
```

`main.c`:

```c
#include <stdio.h>
#include <stdlib.h>
#include <GL/glew.h>
#include <GLFW/glfw3.h>

int main()
{
    glewExperimental = true; // Needed for core profile
    if( !glfwInit() )
    {
        fprintf( stderr, "Failed to initialize GLFW\n" );
        return -1;
    }

    glfwWindowHint(GLFW_SAMPLES, 4); // 4x antialiasing
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3); // We want OpenGL 3.3
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE); // To make MacOS happy; should not be needed
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE); // We don't want the old OpenGL 

    // Open a window and create its OpenGL context
    GLFWwindow* window; // (In the accompanying source code, this variable is global for simplicity)
    window = glfwCreateWindow( 1024, 768, "Tutorial 01", NULL, NULL);
    if( window == NULL ){
        fprintf( stderr, "Failed to open GLFW window. If you have an Intel GPU, they are not 3.3 compatible. Try the 2.1 version of the tutorials.\n" );
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window); // Initialize GLEW
    glewExperimental=true; // Needed in core profile
    if (glewInit() != GLEW_OK) {
        fprintf(stderr, "Failed to initialize GLEW\n");
        return -1;
    }

    // Ensure we can capture the escape key being pressed below
    glfwSetInputMode(window, GLFW_STICKY_KEYS, GL_TRUE);

    do{
        // Clear the screen. It's not mentioned before Tutorial 02, but it can cause flickering, so it's there nonetheless.
        glClear( GL_COLOR_BUFFER_BIT );

        // Draw nothing, see you in tutorial 2 !

        // Swap buffers
        glfwSwapBuffers(window);
        glfwPollEvents();

    } // Check if the ESC key was pressed or the window was closed
    while( glfwGetKey(window, GLFW_KEY_ESCAPE ) != GLFW_PRESS &&
        glfwWindowShouldClose(window) == 0 );
    return 0;
}
```

编译：

```bash
g++ main.cpp -lOpenGL -lglfw -lGLEW -o main
```

运行：

```bash
./main
```

效果：出现一个窗口，按 Esc 键退出。

glfw 主要处理鼠标键盘等事件消息，glew 主要处理图形相关的东西。

注：

1. `glfwWindowHint()`的几行全都可以注释掉，不影响窗口的创建。

### triangle

#### Without shader

为了画三角形，首先需要创建一些顶点，这些顶点在 OpenGL 中以指定的类型存储起来，这种类型的对象叫 VAO (Vertex Array Object):

```cpp
GLuint VertexArrayID;
glGenVertexArrays(1, &VertexArrayID);
glBindVertexArray(VertexArrayID);
```

这里的`VertexArrayID`类似于 Linux 中的设备号。

Do this once your window is created (= after the OpenGL Context creation) and before any other OpenGL call.

实际上是在`glewInit()`后添加的这段代码。


Screen Coordinates:

* X in on your right
* Y is up
* Z is towards your back (yes, behind, not in front of you)  （从屏幕指向你）

The origin of the coordinates is at the center of the display window.

bind buffer:

```cpp
// This will identify our vertex buffer
GLuint vertexbuffer;
// Generate 1 buffer, put the resulting identifier in vertexbuffer
glGenBuffers(1, &vertexbuffer);
// The following commands will talk about our 'vertexbuffer' buffer
glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer);
// Give our vertices to OpenGL.
glBufferData(GL_ARRAY_BUFFER, sizeof(g_vertex_buffer_data), g_vertex_buffer_data, GL_STATIC_DRAW);
```

`glClearColor()SimpleFragmentShader`是如何确定哪个窗口的？

Full codes:

`main.cpp:`

```cpp
#include <stdio.h>
#include <stdlib.h>
#include <GL/glew.h>
#include <GL/glxew.h>
// #include <GL/freeglut.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include "shader.hpp"
using namespace glm;


int main()
{
    glewExperimental = true;

    if (!glfwInit()) {
        fprintf(stderr, "Failed to initialize GLFW\n");
        return -1;
    }

    GLFWwindow* window = glfwCreateWindow(1024, 768, "Triangle", NULL, NULL);
    if (window == NULL) {
        fprintf(stderr, "Failed to open GLFW window.\n");
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(window);

    if (glewInit() != GLEW_OK) {
        fprintf(stderr, "Failed to initialize GLEW\n");
        return -1;
    }

    // Ensure we can capture the escape key being pressed below
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
        // Clear the screen.
        glClear(GL_COLOR_BUFFER_BIT);

		glEnableVertexAttribArray(0);
        glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer);
		glVertexAttribPointer(
			0,                 // attribute 0. No particular reason for 0, but must match the layout in the shader.
			3,                 // size
			GL_FLOAT,          // type
			GL_FALSE,          // normalized?
			0,                 // stride
			(void*)0           // array buffer offset
		);

        // Draw the triangle!
        glDrawArrays(GL_TRIANGLES, 0, 3); // Starting from vertex 0; 3 vertices total -> 1 triangle
        glDisableVertexAttribArray(0);

        // Swap buffers
        glfwSwapBuffers(window);
        glfwPollEvents();

    } while (glfwGetKey(window, GLFW_KEY_ESCAPE) != GLFW_PRESS &&
            glfwWindowShouldClose(window) == 0);

    // Cleanup VBO
	glDeleteBuffers(1, &vertexbuffer);
    glfwTerminate();
    return 0;
}
```

编译：

```bash
g++ -g main.cpp -lGLEW -lglfw -lX11 -lXext -lXrandr -lXinerama -lGL -lrt -lm -lXi -lXcursor -lGLU
```

运行：

```bash
./a.out
```

#### With shader

`main.cpp:`

```cpp
#include <stdio.h>
#include <stdlib.h>
#include <GL/glew.h>
#include <GL/glxew.h>
// #include <GL/freeglut.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include "shader.hpp"
using namespace glm;

int main()
{
    glewExperimental = true; // Needed for core profile
    if( !glfwInit() )
    {
        fprintf( stderr, "Failed to initialize GLFW\n" );
        return -1;
    }

    // Open a window and create its OpenGL context
    GLFWwindow* window;
    window = glfwCreateWindow(1024, 768, "Tutorial 01", NULL, NULL);
    if( window == NULL ) {
        fprintf( stderr, "Failed to open GLFW window. If you have an Intel GPU, they are not 3.3 compatible. Try the 2.1 version of the tutorials.\n" );
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window); // Initialize GLEW

    glewExperimental = true; // Needed in core profile
    if (glewInit() != GLEW_OK) {
        fprintf(stderr, "Failed to initialize GLEW\n");
        return -1;
    }

    // Ensure we can capture the escape key being pressed below
    glfwSetInputMode(window, GLFW_STICKY_KEYS, GL_TRUE);
    glClearColor(0.0f, 0.0f, 0.4f, 0.0f);

    GLuint VertexArrayID;
    glGenVertexArrays(1, &VertexArrayID);
    glBindVertexArray(VertexArrayID);
   	GLuint programID = LoadShaders( "SimpleVertexShader.vertexshader", "SimpleFragmentShader.fragmentshader" );

    static const GLfloat g_vertex_buffer_data[] = {
        -1.0f, -1.0f, 0.0f,
        1.0f, -1.0f, 0.0f,
        0.0f,  1.0f, 0.0f,
    };

    // This will identify our vertex buffer
    GLuint vertexbuffer;
    // Generate 1 buffer, put the resulting identifier in vertexbuffer
    glGenBuffers(1, &vertexbuffer);
    // The following commands will talk about our 'vertexbuffer' buffer
    glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer);
    // Give our vertices to OpenGL.
    glBufferData(GL_ARRAY_BUFFER, sizeof(g_vertex_buffer_data), g_vertex_buffer_data, GL_STATIC_DRAW);

    do {
        // Clear the screen. It's not mentioned before Tutorial 02, but it can cause flickering, so it's there nonetheless.
        glClear( GL_COLOR_BUFFER_BIT );
        glUseProgram(programID);

        // Draw
		glEnableVertexAttribArray(0);
		glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer);
		glVertexAttribPointer(
			0,                  // attribute 0. No particular reason for 0, but must match the layout in the shader.
			3,                  // size
			GL_FLOAT,           // type
			GL_FALSE,           // normalized?
			0,                  // stride
			(void*)0            // array buffer offset
		);

        // Draw the triangle !
        glDrawArrays(GL_TRIANGLES, 0, 3); // Starting from vertex 0; 3 vertices total -> 1 triangle
        glDisableVertexAttribArray(0);

        // Swap buffers
        glfwSwapBuffers(window);
        glfwPollEvents();

    } // Check if the ESC key was pressed or the window was closed
    while( glfwGetKey(window, GLFW_KEY_ESCAPE ) != GLFW_PRESS &&
            glfwWindowShouldClose(window) == 0 );

    // Cleanup VBO  
	glDeleteBuffers(1, &vertexbuffer);
	glDeleteVertexArrays(1, &VertexArrayID);
	glDeleteProgram(programID);
    glfwTerminate();
    return 0;
}
```

`SimpleFragmentShader.fragmentshader`:

```glsl
#version 330 core

// Ouput data
out vec3 color;

void main()
{

	// Output color = red 
	color = vec3(1,0,0);

}
```

`SimpleVertexShader.vertexshader`:

```glsl
#version 330 core

// Input vertex data, different for all executions of this shader.
layout(location = 0) in vec3 vertexPosition_modelspace;

void main(){

    gl_Position.xyz = vertexPosition_modelspace;
    gl_Position.w = 1.0;

}
```

`shader.hpp`:

```cpp
#ifndef SHADER_HPP
#define SHADER_HPP

GLuint LoadShaders(const char * vertex_file_path,const char * fragment_file_path);

#endif
```

`shader.cpp`:

```cpp
#include <stdio.h>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <sstream>
using namespace std;

#include <stdlib.h>
#include <string.h>

#include <GL/glew.h>

#include "shader.hpp"

GLuint LoadShaders(const char * vertex_file_path,const char * fragment_file_path){

	// Create the shaders
	GLuint VertexShaderID = glCreateShader(GL_VERTEX_SHADER);
	GLuint FragmentShaderID = glCreateShader(GL_FRAGMENT_SHADER);

	// Read the Vertex Shader code from the file
	std::string VertexShaderCode;
	std::ifstream VertexShaderStream(vertex_file_path, std::ios::in);
	if(VertexShaderStream.is_open()){
		std::stringstream sstr;
		sstr << VertexShaderStream.rdbuf();
		VertexShaderCode = sstr.str();
		VertexShaderStream.close();
	}else{
		printf("Impossible to open %s. Are you in the right directory ? Don't forget to read the FAQ !\n", vertex_file_path);
		getchar();
		return 0;
	}

	// Read the Fragment Shader code from the file
	std::string FragmentShaderCode;
	std::ifstream FragmentShaderStream(fragment_file_path, std::ios::in);
	if(FragmentShaderStream.is_open()){
		std::stringstream sstr;
		sstr << FragmentShaderStream.rdbuf();
		FragmentShaderCode = sstr.str();
		FragmentShaderStream.close();
	}

	GLint Result = GL_FALSE;
	int InfoLogLength;

	// Compile Vertex Shader
	printf("Compiling shader : %s\n", vertex_file_path);
	char const * VertexSourcePointer = VertexShaderCode.c_str();
	glShaderSource(VertexShaderID, 1, &VertexSourcePointer , NULL);
	glCompileShader(VertexShaderID);

	// Check Vertex Shader
	glGetShaderiv(VertexShaderID, GL_COMPILE_STATUS, &Result);
	glGetShaderiv(VertexShaderID, GL_INFO_LOG_LENGTH, &InfoLogLength);
	if ( InfoLogLength > 0 ){
		std::vector<char> VertexShaderErrorMessage(InfoLogLength+1);
		glGetShaderInfoLog(VertexShaderID, InfoLogLength, NULL, &VertexShaderErrorMessage[0]);
		printf("%s\n", &VertexShaderErrorMessage[0]);
	}

	// Compile Fragment Shader
	printf("Compiling shader : %s\n", fragment_file_path);
	char const * FragmentSourcePointer = FragmentShaderCode.c_str();
	glShaderSource(FragmentShaderID, 1, &FragmentSourcePointer , NULL);
	glCompileShader(FragmentShaderID);

	// Check Fragment Shader
	glGetShaderiv(FragmentShaderID, GL_COMPILE_STATUS, &Result);
	glGetShaderiv(FragmentShaderID, GL_INFO_LOG_LENGTH, &InfoLogLength);
	if ( InfoLogLength > 0 ){
		std::vector<char> FragmentShaderErrorMessage(InfoLogLength+1);
		glGetShaderInfoLog(FragmentShaderID, InfoLogLength, NULL, &FragmentShaderErrorMessage[0]);
		printf("%s\n", &FragmentShaderErrorMessage[0]);
	}

	// Link the program
	printf("Linking program\n");
	GLuint ProgramID = glCreateProgram();
	glAttachShader(ProgramID, VertexShaderID);
	glAttachShader(ProgramID, FragmentShaderID);
	glLinkProgram(ProgramID);

	// Check the program
	glGetProgramiv(ProgramID, GL_LINK_STATUS, &Result);
	glGetProgramiv(ProgramID, GL_INFO_LOG_LENGTH, &InfoLogLength);
	if ( InfoLogLength > 0 ){
		std::vector<char> ProgramErrorMessage(InfoLogLength+1);
		glGetProgramInfoLog(ProgramID, InfoLogLength, NULL, &ProgramErrorMessage[0]);
		printf("%s\n", &ProgramErrorMessage[0]);
	}
	
	glDetachShader(ProgramID, VertexShaderID);
	glDetachShader(ProgramID, FragmentShaderID);
	
	glDeleteShader(VertexShaderID);
	glDeleteShader(FragmentShaderID);

	return ProgramID;
}
```

编译：

```g++
g++ -g shader.cpp main.cpp -lGLEW -lglfw -lX11 -lXext -lXrandr -lXinerama -lGL -lrt -lm -lXi -lXcursor -lGLU
```

## API Reference

### glfw

Ref: <https://www.glfw.org/docs/3.3/modules.html>

* `glfwInit()`

    Syntax:

    ```c
    int glfwInit(void);
    ```

    初始化函数，申请一些资源。在调用`glfwInit()`后，才可以调用大部分的 glfw 函数。

    在程序结束前需要调用`glfwTerminate()`，释放资源。

    Returns

    `GLFW_TRUE` if successful, or `GLFW_FALSE` if an error occurred.

    可以在程序中多次调用这个函数，但是只会生效一次。

* `glfwCreateWindow()`

    Syntax:

    ```c
    GLFWwindow *glfwCreateWindow(
        int width,
        int height,
        const char *title,
        GLFWmonitor *monitor,
        GLFWwindow *share
    );
    ```

    This function creates a window and its associated OpenGL or OpenGL ES context. Most of the options controlling how the window and its context should be created are specified with window hints.

    Successful creation does not change which context is current. Before you can use the newly created context, you need to make it current.

    The created window, framebuffer and context may differ from what you requested, as not all parameters and hints are hard constraints. This includes the size of the window, especially for full screen windows. To query the actual attributes of the created window, framebuffer and context, see `glfwGetWindowAttrib`, `glfwGetWindowSize` and `glfwGetFramebufferSize`.

    To create a full screen window, you need to specify the monitor the window will cover. If no monitor is specified, the window will be windowed mode. Unless you have a way for the user to choose a specific monitor, it is recommended that you pick the primary monitor.

    By default, newly created windows use the placement recommended by the window system. To create the window at a specific position, make it initially invisible using the GLFW_VISIBLE window hint, set its position and then show it.

    Parameters:

    * `width`: The desired width, in screen coordinates, of the window. This must be greater than zero.
    * `height`: The desired height, in screen coordinates, of the window. This must be greater than zero.
    * `title`: The initial, UTF-8 encoded window title.
    * `monitor`: The monitor to use for full screen mode, or `NULL` for windowed mode.
    * `share`: The window whose context to share resources with, or `NULL` to not share resources.

    Returns

    The handle of the created window, or `NULL` if an error occurred.

    Errors

    Possible errors include `GLFW_NOT_INITIALIZED`, `GLFW_INVALID_ENUM`, `GLFW_INVALID_VALUE`, `GLFW_API_UNAVAILABLE`, `GLFW_VERSION_UNAVAILABLE`, `GLFW_FORMAT_UNAVAILABLE` and `GLFW_PLATFORM_ERROR`.

* `glfwTerminate()`

    Syntax:

    ```c
    void glfwTerminate(void);
    ```

    This function destroys all remaining windows and cursors, restores any modified gamma ramps and frees any other allocated resources. Once this function is called, you must again call `glfwInit` successfully before you will be able to use most GLFW functions.

    If GLFW has been successfully initialized, this function should be called before the application exits. If initialization fails, there is no need to call this function, as it is called by `glfwInit` before it returns failure.

    This function has no effect if GLFW is not initialized.

* `glfwMakeContextCurrent()`

    Syntax:

    ```c
    void glfwMakeContextCurrent(GLFWwindow *window);
    ```

    This function makes the OpenGL or OpenGL ES context of the specified window current on the calling thread. A context must only be made current on a single thread at a time and each thread can have only a single current context at a time.

    When moving a context between threads, you must make it non-current on the old thread before making it current on the new one.

    By default, making a context non-current implicitly forces a pipeline flush. On machines that support GL_KHR_context_flush_control, you can control whether a context performs this flush by setting the GLFW_CONTEXT_RELEASE_BEHAVIOR hint.

    The specified window must have an OpenGL or OpenGL ES context. Specifying a window without a context will generate a `GLFW_NO_WINDOW_CONTEXT` error.

    Parameters:
    
    * `window`: The window whose context to make current, or NULL to detach the current context.

    意思好像是要指定在这个窗口上处理鼠标键盘消息。

* `glfwSetInputMode()`

    Sets an input option for the specified window.

    Syntax:

    ```c
    void glfwSetInputMode(GLFWwindow *window, int mode, int value);
    ```

    This function sets an input mode option for the specified window. The mode must be one of GLFW_CURSOR, GLFW_STICKY_KEYS, GLFW_STICKY_MOUSE_BUTTONS, GLFW_LOCK_KEY_MODS or GLFW_RAW_MOUSE_MOTION.

    If the mode is `GLFW_CURSOR`, the value must be one of the following cursor modes:

    * `GLFW_CURSOR_NORMAL` makes the cursor visible and behaving normally.

    * `GLFW_CURSOR_HIDDEN` makes the cursor invisible when it is over the content area of the window but does not restrict the cursor from leaving.

    * `GLFW_CURSOR_DISABLED` hides and grabs the cursor, providing virtual and unlimited cursor movement. This is useful for implementing for example 3D camera controls.

    If the mode is `GLFW_STICKY_KEYS`, the value must be either `GLFW_TRUE` to enable sticky keys, or `GLFW_FALSE` to disable it. If sticky keys are enabled, a key press will ensure that `glfwGetKey` returns `GLFW_PRESS` the next time it is called even if the key had been released before the call. This is useful when you are only interested in whether keys have been pressed but not when or in which order.  不是很懂这个，猜想：如果`glfwGetKey`下次被调用前，按键释放，那么依然会返回`GLFW_PRESS`。猜想二：如果在`glfwGetKey()`被调用前按下了多个键，那么在调用`glfwGetKey()`时会一起返回。

    If the mode is `GLFW_STICKY_MOUSE_BUTTONS`, the value must be either GLFW_TRUE to enable sticky mouse buttons, or GLFW_FALSE to disable it. If sticky mouse buttons are enabled, a mouse button press will ensure that glfwGetMouseButton returns GLFW_PRESS the next time it is called even if the mouse button had been released before the call. This is useful when you are only interested in whether mouse buttons have been pressed but not when or in which order.

    If the mode is `GLFW_LOCK_KEY_MODS`, the value must be either `GLFW_TRUE` to enable lock key modifier bits, or GLFW_FALSE to disable them. If enabled, callbacks that receive modifier bits will also have the GLFW_MOD_CAPS_LOCK bit set when the event was generated with Caps Lock on, and the GLFW_MOD_NUM_LOCK bit when Num Lock was on.

    If the mode is `GLFW_RAW_MOUSE_MOTION`, the value must be either GLFW_TRUE to enable raw (unscaled and unaccelerated) mouse motion when the cursor is disabled, or GLFW_FALSE to disable it. If raw motion is not supported, attempting to set this will emit GLFW_PLATFORM_ERROR. Call glfwRawMouseMotionSupported to check for support.

    Parameters:

    * `window`: The window whose input mode to set.

    * `mode`: One of `GLFW_CURSOR`, `GLFW_STICKY_KEYS`, `GLFW_STICKY_MOUSE_BUTTONS`, `GLFW_LOCK_KEY_MODS` or `GLFW_RAW_MOUSE_MOTION`.

    * `value`: The new value of the specified input mode.

* `glfwSwapBuffers()`

    Syntax:

    ```cpp
    void glfwSwapBuffers(GLFWwindow *window);
    ```

    This function swaps the front and back buffers of the specified window when rendering with OpenGL or OpenGL ES. If the swap interval is greater than zero, the GPU driver waits the specified number of screen updates before swapping the buffers.

    The specified window must have an OpenGL or OpenGL ES context. Specifying a window without a context will generate a `GLFW_NO_WINDOW_CONTEXT` error.

    This function does not apply to Vulkan. If you are rendering with Vulkan, see `vkQueuePresentKHR` instead.

    Parameters:
    
    * `window`: The window whose buffers to swap.

    不懂。为什么要交换 gpu 内存和前端内存？完全没头绪。

* `glfwPollEvents()`

* `glfwGetKey()`

* `glfwWindowShouldClose()`

    Syntax:

    ```c
    int glfwWindowShouldClose(GLFWwindow *window);
    ```

    This function returns the value of the close flag of the specified window.

    When the user attempts to close the window, for example by clicking the close widget or using a key chord like Alt+F4, the close flag of the window is set.

    Returns

    The value of the close flag.

    这个函数通常用于判断是否退出消息循环。

* `glGenVertexArrays()`

* `glBindVertexArray()`

    bind a vertex array object

    Syntax:

    ```cpp
    void glBindVertexArray(GLuint array);
    ```

    Parameters

    * `array`

        Specifies the name of the vertex array to bind. 

    Description

    `glBindVertexArray` binds the vertex array object with name array. array is the name of a vertex array object previously returned from a call to `glGenVertexArrays`, or zero to bind the default vertex array object binding.

    If no vertex array object with name array exists, one is created when array is first bound. If the bind is successful no change is made to the state of the vertex array object, and any previous vertex array object binding is broken.

    不懂。可能是一个窗口需要绑定一个 vertex array？

* `glfwDestroyWindow()`

    Syntax:

    ```c
    void glfwDestroyWindow (GLFWwindow *window);
    ```

    This function destroys the specified window and its context. On calling this function, no further callbacks will be called for that window.

    If the context of the specified window is current on the main thread, it is detached before being destroyed. 实际测的时候，窗口关不掉，可能是因为这个原因。有时间了再看看。


### GLEW

* `glewInit()`

* `glClear()`

    clear buffers to preset values

    Syntax:

    ```cpp
    void glClear(GLbitfield mask);
    ```

    Parameters:

    * `mask`: Bitwise OR of masks that indicate the buffers to be cleared. The three masks are GL_COLOR_BUFFER_BIT, GL_DEPTH_BUFFER_BIT, and GL_STENCIL_BUFFER_BIT.

    `glClear` sets the bitplane area of the window to values previously selected by glClearColor, glClearDepthf, and glClearStencil. Multiple color buffers can be cleared simultaneously by selecting more than one buffer at a time using glDrawBuffers.

    The pixel ownership test, the scissor test, sRGB conversion, dithering, and the buffer writemasks affect the operation of glClear. The scissor box bounds the cleared region. Alpha function, blend function, stenciling, texture mapping, and depth-buffering are ignored by glClear.

    `glClear` takes a single argument that is the bitwise OR of several values indicating which buffer is to be cleared.

    The values are as follows:

    * `GL_COLOR_BUFFER_BIT`: Indicates the buffers currently enabled for color writing.
    * `GL_DEPTH_BUFFER_BIT`: Indicates the depth buffer.
    * `GL_STENCIL_BUFFER_BIT`: Indicates the stencil buffer.

    The value to which each buffer is cleared depends on the setting of the clear value for that buffer. 
    
    常用的是`GL_COLOR_BUFFER_BIT`。

* `glGenBuffers`

    Syntax:

    ```cpp
    void glGenBuffers(GLsizei n, GLuint *buffers);
    ```

    Parameters:

    * `n`: Specifies the number of buffer object names to be generated.

    * `buffers`: Specifies an array in which the generated buffer object names are stored.

    `glGenBuffers` returns `n` buffer object names in buffers. There is no guarantee that the names form a contiguous set of integers; however, it is guaranteed that none of the returned names was in use immediately before the call to glGenBuffers.

    Buffer object names returned by a call to glGenBuffers are not returned by subsequent calls, unless they are first deleted with glDeleteBuffers.

    The names returned in buffers are marked as used, for the purposes of glGenBuffers only, but they acquire state and type only when they are first bound by calling glBindBuffer.

    这个函数功能有点像 Linux 驱动中的申请设备号。

    Errors:

    `GL_INVALID_VALUE` is generated if n is negative.

* `glBindBuffer`

    bind a named buffer object

    Syntax:

    ```cpp
    void glBindBuffer(GLenum target, GLuint buffer);
    ```

    Parameters:

    * `target`: Specifies the target to which the buffer object is bound, which must be one of the buffer binding targets in the following table:

        * `GL_ARRAY_BUFFER`: Vertex attributes
        * `GL_ATOMIC_COUNTER_BUFFER`: Atomic counter storage
        * `GL_COPY_READ_BUFFER`: Buffer copy source
        * `GL_COPY_WRITE_BUFFER`: Buffer copy destination
        * `GL_DISPATCH_INDIRECT_BUFFER`: Indirect compute dispatch commands
        * `GL_DRAW_INDIRECT_BUFFER`: Indirect command arguments
        * `GL_ELEMENT_ARRAY_BUFFER`: Vertex array indices
        * `GL_PIXEL_PACK_BUFFER`: Pixel read target
        * `GL_PIXEL_UNPACK_BUFFER`: Texture data source
        * `GL_SHADER_STORAGE_BUFFER`: Read-write storage for shaders
        * `GL_TEXTURE_BUFFER`: Texture data buffer
        * `GL_TRANSFORM_FEEDBACK_BUFFER`: Transform feedback buffer
        * `GL_UNIFORM_BUFFER`: Uniform block storage

    * `buffer`: Specifies the name of a buffer object.

    `glBindBuffer` binds a buffer object to the specified buffer binding point. Calling glBindBuffer with target set to one of the accepted symbolic constants and buffer set to the name of a buffer object binds that buffer object name to the target. If no buffer object with name buffer exists, one is created with that name. When a buffer object is bound to a target, the previous binding for that target is automatically broken. 

    这个函数看上去是在声明 buffer 的用途。

* `glBufferData`

    creates and initializes a buffer object's data store

    Syntax:

    ```cpp
    void glBufferData(
        GLenum target,
        GLsizeiptr size,
        const void * data,
        GLenum usage
    );
    ```

    Parameters:

    * `target`: 同`glBindBuffer`中的 target

    * `size`: Specifies the size in bytes of the buffer object's new data store. 

    * `data`: Specifies a pointer to data that will be copied into the data store for initialization, or NULL if no data is to be copied.

    * `usage`: Specifies the expected usage pattern of the data store. The symbolic constant must be `GL_STREAM_DRAW`, `GL_STREAM_READ`, `GL_STREAM_COPY`, `GL_STATIC_DRAW`, `GL_STATIC_READ`, `GL_STATIC_COPY`, `GL_DYNAMIC_DRAW`, `GL_DYNAMIC_READ`, or `GL_DYNAMIC_COPY`.

    `glBufferData` creates a new data store for the buffer object currently bound to target. Any pre-existing data store is deleted. The new data store is created with the specified size in bytes and usage. If data is not NULL, the data store is initialized with data from this pointer. In its initial state, the new data store is not mapped, it has a NULL mapped pointer, and its mapped access is `GL_READ_WRITE`.

    `usage` is a hint to the GL implementation as to how a buffer object's data store will be accessed. This enables the GL implementation to make more intelligent decisions that may significantly impact buffer object performance. It does not, however, constrain the actual usage of the data store. usage can be broken down into two parts: first, the frequency of access (modification and usage), and second, the nature of that access. The frequency of access may be one of these:

    * `STREAM`: The data store contents will be modified once and used at most a few times.
    * `STATIC`: The data store contents will be modified once and used many times.
    * `DYNAMIC`: The data store contents will be modified repeatedly and used many times.

    The nature of access m
    * `COPY`: The data store contents are modiay be one of these:

    * `DRAW`: The data store contents are modified by the application, and used as the source for GL drawing and image specification commands.

    * `READ`: The data store contents are modified by reading data from the GL, and used to return that data when queried by the application.

    * `COPY`: The data store contents are modified by reading data from the GL, and used as the source for GL drawing and image specification commands.

    这个函数并没有和具体的 buffer 交互，猜测有可能是注册全局数据。看来是一次性给够所有的数据。

* `glEnableVertexAttribArray`, `glDisableVertexAttribArray`

    Enable or disable a generic vertex attribute array

    Syntax:

    ```cpp
    void glEnableVertexAttribArray(GLuint index);
    void glDisableVertexAttribArray(GLuint index);
    ```

    Parameters:

    * `index`: Specifies the index of the generic vertex attribute to be enabled or disabled.

    `glEnableVertexAttribArray` enables the generic vertex attribute array specified by index. `glDisableVertexAttribArray` disables the generic vertex attribute array specified by index. By default, all generic vertex attribute arrays are disabled. If enabled, the values in the generic vertex attribute array will be accessed and used for rendering when calls are made to vertex array commands such as `glDrawArrays`, `glDrawArraysInstanced`, `glDrawElements`, `glDrawElementsInstanced`, or `glDrawRangeElements`.

    这个函数像是一个开关，告诉哪些 vertices 可以参与渲染，哪些 vertices 被禁止渲染。

* `glClearColor`

    specify clear values for the color buffers

    ```cpp
    void glClearColor(
        GLfloat red,
        GLfloat green,
        GLfloat blue,
        GLfloat alpha
    );
    ```

    Parameters:

    * `red, green, blue, alpha`

        Specify the red, green, blue, and alpha values used when the color buffers are cleared. The initial values are all 0.

    `glClearColor` specifies the red, green, blue, and alpha values used by `glClear` to clear fixed- and floating-point color buffers. Unsigned normalized fixed point RGBA color buffers are cleared to color values derived by clamping each component of the clear color to the range `[0,1]`, then converting the (possibly sRGB converted and/or dithered) color to fixed-point.

* `glVertexAttribPointer()`

    define an array of generic vertex attribute data

    Syntax:

    ```cpp
    void glVertexAttribPointer(
        GLuint index,
        GLint size,
        GLenum type,
        GLboolean normalized,
        GLsizei stride,
        const void * pointer);
    
    void glVertexAttribIPointer(
        GLuint index,
        GLint size,
        GLenum type,
        GLsizei stride,
        const void * pointer);
    ```

    Parameters

    * `index`: Specifies the index of the generic vertex attribute to be modified.

    * `size`: Specifies the number of components per generic vertex attribute. Must be 1, 2, 3, 4. The initial value is 4.

    * `type`: Specifies the data type of each component in the array. The symbolic constants `GL_BYTE`, `GL_UNSIGNED_BYTE`, `GL_SHORT`, `GL_UNSIGNED_SHORT`, `GL_INT`, and `GL_UNSIGNED_INT` are accepted by both functions. Additionally `GL_HALF_FLOAT`, `GL_FLOAT`, `GL_FIXED`, `GL_INT_2_10_10_10_REV`, and `GL_UNSIGNED_INT_2_10_10_10_REV` are accepted by `glVertexAttribPointer`. The initial value is `GL_FLOAT`.

    * `normalized`: For `glVertexAttribPointer`, specifies whether fixed-point data values should be normalized (`GL_TRUE`) or converted directly as fixed-point values (`GL_FALSE`) when they are accessed. This parameter is ignored if type is `GL_FIXED`.

    * `stride`: Specifies the byte offset between consecutive generic vertex attributes. If stride is 0, the generic vertex attributes are understood to be tightly packed in the array. The initial value is 0.

    * `pointer`: Specifies a pointer to the first generic vertex attribute in the array. If a non-zero buffer is currently bound to the `GL_ARRAY_BUFFER` target, pointer specifies an offset of into the array in the data store of that buffer. The initial value is 0. 

    Description

    `glVertexAttribPointer` and `glVertexAttribIPointer` specify the location and data format of the array of generic vertex attributes at index index to use when rendering. `size` specifies the number of components per attribute and must be 1, 2, 3 or 4. type specifies the data type of each component, and stride specifies the byte stride from one attribute to the next, allowing vertices and attributes to be packed into a single array or stored in separate arrays.

    For `glVertexAttribPointer`, if normalized is set to GL_TRUE, it indicates that values stored in an integer format are to be mapped to the range [-1,1] (for signed values) or [0,1] (for unsigned values) when they are accessed and converted to floating point. Otherwise, values will be converted to floats directly without normalization.

    For glVertexAttribIPointer, only the integer types GL_BYTE, GL_UNSIGNED_BYTE, GL_SHORT, GL_UNSIGNED_SHORT, GL_INT, GL_UNSIGNED_INT are accepted. Values are always left as integer values.

    If a non-zero named buffer object is bound to the GL_ARRAY_BUFFER target (see glBindBuffer), pointer is treated as a byte offset into the buffer object's data store and the buffer object binding (GL_ARRAY_BUFFER_BINDING) is saved as generic vertex attribute array state (GL_VERTEX_ATTRIB_ARRAY_BUFFER_BINDING) for index index.

    Client vertex arrays (a binding of zero to the GL_ARRAY_BUFFER target) are only valid in conjunction with the zero named vertex array object. This is provided for backwards compatibility with OpenGL ES 2.0.

    When a generic vertex attribute array is specified, size, type, normalized, stride, and pointer are saved as vertex array state, in addition to the current vertex array buffer object binding.

    To enable and disable a generic vertex attribute array, call glEnableVertexAttribArray and glDisableVertexAttribArray with index. If enabled, the generic vertex attribute array is used when glDrawArrays, glDrawArraysInstanced, glDrawElements, glDrawElementsIntanced, or glDrawRangeElements is called.

    看起来是在描述前面 vertex array 的数据结构。

* `glDrawArrays`

    render primitives from array data

    Syntax:

    ```cpp
    void glDrawArrays(
        GLenum mode,
        GLint first,
        GLsizei count
    );
    ```

    Parameters

    * `mode`: Specifies what kind of primitives to render. Symbolic constants `GL_POINTS`, `GL_LINE_STRIP`, `GL_LINE_LOOP`, `GL_LINES`, `GL_LINE_STRIP_ADJACENCY`, `GL_LINES_ADJACENCY`, `GL_TRIANGLE_STRIP`, `GL_TRIANGLE_FAN`, `GL_TRIANGLES`, `GL_TRIANGLE_STRIP_ADJACENCY`, `GL_TRIANGLES_ADJACENCY` and `GL_PATCHES` are `accepted`.

    * `first`: Specifies the starting index in the enabled arrays.
    
    * `count`: Specifies the number of indices to be rendered.

    Description

    `glDrawArrays` specifies multiple geometric primitives with very few subroutine calls. It is possible to prespecify separate arrays of attributes and use them to construct a sequence of primitives with a single call to glDrawArrays.

    When `glDrawArrays` is called, it uses count sequential elements from each enabled array to construct a sequence of geometric primitives, beginning with element first. mode specifies what kind of primitives are constructed and how the array elements construct those primitives.

    To enable and disable a generic vertex attribute array, call glEnableVertexAttribArray and glDisableVertexAttribArray.

    If an array corresponding to a generic attribute required by a vertex shader is not enabled, then the corresponding element is taken from the current generic attribute state.

    将 array 中的数据画成一些具体的图形。感觉类似于对 array 中的数据再进行一次解释。

* `glDeleteBuffers`

    delete named buffer objects

    Syntax:

    ```cpp
    void glDeleteBuffers(
        GLsizei n,
        const GLuint * buffers
    );
    ```

    Parameters:

    `n`: Specifies the number of buffer objects to be deleted.

    `buffers`: Specifies an array of buffer objects to be deleted.

    Description

    `glDeleteBuffers` deletes `n` buffer objects named by the elements of the array buffers. After a buffer object is deleted it has no contents, and its name is again unused. Unused names in buffers that have been marked as used for the purposes of glGenBuffers are marked as unused again. Unused names in buffers are silently ignored, as is the value zero. If a buffer object is deleted while it is bound, all bindings to that object in the current context are reset to zero. Bindings to that buffer in other contexts are not affected.

    `glDeleteBuffers` silently ignores 0's and names that do not correspond to existing buffer objects. 
