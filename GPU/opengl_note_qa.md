# OpenGL Note QA

[unit]
[u_0]
请使用 glfw3 创建一个空白窗口。
[u_1]
`main.c`:

```c
#include <GLFW/glfw3.h>

int main()
{
    glfwInit();
    GLFWwindow *window = glfwCreateWindow(1024, 768, "opengl qa", NULL, NULL);
    do {
        glfwPollEvents();
    } while(glfwWindowShouldClose(window) == GLFW_FALSE);
    return 0;
}
```

编译：

```bash
gcc main.c -lglfw -o main
```

运行：

```bash
./main
```

[unit]
[u_0]
请使用 glfw3 创建一个空白窗口，既可以点 X 退出，也可以按 esc 键退出。
[u_1]
`main.c`:

```c
#include <GLFW/glfw3.h>

int main()
{
    glfwInit();
    GLFWwindow *window = glfwCreateWindow(1024, 768, "opengl test", NULL, NULL);
    do {
        glfwPollEvents();
    } while (glfwWindowShouldClose(window) == GLFW_FALSE &&
		glfwGetKey(window, GLFW_KEY_ESCAPE) != GLFW_PRESS);
    return 0;
}
```

编译：

```bash
gcc main.c -lglfw -o main
```

运行：

```bash
./main
```

效果：

出现一个黑色窗口，按 Esc 键退出。

[unit]
[u_0]
请给出函数`glGenBuffers()`的原型，给出函数的作用和参数的解释，返回值的判断。
[u_1]
Syntax:

```cpp
void glGenBuffers(GLsizei n, GLuint *buffers);
```

Description:

生成指定数量的非负整数 buffer id，这些 id 不保证是连续的。注意，此时并不会分配内存/显存，只做 id 是否被占用的检查。

Parameters:

* `n`: 生成 n 个 buffer ids。

* `buffers`: 指定一个数组起始地址，用于存储 buffer ids。

Return values:

(empty)

[unit]
[u_0]
请解释函数`glBindBuffer`。
[u_1]
Syntax:

```cpp
void glBindBuffer(GLenum target, GLuint buffer);
```

声明 buffer 的用途。这个函数不会实际分配显存。一个 buffer 可以被多次绑定，之前的绑定会被自动销毁。

Parameters:

* `target`: 指定 buffer 的用途

	常用的有下面几个：

	* `GL_ARRAY_BUFFER`: 顶点
	* `GL_COPY_READ_BUFFER`: Buffer copy source
	* `GL_COPY_WRITE_BUFFER`: Buffer copy destination
	* `GL_ELEMENT_ARRAY_BUFFER`: Vertex array indices

* `buffer`: Specifies the name (buffer 的 id) of a buffer object.

[unit]
[u_0]
请解释`glVertexAttribPointer()`函数。
[u_1]
Syntax:

```cpp
void glVertexAttribPointer(
	GLuint index,
  	GLint size,
  	GLenum type,
  	GLboolean normalized,
  	GLsizei stride,
  	const void * pointer);
```

Specify the location and data format of the array of generic vertex attributes at index index to use when rendering.

Parameters:

* `index`: Specifies the index of the generic vertex attribute to be modified.

* `size`: Specifies the number of components per generic vertex attribute. Must be 1, 2, 3, 4.

	Additionally, the symbolic constant GL_BGRA is accepted by glVertexAttribPointer. The initial value is 4.

* `type`: Specifies the data type of each component in the array. The symbolic constants `GL_BYTE`, `GL_UNSIGNED_BYTE`, `GL_SHORT`, `GL_UNSIGNED_SHORT`, `GL_INT`, and `GL_UNSIGNED_INT` are accepted by `glVertexAttribPointer`.

	Additionally `GL_HALF_FLOAT`, `GL_FLOAT`, `GL_DOUBLE`, `GL_FIXED`, `GL_INT_2_10_10_10_REV`, `GL_UNSIGNED_INT_2_10_10_10_REV` and `GL_UNSIGNED_INT_10F_11F_11F_REV` are accepted by `glVertexAttribPointer`.

* `normalized`: For glVertexAttribPointer, specifies whether fixed-point data values should be normalized (GL_TRUE) or converted directly as fixed-point values (GL_FALSE) when they are accessed.

	（不清楚这个是怎么个归一化法）

* `stride`: Specifies the byte offset between consecutive generic vertex attributes. If stride is 0, the generic vertex attributes are understood to be tightly packed in the array. The initial value is 0.

	（不懂）

* `pointer`: Specifies a offset of the first component of the first generic vertex attribute in the array in the data store of the buffer currently bound to the GL_ARRAY_BUFFER target. The initial value is 0.

	（不懂）

[unit]
[u_0]
请不使用 shader 画出一个三角形。
[u_1]
```cpp
#include <GL/glew.h>
#include <GLFW/glfw3.h>

int main()
{
    glfwInit();
    GLFWwindow *window = glfwCreateWindow(1024, 768, "opengl qa", NULL, NULL);
    glfwMakeContextCurrent(window);
    glewInit();
    float vtxs[3][3] = {
        {-0.5, 0, 0},
        {0, 1, 0},
        {0.5, 0, 0}
    };
    GLuint oglbuf_vtxs;
    glGenBuffers(1, &oglbuf_vtxs);
    glBindBuffer(GL_ARRAY_BUFFER, oglbuf_vtxs);
    glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 9, vtxs, GL_STATIC_DRAW);

    glClearColor(0, 0, 0, 0);
    glEnableVertexAttribArray(0);
    do {
        glClear(GL_COLOR_BUFFER_BIT);
        glBindBuffer(GL_ARRAY_BUFFER, oglbuf_vtxs);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, NULL);
        glDrawArrays(GL_TRIANGLES, 0, 3);
        glfwSwapBuffers(window);
        glfwPollEvents();
    } while (glfwWindowShouldClose(window) != GLFW_TRUE &&
        glfwGetKey(window, GLFW_KEY_ESCAPE) != GLFW_PRESS);
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
请解释函数`glCreateShader()`
[u_1]
Syntax:

```c
GLuint glCreateShader(GLenum shaderType);
```

创建并返回一个 shader id。

常用的`shaderType`有`GL_VERTEX_SHADER`，`GL_FRAGMENT_SHADER`，`GL_COMPUTE_SHADER`。

[unit]
[u_0]
请解释函数`glShaderSource()`
[u_1]
Replaces the source code in a shader object

Syntax:

```c
void glShaderSource(
	GLuint shader,
	GLsizei count,
	const GLchar **string,
	const GLint *length
);
```

Parameters:

* `shader`

	Specifies the handle of the shader object whose source code is to be replaced.

* `count`

	Specifies the number of elements in the string and length arrays.

	通常填 1。

* `string`

	Specifies an array of pointers to strings containing the source code to be loaded into the shader.

	注意这里需要填指针的指针。
	
* `length`

	Specifies an array of string lengths.

	通常填 NULL.

[unit]
[u_0]
请写出加载 shader 的代码。
[u_1]
`main.cpp`

```cpp
#include <GL/glew.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

GLuint load_program(const char *vtx_shader_path, const char *frag_shader_path)
{
    GLuint vtx_shader, frag_shader;
    vtx_shader = glCreateShader(GL_VERTEX_SHADER);
    frag_shader = glCreateShader(GL_FRAGMENT_SHADER);
    char *buf_read = (char*) malloc(1024);
    FILE *f = fopen(vtx_shader_path, "r");
    memset(buf_read, 0, 1024);
    fread(buf_read, 1024, 1, f);
    fclose(f);
    glShaderSource(vtx_shader, 1, &buf_read, NULL);
    f = fopen(frag_shader_path, "r");
    memset(buf_read, 0, 1024);
    fread(buf_read, 1024, 1, f);
    fclose(f);
    glShaderSource(frag_shader, 1, &buf_read, NULL);
    glCompileShader(vtx_shader);
    GLint Result;
    GLsizei InfoLogLength;
    glCompileShader(frag_shader);
    GLuint program_id;
    program_id = glCreateProgram();
    glAttachShader(program_id, vtx_shader);
    glAttachShader(program_id, frag_shader);
    glLinkProgram(program_id);
    glDetachShader(program_id, vtx_shader);
    glDetachShader(program_id, frag_shader);
    glDeleteShader(vtx_shader);
    glDeleteShader(frag_shader);
    free(buf_read);
    return program_id;
}
```

[unit]
[u_0]
请使用 shader 画一个绿色三角形。
[u_1]
(2024.03.18 version)

`main.cpp`:

```cpp
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <stdlib.h>
#include <stdio.h>

GLuint load_shader(const char *vtx_shader_path, const char *frag_shader_path)
{
    GLuint vtx_shader = glCreateShader(GL_VERTEX_SHADER);
    GLuint frag_shader = glCreateShader(GL_FRAGMENT_SHADER);
    FILE *f = fopen(vtx_shader_path, "r");
    fseek(f, 0, SEEK_END);
    size_t len = ftell(f);
    char *content = (char*) malloc(len);
    fseek(f, 0, SEEK_SET);
    fread(content, len, 1, f);
    glShaderSource(vtx_shader, 1, &content, (GLint*) &len);
    free(content);
    fclose(f);
    f = fopen(frag_shader_path, "r");
    fseek(f, 0, SEEK_END);
    len = ftell(f);
    content = (char*) malloc(len);
    fseek(f, 0, SEEK_SET);
    fread(content, len, 1, f);
    glShaderSource(frag_shader, 1, &content, (GLint*) &len);
    free(content);
    fclose(f);
    glCompileShader(vtx_shader);
    glCompileShader(frag_shader);
    GLuint prog = glCreateProgram();
    glAttachShader(prog, vtx_shader);
    glAttachShader(prog, frag_shader);
    glLinkProgram(prog);
    glDetachShader(prog, vtx_shader);
    glDetachShader(prog, frag_shader);
    glDeleteShader(vtx_shader);
    glDeleteShader(frag_shader);
    return prog;
}

int main()
{
    glfwInit();
    GLFWwindow *window = glfwCreateWindow(700, 500, "triangle", NULL, NULL);
    glfwMakeContextCurrent(window);
    glewInit();
    GLuint prog = load_shader("shader.vert", "shader.frag");
    float vtxs[9] = {
        -0.5, 0, 0,
        0, 1, 0,
        0.5, 0, 0
    };
    GLuint vtx_buf;
    glGenBuffers(1, &vtx_buf);
    glBindBuffer(GL_ARRAY_BUFFER, vtx_buf);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vtxs), vtxs, GL_STATIC_DRAW);
    glClearColor(0, 0, 0, 0);
    glEnableVertexAttribArray(0);
    while (!glfwWindowShouldClose(window))
    {
        glfwPollEvents();
        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        {
            glfwSetWindowShouldClose(window, GLFW_TRUE);
            continue;
        }
        glClear(GL_COLOR_BUFFER_BIT);
        glUseProgram(prog);
        glBindBuffer(GL_ARRAY_BUFFER, vtx_buf);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, NULL);
        glDrawArrays(GL_TRIANGLES, 0, 3);
        glfwSwapBuffers(window);
    }

    return 0;
}
```

`shader.vert`:

```cpp
#version 330

layout(location = 0) in vec3 pos;

void main()
{
    gl_Position = vec4(pos, 1);
}
```

`shader.frag`:

```cpp
#version 330

out vec3 color;

void main()
{
    color = vec3(0.5, 0.8, 0.5);
}
```

`main.cpp`:

```cpp
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <unistd.h>
#include <sstream>
#include <fstream>
#include <string>
#include <stdio.h>
#include <vector>

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


int main()
{
    glfwInit();
    GLFWwindow* window = glfwCreateWindow(1024, 768, "opengl qa test", NULL, NULL);
    glfwMakeContextCurrent(window);
    glewInit();  // glewInit() 必须写在 glfwMakeContextCurrent() 的后面，不然会报错。
    GLuint program = LoadShaders("./vtx.glsl", "./fce.glsl");
    glUseProgram(program);
    
    const GLfloat g_vertex_buffer_data[] = {
        -0.5f, -0.0f, 0.0f,
        0.5f, 0.0f, 0.0f,
        0.0f,  1.0f, 0.0f,
    };
    GLuint vtx_buf;
    glGenBuffers(1, &vtx_buf);
    glBindBuffer(GL_ARRAY_BUFFER, vtx_buf);
    glBufferData(GL_ARRAY_BUFFER, sizeof(g_vertex_buffer_data), g_vertex_buffer_data, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, NULL);

    glEnableVertexAttribArray(0);
    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    do {
        glClear(GL_COLOR_BUFFER_BIT);
        glDrawArrays(GL_LINE_LOOP, 0, 3);
        glfwSwapBuffers(window);
        glfwPollEvents();
        usleep(100000);
    } while (glfwWindowShouldClose(window) == 0);
    glDisableVertexAttribArray(0);
	glDeleteBuffers(1, &vtx_buf);
    glfwTerminate();
    return 0;
}
```

`vtx.glsl`:

```glsl
layout(location = 0) in vec3 vertexPosition_modelspace;

void main()
{
    gl_Position.xyz = vertexPosition_modelspace;
    gl_Position.w = 1.0;
}
```

`fce.glsl`:

```glsl
out vec3 color;

void main()
{
    color = vec3(0.5, 0.8, 0.5);
}
```

编译：

```bash
g++ main.cpp -lGLEW -lglfw -lGL -o main
```

运行：

```bash
./main
```

[unit]
[u_0]
请使用 glm 生成一个 MVP 矩阵。
[u_1]
```cpp
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

int main()
{
	glm::mat4 M_s = glm::scale(glm::mat4(1), {2, 2, 2});
	glm::mat4 M_r = glm::rotate(glm::mat4(1), 0.2f, {0, 1, 0});
	glm::mat4 M_t = glm::translate(glm::mat4(1), {0.1, 0.2, -1});
	glm::mat4 M_m = M_t * M_r * M_s;  // model matrix
	glm::vec3 eye{0, 0, 1};
	glm::vec3 center{0, 0, 0};
	glm::vec3 up{0, 1, 0};
	glm::mat4 M_v = glm::lookAt(eye, center, up);  // view matrix
	glm::mat4 M_p = glm::perspective(glm::radians(45.0), 1024.0 / 768.0, 0.1, 100.0);  // perspective matrix
	glm::mat4 M_mvp = M_p * M_v * M_m;  // the MVP matrix
	return 0;
}
```

[unit]
[u_0]
请使一个三角形绕 y 轴旋转。
[u_1]
`main.cpp`:

```cpp
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
using namespace glm;

GLuint load_shader(const char *vtx_shader_path, const char *frag_shader_path)
{
    GLuint vtx_shader, frag_shader;
    vtx_shader = glCreateShader(GL_VERTEX_SHADER);
    frag_shader = glCreateShader(GL_FRAGMENT_SHADER);
    char *buf = (char*) malloc(1024);
    FILE *f = fopen(vtx_shader_path, "r");
    memset(buf, 0, 1024);
    fread(buf, 1024, 1, f);
    fclose(f);
    glShaderSource(vtx_shader, 1, &buf, NULL);
    f = fopen(frag_shader_path, "r");
    memset(buf, 0, 1024);
    fread(buf, 1024, 1, f);
    fclose(f);
    glShaderSource(frag_shader, 1, &buf, NULL);
    glCompileShader(vtx_shader);
    glCompileShader(frag_shader);
    GLuint program_id = glCreateProgram();
    glAttachShader(program_id, vtx_shader);
    glAttachShader(program_id, frag_shader);
    glLinkProgram(program_id);
    glDetachShader(program_id, vtx_shader);
    glDetachShader(program_id, frag_shader);
    glDeleteShader(vtx_shader);
    glDeleteShader(frag_shader);
    free(buf);
    return program_id;
}

int main()
{
    glfwInit();
    GLFWwindow *window = glfwCreateWindow(1024, 768, "aaa", NULL, NULL);
    glfwMakeContextCurrent(window);
    glewInit();

    GLuint program_id = load_shader("./vtx_shader.glsl", "./frag_shader.glsl");
    GLuint mvp_id = glGetUniformLocation(program_id, "MVP");
    glUseProgram(program_id);

    float vtxs[9] = {
        -0.5, 0, 0,
        0, 1, 0,
        0.5, 0, 0
    };
    GLuint buf_vtxs;
    glGenBuffers(1, &buf_vtxs);
    glBindBuffer(GL_ARRAY_BUFFER, buf_vtxs);
    glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 9, vtxs, GL_STATIC_DRAW);

    mat4 M_s = scale(mat4(1), {1, 1, 1});
    mat4 M_r = rotate(mat4(1), 0.0f, {0, 1, 0});
    mat4 M_t = translate(mat4(1), {0, 0, 0});
    mat4 M_m = M_t * M_r * M_s;
    vec3 eye = {0, 0, 1};
    vec3 center = {0, 0, 0};
    vec3 up = {0, 1, 0};
    mat4 M_v = lookAt(eye, center, up);
    mat4 M_p = perspective(radians(90.0f), 1024.0f / 768.0f, 0.1f, 100.0f);
    mat4 MVP = M_p * M_v * M_m;

    float theta = 0.0f;
    glClearColor(0, 0, 0, 0);
    glEnableVertexAttribArray(0);
    while (!glfwWindowShouldClose(window))
    {
        glClear(GL_COLOR_BUFFER_BIT);

        M_r = rotate(mat4(1), theta, {0, 1, 0});
        M_m = M_t * M_r * M_s;
        MVP = M_p * M_v * M_m;
        glUniformMatrix4fv(mvp_id, 1, GL_FALSE, &MVP[0][0]);
        theta += 0.001f;
        if (theta > pi<float>() * 2)
            theta = 0;

        glBindBuffer(GL_ARRAY_BUFFER, buf_vtxs);
        glVertexAttribPointer(0, 3, GL_FLOAT, false, 0, NULL);
        glDrawArrays(GL_TRIANGLES, 0, 3);
        glfwSwapBuffers(window);
        glfwPollEvents();
        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
            glfwSetWindowShouldClose(window, GLFW_TRUE);
    }
    return 0;
}
```

`vtx_shader.glsl`:

```glsl
#version 330 core

layout(location = 0) in vec3 pos;
uniform mat4 MVP;

void main()
{
    gl_Position = MVP * vec4(pos, 1);
}
```

`frag_shader.glsl`:

```glsl
#version 330 core

out vec3 color;

void main()
{
    color = vec3(0.5, 0.8, 0.5);
}
```

编译：

```bash
g++ -g main.cpp -lGLEW -lglfw -lGL -o main
```

运行：

```bash
./main
```

[unit]
[u_0]
请给一个三角形加上纹理贴图。
[u_1]
`main.cpp`:

```cpp
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <string>
#include <fstream>
#include <sstream>
#include <vector>
#include <cstring>

GLuint LoadShaders(const char * vertex_file_path,const char * fragment_file_path){
	// Create the shaders
	GLuint VertexShaderID = glCreateShader(GL_VERTEX_SHADER);
	GLuint FragmentShaderID = glCreateShader(GL_FRAGMENT_SHADER);

	// Read the Vertex Shader code from the file
	std::string VertexShaderCode;
	std::ifstream VertexShaderStream(vertex_file_path, std::ios::in);
	if (VertexShaderStream.is_open()) {
		std::stringstream sstr;
		sstr << VertexShaderStream.rdbuf();
		VertexShaderCode = sstr.str();
		VertexShaderStream.close();
	} else {
		printf("Impossible to open %s. Are you in the right directory ? Don't forget to read the FAQ !\n", vertex_file_path);
		getchar();
		return 0;
	}

	// Read the Fragment Shader code from the file
	std::string FragmentShaderCode;
	std::ifstream FragmentShaderStream(fragment_file_path, std::ios::in);
	if (FragmentShaderStream.is_open()){
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
	if (InfoLogLength > 0) {
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
	if (InfoLogLength > 0) {
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
	if (InfoLogLength > 0) {
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

GLuint loadBMP_custom(const char * imagepath){
	printf("Reading image %s\n", imagepath);

	// Data read from the header of the BMP file
	unsigned char header[54];
	unsigned int dataPos;
	unsigned int imageSize;
	unsigned int width, height;
	// Actual RGB data
	unsigned char * data;

	// Open the file
	FILE * file = fopen(imagepath,"rb");
	if (!file) {
		printf("%s could not be opened. Are you in the right directory ? Don't forget to read the FAQ !\n", imagepath);
		getchar();
		return 0;
	}

	// Read the header, i.e. the 54 first bytes

	// If less than 54 bytes are read, problem
	if (fread(header, 1, 54, file) != 54) { 
		printf("Not a correct BMP file\n");
		fclose(file);
		return 0;
	}
	// A BMP files always begins with "BM"
	if (header[0]!='B' || header[1]!='M') {
		printf("Not a correct BMP file\n");
		fclose(file);
		return 0;
	}

	// Make sure this is a 24bpp file
	if (*(int*)&(header[0x1E])!=0) {
		printf("Not a correct BMP file\n");
		fclose(file);
		return 0;
	}

	if (*(int*)&(header[0x1C]) != 24) {
		printf("Not a correct BMP file\n");
		fclose(file);
		return 0;
	}

	// Read the information about the image
	dataPos    = *(int*)&(header[0x0A]);
	imageSize  = *(int*)&(header[0x22]);
	width      = *(int*)&(header[0x12]);
	height     = *(int*)&(header[0x16]);

	// Some BMP files are misformatted, guess missing information
	if (imageSize==0)    imageSize=width*height*3; // 3 : one byte for each Red, Green and Blue component
	if (dataPos==0)      dataPos=54; // The BMP header is done that way

	// Create a buffer
	data = new unsigned char [imageSize];

	// Read the actual data from the file into the buffer
	fread(data,1,imageSize,file);

	// Everything is in memory now, the file can be closed.
	fclose (file);

	// Create one OpenGL texture
	GLuint textureID;
	glGenTextures(1, &textureID);
	glBindTexture(GL_TEXTURE_2D, textureID);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_BGR, GL_UNSIGNED_BYTE, data);

	// OpenGL has now copied the data. Free our own version
	delete [] data;

	// Poor filtering, or ...
	// glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	// glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST); 

	// ... nice trilinear filtering ...
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);

	// Generate mipmaps automatically
	glGenerateMipmap(GL_TEXTURE_2D); 
	return textureID;
}

int main()
{
    glfwInit();
    GLFWwindow *window = glfwCreateWindow(1024, 768, "opengl test", NULL, NULL);
    glfwMakeContextCurrent(window);
    glewInit();

    GLuint program_id = LoadShaders("./vtx.glsl", "./frag.glsl");
	glUseProgram(program_id);
    
    GLuint img = loadBMP_custom("./img.bmp");  // 主要用于创建并填充 texture 的 buffer
    GLuint img_id  = glGetUniformLocation(program_id, "myTextureSampler");  // 获取 glsl 中 myTextureSampler 变量的 id，后续我们会把 texture 的 buffer 的数据绑定到 glsl 中的 myTextureSampler 变量下

    float tri_vtxs[3][3] = {
        -0.5, 0, 0,
        0.5, 0, 0,
        0, 0.5, 0,
    };
    GLuint tri_vtxs_buf;
    glGenBuffers(1, &tri_vtxs_buf);
    glBindBuffer(GL_ARRAY_BUFFER, tri_vtxs_buf);
    glBufferData(GL_ARRAY_BUFFER, sizeof(tri_vtxs), tri_vtxs, GL_STATIC_DRAW);

    float uv_vtxs[3][2] = {
        0, 0,
        1, 0,
        0.5, 1
    };  // 注意这些点的顺序要和对应的 mesh 保持一致，顶点的顺序会影响贴图的方向
    GLuint uv_vtxs_buf;
    glGenBuffers(1, &uv_vtxs_buf);  // 纹理贴图的顶点 buffer 只是常规 buffer
    glBindBuffer(GL_ARRAY_BUFFER, uv_vtxs_buf);
    glBufferData(GL_ARRAY_BUFFER, sizeof(uv_vtxs), uv_vtxs, GL_STATIC_DRAW);
    
    glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LESS);
    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);
    glClearColor(0, 0, 0, 0);
    do {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, img);  // 绑定 buffer
		glUniform1i(img_id, 0);  // 将 texture buffer 绑定到 glsl 中的变量

        glBindBuffer(GL_ARRAY_BUFFER, tri_vtxs_buf);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, NULL);  // 将 buffer 绑定到 glsl 中的 location 0

        glBindBuffer(GL_ARRAY_BUFFER, uv_vtxs_buf);
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, NULL);  // 将 buffer 绑定到 glsl 中的 location 1

        glDrawArrays(GL_TRIANGLES, 0, 3);  // 真正的绘制只需要这一行就行了，剩下的工作交给 glsl shader

        glfwSwapBuffers(window);  // 这一行别忘了，不然不出图
        glfwPollEvents();
    } while (
        glfwWindowShouldClose(window) == GL_FALSE &&
        glfwGetKey(window, GLFW_KEY_ESCAPE) != GLFW_PRESS
    );
    glDisableVertexAttribArray(0);
    return 0;
}
```

`vtx.glsl`:

```glsl
layout(location = 0) in vec4 vtx_pos;  // 接受 c 程序中传进来的数据，下一行同理
layout(location = 1) in vec2 uv_pos;
out vec2 uv;  // 这个变量名 uv 必须和 frag.glsl 中的 in vec2 us; 中的变量名 uv 保持一致才行。看来他们是全局的

void main()
{
    gl_Position = vtx_pos;
    uv = uv_pos;  // 把 uv_pos 数据传递给下一级
}
```

`frag.glsl`:

```glsl
out vec3 color;
uniform sampler2D myTextureSampler;  // 就是 c 代码里传进来的图片数据
in vec2 uv;

void main()
{
    color = texture(myTextureSampler, uv).rgb;  // 不懂，先记住这个语法
}
```

编译：

```bash
g++ -g main.cpp -lGLEW -lglfw -lGL -o main
```

运行：

```bash
./main
```

（当前目录下需有一张`img.bmp`图片）

[unit]
[u_0]
使用 element draw 画一个 cube。
[u_1]
`main.cpp`:

```c
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <stdio.h>
#include <stdlib.h>

GLuint load_shader(const char *vert_shader_path, const char *frag_shader_path)
{
    GLuint vert_shader = glCreateShader(GL_VERTEX_SHADER);
    GLuint frag_shader = glCreateShader(GL_FRAGMENT_SHADER);
    FILE *f = fopen(vert_shader_path, "r");
    fseek(f, 0, SEEK_END);
    size_t len = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *shader_src = (char*) malloc(len);
    fread(shader_src, len, 1, f);
    glShaderSource(vert_shader, 1, &shader_src, (const GLint *)&len);
    glCompileShader(vert_shader);
    free(shader_src);
    fclose(f);
    f = fopen(frag_shader_path, "r");
    fseek(f, 0, SEEK_END);
    len = ftell(f);
    fseek(f, 0, SEEK_SET);
    shader_src = (char*) malloc(len);
    fread(shader_src, len, 1, f);
    glShaderSource(frag_shader, 1, &shader_src, (const GLint *)&len);
    glCompileShader(frag_shader);
    free(shader_src);
    fclose(f);
    GLuint prog = glCreateProgram();
    glAttachShader(prog, vert_shader);
    glAttachShader(prog, frag_shader);
    glLinkProgram(prog);
    glDetachShader(prog, vert_shader);
    glDetachShader(prog, frag_shader);
    glDeleteShader(vert_shader);
    glDeleteShader(frag_shader);
    return prog;
}

int main()
{
    glfwInit();
    GLFWwindow *window = glfwCreateWindow(700, 500, "hello", NULL, NULL);
    glfwMakeContextCurrent(window);
    glewInit();
    GLuint prog_id = load_shader("vert.glsl", "frag.glsl");
    glUseProgram(prog_id);

    float vtxs[] = {
        -0.5, -0.5, -0.5,
        -0.5, -0.5, 0.5,
        -0.5, 0.5, 0.5,
        -0.5, 0.5, -0.5,
        0.5, -0.5, -0.5,
        0.5, -0.5, 0.5,
        0.5, 0.5, 0.5,
        0.5, 0.5, -0.5
    };

    uint32_t inds[][3] = {
        1, 2, 0,
        3, 0, 2,
        1, 5, 6,
        1, 6, 2,
        5, 4, 6,
        4, 7, 6,
        0, 3, 7,
        0, 7, 4,
        6, 3, 2,
        6, 7, 3,
        1, 0, 5,
        5, 0, 4
    };

    GLuint vtx_buf, ind_buf;
    glGenBuffers(1, &vtx_buf);
    glBindBuffer(GL_ARRAY_BUFFER, vtx_buf);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vtxs), vtxs, GL_STATIC_DRAW);
    glGenBuffers(1, &ind_buf);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ind_buf);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(inds), inds, GL_STATIC_DRAW);

    glClearColor(0, 0, 0, 0);
    glEnableVertexAttribArray(0);
    while (glfwWindowShouldClose(window) != GLFW_TRUE)
    {
        glClear(GL_COLOR_BUFFER_BIT0_QCOM);

        glUseProgram(prog_id);
        glBindBuffer(GL_ARRAY_BUFFER, vtx_buf);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(float) * 3, NULL);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ind_buf);
        glDrawElements(GL_TRIANGLES, 12, GL_UNSIGNED_INT, NULL);
        glfwSwapBuffers(window);
        glfwPollEvents();
		
        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        {
            glfwSetWindowShouldClose(window, GLFW_TRUE);
        }
    }
    return 0;
}
```

`vert.glsl`:

```glsl
#version 330 core

layout(location = 0) in vec3 pos;

void main()
{
    gl_Position = vec4(pos, 1);
}
```

`frag.glsl`:

```glsl
#version 330 core

out vec3 color;

void main()
{
    color = vec3(0.5, 0.8, 0.5);
}
```

`Makefile`:

```makefile
main: main.cpp
	g++ -g main.cpp -lglfw -lGLEW -lGL -o main
```

compile:

```bash
make
```

run:

```bash
./main
```

[unit]
[u_0]
请画一个彩色正方体，并使之绕 y 轴旋转。
[u_1]
`main.cpp`:

```cpp
#include <stdio.h>
#include <stdlib.h>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <unistd.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

static const GLfloat g_color_buffer_data[] = {
    0.583f,  0.771f,  0.014f,
    0.609f,  0.115f,  0.436f,
    0.327f,  0.483f,  0.844f,
    0.822f,  0.569f,  0.201f,
    0.435f,  0.602f,  0.223f,
    0.310f,  0.747f,  0.185f,
    0.597f,  0.770f,  0.761f,
    0.559f,  0.436f,  0.730f,
    0.359f,  0.583f,  0.152f,
    0.483f,  0.596f,  0.789f,
    0.559f,  0.861f,  0.639f,
    0.195f,  0.548f,  0.859f,
    0.014f,  0.184f,  0.576f,
    0.771f,  0.328f,  0.970f,
    0.406f,  0.615f,  0.116f,
    0.676f,  0.977f,  0.133f,
    0.971f,  0.572f,  0.833f,
    0.140f,  0.616f,  0.489f,
    0.997f,  0.513f,  0.064f,
    0.945f,  0.719f,  0.592f,
    0.543f,  0.021f,  0.978f,
    0.279f,  0.317f,  0.505f,
    0.167f,  0.620f,  0.077f,
    0.347f,  0.857f,  0.137f,
    0.055f,  0.953f,  0.042f,
    0.714f,  0.505f,  0.345f,
    0.783f,  0.290f,  0.734f,
    0.722f,  0.645f,  0.174f,
    0.302f,  0.455f,  0.848f,
    0.225f,  0.587f,  0.040f,
    0.517f,  0.713f,  0.338f,
    0.053f,  0.959f,  0.120f,
    0.393f,  0.621f,  0.362f,
    0.673f,  0.211f,  0.457f,
    0.820f,  0.883f,  0.371f,
    0.982f,  0.099f,  0.879f
};

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

int main()
{
    GLfloat g_vertex_buffer_data[] = {
        -1.0f,-1.0f,-1.0f, // triangle 1 : begin
        -1.0f,-1.0f, 1.0f,
        -1.0f, 1.0f, 1.0f, // triangle 1 : end
        1.0f, 1.0f,-1.0f, // triangle 2 : begin
        -1.0f,-1.0f,-1.0f,
        -1.0f, 1.0f,-1.0f, // triangle 2 : end
        1.0f,-1.0f, 1.0f,
        -1.0f,-1.0f,-1.0f,
        1.0f,-1.0f,-1.0f,
        1.0f, 1.0f,-1.0f,
        1.0f,-1.0f,-1.0f,
        -1.0f,-1.0f,-1.0f,
        -1.0f,-1.0f,-1.0f,
        -1.0f, 1.0f, 1.0f,
        -1.0f, 1.0f,-1.0f,
        1.0f,-1.0f, 1.0f,
        -1.0f,-1.0f, 1.0f,
        -1.0f,-1.0f,-1.0f,
        -1.0f, 1.0f, 1.0f,
        -1.0f,-1.0f, 1.0f,
        1.0f,-1.0f, 1.0f,
        1.0f, 1.0f, 1.0f,
        1.0f,-1.0f,-1.0f,
        1.0f, 1.0f,-1.0f,
        1.0f,-1.0f,-1.0f,
        1.0f, 1.0f, 1.0f,
        1.0f,-1.0f, 1.0f,
        1.0f, 1.0f, 1.0f,
        1.0f, 1.0f,-1.0f,
        -1.0f, 1.0f,-1.0f,
        1.0f, 1.0f, 1.0f,
        -1.0f, 1.0f,-1.0f,
        -1.0f, 1.0f, 1.0f,
        1.0f, 1.0f, 1.0f,
        -1.0f, 1.0f, 1.0f,
        1.0f,-1.0f, 1.0f
    };
    // for (int i = 0; i < sizeof(g_vertex_buffer_data) / sizeof(float); ++i)
    // {
    //     g_vertex_buffer_data[i] /= 3.0;
    // }
    glfwInit();
    GLFWwindow* window = glfwCreateWindow(1024, 768, "opengl qa test", NULL, NULL);
    glfwMakeContextCurrent(window);
    glewInit();

    GLuint program = LoadShaders("./vtx.glsl","./fce.glsl");
    glUseProgram(program);

    glfwSetInputMode(window, GLFW_STICKY_KEYS, GL_TRUE);
    glClearColor(0.5f, 0.5f, 0.5f, 0.0f);
    // static const GLfloat g_vertex_buffer_data[] = {
    //     -1.0f, -1.0f, 0.0f,
    //     1.0f, -1.0f, 0.0f,
    //     0.0f,  1.0f, 0.0f,
    // };
    GLuint vertexbuffer;
    glGenBuffers(1, &vertexbuffer);
    glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer);
    glBufferData(GL_ARRAY_BUFFER, sizeof(g_vertex_buffer_data), g_vertex_buffer_data, GL_STATIC_DRAW);


    GLuint colorbuffer;
    glGenBuffers(1, &colorbuffer);
    glBindBuffer(GL_ARRAY_BUFFER, colorbuffer);
    glBufferData(GL_ARRAY_BUFFER, sizeof(g_color_buffer_data), g_color_buffer_data, GL_STATIC_DRAW);

    glEnableVertexAttribArray(1);
    glBindBuffer(GL_ARRAY_BUFFER, colorbuffer);
    glVertexAttribPointer(
            1,                                // attribute. No particular reason for 1, but must match the layout in the shader.
            3,                                // size
            GL_FLOAT,                         // type
            GL_FALSE,                         // normalized?
            0,                                // stride
            (void*)0                          // array buffer offset
        );

	GLuint MatrixID = glGetUniformLocation(program, "MVP");
    GLuint rot_mat_id = glGetUniformLocation(program, "rot_mat");
	glm::mat4 Projection = glm::perspective(glm::radians(45.0f), 4.0f / 3.0f, 0.1f, 100.0f);
	glm::mat4 View       = glm::lookAt(
								glm::vec3(4,3,-3), // Camera is at (4,3,-3), in World Space
								glm::vec3(0,0,0), // and looks at the origin
								glm::vec3(0,1,0)  // Head is up (set to 0,-1,0 to look upside-down)
						   );
	// Model matrix : an identity matrix (model will be at the origin)
	glm::mat4 Model      = glm::mat4(1.0f);
	// Our ModelViewProjection : multiplication of our 3 matrices
	glm::mat4 MVP        = Projection * View * Model; 
    glm::vec3 rot_dir(0, 1, 0);
    glm::mat4 one(1.0f);
    glm::mat4 rot_mat = glm::rotate(glm::mat4(1.0f), 0.02f, rot_dir);

    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);

    // 猜测 enable 索引用于控制渲染管线中的 buffer 是否生效
    // 如果 buffer 不生效，其实 glsl 是不会报错的
    // 不清楚这个错误处理机制是怎样的
    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);

    int i = 0;
    do {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glUniformMatrix4fv(MatrixID, 1, GL_FALSE, &MVP[0][0]);
        rot_mat = glm::rotate(glm::mat4(1.0f), 3.1415926535f * 2 / 1000 * i, rot_dir);
        glUniformMatrix4fv(rot_mat_id, 1, GL_FALSE, &rot_mat[0][0]);
        // printf("%d\n", i);
        ++i;
        if (i == 1001)
            i = 0;

		// 猜测：每绑定一次，都会去自动调用一次 glsl 中的代码
        // 多次调用并未使正方体发生旋转 => vertexbuffer 中的数据不会发生改变，
        // 说明计算得到的临时输出有另外的缓冲区存储
        glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer);
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, NULL);
        
        glBindBuffer(GL_ARRAY_BUFFER, colorbuffer);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, NULL);

        glDrawArrays(GL_TRIANGLES, 0, 12 * 3);

        glfwSwapBuffers(window);
        glfwPollEvents();
        usleep(20000);
    } while (glfwGetKey(window, GLFW_KEY_ESCAPE) != GLFW_PRESS &&
        glfwWindowShouldClose(window) == 0);

    glDisableVertexAttribArray(0);
	glDisableVertexAttribArray(1);

	glDeleteBuffers(1, &vertexbuffer);
    glfwTerminate();
    return 0;
}
```

[unit]
[u_0]
请使用冯氏模型为一个三角形打光。
[u_1]
`main.cpp`:

```cpp
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <iostream>
#include <unistd.h>
using namespace std;
using namespace glm;


GLuint load_shader(const char *vtx_shader_path, const char *frag_shader_path)
{
    GLuint vtx_shader, frag_shader;
    vtx_shader = glCreateShader(GL_VERTEX_SHADER);
    frag_shader = glCreateShader(GL_FRAGMENT_SHADER);
    const int buf_size = 1024;
    char *buf = (char*) malloc(buf_size);
    FILE *f = fopen(vtx_shader_path, "r");
    memset(buf, 0, buf_size);
    fread(buf, buf_size, 1, f);
    fclose(f);
    glShaderSource(vtx_shader, 1, &buf, NULL);
    glCompileShader(vtx_shader);
    fopen(frag_shader_path, "r");
    memset(buf, 0, buf_size);
    fread(buf, buf_size, 1, f);
    fclose(f);
    glShaderSource(frag_shader, 1, &buf, NULL);
    glCompileShader(frag_shader);
    GLuint program_id = glCreateProgram();
    glAttachShader(program_id, vtx_shader);
    glAttachShader(program_id, frag_shader);
    glLinkProgram(program_id);
    glDetachShader(program_id, vtx_shader);
    glDetachShader(program_id, frag_shader);
    glDeleteShader(vtx_shader);
    glDeleteShader(frag_shader);
    free(buf);
    return program_id;
}

struct MVP
{
    MVP() {
        m_scale = {1, 1, 1};
        m_translate = {0, 0, 0};
        rotate_dir = {0, 1, 0};
        rotate_rad = 0;
        eye = {0, 0, 1};
        center = {0, 0, 0};
        up = {0, 1, 0};
    }

    mat4 get_rotate_mat() {
        mat4 M_r = rotate(mat4(1), rotate_rad, rotate_dir);
        return M_r;
    }

    mat4 get_mvp() {
        mat4 M_s = scale(mat4(1), m_scale);
        mat4 M_r = rotate(mat4(1), rotate_rad, rotate_dir);
        mat4 M_t = translate(mat4(1), m_translate);
        mat4 M_m = M_t * M_r * M_s;
        mat4 M_v = lookAt(eye, center, up);
        mat4 M_p = perspective(radians(90.0), 1024 / 768.0, 0.1, 100.0);
        mat4 mvp = M_p * M_v * M_m;
        return mvp;
    }

    vec3 m_scale;
    vec3 m_translate;
    vec3 rotate_dir;
    float rotate_rad;
    vec3 eye, center, up;    
};

struct ModelMat
{
    vec3 m_scale;
    vec3 m_translate;
    vec3 rotate_dir;
    float rotate_rad;
    mat4 M_m;

    ModelMat() {
        m_scale = {1, 1, 1};
        m_translate = {0, 0, 0};
        rotate_dir = {0, 1, 0};
        rotate_rad = 0;
    }

    mat4& get_model_mat() {
        mat4 M_s = scale(mat4(1), m_scale);
        mat4 M_r = rotate(mat4(1), rotate_rad, rotate_dir);
        mat4 M_t = translate(mat4(1), m_translate);
        M_m = M_t * M_r * M_s;
        return M_m;
    }

    mat4 get_rotate_mat() {
        return rotate(mat4(1), rotate_rad, rotate_dir);
    }
};

struct ViewMat
{
    vec3 eye, center, up;
    mat4 M_v;

    ViewMat() {
        eye = {0, 0, 1};
        center = {0, 0, 0};
        up = {0, 1, 0};
    }

    mat4& get_view_mat() {
        M_v = lookAt(eye, center, up);
        return M_v;
    }
};

mat4 get_mvp(mat4 &model, mat4 &view)
{
    mat4 M_p = perspective(radians(90.0), 1024 / 768.0, 0.1, 100.0);
    return M_p * view * model;
}


int main()
{
    glfwInit();
    GLFWwindow *window = glfwCreateWindow(1024, 768, "gl light", NULL, NULL);
    glfwMakeContextCurrent(window);
    glewInit();

    GLuint program_id = load_shader("./vtx.glsl", "./frag.glsl");
    GLuint pid_light = load_shader("./vtx_light.glsl", "./frag_light.glsl");

    GLuint norm_id = glGetUniformLocation(program_id, "norm");
    GLuint mvp_id = glGetUniformLocation(program_id, "mvp");
    GLuint eye_id = glGetUniformLocation(program_id, "eye");
    GLuint light_mvp_id = glGetUniformLocation(pid_light, "mvp");
    

    float vtxs[3][3] = {
        {-0.5, 0, 0},
        {0, 1, 0},
        {0.5, 0, 0}
    };
    GLuint buf_vtxs;
    glGenBuffers(1, &buf_vtxs);
    glBindBuffer(GL_ARRAY_BUFFER, buf_vtxs);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vtxs), vtxs, GL_STATIC_DRAW);

    float vtxs_line[2][3];
    GLuint buf_line;
    glGenBuffers(1, &buf_line);

    float vtxs_coord[6][3] = {
        {0, 0, 0},
        {1, 0, 0},
        {0, 0, 0},
        {0, 1, 0},
        {0, 0, 0},
        {0, 0, 1}
    };
    GLuint buf_coord;
    glGenBuffers(1, &buf_coord);
    glBindBuffer(GL_ARRAY_BUFFER, buf_coord);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vtxs_coord), vtxs_coord, GL_STATIC_DRAW);

    ModelMat tri_mat, light_mat, coord_mat;
    ViewMat view_mat;

    light_mat.m_scale = {0.1, 0.1, 0.1};
    light_mat.m_translate = {0.7, 0.7, 0};
    view_mat.eye = {0, 0.5, 2};

    MVP MVP, MVP_light, MVP_line, MVP_coord;
    MVP_light.m_translate = {0.7, 0.7, 0};
    MVP_light.m_scale = {0.1, 0.1, 0.1};
    mat4 light_mvp, mvp, mvp_line, mvp_coord;
    vec3 norm;
    float theta = 0;
    float theta_obj = 0;
    glEnableVertexAttribArray(0);
    glClearColor(0, 0, 0, 0);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);
    while (!glfwWindowShouldClose(window))
    {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // rotate camera
        view_mat.eye = {sin(theta) * sqrt(2), 0.5, cos(theta) * sqrt(2)};
        theta += 0.001;
        if (theta > pi<float>() * 2)
            theta = 0;

        // darw coordinate
        glUseProgram(pid_light);
        mvp_coord = get_mvp(coord_mat.get_model_mat(), view_mat.get_view_mat());
        glUniformMatrix4fv(light_mvp_id, 1, GL_FALSE, &mvp_coord[0][0]);
        glBindBuffer(GL_ARRAY_BUFFER, buf_coord);
        glVertexAttribPointer(0, 3, GL_FLOAT, false, 0, NULL);
        glDrawArrays(GL_LINES, 0, 6);

        // draw light triangle
        light_mvp = get_mvp(light_mat.get_model_mat(), view_mat.get_view_mat());
        glUseProgram(pid_light);
        glUniformMatrix4fv(light_mvp_id, 1, GL_FALSE, &light_mvp[0][0]);
        glBindBuffer(GL_ARRAY_BUFFER, buf_vtxs);
        glVertexAttribPointer(0, 3, GL_FLOAT, false, 0, NULL);
        glDrawArrays(GL_TRIANGLES, 0, 3);

        // draw triangle
        tri_mat.rotate_rad = theta_obj;
        theta_obj += 0.01;
        if (theta_obj > pi<float>() * 2)
            theta_obj = 0;
        mvp = get_mvp(tri_mat.get_model_mat(), view_mat.get_view_mat());
        norm = cross(vec3{0.5, 0, 0} - vec3{-0.5, 0, 0}, vec3{0, 1, 0} - vec3{-0.5, 0, 0});
        norm = tri_mat.get_rotate_mat() * vec4(norm, 1);

        glUseProgram(program_id);
        glUniform3fv(norm_id, 1, &norm[0]);
        glUniformMatrix4fv(mvp_id, 1, GL_FALSE, &mvp[0][0]);
        glUniform3fv(eye_id, 1, &view_mat.eye[0]);
        glBindBuffer(GL_ARRAY_BUFFER, buf_vtxs);
        glVertexAttribPointer(0, 3, GL_FLOAT, false, 0, NULL);
        glDrawArrays(GL_TRIANGLES, 0, 3);

        // draw norm line
        vtxs_line[1][0] = norm[0] / 2;
        vtxs_line[1][1] = norm[1] / 2;
        vtxs_line[1][2] = norm[2] / 2;
        glBindBuffer(GL_ARRAY_BUFFER, buf_line);
        glBufferData(GL_ARRAY_BUFFER, sizeof(vtxs_line), vtxs_line, GL_STATIC_DRAW);

        glUseProgram(pid_light);
        mvp_line = get_mvp(coord_mat.get_model_mat(), view_mat.get_view_mat());
        glUniformMatrix4fv(light_mvp_id, 1, GL_FALSE, &mvp_line[0][0]);
        glBindBuffer(GL_ARRAY_BUFFER, buf_line);
        glVertexAttribPointer(0, 3, GL_FLOAT, false, 0, NULL);
        glDrawArrays(GL_LINES, 0, 2);

        glfwSwapBuffers(window);
        glfwPollEvents();
        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        {
            glfwSetWindowShouldClose(window, true);
        }
        usleep(10000);
    }
    return 0;
}
```

`vtx_light:glsl`:

```glsl
#version 330 core

layout(location = 0) in vec3 pos;
uniform mat4 mvp;

void main()
{
    gl_Position = mvp * vec4(pos, 1);
}
```

`frag_light.glsl`:

```glsl
#version 330 core

out vec3 color;

void main()
{
    color = vec3(1, 1, 1);
}
```

`vtx.glsl`:

```glsl
#version 330 core

layout(location = 0) in vec3 pos;
uniform mat4 mvp;
uniform vec3 norm;
out vec3 norm_frag;
out vec3 pos_frag;

void main()
{
    gl_Position = mvp * vec4(pos, 1);
    norm_frag = norm;
    pos_frag = pos;
}
```

`frag.glsl`:

```glsl
#version 330 core

out vec3 color;
in vec3 norm_frag;
in vec3 pos_frag;
uniform vec3 eye;

void main()
{
    vec3 lightColor = vec3(1, 1, 1);
    vec3 ambient = lightColor * 0.1;
    vec3 objectColor = vec3(0.5, 0.8, 0.5);
    vec3 lightPos = vec3(0.7, 0.7, 0);
    vec3 FragPos = pos_frag;
    vec3 norm = normalize(norm_frag);
    vec3 lightDir = normalize(lightPos - FragPos);
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = diff * lightColor;
    float specularStrength = 0.5;
    vec3 viewPos = eye;
    vec3 viewDir = normalize(viewPos - FragPos);
    vec3 reflectDir = reflect(-lightDir, norm);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32);
    vec3 specular = specularStrength * spec * lightColor;
    vec3 result = (ambient + diffuse) * objectColor;
    result = (ambient + diffuse + specular) * objectColor;
    // FragColor = vec4(result, 1.0);
    color = result;
}
```

编译：

```bash
g++ -g main.cpp -lGLEW -lglfw -lGL -o main
```

运行：

```bash
./main
```
