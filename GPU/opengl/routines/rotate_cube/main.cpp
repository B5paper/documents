#include <iostream>

// 1. 包含 GLEW (必须在 GLFW 之前)
#define GLEW_STATIC // 如果你使用的是静态库，请保留此行
#include <GL/glew.h>

// 2. 包含 GLFW
#include <GLFW/glfw3.h>

// 3. 包含 GLM
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

// 着色器源码 (与之前相同)
const char* vertexShaderSource = "#version 330 core\n"
"layout (location = 0) in vec3 aPos;\n"
"layout (location = 1) in vec3 aColor;\n"
"out vec3 ourColor;\n"
"uniform mat4 model;\n"
"uniform mat4 view;\n"
"uniform mat4 projection;\n"
"void main() {\n"
"   gl_Position = projection * view * model * vec4(aPos, 1.0);\n"
"   ourColor = aColor;\n"
"}\0";

const char* fragmentShaderSource = "#version 330 core\n"
"out vec4 FragColor;\n"
"in vec3 ourColor;\n"
"void main() {\n"
"   FragColor = vec4(ourColor, 1.0f);\n"
"}\n\0";

int main() {
    // 初始化 GLFW
    if (!glfwInit()) return -1;
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* window = glfwCreateWindow(800, 600, "GLEW Rotating Cube", NULL, NULL);
    if (!window) {
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);

    // --- 核心区别：初始化 GLEW ---
    glewExperimental = GL_TRUE;  // 开启现代 OpenGL 实验性功能支持
    if (glewInit() != GLEW_OK) {
        std::cout << "Failed to initialize GLEW" << std::endl;
        return -1;
    }

    glEnable(GL_DEPTH_TEST);

    // 编译着色器
    unsigned int vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vertexShaderSource, NULL);
    glCompileShader(vertexShader);

    unsigned int fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragmentShaderSource, NULL);
    glCompileShader(fragmentShader);

    unsigned int shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);

    // 顶点数据 (位置 + 颜色)
    float vertices[] = {
        -0.5f, -0.5f, -0.5f,  1.0f, 0.0f, 0.0f,
         0.5f, -0.5f, -0.5f,  0.0f, 1.0f, 0.0f,
         0.5f,  0.5f, -0.5f,  0.0f, 0.0f, 1.0f,
         0.5f,  0.5f, -0.5f,  0.0f, 0.0f, 1.0f,
        -0.5f,  0.5f, -0.5f,  1.0f, 1.0f, 0.0f,
        -0.5f, -0.5f, -0.5f,  1.0f, 0.0f, 0.0f,

        -0.5f, -0.5f,  0.5f,  0.0f, 1.0f, 1.0f,
         0.5f, -0.5f,  0.5f,  1.0f, 0.0f, 1.0f,
         0.5f,  0.5f,  0.5f,  1.0f, 1.0f, 1.0f,
         0.5f,  0.5f,  0.5f,  1.0f, 1.0f, 1.0f,
        -0.5f,  0.5f,  0.5f,  0.5f, 0.5f, 0.5f,
        -0.5f, -0.5f,  0.5f,  0.0f, 1.0f, 1.0f,

        -0.5f,  0.5f,  0.5f,  1.0f, 0.5f, 0.0f,
        -0.5f,  0.5f, -0.5f,  0.5f, 1.0f, 0.0f,
        -0.5f, -0.5f, -0.5f,  0.0f, 0.5f, 1.0f,
        -0.5f, -0.5f, -0.5f,  0.0f, 0.5f, 1.0f,
        -0.5f, -0.5f,  0.5f,  1.0f, 0.0f, 0.5f,
        -0.5f,  0.5f,  0.5f,  1.0f, 0.5f, 0.0f,

         0.5f,  0.5f,  0.5f,  0.2f, 0.8f, 0.2f,
         0.5f,  0.5f, -0.5f,  0.8f, 0.2f, 0.2f,
         0.5f, -0.5f, -0.5f,  0.2f, 0.2f, 0.8f,
         0.5f, -0.5f, -0.5f,  0.2f, 0.2f, 0.8f,
         0.5f, -0.5f,  0.5f,  0.8f, 0.8f, 0.2f,
         0.5f,  0.5f,  0.5f,  0.2f, 0.8f, 0.2f,

        -0.5f, -0.5f, -0.5f,  0.1f, 0.9f, 0.4f,
         0.5f, -0.5f, -0.5f,  0.9f, 0.1f, 0.4f,
         0.5f, -0.5f,  0.5f,  0.4f, 0.9f, 0.1f,
         0.5f, -0.5f,  0.5f,  0.4f, 0.9f, 0.1f,
        -0.5f, -0.5f,  0.5f,  0.1f, 0.4f, 0.9f,
        -0.5f, -0.5f, -0.5f,  0.1f, 0.9f, 0.4f,

        -0.5f,  0.5f, -0.5f,  0.7f, 0.3f, 0.9f,
         0.5f,  0.5f, -0.5f,  0.3f, 0.7f, 0.9f,
         0.5f,  0.5f,  0.5f,  0.9f, 0.7f, 0.3f,
         0.5f,  0.5f,  0.5f,  0.9f, 0.7f, 0.3f,
        -0.5f,  0.5f,  0.5f,  0.7f, 0.9f, 0.3f,
        -0.5f,  0.5f, -0.5f,  0.7f, 0.3f, 0.9f
    };

    unsigned int VBO, VAO;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);

    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);

    while (!glfwWindowShouldClose(window)) {
        glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glUseProgram(shaderProgram);

        // 旋转逻辑
        glm::mat4 model = glm::mat4(1.0f);
        glm::mat4 view  = glm::mat4(1.0f);
        glm::mat4 projection = glm::mat4(1.0f);

        // 绕 Z 轴缓慢旋转
        float time = (float)glfwGetTime();
        model = glm::rotate(model, time * glm::radians(30.0f), glm::vec3(0.0f, 0.0f, 1.0f));
        // 微调 X 轴旋转增加立体感
        model = glm::rotate(model, time * glm::radians(15.0f), glm::vec3(1.0f, 0.0f, 0.0f));

        view = glm::translate(view, glm::vec3(0.0f, 0.0f, -3.5f));
        projection = glm::perspective(glm::radians(45.0f), 800.0f / 600.0f, 0.1f, 100.0f);

        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "model"), 1, GL_FALSE, glm::value_ptr(model));
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "view"), 1, GL_FALSE, glm::value_ptr(view));
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "projection"), 1, GL_FALSE, glm::value_ptr(projection));

        glBindVertexArray(VAO);
        glDrawArrays(GL_TRIANGLES, 0, 36);

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
    glfwTerminate();
    return 0;
}

