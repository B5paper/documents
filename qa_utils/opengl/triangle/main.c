#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <stdlib.h>
#include <stdio.h>

void get_file_content(const char *file_path, char **buf, size_t *len)
{
    FILE *f = fopen(file_path, "r");
    fseek(f, 0, SEEK_END);
    *len = ftell(f);
    fseek(f, 0, SEEK_SET);
    *buf = (char*) malloc(*len);
    fread(*buf, *len, 1, f);
    fclose(f);
}

GLuint load_shader(const char *vert_shader_path,
    const char *frag_shader_path)
{
    GLuint vert_shader, frag_shader;
    vert_shader = glCreateShader(GL_VERTEX_SHADER);
    frag_shader = glCreateShader(GL_FRAGMENT_SHADER);
    char *buf = NULL;
    size_t len = 0;
    get_file_content(vert_shader_path, &buf, &len);
    glShaderSource(vert_shader, 1, &buf, (GLint*) &len);
    glCompileShader(vert_shader);
    free(buf);
    get_file_content(frag_shader_path, &buf, &len);
    glShaderSource(frag_shader, 1, &buf, (GLint*) &len);
    glCompileShader(frag_shader);
    free(buf);
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

    GLuint prog = load_shader("vert.glsl", "frag.glsl");

    float vtxs[][3] = {
        -0.5, 0, 0,
        0.5, 0, 0,
        0, 1, 0
    };
    GLuint vtx_buf;
    glGenBuffers(1, &vtx_buf);
    glBindBuffer(GL_ARRAY_BUFFER, vtx_buf);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vtxs), vtxs, GL_STATIC_DRAW); 

    glClearColor(0, 0, 0, 0);
    glEnableVertexAttribArray(0);
    while (glfwWindowShouldClose(window) != GLFW_TRUE)
    {
        glClear(GL_COLOR_BUFFER_BIT);
        glUseProgram(prog);

        glBindBuffer(GL_ARRAY_BUFFER, vtx_buf);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, NULL);
        glDrawArrays(GL_TRIANGLES, 0, 3);

        glfwSwapBuffers(window);
        glfwPollEvents();
        if (glfwGetKey(window,GLFW_KEY_ESCAPE) == GLFW_PRESS)
        {
            glfwSetWindowShouldClose(window, GLFW_TRUE);
        }
    }
    return 0;
}