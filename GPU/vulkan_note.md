# Vulkan Note

## 安装

```bash
sudo apt install vulkan-tools
sudo apt install libvulkan-dev
sudo apt install vulkan-validationlayers-dev spirv-tools
```

测试：

```bash
vkcube
```

## 第一份代码

`main.cpp`

```cpp
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/vec4.hpp>
#include <glm/mat4x4.hpp>
#include <iostream>

int main() {
    glfwInit();
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    GLFWwindow* window = glfwCreateWindow(800, 600, "Vulkan window", nullptr, nullptr);
    uint32_t extensionCount = 0;
    vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, nullptr);
    std::cout << extensionCount << " extensions supported\n";
    glm::mat4 matrix;
    glm::vec4 vec;
    auto test = matrix * vec;
    while(!glfwWindowShouldClose(window)) {
        glfwPollEvents();
    }
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
```

编译：

```bash
g++ -g main.cpp -lglfw -lvulkan -ldl -lpthread -lX11 -lXxf86vm -lXrandr -lXi -o main
```

运行：

```bash
./main
```

## 画一个三角形

### 1. base code

```cpp
#include <vulkan/vulkan.h>
#include <iostream>
#include <stdexcept>
#include <cstdlib>
class HelloTriangleApplication {
public:
    void run() {
        initVulkan();
        mainLoop();
        cleanup();
    }

private:
    void initVulkan() {

    }

    void mainLoop() {

    }

    void cleanup() {

    }
};

int main() {
    HelloTriangleApplication app;
    try {
        app.run();
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
```

编译：

```
g++ -g main.cpp -lglfw -lvulkan -ldl -lpthread -lX11 -lXxf86vm -lXrandr -lXi -o main
```

运行：

```
./main
```

### 2. instance

`main.cpp`:

```cpp
#define GLFW_INCLUDE_VULKAN  // 不写这一行会编译报错
#include <GLFW/glfw3.h>
#include <iostream>
using namespace std;

int main()
{
    VkApplicationInfo app_info{};
    app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app_info.pApplicationName = "hello";
    app_info.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    app_info.pEngineName = "no engine";
    app_info.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    app_info.apiVersion = VK_API_VERSION_1_0;

    VkInstanceCreateInfo inst_crt_info{};  // 清空所有字段，尤其是 pNext
    inst_crt_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    inst_crt_info.pApplicationInfo = &app_info;
    uint32_t glfwExtensionCount = 0;
    glfwInit();  // 如果不初始化 glfw 环境，下面的 glfwExtensions 会是 NULL
    const char** glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
    if (glfwExtensions == nullptr) {
        cout << "fail to initialize glfw env" << endl;
        return -1;
    }
    inst_crt_info.enabledExtensionCount = glfwExtensionCount;
    inst_crt_info.ppEnabledExtensionNames = glfwExtensions;
    inst_crt_info.enabledLayerCount = 0;
    VkInstance instance;
    VkResult result = vkCreateInstance(&inst_crt_info, nullptr, &instance);
    if (result != VK_SUCCESS) {
        cout << "failed to create instance." << endl;
        return -1;
    } else {
        cout << "successfully create a vk instance." << endl;
    }
    return 0;
}
```

编译：

```bash
g++ -g main.cpp -lglfw -lvulkan -ldl -lpthread -lX11 -lXxf86vm -lXrandr -lXi -o main
```

运行：

```bash
./main
```

输出：

```
successfully create a vk instance.
```