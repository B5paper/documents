# Vulkan Note QA

[unit]
[u_0]
写一个`VkApplicationInfo` struct。
[u_1]
```cpp
VkApplicationInfo appInfo;
appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
appInfo.pApplicationName = "Hello Triangle";
appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
appInfo.pEngineName = "No Engine";
appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
appInfo.apiVersion = VK_API_VERSION_1_0;
```

[unit]
[u_0]
创建一个 vulkan instance。
[u_1]
`main.cpp`:

```cpp
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

int main()
{
    VkApplicationInfo app_info{};
    app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app_info.pApplicationName = "hello";
    app_info.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    app_info.pEngineName = "no engine";
    app_info.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    app_info.apiVersion = VK_API_VERSION_1_0;

    VkInstanceCreateInfo inst_crt_info{};
    inst_crt_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    inst_crt_info.pApplicationInfo = &app_info;
    uint32_t glfwExtensionCount = 0;
    glfwInit();
    const char** glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
    inst_crt_info.enabledExtensionCount = glfwExtensionCount;
    inst_crt_info.ppEnabledExtensionNames = glfwExtensions;
    inst_crt_info.enabledLayerCount = 0;
    VkInstance instance;
    vkCreateInstance(&inst_crt_info, nullptr, &instance);
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
