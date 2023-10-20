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

枚举 vulkan 支持的 extension 和 glfw 需要的 extension：

```cpp
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <iostream>
#include <vector>
using namespace std;

int main()
{
    unsigned int glfw_extension_count = 0;
    glfwInit();
    const char **glfw_extensions = glfwGetRequiredInstanceExtensions(&glfw_extension_count);

    uint32_t extension_count = -1;
    vkEnumerateInstanceExtensionProperties(nullptr, &extension_count, nullptr);
    vector<VkExtensionProperties> extensions(extension_count);
    vkEnumerateInstanceExtensionProperties(nullptr, &extension_count, extensions.data());

    std::cout << "available extensions:\n";
    for (const auto& extension : extensions) {
        cout << '\t' << extension.extensionName << '\n';
    }

    cout << "glfw required instance extensions" << endl;
    for (int i = 0; i < glfw_extension_count; ++i) {
        cout << glfw_extensions[i] << endl;
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
available extensions:
        VK_KHR_device_group_creation
        VK_KHR_display
        VK_KHR_external_fence_capabilities
        VK_KHR_external_memory_capabilities
        VK_KHR_external_semaphore_capabilities
        VK_KHR_get_display_properties2
        VK_KHR_get_physical_device_properties2
        VK_KHR_get_surface_capabilities2
        VK_KHR_surface
        VK_KHR_surface_protected_capabilities
        VK_KHR_wayland_surface
        VK_KHR_xcb_surface
        VK_KHR_xlib_surface
        VK_EXT_acquire_drm_display
        VK_EXT_acquire_xlib_display
        VK_EXT_debug_report
        VK_EXT_direct_mode_display
        VK_EXT_display_surface_counter
        VK_EXT_debug_utils
glfw required instance extensions
VK_KHR_surface
VK_KHR_xcb_surface
```

### validation layer

There were formerly two different types of validation layers in Vulkan: instance and device specific.

The idea was that instance layers would only check calls related to global Vulkan objects like instances, and device specific layers would only check calls related to a specific GPU. Device specific layers have now been deprecated, which means that instance validation layers apply to all Vulkan calls. The specification document still recommends that you enable validation layers at device level as well for compatibility,

```cpp
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <iostream>
#include <vector>
#include <string>
using namespace std;

int main()
{
    unsigned int glfw_extension_count = 0;
    glfwInit();
    const char **glfw_extensions = glfwGetRequiredInstanceExtensions(&glfw_extension_count);

    uint32_t extension_count = -1;
    vkEnumerateInstanceExtensionProperties(nullptr, &extension_count, nullptr);
    vector<VkExtensionProperties> extensions(extension_count);
    vkEnumerateInstanceExtensionProperties(nullptr, &extension_count, extensions.data());

    cout << "available extensions:" << endl;
    for (const auto& extension : extensions) {
        cout << '\t' << extension.extensionName << '\n';
    }

    cout << "glfw required instance extensions" << endl;
    for (int i = 0; i < glfw_extension_count; ++i) {
        cout << glfw_extensions[i] << endl;
    }

    // list available layers
    uint32_t layer_count = -1;
    vkEnumerateInstanceLayerProperties(&layer_count, nullptr);
    vector<VkLayerProperties> available_layers(layer_count);
    vkEnumerateInstanceLayerProperties(&layer_count, available_layers.data());
    cout << "------------" << endl;
    cout << "available layers:" << endl;
    for (VkLayerProperties &layer: available_layers) {
        cout << layer.layerName << endl;
    }

    // check if validation layer exists
    const char *validation_layer_name = "VK_LAYER_KHRONOS_validation";
    bool is_validation_layer_exist = false;
    for (VkLayerProperties &layer: available_layers) {
        if (string(layer.layerName) == validation_layer_name) {
            is_validation_layer_exist = true;
            break;
        }
    }
    cout << "------------" << endl;
    if (!is_validation_layer_exist) {
        cout << "error: validation layer doesn't exist" << endl;
        return -1;
    } else {
        cout << "OK: validation layer exists" << endl;
    }
    return 0;
}
```

输出：

```
available extensions:
	VK_KHR_device_group_creation
	VK_KHR_display
	VK_KHR_external_fence_capabilities
	VK_KHR_external_memory_capabilities
	VK_KHR_external_semaphore_capabilities
	VK_KHR_get_display_properties2
	VK_KHR_get_physical_device_properties2
	VK_KHR_get_surface_capabilities2
	VK_KHR_surface
	VK_KHR_surface_protected_capabilities
	VK_KHR_wayland_surface
	VK_KHR_xcb_surface
	VK_KHR_xlib_surface
	VK_EXT_acquire_drm_display
	VK_EXT_acquire_xlib_display
	VK_EXT_debug_report
	VK_EXT_direct_mode_display
	VK_EXT_display_surface_counter
	VK_EXT_debug_utils
glfw required instance extensions
VK_KHR_surface
VK_KHR_xcb_surface
------------
available layers:
VK_LAYER_MESA_device_select
VK_LAYER_KHRONOS_validation
VK_LAYER_MESA_overlay
------------
OK: validation layer exists
```

validation layer 对应的名称为`"VK_LAYER_KHRONOS_validation"`。

为了使用 callback function，需要使用`VK_EXT_DEBUG_UTILS_EXTENSION_NAME` extension，这个宏等价于`"VK_EXT_debug_utils"`。

Example code:

```cpp
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <iostream>
#include <vector>
#include <string>
using namespace std;

static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity, VkDebugUtilsMessageTypeFlagsEXT messageType, const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData, void* pUserData) {
    std::cerr << "validation layer: " << pCallbackData->pMessage << std::endl;

    return VK_FALSE;
}

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
    if (glfwExtensions == nullptr) {
        cout << "fail to initialize glfw env" << endl;
        return -1;
    }
    vector<const char*> required_extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);
    required_extensions.push_back("VK_EXT_debug_utils");  // this is required for outputing debug info
    inst_crt_info.enabledExtensionCount = required_extensions.size();
    inst_crt_info.ppEnabledExtensionNames = required_extensions.data();

    // list available layers
    uint32_t layer_count = -1;
    vkEnumerateInstanceLayerProperties(&layer_count, nullptr);
    vector<VkLayerProperties> available_layers(layer_count);
    vkEnumerateInstanceLayerProperties(&layer_count, available_layers.data());
    cout << "------------" << endl;
    cout << "available layers:" << endl;
    for (VkLayerProperties &layer: available_layers) {
        cout << layer.layerName << endl;
    }

    // check if validation layer exists
    const char *validation_layer_name = "VK_LAYER_KHRONOS_validation";
    bool is_validation_layer_exist = false;
    for (VkLayerProperties &layer: available_layers) {
        if (string(layer.layerName) == validation_layer_name) {
            is_validation_layer_exist = true;
            break;
        }
    }
    cout << "------------" << endl;
    if (!is_validation_layer_exist) {
        cout << "error: validation layer doesn't exist" << endl;
        return -1;
    } else {
        cout << "OK: validation layer exists" << endl;
    }

    // set validation layer name as one of enabled layer nemes
    inst_crt_info.enabledLayerCount = 1;
    vector<const char*> required_layers{validation_layer_name};
    inst_crt_info.ppEnabledLayerNames = required_layers.data();

    // set debug info callback function
    VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo{};
    debugCreateInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
    debugCreateInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    debugCreateInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    debugCreateInfo.pfnUserCallback = debugCallback;
    inst_crt_info.pNext = (VkDebugUtilsMessengerCreateInfoEXT*) &debugCreateInfo;

    // create instance
    cout << "------------" << endl;
    cout << "start to create instance..." << endl;
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

输出：

```
------------
available layers:
VK_LAYER_MESA_device_select
VK_LAYER_KHRONOS_validation
VK_LAYER_MESA_overlay
------------
OK: validation layer exists
------------
start to create instance...
validation layer: Searching for ICD drivers named /usr/lib/i386-linux-gnu/libvulkan_lvp.so
validation layer: Searching for ICD drivers named /usr/lib/x86_64-linux-gnu/libvulkan_radeon.so
validation layer: Searching for ICD drivers named /usr/lib/i386-linux-gnu/libvulkan_radeon.so
validation layer: Searching for ICD drivers named /usr/lib/x86_64-linux-gnu/libvulkan_lvp.so
validation layer: Searching for ICD drivers named /usr/lib/i386-linux-gnu/libvulkan_intel.so
validation layer: Searching for ICD drivers named /usr/lib/x86_64-linux-gnu/libvulkan_intel.so
validation layer: Build ICD instance extension list
validation layer: Instance Extension: VK_KHR_device_group_creation (/usr/lib/x86_64-linux-gnu/libvulkan_radeon.so) version 0.0.1
validation layer: Instance Extension: VK_KHR_display (/usr/lib/x86_64-linux-gnu/libvulkan_radeon.so) version 0.0.23
validation layer: Instance Extension: VK_KHR_external_fence_capabilities (/usr/lib/x86_64-linux-gnu/libvulkan_radeon.so) version 0.0.1
validation layer: Instance Extension: VK_KHR_external_memory_capabilities (/usr/lib/x86_64-linux-gnu/libvulkan_radeon.so) version 0.0.1
validation layer: Instance Extension: VK_KHR_external_semaphore_capabilities (/usr/lib/x86_64-linux-gnu/libvulkan_radeon.so) version 0.0.1
validation layer: Instance Extension: VK_KHR_get_display_properties2 (/usr/lib/x86_64-linux-gnu/libvulkan_radeon.so) version 0.0.1
validation layer: Instance Extension: VK_KHR_get_physical_device_properties2 (/usr/lib/x86_64-linux-gnu/libvulkan_radeon.so) version 0.0.2
validation layer: Instance Extension: VK_KHR_get_surface_capabilities2 (/usr/lib/x86_64-linux-gnu/libvulkan_radeon.so) version 0.0.1
validation layer: Instance Extension: VK_KHR_surface (/usr/lib/x86_64-linux-gnu/libvulkan_radeon.so) version 0.0.25
validation layer: Instance Extension: VK_KHR_surface_protected_capabilities (/usr/lib/x86_64-linux-gnu/libvulkan_radeon.so) version 0.0.1
validation layer: Instance Extension: VK_KHR_wayland_surface (/usr/lib/x86_64-linux-gnu/libvulkan_radeon.so) version 0.0.6
validation layer: Instance Extension: VK_KHR_xcb_surface (/usr/lib/x86_64-linux-gnu/libvulkan_radeon.so) version 0.0.6
validation layer: Instance Extension: VK_KHR_xlib_surface (/usr/lib/x86_64-linux-gnu/libvulkan_radeon.so) version 0.0.6
validation layer: Instance Extension: VK_EXT_acquire_drm_display (/usr/lib/x86_64-linux-gnu/libvulkan_radeon.so) version 0.0.1
validation layer: Instance Extension: VK_EXT_acquire_xlib_display (/usr/lib/x86_64-linux-gnu/libvulkan_radeon.so) version 0.0.1
validation layer: Instance Extension: VK_EXT_debug_report (/usr/lib/x86_64-linux-gnu/libvulkan_radeon.so) version 0.0.10
validation layer: Instance Extension: VK_EXT_direct_mode_display (/usr/lib/x86_64-linux-gnu/libvulkan_radeon.so) version 0.0.1
validation layer: Instance Extension: VK_EXT_display_surface_counter (/usr/lib/x86_64-linux-gnu/libvulkan_radeon.so) version 0.0.1
validation layer: Instance Extension: VK_KHR_device_group_creation (/usr/lib/x86_64-linux-gnu/libvulkan_lvp.so) version 0.0.1
validation layer: Instance Extension: VK_KHR_external_fence_capabilities (/usr/lib/x86_64-linux-gnu/libvulkan_lvp.so) version 0.0.1
validation layer: Instance Extension: VK_KHR_external_memory_capabilities (/usr/lib/x86_64-linux-gnu/libvulkan_lvp.so) version 0.0.1
validation layer: Instance Extension: VK_KHR_external_semaphore_capabilities (/usr/lib/x86_64-linux-gnu/libvulkan_lvp.so) version 0.0.1
validation layer: Instance Extension: VK_KHR_get_physical_device_properties2 (/usr/lib/x86_64-linux-gnu/libvulkan_lvp.so) version 0.0.2
validation layer: Instance Extension: VK_KHR_get_surface_capabilities2 (/usr/lib/x86_64-linux-gnu/libvulkan_lvp.so) version 0.0.1
validation layer: Instance Extension: VK_KHR_surface (/usr/lib/x86_64-linux-gnu/libvulkan_lvp.so) version 0.0.25
validation layer: Instance Extension: VK_KHR_surface_protected_capabilities (/usr/lib/x86_64-linux-gnu/libvulkan_lvp.so) version 0.0.1
validation layer: Instance Extension: VK_KHR_wayland_surface (/usr/lib/x86_64-linux-gnu/libvulkan_lvp.so) version 0.0.6
validation layer: Instance Extension: VK_KHR_xcb_surface (/usr/lib/x86_64-linux-gnu/libvulkan_lvp.so) version 0.0.6
validation layer: Instance Extension: VK_KHR_xlib_surface (/usr/lib/x86_64-linux-gnu/libvulkan_lvp.so) version 0.0.6
validation layer: Instance Extension: VK_EXT_debug_report (/usr/lib/x86_64-linux-gnu/libvulkan_lvp.so) version 0.0.10
validation layer: Instance Extension: VK_KHR_device_group_creation (/usr/lib/x86_64-linux-gnu/libvulkan_intel.so) version 0.0.1
validation layer: Instance Extension: VK_KHR_display (/usr/lib/x86_64-linux-gnu/libvulkan_intel.so) version 0.0.23
validation layer: Instance Extension: VK_KHR_external_fence_capabilities (/usr/lib/x86_64-linux-gnu/libvulkan_intel.so) version 0.0.1
validation layer: Instance Extension: VK_KHR_external_memory_capabilities (/usr/lib/x86_64-linux-gnu/libvulkan_intel.so) version 0.0.1
validation layer: Instance Extension: VK_KHR_external_semaphore_capabilities (/usr/lib/x86_64-linux-gnu/libvulkan_intel.so) version 0.0.1
validation layer: Instance Extension: VK_KHR_get_display_properties2 (/usr/lib/x86_64-linux-gnu/libvulkan_intel.so) version 0.0.1
validation layer: Instance Extension: VK_KHR_get_physical_device_properties2 (/usr/lib/x86_64-linux-gnu/libvulkan_intel.so) version 0.0.2
validation layer: Instance Extension: VK_KHR_get_surface_capabilities2 (/usr/lib/x86_64-linux-gnu/libvulkan_intel.so) version 0.0.1
validation layer: Instance Extension: VK_KHR_surface (/usr/lib/x86_64-linux-gnu/libvulkan_intel.so) version 0.0.25
validation layer: Instance Extension: VK_KHR_surface_protected_capabilities (/usr/lib/x86_64-linux-gnu/libvulkan_intel.so) version 0.0.1
validation layer: Instance Extension: VK_KHR_wayland_surface (/usr/lib/x86_64-linux-gnu/libvulkan_intel.so) version 0.0.6
validation layer: Instance Extension: VK_KHR_xcb_surface (/usr/lib/x86_64-linux-gnu/libvulkan_intel.so) version 0.0.6
validation layer: Instance Extension: VK_KHR_xlib_surface (/usr/lib/x86_64-linux-gnu/libvulkan_intel.so) version 0.0.6
validation layer: Instance Extension: VK_EXT_acquire_drm_display (/usr/lib/x86_64-linux-gnu/libvulkan_intel.so) version 0.0.1
validation layer: Instance Extension: VK_EXT_acquire_xlib_display (/usr/lib/x86_64-linux-gnu/libvulkan_intel.so) version 0.0.1
validation layer: Instance Extension: VK_EXT_debug_report (/usr/lib/x86_64-linux-gnu/libvulkan_intel.so) version 0.0.10
validation layer: Instance Extension: VK_EXT_direct_mode_display (/usr/lib/x86_64-linux-gnu/libvulkan_intel.so) version 0.0.1
validation layer: Instance Extension: VK_EXT_display_surface_counter (/usr/lib/x86_64-linux-gnu/libvulkan_intel.so) version 0.0.1
validation layer: Build ICD instance extension list
validation layer: Instance Extension: VK_KHR_device_group_creation (/usr/lib/x86_64-linux-gnu/libvulkan_radeon.so) version 0.0.1
validation layer: Instance Extension: VK_KHR_display (/usr/lib/x86_64-linux-gnu/libvulkan_radeon.so) version 0.0.23
validation layer: Instance Extension: VK_KHR_external_fence_capabilities (/usr/lib/x86_64-linux-gnu/libvulkan_radeon.so) version 0.0.1
validation layer: Instance Extension: VK_KHR_external_memory_capabilities (/usr/lib/x86_64-linux-gnu/libvulkan_radeon.so) version 0.0.1
validation layer: Instance Extension: VK_KHR_external_semaphore_capabilities (/usr/lib/x86_64-linux-gnu/libvulkan_radeon.so) version 0.0.1
validation layer: Instance Extension: VK_KHR_get_display_properties2 (/usr/lib/x86_64-linux-gnu/libvulkan_radeon.so) version 0.0.1
validation layer: Instance Extension: VK_KHR_get_physical_device_properties2 (/usr/lib/x86_64-linux-gnu/libvulkan_radeon.so) version 0.0.2
validation layer: Instance Extension: VK_KHR_get_surface_capabilities2 (/usr/lib/x86_64-linux-gnu/libvulkan_radeon.so) version 0.0.1
validation layer: Instance Extension: VK_KHR_surface (/usr/lib/x86_64-linux-gnu/libvulkan_radeon.so) version 0.0.25
validation layer: Instance Extension: VK_KHR_surface_protected_capabilities (/usr/lib/x86_64-linux-gnu/libvulkan_radeon.so) version 0.0.1
validation layer: Instance Extension: VK_KHR_wayland_surface (/usr/lib/x86_64-linux-gnu/libvulkan_radeon.so) version 0.0.6
validation layer: Instance Extension: VK_KHR_xcb_surface (/usr/lib/x86_64-linux-gnu/libvulkan_radeon.so) version 0.0.6
validation layer: Instance Extension: VK_KHR_xlib_surface (/usr/lib/x86_64-linux-gnu/libvulkan_radeon.so) version 0.0.6
validation layer: Instance Extension: VK_EXT_acquire_drm_display (/usr/lib/x86_64-linux-gnu/libvulkan_radeon.so) version 0.0.1
validation layer: Instance Extension: VK_EXT_acquire_xlib_display (/usr/lib/x86_64-linux-gnu/libvulkan_radeon.so) version 0.0.1
validation layer: Instance Extension: VK_EXT_debug_report (/usr/lib/x86_64-linux-gnu/libvulkan_radeon.so) version 0.0.10
validation layer: Instance Extension: VK_EXT_direct_mode_display (/usr/lib/x86_64-linux-gnu/libvulkan_radeon.so) version 0.0.1
validation layer: Instance Extension: VK_EXT_display_surface_counter (/usr/lib/x86_64-linux-gnu/libvulkan_radeon.so) version 0.0.1
validation layer: Build ICD instance extension list
validation layer: Instance Extension: VK_KHR_device_group_creation (/usr/lib/x86_64-linux-gnu/libvulkan_lvp.so) version 0.0.1
validation layer: Instance Extension: VK_KHR_external_fence_capabilities (/usr/lib/x86_64-linux-gnu/libvulkan_lvp.so) version 0.0.1
validation layer: Instance Extension: VK_KHR_external_memory_capabilities (/usr/lib/x86_64-linux-gnu/libvulkan_lvp.so) version 0.0.1
validation layer: Instance Extension: VK_KHR_external_semaphore_capabilities (/usr/lib/x86_64-linux-gnu/libvulkan_lvp.so) version 0.0.1
validation layer: Instance Extension: VK_KHR_get_physical_device_properties2 (/usr/lib/x86_64-linux-gnu/libvulkan_lvp.so) version 0.0.2
validation layer: Instance Extension: VK_KHR_get_surface_capabilities2 (/usr/lib/x86_64-linux-gnu/libvulkan_lvp.so) version 0.0.1
validation layer: Instance Extension: VK_KHR_surface (/usr/lib/x86_64-linux-gnu/libvulkan_lvp.so) version 0.0.25
validation layer: Instance Extension: VK_KHR_surface_protected_capabilities (/usr/lib/x86_64-linux-gnu/libvulkan_lvp.so) version 0.0.1
validation layer: Instance Extension: VK_KHR_wayland_surface (/usr/lib/x86_64-linux-gnu/libvulkan_lvp.so) version 0.0.6
validation layer: Instance Extension: VK_KHR_xcb_surface (/usr/lib/x86_64-linux-gnu/libvulkan_lvp.so) version 0.0.6
validation layer: Instance Extension: VK_KHR_xlib_surface (/usr/lib/x86_64-linux-gnu/libvulkan_lvp.so) version 0.0.6
validation layer: Instance Extension: VK_EXT_debug_report (/usr/lib/x86_64-linux-gnu/libvulkan_lvp.so) version 0.0.10
validation layer: Build ICD instance extension list
validation layer: Instance Extension: VK_KHR_device_group_creation (/usr/lib/x86_64-linux-gnu/libvulkan_intel.so) version 0.0.1
validation layer: Instance Extension: VK_KHR_display (/usr/lib/x86_64-linux-gnu/libvulkan_intel.so) version 0.0.23
validation layer: Instance Extension: VK_KHR_external_fence_capabilities (/usr/lib/x86_64-linux-gnu/libvulkan_intel.so) version 0.0.1
validation layer: Instance Extension: VK_KHR_external_memory_capabilities (/usr/lib/x86_64-linux-gnu/libvulkan_intel.so) version 0.0.1
validation layer: Instance Extension: VK_KHR_external_semaphore_capabilities (/usr/lib/x86_64-linux-gnu/libvulkan_intel.so) version 0.0.1
validation layer: Instance Extension: VK_KHR_get_display_properties2 (/usr/lib/x86_64-linux-gnu/libvulkan_intel.so) version 0.0.1
validation layer: Instance Extension: VK_KHR_get_physical_device_properties2 (/usr/lib/x86_64-linux-gnu/libvulkan_intel.so) version 0.0.2
validation layer: Instance Extension: VK_KHR_get_surface_capabilities2 (/usr/lib/x86_64-linux-gnu/libvulkan_intel.so) version 0.0.1
validation layer: Instance Extension: VK_KHR_surface (/usr/lib/x86_64-linux-gnu/libvulkan_intel.so) version 0.0.25
validation layer: Instance Extension: VK_KHR_surface_protected_capabilities (/usr/lib/x86_64-linux-gnu/libvulkan_intel.so) version 0.0.1
validation layer: Instance Extension: VK_KHR_wayland_surface (/usr/lib/x86_64-linux-gnu/libvulkan_intel.so) version 0.0.6
validation layer: Instance Extension: VK_KHR_xcb_surface (/usr/lib/x86_64-linux-gnu/libvulkan_intel.so) version 0.0.6
validation layer: Instance Extension: VK_KHR_xlib_surface (/usr/lib/x86_64-linux-gnu/libvulkan_intel.so) version 0.0.6
validation layer: Instance Extension: VK_EXT_acquire_drm_display (/usr/lib/x86_64-linux-gnu/libvulkan_intel.so) version 0.0.1
validation layer: Instance Extension: VK_EXT_acquire_xlib_display (/usr/lib/x86_64-linux-gnu/libvulkan_intel.so) version 0.0.1
validation layer: Instance Extension: VK_EXT_debug_report (/usr/lib/x86_64-linux-gnu/libvulkan_intel.so) version 0.0.10
validation layer: Instance Extension: VK_EXT_direct_mode_display (/usr/lib/x86_64-linux-gnu/libvulkan_intel.so) version 0.0.1
validation layer: Instance Extension: VK_EXT_display_surface_counter (/usr/lib/x86_64-linux-gnu/libvulkan_intel.so) version 0.0.1
successfully create a vk instance.
```

### debug messenger

为了使用 debug messenger，我们首先需要启用`"VK_EXT_debug_utils"` extension，这个字符串等价于`VK_EXT_DEBUG_UTILS_EXTENSION_NAME`宏。

`messageType`对应的三种：

* VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT: Some event has hap-
pened that is unrelated to the specification or performance
* VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT: Something has
happened that violates the specification or indicates a possible mistake
* VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT: Potential non-
optimal use of Vulkan

`pCallbackData`比较重要的几个值：

* `pMessage`: The debug message as a null-terminated string
* `pObjects`: Array of Vulkan object handles related to the message
* `objectCount`: Number of objects in array