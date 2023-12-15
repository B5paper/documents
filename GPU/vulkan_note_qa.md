# Vulkan Note QA

[unit]
[u_0]
写一个`VkApplicationInfo` struct。
[u_1]
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
    return 0;
}
```

[unit]
[u_0]
collect glfw extensions
[u_1]
(2023.12.15 version)
```cpp
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <vector>
using std::vector;
#include <iostream>
using std::cout, std::endl;

void collect_glfw_required_inst_exts(vector<const char*> &enabled_extensions)
{
    uint32_t ext_count;
    const char **glfw_exts = glfwGetRequiredInstanceExtensions(&ext_count);
    cout << "glfw requires " << ext_count << " extensions:" << endl;
    for (int i = 0; i < ext_count; ++i)
    {
        cout << glfw_exts[i] << endl;
        enabled_extensions.push_back(glfw_exts[i]);
    }
}

int main()
{
    glfwInit();
    vector<const char*> enabled_extensions;
    collect_glfw_required_inst_exts(enabled_extensions);
    return 0;
}
```

```cpp
void collect_glfw_extensions(vector<const char*> &enabled_extensions)
{
    uint32_t count;
    const char **glfw_required_extensions = glfwGetRequiredInstanceExtensions(&count);
    cout << "glfw requires " << count << " extensions:" << endl;
    for (int i = 0; i < count; ++i)
    {
        cout << glfw_required_extensions[i] << endl;
        enabled_extensions.push_back(glfw_required_extensions[i]);
    }
}
```

[unit]
[u_0]
创建一个 vulkan instance。
[u_1]
(2023.12.15 version)
```cpp
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <vector>
using std::vector;
#include <iostream>
using std::cout, std::endl;

void collect_glfw_required_inst_exts(vector<const char*> &enabled_extensions)
{
    uint32_t ext_count;
    const char **glfw_exts = glfwGetRequiredInstanceExtensions(&ext_count);
    cout << "glfw requires " << ext_count << " extensions:" << endl;
    for (int i = 0; i < ext_count; ++i)
    {
        cout << glfw_exts[i] << endl;
        enabled_extensions.push_back(glfw_exts[i]);
    }
}

int main()
{
    glfwInit();
    vector<const char*> enabled_extensions;
    collect_glfw_required_inst_exts(enabled_extensions);

    VkApplicationInfo app_info{};
    app_info.apiVersion = VK_API_VERSION_1_0;
    app_info.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    app_info.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    app_info.pApplicationName = "hello";
    app_info.pEngineName = "no engine";
    app_info.pNext = NULL;
    app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;

    VkInstanceCreateInfo inst_crt_info{};
    inst_crt_info.enabledExtensionCount = enabled_extensions.size();
    inst_crt_info.enabledLayerCount = 0;
    inst_crt_info.flags = 0;
    inst_crt_info.pApplicationInfo = &app_info;
    inst_crt_info.pNext = NULL;
    inst_crt_info.ppEnabledExtensionNames = enabled_extensions.data();
    inst_crt_info.ppEnabledLayerNames = nullptr;
    inst_crt_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;

    VkResult result;
    VkInstance inst;
    result = vkCreateInstance(&inst_crt_info, nullptr, &inst);
    if (result != VK_SUCCESS)
    {
        cout << "fail to create vulkan instance" << endl;
        exit(-1);
    }
    return 0;
}
```


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
g++ -g main.cpp -lglfw -lvulkan -o main
```

运行：

```bash
./main
```

[unit]
[u_0]
列出所有可用的 instance layer。
[u_1]
(2023.12.15 version)
```cpp
void collect_available_inst_layers(vector<VkLayerProperties> &available_inst_layers)
{
    uint32_t layer_count;
    vkEnumerateInstanceLayerProperties(&layer_count, nullptr);
    available_inst_layers.resize(layer_count);
    vkEnumerateInstanceLayerProperties(&layer_count, available_inst_layers.data());
    cout << "ther are " << layer_count << " available instance layers:" << endl;
    for (int i = 0; i < layer_count; ++i)
    {
        cout << available_inst_layers[i].layerName << endl;
    }
}
```

`main.cpp`:

```cpp
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <iostream>
#include <vector>
#include <string>
using namespace std;

int main()
{
    // list available layers
    uint32_t layer_count = -1;
    vkEnumerateInstanceLayerProperties(&layer_count, nullptr);
    vector<VkLayerProperties> available_layers(layer_count);
    vkEnumerateInstanceLayerProperties(&layer_count, available_layers.data());
    cout << "available layers:" << endl;
    for (VkLayerProperties &layer: available_layers) {
        cout << layer.layerName << endl;
    }
    return 0;
}
```

[unit]
[u_0]
创建 debug message create info structure，并写出对应的 callback 函数。
[u_1]
(2023.12.15 version)
```cpp
VkBool32 debug_callback(VkDebugUtilsMessageSeverityFlagBitsEXT message_severity,
    VkDebugUtilsMessageTypeFlagsEXT message_type,
    const VkDebugUtilsMessengerCallbackDataEXT *p_callback_data,
    void *p_user_data)
{
    cout << "validation layer: " << p_callback_data->pMessage << endl;
    return VK_FALSE;
}
```

```cpp
VkBool32 debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity, VkDebugUtilsMessageTypeFlagsEXT messageType, const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData, void* pUserData)
{
    std::cerr << "validation layer: " << pCallbackData->pMessage << std::endl;
    return VK_FALSE;
}

int main()
{
    VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo{};
    debugCreateInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
    debugCreateInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    debugCreateInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    debugCreateInfo.pfnUserCallback = debugCallback;
}
```

[unit]
[u_0]
写出 validation 所需要的 instance extension 和 layer name。
[u_1]
instance extension name: `VK_EXT_debug_utils`

instance layer name: `VK_LAYER_KHRONOS_validation`

[unit]
[u_0]
使用 callback validation layer 创建 instance。
[u_1]
(2023.12.15 version)
```cpp
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <vector>
using std::vector;
#include <iostream>
using std::cout, std::endl;

void collect_glfw_required_inst_exts(vector<const char*> &enabled_extensions)
{
    uint32_t ext_count;
    const char **glfw_exts = glfwGetRequiredInstanceExtensions(&ext_count);
    cout << "glfw requires " << ext_count << " extensions:" << endl;
    for (int i = 0; i < ext_count; ++i)
    {
        cout << glfw_exts[i] << endl;
        enabled_extensions.push_back(glfw_exts[i]);
    }
}

void collect_available_inst_layers(vector<VkLayerProperties> &available_inst_layers)
{
    uint32_t layer_count;
    vkEnumerateInstanceLayerProperties(&layer_count, nullptr);
    available_inst_layers.resize(layer_count);
    vkEnumerateInstanceLayerProperties(&layer_count, available_inst_layers.data());
    cout << layer_count << " available instance layers:" << endl;
    for (int i = 0; i < layer_count; ++i)
    {
        cout << available_inst_layers[i].layerName << endl;
    }
}

void collect_available_inst_exts(vector<VkExtensionProperties> &available_inst_exts)
{
    uint32_t inst_ext_count;
    vkEnumerateInstanceExtensionProperties(nullptr, &inst_ext_count, nullptr);
    available_inst_exts.resize(inst_ext_count);
    vkEnumerateInstanceExtensionProperties(nullptr, &inst_ext_count, available_inst_exts.data());
    cout << "there are " << inst_ext_count << " instance extensions available:" << endl;
    for (int i = 0; i < inst_ext_count; ++i)
    {
        cout << available_inst_exts[i].extensionName << endl;
    }
}

VkBool32 debug_callback(VkDebugUtilsMessageSeverityFlagBitsEXT message_severity,
    VkDebugUtilsMessageTypeFlagsEXT message_type,
    const VkDebugUtilsMessengerCallbackDataEXT *p_callback_data,
    void *p_user_data)
{
    cout << "validation layer: " << p_callback_data->pMessage << endl;
    return VK_FALSE;
}

int main()
{
    glfwInit();
    
    vector<VkExtensionProperties> available_inst_exts;
    collect_available_inst_exts(available_inst_exts);

    vector<VkLayerProperties> available_inst_layers;
    collect_available_inst_layers(available_inst_layers);

    vector<const char*> enabled_extensions;
    collect_glfw_required_inst_exts(enabled_extensions);
    enabled_extensions.push_back("VK_EXT_debug_utils");

    vector<const char*> enabled_inst_layers;
    enabled_inst_layers.push_back("VK_LAYER_KHRONOS_validation");

    VkApplicationInfo app_info{};
    app_info.apiVersion = VK_API_VERSION_1_0;
    app_info.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    app_info.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    app_info.pApplicationName = "hello";
    app_info.pEngineName = "no engine";
    app_info.pNext = NULL;
    app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;

    VkDebugUtilsMessengerCreateInfoEXT debug_messenger_crt_info{};
    debug_messenger_crt_info.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
    debug_messenger_crt_info.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT |
        VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
        VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    debug_messenger_crt_info.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
        VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
        VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    debug_messenger_crt_info.pfnUserCallback = debug_callback;

    VkInstanceCreateInfo inst_crt_info{};
    inst_crt_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    inst_crt_info.pApplicationInfo = &app_info;
    inst_crt_info.enabledExtensionCount = enabled_extensions.size();
    inst_crt_info.ppEnabledExtensionNames = enabled_extensions.data();
    inst_crt_info.enabledLayerCount = enabled_inst_layers.size();
    inst_crt_info.ppEnabledLayerNames = enabled_inst_layers.data();
    inst_crt_info.flags = 0;
    inst_crt_info.pNext = &debug_messenger_crt_info;

    VkResult result;
    VkInstance inst;
    result = vkCreateInstance(&inst_crt_info, nullptr, &inst);
    if (result != VK_SUCCESS)
    {
        cout << "fail to create vulkan instance" << endl;
        exit(-1);
    }
    return 0;
}
```

```cpp
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <iostream>
using namespace std;

VkBool32 debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity, VkDebugUtilsMessageTypeFlagsEXT messageType, const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData, void* pUserData) {
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
    uint32_t ext_count;
    const char **glfw_required_extensions = glfwGetRequiredInstanceExtensions(&ext_count);
    inst_crt_info.enabledExtensionCount = ext_count;
    inst_crt_info.ppEnabledExtensionNames = glfw_required_extensions;
    inst_crt_info.enabledLayerCount = 1;
    const char* layer_nemas[1] = {"VK_LAYER_KHRONOS_validation"};
    inst_crt_info.ppEnabledLayerNames = layer_nemas;

    VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo{};
    debugCreateInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
    debugCreateInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    debugCreateInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    debugCreateInfo.pfnUserCallback = debugCallback;

    inst_crt_info.pNext = &debugCreateInfo;
    VkInstance vk_inst;
    VkResult rtv = vkCreateInstance(&inst_crt_info, nullptr, &vk_inst);
    if (rtv != VK_SUCCESS) {
        cout << "fail to create instance" << endl;
    } else {
        cout << "successfully create instance" << endl;
    }
    
    return 0;
}
```

[unit]
[u_0]
列出所有可用的 physical device。
[u_1]
```cpp
void collect_available_physical_devices(const VkInstance inst,
    vector<VkPhysicalDevice> &phy_devs)
{
    uint32_t phy_dev_count;
    vkEnumeratePhysicalDevices(inst, &phy_dev_count, nullptr);
    phy_devs.resize(phy_dev_count);
    vkEnumeratePhysicalDevices(inst, &phy_dev_count, phy_devs.data());
}
```

[unit]
[u_0]
get physical device queue family
[u_1]
```cpp
void get_phy_dev_queue_families(VkPhysicalDevice phy_dev, 
    vector<VkQueueFamilyProperties> &queue_family_props)
{
    uint32_t queueFamilyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(phy_dev, &queueFamilyCount, nullptr);
    queue_family_props.resize(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(phy_dev, &queueFamilyCount, queue_family_props.data());
}
```

[unit]
[u_0]
select graphics queue family index
[u_1]
```cpp
uint32_t select_graphcs_queue_family_idx(
    const vector<VkQueueFamilyProperties> &queue_family_props,
    bool &valid)
{
    valid = false;
    for (int i = 0; i < queue_family_props.size(); ++i)
    {
        if (queue_family_props[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)
        {
            valid = true;
            return i;
        }
    }
    return 0;
}
```