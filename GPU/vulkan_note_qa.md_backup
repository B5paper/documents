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
(2024.01.15 version)
```cpp
void collect_glfw_required_inst_exts(vector<const char*> &glfw_required_inst_exts)
{
    uint32_t ext_count;
    glfwGetRequiredInstanceExtensions(&ext_count);
    glfw_required_inst_exts.resize(ext_count);
    const char **ext_names = glfwGetRequiredInstanceExtensions(&ext_count);
    for (int i = 0; i < ext_count; ++i)
        glfw_required_inst_exts[i] = ext_names[i];

    printf("glfw requires %d instance extensions:\n", ext_count);
    for (int i = 0; i < ext_count; ++i)
        printf("%d: %s\n", i, glfw_required_inst_exts[i]);
}
```

(2023.12.18 version)

```cpp
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <vector>
using std::vector;
#include <iostream>
using std::cout, std::endl;

void collect_glfw_required_inst_exts(vector<const char*> &glfw_required_inst_exts)
{
    uint32_t ext_count;
    const char **glfw_required_inst_extensions = glfwGetRequiredInstanceExtensions(&ext_count);
    glfw_required_inst_exts.resize(ext_count);
    cout << "glfw requires " << ext_count << " instance extensions:" << endl;
    for (int i = 0; i < ext_count; ++i)
    {
        glfw_required_inst_exts[i] = glfw_required_inst_extensions[i];
        cout << glfw_required_inst_extensions[i] << endl;
    }
}

int main()
{
    glfwInit();
    vector<const char*> glfw_required_inst_exts;
    collect_glfw_required_inst_exts(glfw_required_inst_exts);
    return 0;
}
```

output:

```
glfw requires 2 instance extensions:
VK_KHR_surface
VK_KHR_xcb_surface
```

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

[unit]
[u_0]
创建一个 vulkan instance。不需要创建 debug messenger。
[u_1]
(2023.12.18 version)
`main.cpp`:

```cpp
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

int main()
{
    glfwInit();
    uint32_t glfw_req_inst_ext_cnt;
    const char **glfw_req_inst_exts = glfwGetRequiredInstanceExtensions(&glfw_req_inst_ext_cnt);

    VkInstance inst;
    VkInstanceCreateInfo inst_crt_info{};
    inst_crt_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    inst_crt_info.enabledExtensionCount = glfw_req_inst_ext_cnt;
    inst_crt_info.ppEnabledExtensionNames = glfw_req_inst_exts;
    VkApplicationInfo app_info{};
    app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app_info.apiVersion = VK_API_VERSION_1_0;
    app_info.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    app_info.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    app_info.pApplicationName = "hello";
    app_info.pEngineName = "no engine"; 
    inst_crt_info.pApplicationInfo = &app_info;
    VkResult result;
    result = vkCreateInstance(&inst_crt_info, nullptr, &inst);
    if (result != VK_SUCCESS)
    {
        printf("fail to create vk instance.\n");
        exit(-1);
    }
    printf("successfully create a vulkan instance.\n");
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

```cpp
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <vector>
using std::vector;
#include <iostream>
using std::cout, std::endl;

void collect_glfw_required_inst_exts(vector<const char*> &glfw_required_inst_exts)
{
    uint32_t ext_count;
    const char **glfw_required_inst_extensions = glfwGetRequiredInstanceExtensions(&ext_count);
    glfw_required_inst_exts.resize(ext_count);
    cout << "glfw requires " << ext_count << " instance extensions:" << endl;
    for (int i = 0; i < ext_count; ++i)
    {
        glfw_required_inst_exts[i] = glfw_required_inst_extensions[i];
        cout << glfw_required_inst_extensions[i] << endl;
    }
}

void fill_vk_app_info(VkApplicationInfo &app_info)
{
    app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app_info.apiVersion = VK_API_VERSION_1_2;
    app_info.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    app_info.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    app_info.pApplicationName = "hello";
    app_info.pEngineName = "no engine";
}

VkInstance create_vk_inst(const VkApplicationInfo &app_info,
    const vector<const char*> &enabled_inst_exts,
    const vector<const char*> &enabled_inst_layers)
{

    VkInstanceCreateInfo inst_crt_info{};
    inst_crt_info.enabledExtensionCount = enabled_inst_exts.size();
    inst_crt_info.enabledLayerCount = enabled_inst_layers.size();
    inst_crt_info.pApplicationInfo = &app_info;
    inst_crt_info.ppEnabledExtensionNames = enabled_inst_exts.data();
    inst_crt_info.ppEnabledLayerNames = enabled_inst_layers.data();
    inst_crt_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    VkInstance inst;
    VkResult result = vkCreateInstance(&inst_crt_info, nullptr, &inst);
    if (result != VK_SUCCESS)
    {
        cout << "fail to create vk instance" << endl;
        cout << "error code: " << result << endl;
        exit(-1);
    }
    return inst;
}

int main()
{
    glfwInit();
    vector<const char*> glfw_required_inst_exts;
    collect_glfw_required_inst_exts(glfw_required_inst_exts);
    VkApplicationInfo app_info{};
    fill_vk_app_info(app_info);
    VkInstance vk_inst = create_vk_inst(app_info, glfw_required_inst_exts, {});
    cout << "successfully create vk instance" << endl;
    return 0;
}
```

output:

```
glfw requires 2 instance extensions:
VK_KHR_surface
VK_KHR_xcb_surface
successfully create vk instance
```

[unit]
[u_0]
列出所有可用的 instance layer。
[u_1]
(2023.12.18 version)

```cpp
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <vector>
using std::vector;
#include <iostream>
using std::cout, std::endl;

void collect_available_instance_layers(vector<VkLayerProperties> &available_inst_layers)
{
    uint32_t layer_count;
    vkEnumerateInstanceLayerProperties(&layer_count, nullptr);
    available_inst_layers.resize(layer_count);
    vkEnumerateInstanceLayerProperties(&layer_count, available_inst_layers.data());
    cout << "vulkan instance has " << layer_count << " available layers:" << endl;
    for (int i = 0; i < layer_count; ++i)
    {
        cout << i << ": " << available_inst_layers[i].layerName << endl;
    }
}


int main()
{
    vector<VkLayerProperties> available_inst_layers;
    collect_available_instance_layers(available_inst_layers);
    return 0;
}
```

output:

```
vulkan instance has 4 available layers:
0: VK_LAYER_RENDERDOC_Capture
1: VK_LAYER_MESA_device_select
2: VK_LAYER_KHRONOS_validation
3: VK_LAYER_MESA_overlay
```

(2023.12.15 version)

```cpp
void collect_available_inst_layers(vector<VkLayerProperties> &available_inst_layers)
{
    uint32_t layer_count;
    vkEnumerateInstanceLayerProperties(&layer_count, nullptr);
    available_inst_layers.resize(layer_count);
    vkEnumerateInstanceLayerProperties(&layer_count, available_inst_layers.data());
    cout << "there are " << layer_count << " available instance layers:" << endl;
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
[idx]
3
[id]
<hash of the current time>
[dep]
(empty)
[u_1]
(2023.12.18 version)

```cpp
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <vector>
using std::vector;
#include <iostream>
using std::cout, std::endl;

VkBool32 dbg_callback(VkDebugUtilsMessageSeverityFlagBitsEXT msg_severity,
    VkDebugUtilsMessageTypeFlagsEXT msg_type,
    const VkDebugUtilsMessengerCallbackDataEXT *p_callback_data,
    void *p_user_data)
{
    cout << "validation layer: " << p_callback_data->pMessage << endl;
    return VK_FALSE;
}

void fill_debug_msg_crt_info(VkDebugUtilsMessengerCreateInfoEXT &dbg_msg_crt_info)
{
    dbg_msg_crt_info.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT |
        VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
        VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    dbg_msg_crt_info.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
        VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT |
        VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT;
    dbg_msg_crt_info.pfnUserCallback = dbg_callback;
    dbg_msg_crt_info.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
}

int main()
{
    VkDebugUtilsMessengerCreateInfoEXT dbg_msg_crt_info{};
    fill_debug_msg_crt_info(dbg_msg_crt_info);
    return 0;
}
```

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
(2023.12.18 version)

```cpp
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <vector>
using std::vector;
#include <iostream>
using std::cout, std::endl;

VkBool32 dbg_callback(VkDebugUtilsMessageSeverityFlagBitsEXT msg_severity,
    VkDebugUtilsMessageTypeFlagsEXT msg_type,
    const VkDebugUtilsMessengerCallbackDataEXT *p_callback_data,
    void *p_user_data)
{
    cout << "validation layer: " << p_callback_data->pMessage << endl;
    return VK_FALSE;
}

void fill_debug_msg_crt_info(VkDebugUtilsMessengerCreateInfoEXT &dbg_msg_crt_info)
{
    dbg_msg_crt_info.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT |
        VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
        VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    dbg_msg_crt_info.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
        VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT |
        VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT;
    dbg_msg_crt_info.pfnUserCallback = dbg_callback;
    dbg_msg_crt_info.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
}

void collect_glfw_required_inst_exts(vector<const char*> &enabled_inst_exts)
{
    uint32_t ext_count;
    const char **glfw_required_inst_exts = glfwGetRequiredInstanceExtensions(&ext_count);
    cout << "glfw requires " << ext_count << " instance extensions:" << endl;
    for (int i = 0; i < ext_count; ++i)
    {
        enabled_inst_exts.push_back(glfw_required_inst_exts[i]);
        cout << i << ": " << glfw_required_inst_exts[i] << endl;
    }
}

void collect_validation_required_inst_exts(vector<const char*> &validation_required_inst_exts)
{
    validation_required_inst_exts.push_back("VK_EXT_debug_utils");
}

void collect_validation_required_inst_layers(vector<const char*> &validation_required_inst_layers)
{
    validation_required_inst_layers.push_back("VK_LAYER_KHRONOS_validation");
}

void fill_app_info(VkApplicationInfo &app_info)
{
    app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app_info.apiVersion = VK_API_VERSION_1_2;
    app_info.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    app_info.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    app_info.pApplicationName = "hello";
    app_info.pEngineName = "no engine";
    app_info.pNext = nullptr;
}

VkInstance create_vk_instance(const VkApplicationInfo &app_info,
    const vector<const char*> &enabled_inst_exts,
    const vector<const char*> &enabled_inst_layers,
    const VkDebugUtilsMessengerCreateInfoEXT &dbg_msg_crt_info)
{
    VkInstanceCreateInfo inst_crt_info{};
    inst_crt_info.enabledExtensionCount = enabled_inst_exts.size();
    inst_crt_info.enabledLayerCount = enabled_inst_layers.size();
    inst_crt_info.flags = 0;
    inst_crt_info.pApplicationInfo = &app_info;
    inst_crt_info.pNext = &dbg_msg_crt_info;
    inst_crt_info.ppEnabledExtensionNames = enabled_inst_exts.data();
    inst_crt_info.ppEnabledLayerNames = enabled_inst_layers.data();
    inst_crt_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    VkInstance inst;
    VkResult result = vkCreateInstance(&inst_crt_info, nullptr, &inst);
    if (result != VK_SUCCESS)
    {
        cout << "fail to create vulkan instance" << endl;
        cout << "error code: " << result << endl;
        exit(-1);
    }
    return inst;
}

int main()
{
    glfwInit();
    vector<const char*> enabled_inst_exts;
    collect_glfw_required_inst_exts(enabled_inst_exts);
    collect_validation_required_inst_exts(enabled_inst_exts);
    vector<const char*> enabled_inst_layers;
    collect_validation_required_inst_layers(enabled_inst_layers);
    VkDebugUtilsMessengerCreateInfoEXT dbg_msg_crt_info{};
    fill_debug_msg_crt_info(dbg_msg_crt_info);
    VkApplicationInfo app_info{};
    fill_app_info(app_info);
    VkInstance inst = create_vk_instance(app_info,
        enabled_inst_exts,
        enabled_inst_layers,
        dbg_msg_crt_info);
    cout << "successfully create a vulkan instance" << endl;
    return 0;
}
```

output:

```
glfw requires 2 instance extensions:
0: VK_KHR_surface
1: VK_KHR_xcb_surface
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
successfully create a vulkan instance
```

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
(2023.12.18 version)

```cpp
void get_phy_dev_queue_family(const VkPhysicalDevice phy_dev,
    vector<VkQueueFamilyProperties> &queue_family_props)
{
    uint32_t queue_family_count;
    vkGetPhysicalDeviceQueueFamilyProperties(phy_dev, &queue_family_count, nullptr);
    queue_family_props.resize(queue_family_count);
    vkGetPhysicalDeviceQueueFamilyProperties(phy_dev, &queue_family_count, queue_family_props.data());
}
```

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

[unit]
[u_0]
select present queue family index
[u_1]
```cpp
uint32_t select_present_queue_family_idx(
    const VkPhysicalDevice phs_dev,
    const VkSurfaceKHR surface,
    const vector<VkQueueFamilyProperties> &queue_family_props,
    bool &valid)
{
    valid = false;
    VkBool32 presentSupport = false;
    for (int i = 0; i < queue_family_props.size(); ++i)
    {
        vkGetPhysicalDeviceSurfaceSupportKHR(phs_dev, i, surface, &presentSupport);
        if (presentSupport)
        {
            valid = true;
            return i;
        }
    }
    return 0;
}
```

[unit]
[u_0]
create window
[u_1]
```cpp
GLFWwindow* create_window(int width, int height, const char *title)
{
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
    GLFWwindow *window = glfwCreateWindow(width, height, title, NULL, NULL);
    return window;
}
```

[unit]
[u_0]
create window surface
[u_1]
```cpp
void create_window_surface(VkSurfaceKHR &surface, GLFWwindow *window, const VkInstance vk_inst)
{
    VkResult ret_val;
    ret_val = glfwCreateWindowSurface(vk_inst, window, nullptr, &surface);
    if (ret_val != VK_SUCCESS) {
        cout << "failed to create window surface!" << endl;;
        exit(-1);
    }
}
```
在执行 glfw create window 之前，必须要有`glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);`这一行。

[unit]
[u_0]
enable swapchain device extension
[u_1]
```cpp
void enable_swapchain_device_extension(vector<const char*> &enabled_device_extensions)
{
    enabled_device_extensions.push_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
}
```

[unit]
[u_0]
create vulkan logic device.
[u_1]
```cpp
VkDevice create_logic_device(
    VkPhysicalDevice phy_dev,
    const uint32_t graphics_queue_family_idx,
    const uint32_t present_queue_family_idx,
    const vector<const char*> &enabled_dev_extensions,
    const vector<const char*> &enabled_dev_layers)
{
    vector<uint32_t> queue_families;
    if (graphics_queue_family_idx == present_queue_family_idx)
    {
        queue_families.resize(1);
        queue_families[0] = graphics_queue_family_idx;
    }
    else
    {
        queue_families.resize(2);
        queue_families[0] = graphics_queue_family_idx;
        queue_families[1] = present_queue_family_idx;
    }

    vector<VkDeviceQueueCreateInfo> queueCreateInfos;
    const float queuePriority = 1.0f;
    for (uint32_t queue_family_idx: queue_families) {
        VkDeviceQueueCreateInfo queueCreateInfo{};
        queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queueCreateInfo.queueFamilyIndex = queue_family_idx;
        queueCreateInfo.queueCount = 1;
        queueCreateInfo.pQueuePriorities = &queuePriority;
        queueCreateInfos.push_back(queueCreateInfo);
    }

    VkDeviceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    VkPhysicalDeviceFeatures deviceFeatures{};
    createInfo.pEnabledFeatures = &deviceFeatures;
    createInfo.queueCreateInfoCount = queueCreateInfos.size();
    createInfo.pQueueCreateInfos = queueCreateInfos.data();
    createInfo.enabledExtensionCount = enabled_dev_extensions.size();
    createInfo.ppEnabledExtensionNames = enabled_dev_extensions.data();
    createInfo.enabledLayerCount = enabled_dev_layers.size();
    createInfo.ppEnabledLayerNames = enabled_dev_layers.data();

    VkDevice device;
    VkResult ret_val = vkCreateDevice(phy_dev, &createInfo, nullptr, &device);
    if (ret_val != VK_SUCCESS) {
        cout << "failed to create logical device!" << endl;
        cout << "error code: " << ret_val << endl;
    }
    else {
        cout << "successfully create logical device" << endl;
    }

    return device;
}
```

[unit]
[u_0]
创建一个 1MB 的 buffer，要求使用显存，主机可见，并且绑定到 memory 对象上。
[u_1]
```cpp
void create_vk_buffer(VkBuffer &buffer,
    VkDeviceMemory &bufferMemory,
    const VkPhysicalDevice phy_dev,
    const VkDevice device,
    const VkDeviceSize size,
    const VkBufferUsageFlags usage,
    const VkMemoryPropertyFlags properties)
{
    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = size;
    bufferInfo.usage = usage;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VkResult result = vkCreateBuffer(device, &bufferInfo, nullptr, &buffer);
    if (result != VK_SUCCESS)
    {
        printf("fail to create buffer, error code: %d\n", result);
        exit(-1);
    }

    VkMemoryRequirements mem_req;
    vkGetBufferMemoryRequirements(device, buffer, &mem_req);

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = mem_req.size;
    VkPhysicalDeviceMemoryProperties mem_prop;
    vkGetPhysicalDeviceMemoryProperties(phy_dev, &mem_prop);
    bool valid_mem_type = false;
    uint32_t mem_type_idx = -1;
    for (uint32_t i = 0; i < mem_prop.memoryTypeCount; i++)
    {
        if ((mem_req.memoryTypeBits & (1 << i)) && (mem_prop.memoryTypes[i].propertyFlags & properties) == properties)
        {
            valid_mem_type = true;
            mem_type_idx = i;
            break;
        }
    }
    if (!valid_mem_type)
    {
        printf("fail to find an appropriate memoty type.\n");
        exit(-1);
    }
    allocInfo.memoryTypeIndex = mem_type_idx;

    result = vkAllocateMemory(device, &allocInfo, nullptr, &bufferMemory);
    if (result != VK_SUCCESS)
    {
        printf("fail to allocate vk memory, error code: %d\n", result);
        exit(-1);
    }

    vkBindBufferMemory(device, buffer, bufferMemory, 0);
}

int main()
{
    create_vk_buffer(buf, buf_mem, phy_dev, dev, 1024, VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    return 0;
}
```

[unit]
[u_0]
解释`vkGetPhysicalDeviceMemoryProperties()`得到的信息的含义。
[u_1]
函数原型：

```c
void vkGetPhysicalDeviceMemoryProperties(
    VkPhysicalDevice                            physicalDevice,
    VkPhysicalDeviceMemoryProperties*           pMemoryProperties);
```

得到的信息是不同属性组合的内存类型。


