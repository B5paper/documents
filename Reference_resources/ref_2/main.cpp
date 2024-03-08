#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <stdlib.h>
#include <stdio.h>
#include <memory.h>
#include <unistd.h>

VkBool32 dbg_callback(VkDebugUtilsMessageSeverityFlagBitsEXT severity,
    VkDebugUtilsMessageTypeFlagsEXT msg_type,
    const VkDebugUtilsMessengerCallbackDataEXT *p_msg,
    void *p_user_data)
{
    printf("validation layer: %s\n", p_msg->pMessage);
    return VK_FALSE;
}

int main()
{
    glfwInit();

    // create vk instance
    VkInstanceCreateInfo inst_crt_info{};
    inst_crt_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    inst_crt_info.enabledLayerCount = 1;
    const char* enabled_inst_layers[] = {"VK_LAYER_KHRONOS_validation"};
    inst_crt_info.ppEnabledLayerNames = enabled_inst_layers;
    uint32_t glfw_required_inst_ext_count;
    const char **glfw_required_inst_exts = glfwGetRequiredInstanceExtensions(&glfw_required_inst_ext_count);
    inst_crt_info.enabledExtensionCount = glfw_required_inst_ext_count + 1;
    const char **enabled_inst_exts = (const char**) malloc(sizeof(char*) * inst_crt_info.enabledExtensionCount);
    for (int i = 0; i < glfw_required_inst_ext_count; ++i)
        enabled_inst_exts[i] = glfw_required_inst_exts[i];
    enabled_inst_exts[inst_crt_info.enabledExtensionCount - 1] = VK_EXT_DEBUG_UTILS_EXTENSION_NAME;
    inst_crt_info.ppEnabledExtensionNames = enabled_inst_exts;
    VkApplicationInfo app_info{};
    app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app_info.apiVersion = VK_API_VERSION_1_0;
    app_info.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    app_info.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    app_info.pApplicationName = "hello";
    app_info.pEngineName = "no engine";
    inst_crt_info.pApplicationInfo = &app_info;
    VkDebugUtilsMessengerCreateInfoEXT dbg_msgr_crt_info{};
    dbg_msgr_crt_info.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
    dbg_msgr_crt_info.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | 
        VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT |
        VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT;
    dbg_msgr_crt_info.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
        VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT |
        VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT;
    dbg_msgr_crt_info.pfnUserCallback = dbg_callback;
    inst_crt_info.pNext = &dbg_msgr_crt_info;
    VkInstance inst;
    VkResult result = vkCreateInstance(&inst_crt_info, nullptr, &inst);
    if (result != VK_SUCCESS)
    {
        printf("fail to create instance\n");
        exit(-1);
    }
    free(enabled_inst_exts);

    // selece physical device
    uint32_t phy_dev_cnt;
    vkEnumeratePhysicalDevices(inst, &phy_dev_cnt, nullptr);
    VkPhysicalDevice *phy_devs = (VkPhysicalDevice*) malloc(sizeof(VkPhysicalDevice*) * phy_dev_cnt);
    vkEnumeratePhysicalDevices(inst, &phy_dev_cnt, phy_devs);
    VkPhysicalDevice phy_dev;
    VkPhysicalDeviceProperties phy_dev_props;
    printf("Available physical devices:\n");
    for (int i = 0; i < phy_dev_cnt; ++i)
    {
        vkGetPhysicalDeviceProperties(phy_devs[i], &phy_dev_props);
        printf("%d: device name: %s, device type: %d\n",
            i, phy_dev_props.deviceName, phy_dev_props.deviceType);

        uint32_t queue_family_cnt;
        vkGetPhysicalDeviceQueueFamilyProperties(phy_devs[i], &queue_family_cnt, nullptr);
        VkQueueFamilyProperties * queue_family_props = (VkQueueFamilyProperties*) malloc(sizeof(VkQueueFamilyProperties*) * queue_family_cnt);
        printf("queue families:\n");
        for (int j = 0; j < queue_family_cnt; ++j)
        {
            printf("idx: %d,  queue count: %d, queue flags: %d\n",
                j, queue_family_props[j].queueCount, queue_family_props[j].queueFlags);
        }
        free(queue_family_props);
    }
    phy_dev = phy_devs[0];
    uint32_t queue_family_idx = 0;
    free(phy_devs);

    // create device
    VkDevice device;
    VkQueue queue;
    VkDeviceCreateInfo device_crt_info{};
    device_crt_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    device_crt_info.enabledExtensionCount = 1;
    const char *enabled_device_exts[] = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};
    device_crt_info.ppEnabledExtensionNames = enabled_device_exts;
    device_crt_info.enabledLayerCount = 0;
    device_crt_info.queueCreateInfoCount = 1;
    VkDeviceQueueCreateInfo dev_queue_crt_info{};
    dev_queue_crt_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    dev_queue_crt_info.queueCount = 1;
    const float queue_priprity = 1.0;
    dev_queue_crt_info.pQueuePriorities = &queue_priprity;
    dev_queue_crt_info.queueFamilyIndex = queue_family_idx;
    device_crt_info.pQueueCreateInfos = &dev_queue_crt_info;
    result = vkCreateDevice(phy_dev, &device_crt_info, nullptr, &device);
    if (result != VK_SUCCESS)
    {
        printf("fail to create device.\n");
        exit(-1);
    }
    vkGetDeviceQueue(device, queue_family_idx, 0, &queue);

    // create window surface
    VkSurfaceKHR surf;
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    GLFWwindow *window = glfwCreateWindow(700, 500, "hello", nullptr, nullptr);
    result = glfwCreateWindowSurface(inst, window, nullptr, &surf);
    if (result != VK_SUCCESS)
    {
        printf("fail to create window surface.\n");
        exit(-1);
    }

    // get surface info
    VkSurfaceCapabilitiesKHR surf_caps;
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(phy_dev, surf, &surf_caps);
    VkExtent2D surf_extent = surf_caps.currentExtent;
    VkSurfaceTransformFlagBitsKHR surf_trans = surf_caps.currentTransform;
    uint32_t surf_max_img_cnt = surf_caps.maxImageCount;
    uint32_t surf_min_img_cnt = surf_caps.minImageCount;
    uint32_t surf_max_img_arr_layer = surf_caps.maxImageArrayLayers;
    uint32_t surf_fmt_cnt;
    VkSurfaceFormatKHR *surf_fmts;
    VkFormat surf_fmt;
    VkColorSpaceKHR surf_color_space;
    vkGetPhysicalDeviceSurfaceFormatsKHR(phy_dev, surf, &surf_fmt_cnt, nullptr);
    surf_fmts = (VkSurfaceFormatKHR*) malloc(sizeof(VkSurfaceFormatKHR*) * surf_fmt_cnt);
    vkGetPhysicalDeviceSurfaceFormatsKHR(phy_dev, surf, &surf_fmt_cnt, surf_fmts);
    printf("surface info:\n");
    printf("extent: (%d, %d), min/max img count: %d/%d, max img array layer: %d\n",
        surf_extent.width, surf_extent.height,
        surf_min_img_cnt, surf_max_img_cnt, surf_max_img_arr_layer);
    printf("surface formats:\n");
    for (int i = 0; i < surf_fmt_cnt; ++i)
    {
        printf("%d: format: %d, color space: %d\n",
            i, surf_fmts[i].format, surf_fmts[i].colorSpace);
    }
    surf_fmt = surf_fmts[0].format;
    surf_color_space = surf_fmts[0].colorSpace;
    free(surf_fmts);

    // create swapchain
    VkSwapchainCreateInfoKHR swpch_crt_info{};
    swpch_crt_info.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    swpch_crt_info.surface = surf;
    swpch_crt_info.clipped = VK_FALSE;
    swpch_crt_info.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    swpch_crt_info.imageArrayLayers = 1;
    swpch_crt_info.imageColorSpace = surf_color_space;
    swpch_crt_info.imageExtent = surf_extent;
    swpch_crt_info.imageFormat = surf_fmt;
    swpch_crt_info.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    swpch_crt_info.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;  // 这个有啥用？
    swpch_crt_info.minImageCount = surf_min_img_cnt;
    // swpch_crt_info.minImageCount = surf_max_img_cnt;
    swpch_crt_info.minImageCount = surf_min_img_cnt;
    swpch_crt_info.pQueueFamilyIndices = &queue_family_idx;
    swpch_crt_info.queueFamilyIndexCount = 1;
    swpch_crt_info.presentMode = VK_PRESENT_MODE_FIFO_KHR;
    swpch_crt_info.preTransform = surf_trans;
    VkSwapchainKHR swpch;
    result = vkCreateSwapchainKHR(device, &swpch_crt_info, nullptr, &swpch);
    if (result != VK_SUCCESS)
    {
        printf("fail to create swapchain.\n");
        exit(-1);
    }

    // get swapchain info
    uint32_t swpch_img_cnt;
    VkImage *swpch_imgs;
    vkGetSwapchainImagesKHR(device, swpch, &swpch_img_cnt, nullptr);
    swpch_imgs = (VkImage*) malloc(sizeof(VkImage*) * swpch_img_cnt);
    vkGetSwapchainImagesKHR(device, swpch, &swpch_img_cnt, swpch_imgs);

    // create shader module
    size_t shader_code_size;
    char *shader_content;
    FILE *f = fopen("./vert.spv", "rb");
    fseek(f, 0, SEEK_END);
    shader_code_size = ftell(f);
    shader_content = (char*) malloc(shader_code_size);
    fseek(f, 0, SEEK_SET);
    fread(shader_content, shader_code_size, 1, f);
    fclose(f);
    VkShaderModuleCreateInfo shader_mod_crt_info{};
    shader_mod_crt_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    shader_mod_crt_info.pCode = (const uint32_t*) shader_content;
    shader_mod_crt_info.codeSize = shader_code_size;
    VkShaderModule vert_mod;
    result = vkCreateShaderModule(device, &shader_mod_crt_info, nullptr, &vert_mod);
    if (result != VK_SUCCESS)
    {
        printf("fail to create vertex shader module.\n");
        exit(-1);
    }
    free(shader_content);

    f = fopen("./frag.spv", "rb");
    fseek(f, 0, SEEK_END);
    shader_code_size = ftell(f);
    shader_content = (char*) malloc(shader_code_size);
    fseek(f, 0, SEEK_SET);
    fread(shader_content, shader_code_size, 1, f);
    shader_mod_crt_info.codeSize = shader_code_size;
    shader_mod_crt_info.pCode = (const uint32_t*) shader_content;
    VkShaderModule frag_mod;
    result = vkCreateShaderModule(device, &shader_mod_crt_info, nullptr, &frag_mod);
    if (result != VK_SUCCESS)
    {
        printf("fail to create fragment shader module.\n");
        exit(-1);
    }
    free(shader_content);
    fclose(f);

    // create render pass
    VkRenderPass rdpass;
    VkRenderPassCreateInfo rdpass_crt_info{};
    rdpass_crt_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    rdpass_crt_info.attachmentCount = 1;
    VkAttachmentDescription attachment_desc{};
    attachment_desc.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    attachment_desc.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    attachment_desc.format = surf_fmt;
    attachment_desc.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    attachment_desc.samples = VK_SAMPLE_COUNT_1_BIT;
    attachment_desc.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    attachment_desc.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    attachment_desc.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    rdpass_crt_info.pAttachments = &attachment_desc;
    rdpass_crt_info.subpassCount = 1;
    VkSubpassDescription subpass_desc{};
    subpass_desc.colorAttachmentCount = 1;
    VkAttachmentReference attachment_ref{};
    attachment_ref.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    attachment_ref.attachment = 0;
    subpass_desc.pColorAttachments = &attachment_ref;
    rdpass_crt_info.pSubpasses = &subpass_desc;
    result = vkCreateRenderPass(device, &rdpass_crt_info, nullptr, &rdpass);
    if (result != VK_SUCCESS)
    {
        printf("fail to create render pass.\n");
        exit(-1);
    }

    // create pipeline
    VkGraphicsPipelineCreateInfo gpipe_crt_info{};
    gpipe_crt_info.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    gpipe_crt_info.basePipelineHandle = nullptr;
    VkPipelineLayoutCreateInfo pipe_layout_crt_info{};
    pipe_layout_crt_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    VkPipelineLayout pipe_layout;
    result = vkCreatePipelineLayout(device, &pipe_layout_crt_info, nullptr, &pipe_layout);
    gpipe_crt_info.layout = pipe_layout;
    VkPipelineColorBlendStateCreateInfo color_blend_state_crt_info{};
    color_blend_state_crt_info.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    color_blend_state_crt_info.attachmentCount = 1;
    VkPipelineColorBlendAttachmentState color_blend_attachment_state{};
    color_blend_attachment_state.blendEnable = VK_FALSE;
    color_blend_attachment_state.colorWriteMask = VK_COLOR_COMPONENT_R_BIT |
        VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;  // 这行不能删掉，否则会报错
    color_blend_state_crt_info.pAttachments = &color_blend_attachment_state;
    color_blend_state_crt_info.logicOpEnable = VK_FALSE;
    gpipe_crt_info.pColorBlendState = &color_blend_state_crt_info;
    VkPipelineDynamicStateCreateInfo dynamic_state_crt_info{};
    VkDynamicState dynamic_states[2] = {
        VK_DYNAMIC_STATE_VIEWPORT,
        VK_DYNAMIC_STATE_SCISSOR
    };
    dynamic_state_crt_info.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamic_state_crt_info.dynamicStateCount = 2;
    dynamic_state_crt_info.pDynamicStates = dynamic_states;
    gpipe_crt_info.pDynamicState = &dynamic_state_crt_info;
    VkPipelineInputAssemblyStateCreateInfo input_assembly_state_crt_info{};
    input_assembly_state_crt_info.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    input_assembly_state_crt_info.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    input_assembly_state_crt_info.primitiveRestartEnable = VK_FALSE;
    gpipe_crt_info.pInputAssemblyState = &input_assembly_state_crt_info;
    VkPipelineMultisampleStateCreateInfo multisample_state_crt_info{};
    multisample_state_crt_info.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisample_state_crt_info.sampleShadingEnable = VK_FALSE;
    multisample_state_crt_info.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    gpipe_crt_info.pMultisampleState = &multisample_state_crt_info;
    VkPipelineRasterizationStateCreateInfo rasterization_state_crt_info{};
    rasterization_state_crt_info.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterization_state_crt_info.depthBiasClamp = VK_FALSE;
    rasterization_state_crt_info.rasterizerDiscardEnable = VK_FALSE;
    rasterization_state_crt_info.polygonMode = VK_POLYGON_MODE_FILL;
    rasterization_state_crt_info.lineWidth = 1.0;
    rasterization_state_crt_info.cullMode = VK_CULL_MODE_BACK_BIT;
    rasterization_state_crt_info.frontFace = VK_FRONT_FACE_CLOCKWISE;
    rasterization_state_crt_info.depthBiasEnable = VK_FALSE;
    gpipe_crt_info.pRasterizationState = &rasterization_state_crt_info;
    gpipe_crt_info.stageCount = 2;
    VkPipelineShaderStageCreateInfo shader_stage_crt_info[2];
    memset(shader_stage_crt_info, 0, sizeof(shader_stage_crt_info));
    shader_stage_crt_info[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    shader_stage_crt_info[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    shader_stage_crt_info[0].module = vert_mod;
    shader_stage_crt_info[0].pName = "main";
    shader_stage_crt_info[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    shader_stage_crt_info[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    shader_stage_crt_info[1].module = frag_mod;
    shader_stage_crt_info[1].pName = "main";
    gpipe_crt_info.pStages = shader_stage_crt_info;
    VkPipelineVertexInputStateCreateInfo vtx_input_state_crt_info{};
    vtx_input_state_crt_info.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vtx_input_state_crt_info.vertexBindingDescriptionCount = 1;
    VkVertexInputBindingDescription vtx_input_binding_desc;
    vtx_input_binding_desc.binding = 0;
    vtx_input_binding_desc.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
    vtx_input_binding_desc.stride = sizeof(float) * 3;
    vtx_input_state_crt_info.pVertexBindingDescriptions = &vtx_input_binding_desc;
    vtx_input_state_crt_info.vertexAttributeDescriptionCount = 1;
    VkVertexInputAttributeDescription vtx_input_attrib_desc;
    vtx_input_attrib_desc.binding = 0;
    vtx_input_attrib_desc.location = 0;
    vtx_input_attrib_desc.format = VK_FORMAT_R32G32B32_SFLOAT;
    vtx_input_attrib_desc.offset = 0;
    vtx_input_state_crt_info.pVertexAttributeDescriptions = &vtx_input_attrib_desc;
    gpipe_crt_info.pVertexInputState = &vtx_input_state_crt_info;
    VkPipelineViewportStateCreateInfo viewport_state_crt_info{};
    viewport_state_crt_info.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewport_state_crt_info.viewportCount = 1;  // 没有赋值指针，可以这样吗？
    viewport_state_crt_info.scissorCount = 1;
    gpipe_crt_info.pViewportState = &viewport_state_crt_info;
    gpipe_crt_info.renderPass = rdpass;
    gpipe_crt_info.subpass = 0;
    VkPipeline gpipe;
    result =  vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &gpipe_crt_info, nullptr, &gpipe);
    if (result != VK_SUCCESS)
    {
        printf("fail to create pipeline.\n");
        exit(-1);
    }

    // create input buffer
    float vtxs[9] = {
        -0.5, 0, 0,
        0, -0.5, 0,
        0.5, 0, 0
    };
    VkBuffer vtx_buf;
    VkDeviceMemory vtx_mem;
    VkBufferCreateInfo buf_crt_info{};
    buf_crt_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buf_crt_info.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
    buf_crt_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    buf_crt_info.size = sizeof(float) * 9;
    result = vkCreateBuffer(device, &buf_crt_info, nullptr, &vtx_buf);
    if (result != VK_SUCCESS)
    {
        printf("fail to create buffer.\n");
        exit(-1);
    }
    VkMemoryRequirements mem_reqs;
    vkGetBufferMemoryRequirements(device, vtx_buf, &mem_reqs);
    VkMemoryAllocateInfo mem_alc_info{};
    mem_alc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    mem_alc_info.allocationSize = mem_reqs.size;
    VkPhysicalDeviceMemoryProperties phy_dev_mem_props;
    vkGetPhysicalDeviceMemoryProperties(phy_dev, &phy_dev_mem_props);
    for (int i = 0; i < phy_dev_mem_props.memoryTypeCount; ++i)
    {
        if (mem_reqs.memoryTypeBits & (1 << i) == 0)
            continue;
        if ((phy_dev_mem_props.memoryTypes[i].propertyFlags &
            (VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT))
            == (VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT))
        {
            mem_alc_info.memoryTypeIndex = i;
            break;
        }
    }
    result = vkAllocateMemory(device, &mem_alc_info, nullptr, &vtx_mem);
    if (result != VK_SUCCESS)
    {
        printf("fail to allocate GPU memory\n");
        exit(-1);
    }
    void *mem_addr;
    vkMapMemory(device, vtx_mem, 0, mem_reqs.size, 0, &mem_addr);
    memcpy(mem_addr, vtxs, mem_reqs.size);
    vkUnmapMemory(device, vtx_mem);
    vkBindBufferMemory(device, vtx_buf, vtx_mem, 0);

    // create command pool, command buffer
    VkCommandPoolCreateInfo cmd_pool_crt_info{};
    cmd_pool_crt_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    cmd_pool_crt_info.queueFamilyIndex = queue_family_idx;
    cmd_pool_crt_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    VkCommandPool cmd_pool;
    result = vkCreateCommandPool(device, &cmd_pool_crt_info, nullptr, &cmd_pool);
    if (result != VK_SUCCESS)
    {
        printf("fail to create command pool.\n");
        exit(-1);
    }
    VkCommandBuffer cmd_buf;
    VkCommandBufferAllocateInfo cmd_buf_alc_info{};
    cmd_buf_alc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cmd_buf_alc_info.commandPool = cmd_pool;
    cmd_buf_alc_info.commandBufferCount = 1;
    cmd_buf_alc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    result = vkAllocateCommandBuffers(device, &cmd_buf_alc_info, &cmd_buf);
    if (result != VK_SUCCESS)
    {
        printf("fail to allocate command buffer\n");
        exit(-1);
    }

    // create fence and semaphores
    VkFence fence;
    VkFenceCreateInfo fence_crt_info{};
    fence_crt_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fence_crt_info.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    result = vkCreateFence(device, &fence_crt_info, nullptr, &fence);
    VkSemaphoreCreateInfo sem_crt_info{};
    sem_crt_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    VkSemaphore sem_render_finished, sem_image_available;
    result = vkCreateSemaphore(device, &sem_crt_info, nullptr, &sem_render_finished);
    result = vkCreateSemaphore(device, &sem_crt_info, nullptr, &sem_image_available);

    // create framebuffer
    VkImageView *swpch_img_views;
    swpch_img_views = (VkImageView*) malloc(sizeof(VkImageView*) * swpch_img_cnt);
    VkImageViewCreateInfo img_view_crt_info{};
    img_view_crt_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    img_view_crt_info.components = {VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY,
        VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY};
    img_view_crt_info.format = surf_fmt;
    img_view_crt_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    img_view_crt_info.subresourceRange.baseArrayLayer = 0;
    img_view_crt_info.subresourceRange.baseMipLevel = 0;
    img_view_crt_info.subresourceRange.layerCount = 1;
    img_view_crt_info.subresourceRange.levelCount = 1;
    img_view_crt_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
    VkFramebufferCreateInfo framebuf_crt_info{};
    framebuf_crt_info.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    framebuf_crt_info.attachmentCount = 1;
    framebuf_crt_info.height = surf_extent.height;
    framebuf_crt_info.layers = 1;
    framebuf_crt_info.renderPass = rdpass;
    framebuf_crt_info.width = surf_extent.width;
    VkFramebuffer *swpch_img_framebufs;
    swpch_img_framebufs = (VkFramebuffer*) malloc(sizeof(VkFramebuffer*) * swpch_img_cnt);
    for (int i = 0; i < swpch_img_cnt; ++i)
    {
        img_view_crt_info.image = swpch_imgs[i];
        result = vkCreateImageView(device, &img_view_crt_info, nullptr, swpch_img_views + i);
        framebuf_crt_info.pAttachments = &swpch_img_views[i];
        result = vkCreateFramebuffer(device, &framebuf_crt_info, nullptr, swpch_img_framebufs + i);
    }    

    // draw
    vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX);
    vkResetFences(device, 1, &fence);

    uint32_t swpch_img_idx;
    vkAcquireNextImageKHR(device, swpch, UINT64_MAX, sem_image_available, VK_NULL_HANDLE, &swpch_img_idx);

    vkResetCommandBuffer(cmd_buf, 0);
    VkCommandBufferBeginInfo cmd_buf_beg_info{};
    cmd_buf_beg_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    cmd_buf_beg_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    result = vkBeginCommandBuffer(cmd_buf, &cmd_buf_beg_info);
        VkRenderPassBeginInfo rdpass_beg_info{};
        rdpass_beg_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        rdpass_beg_info.clearValueCount = 1;
        VkClearValue clr_val = {0, 0, 0, 1};
        rdpass_beg_info.pClearValues = &clr_val;
        rdpass_beg_info.framebuffer = swpch_img_framebufs[swpch_img_idx];
        rdpass_beg_info.renderArea.extent = surf_extent;
        rdpass_beg_info.renderArea.offset = {0, 0};
        rdpass_beg_info.renderPass = rdpass;
        vkCmdBeginRenderPass(cmd_buf, &rdpass_beg_info, VK_SUBPASS_CONTENTS_INLINE);
            vkCmdBindPipeline(cmd_buf, VK_PIPELINE_BIND_POINT_GRAPHICS, gpipe);
            VkDeviceSize vtx_offset = 0;
            vkCmdBindVertexBuffers(cmd_buf, 0, 1, &vtx_buf, &vtx_offset);
            VkViewport viewport{};
            viewport.height = surf_extent.height;
            viewport.width = surf_extent.width;
            viewport.x = 0;
            viewport.y = 0;
            viewport.minDepth = 0;
            viewport.maxDepth = 1;
            vkCmdSetViewport(cmd_buf, 0, 1, &viewport);
            VkRect2D scissor;
            scissor.extent = surf_extent;
            scissor.offset = {0, 0};
            vkCmdSetScissor(cmd_buf, 0, 1, &scissor);
            vkCmdDraw(cmd_buf, 3, 1, 0, 0);
        vkCmdEndRenderPass(cmd_buf);
    vkEndCommandBuffer(cmd_buf);
    VkSubmitInfo submit_info{};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.waitSemaphoreCount = 1;
    submit_info.pWaitSemaphores = &sem_image_available;
    submit_info.signalSemaphoreCount = 1;
    submit_info.pSignalSemaphores = &sem_render_finished;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &cmd_buf;
    VkPipelineStageFlags pipeline_stage_flags = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    submit_info.pWaitDstStageMask = &pipeline_stage_flags;
    result = vkQueueSubmit(queue, 1, &submit_info, VK_NULL_HANDLE);
    if (result != VK_SUCCESS)
    {
        printf("fail to queue submit\n");
        exit(-1);
    }

    VkPresentInfoKHR present_info{};
    present_info.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    present_info.pImageIndices = &swpch_img_idx;
    present_info.swapchainCount = 1;
    present_info.pSwapchains = &swpch;
    present_info.pWaitSemaphores = &sem_render_finished;
    present_info.waitSemaphoreCount = 1;
    vkQueuePresentKHR(queue, &present_info);

    while (!glfwWindowShouldClose(window))
    {
        glfwPollEvents();
        sleep(1);
    }

    vkFreeCommandBuffers(device, cmd_pool, 1, &cmd_buf);
    vkDestroyCommandPool(device, cmd_pool, nullptr);
    free(swpch_img_framebufs);
    free(swpch_img_views);
    free(swpch_imgs);
    return 0;
}