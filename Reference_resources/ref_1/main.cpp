#include "../simple_vulkan/simple_vk.hpp"

int main()
{
    VkResult result;
    glfwInit();
    VkInstance inst;
    create_vk_instance(inst);
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    GLFWwindow *window = glfwCreateWindow(700, 500, "hello", nullptr, nullptr);
    VkSurfaceKHR surf;
    glfwCreateWindowSurface(inst, window, nullptr, &surf);
    VkPhysicalDevice phy_dev;
    uint32_t queue_family_idx;
    select_vk_physical_device(phy_dev, queue_family_idx, queue_family_idx, inst, surf);
    VkDevice device;
    VkQueue queue;
    create_vk_device(device, queue, queue, phy_dev, queue_family_idx, queue_family_idx);
    VkSwapchainKHR swpch;
    create_vk_swapchain(swpch, device, surf, queue_family_idx);
    VkRenderPass render_pass = create_render_pass(VK_FORMAT_B8G8R8A8_SRGB, device);

    // create pipeline
    auto vert_shader_code = read_file("vert_2.spv");  // "vert_2.spv"
    auto frag_shader_code = read_file("frag_2.spv");  // "frag_2.spv"
    VkShaderModule vertShaderModule = create_shader_module(vert_shader_code, device);
    VkShaderModule fragShaderModule = create_shader_module(frag_shader_code, device);
    VkPipelineShaderStageCreateInfo vertShaderStageInfo{};
    vertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
    vertShaderStageInfo.module = vertShaderModule;
    vertShaderStageInfo.pName = "main";
    VkPipelineShaderStageCreateInfo fragShaderStageInfo{};
    fragShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    fragShaderStageInfo.module = fragShaderModule;
    fragShaderStageInfo.pName = "main";
    VkPipelineShaderStageCreateInfo shaderStages[] = {vertShaderStageInfo, fragShaderStageInfo};
    
    VkVertexInputBindingDescription vtx_binding_desc{};
    vtx_binding_desc.binding = 0;
    vtx_binding_desc.stride = sizeof(float) * 3;  // (x, y, z) 三个分量
    vtx_binding_desc.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

    VkVertexInputAttributeDescription vtx_attr_desc{};
    vtx_attr_desc.binding = 0;
    vtx_attr_desc.location = 0;
    vtx_attr_desc.format = VK_FORMAT_R32G32B32_SFLOAT;
    vtx_attr_desc.offset = 0;

    VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
    vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertexInputInfo.vertexBindingDescriptionCount = 1;
    vertexInputInfo.pVertexBindingDescriptions = &vtx_binding_desc;
    vertexInputInfo.vertexAttributeDescriptionCount = 1;
    vertexInputInfo.pVertexAttributeDescriptions = &vtx_attr_desc;

    VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
    inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    inputAssembly.primitiveRestartEnable = VK_FALSE;

    VkPipelineViewportStateCreateInfo viewportState{};
    viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = 1;
    viewportState.scissorCount = 1;

    VkPipelineRasterizationStateCreateInfo rasterizer{};
    rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizer.depthClampEnable = VK_FALSE;
    rasterizer.rasterizerDiscardEnable = VK_FALSE;
    rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
    rasterizer.lineWidth = 1.0f;
    rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
    rasterizer.frontFace = VK_FRONT_FACE_CLOCKWISE;
    rasterizer.depthBiasEnable = VK_FALSE;

    VkPipelineMultisampleStateCreateInfo multisampling{};
    multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampling.sampleShadingEnable = VK_FALSE;
    multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineColorBlendAttachmentState colorBlendAttachment{};
    colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    colorBlendAttachment.blendEnable = VK_FALSE;

    VkPipelineColorBlendStateCreateInfo colorBlending{};
    colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlending.logicOpEnable = VK_FALSE;
    colorBlending.logicOp = VK_LOGIC_OP_COPY;
    colorBlending.attachmentCount = 1;
    colorBlending.pAttachments = &colorBlendAttachment;
    colorBlending.blendConstants[0] = 0.0f;
    colorBlending.blendConstants[1] = 0.0f;
    colorBlending.blendConstants[2] = 0.0f;
    colorBlending.blendConstants[3] = 0.0f;

    std::vector<VkDynamicState> dynamicStates = {
        VK_DYNAMIC_STATE_VIEWPORT,
        VK_DYNAMIC_STATE_SCISSOR
    };
    VkPipelineDynamicStateCreateInfo dynamicState{};
    dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
    dynamicState.pDynamicStates = dynamicStates.data();

    VkDescriptorPoolCreateInfo desc_pool_crt_info{};
    desc_pool_crt_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    desc_pool_crt_info.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    desc_pool_crt_info.maxSets = 5;
    desc_pool_crt_info.poolSizeCount = 1;
    VkDescriptorPoolSize desc_pool_size{};
    desc_pool_size.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    desc_pool_size.descriptorCount = 2;
    desc_pool_crt_info.pPoolSizes = &desc_pool_size;
    VkDescriptorPool desc_pool;
    result = vkCreateDescriptorPool(device, &desc_pool_crt_info, nullptr, &desc_pool);
    if (result != VK_SUCCESS)
    {
        printf("fail to create descriptor pool, error code: %d\n", result);
        exit(-1);
    }

    VkDescriptorSetLayout desc_set_layout;
    VkDescriptorSetLayoutCreateInfo desc_set_layout_crt_info{};
    desc_set_layout_crt_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    desc_set_layout_crt_info.bindingCount = 1;
    VkDescriptorSetLayoutBinding desc_set_layout_binding{};
    desc_set_layout_binding.binding = 0;
    desc_set_layout_binding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    desc_set_layout_binding.descriptorCount = 1;
    desc_set_layout_binding.stageFlags = VK_SHADER_STAGE_ALL;
    desc_set_layout_crt_info.pBindings = &desc_set_layout_binding;
    result = vkCreateDescriptorSetLayout(device, &desc_set_layout_crt_info, nullptr, &desc_set_layout);
    if (result != VK_SUCCESS)
    {
        printf("fail to create descriptor set layout, error code: %d\n", result);
        exit(-1);
    }

    VkDescriptorSet desc_set;
    VkDescriptorSetAllocateInfo desc_set_allo_info{};
    desc_set_allo_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    desc_set_allo_info.pSetLayouts = &desc_set_layout;
    desc_set_allo_info.descriptorSetCount = 1;
    desc_set_allo_info.descriptorPool = desc_pool;
    result = vkAllocateDescriptorSets(device, &desc_set_allo_info, &desc_set);
    if (result != VK_SUCCESS)
    {
        printf("fail to allocate descriptor set, error code: %d\n", result);
        exit(-1);
    }

    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &desc_set_layout;
    pipelineLayoutInfo.pushConstantRangeCount = 0;
    VkPipelineLayout pipelineLayout;
    vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &pipelineLayout);

    VkGraphicsPipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipelineInfo.stageCount = 2;
    pipelineInfo.pStages = shaderStages;
    pipelineInfo.pVertexInputState = &vertexInputInfo;
    pipelineInfo.pInputAssemblyState = &inputAssembly;
    pipelineInfo.pViewportState = &viewportState;
    pipelineInfo.pRasterizationState = &rasterizer;
    pipelineInfo.pMultisampleState = &multisampling;
    pipelineInfo.pColorBlendState = &colorBlending;
    pipelineInfo.pDynamicState = &dynamicState;
    pipelineInfo.layout = pipelineLayout;
    pipelineInfo.renderPass = render_pass;
    pipelineInfo.subpass = 0;
    pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;

    VkPipeline pipeline;
    if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipeline) != VK_SUCCESS) {
        throw std::runtime_error("failed to create graphics pipeline!");
    }

    // VkPipeline pipeline = create_pipeline("./vert_2.spv", "frag_2.spv", 9 * sizeof(float), {0, 3 * sizeof(float)}, device, {700, 500}, render_pass);
    
    uint32_t swpch_img_count;
    vkGetSwapchainImagesKHR(device, swpch, &swpch_img_count, nullptr);
    std::vector<VkImage> swpch_imgs(swpch_img_count);
    vkGetSwapchainImagesKHR(device, swpch, &swpch_img_count, swpch_imgs.data());

    std::vector<VkImageView> swpch_img_views(swpch_img_count);
    VkImageViewCreateInfo img_view_crt_info{};
    img_view_crt_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    img_view_crt_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
    img_view_crt_info.format = VK_FORMAT_B8G8R8A8_SRGB;
    img_view_crt_info.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    for (int i = 0; i < swpch_img_count; ++i)
    {
        img_view_crt_info.image = swpch_imgs[i];
        result = vkCreateImageView(device, &img_view_crt_info, nullptr, &swpch_img_views[i]);
        if (result != VK_SUCCESS)
        {
            printf("fail to create image view, error code %d\n", result);
            exit(-1);
        }
    }

    VkFramebufferCreateInfo frame_buf_crt_info{};
    frame_buf_crt_info.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    frame_buf_crt_info.renderPass = render_pass;
    frame_buf_crt_info.attachmentCount = 1;
    frame_buf_crt_info.width = 700;
    frame_buf_crt_info.height = 500;
    frame_buf_crt_info.layers = 1;
    std::vector<VkFramebuffer> frame_bufs(swpch_img_count);
    for (int i = 0; i < swpch_img_count; ++i)
    {
        frame_buf_crt_info.pAttachments = &swpch_img_views[i];
        result = vkCreateFramebuffer(device, &frame_buf_crt_info, nullptr, &frame_bufs[i]);
        if (result != VK_SUCCESS)
        {
            printf("fail to create frame buffer, error code: %d\n", result);
            exit(-1);
        }
    }

    VkCommandPool cmd_pool = create_command_pool(device, phy_dev, queue_family_idx);
    cmd_pool = create_command_pool(device, phy_dev, queue_family_idx);
    VkCommandBuffer cmd_buf;
    VkCommandBufferAllocateInfo cmd_buf_alc_info{};
    cmd_buf_alc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cmd_buf_alc_info.commandBufferCount = 1;
    cmd_buf_alc_info.commandPool = cmd_pool;
    cmd_buf_alc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    result = vkAllocateCommandBuffers(device, &cmd_buf_alc_info, &cmd_buf);
    if (result != VK_SUCCESS)
    {
        printf("fail to allocate command buffer\n");
        exit(-1);
    }

    float vtxs[9] = {
        0, -1, 0,
        1, 1, 0,
        -1, 1, 0
    };
    VkBuffer vtx_buf;
    VkDeviceMemory vtx_buf_mem;
    create_vk_buffer(vtx_buf, vtx_buf_mem, phy_dev, device, 3 * 3 * sizeof(float), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
    float *p_mem_data = nullptr;
    result = vkMapMemory(device, vtx_buf_mem, 0, VK_WHOLE_SIZE, 0, (void**) &p_mem_data);
    memcpy(p_mem_data, vtxs, sizeof(vtxs));
    vkUnmapMemory(device, vtx_buf_mem);
    vkDeviceWaitIdle(device);

    float temp_mem[9];
    vkMapMemory(device, vtx_buf_mem, 0, VK_WHOLE_SIZE, 0, (void**) &p_mem_data);
    memcpy(temp_mem, p_mem_data, sizeof(vtxs));
    vkUnmapMemory(device, vtx_buf_mem);

    uint32_t idxs[3] = {0, 1, 2};
    VkBuffer idx_buf;
    VkDeviceMemory idx_buf_mem;
    create_vk_buffer(idx_buf, idx_buf_mem, phy_dev, device, sizeof(idxs), VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
    vkMapMemory(device, idx_buf_mem, 0, VK_WHOLE_SIZE, 0, (void**) &p_mem_data);
    memcpy(p_mem_data, idxs, sizeof(idxs));
    vkUnmapMemory(device, idx_buf_mem);

    float rgb[3] = {0.8, 0.5, 0.5};
    VkBuffer color_buf;
    VkDeviceMemory color_buf_mem;
    create_vk_buffer(color_buf, color_buf_mem, phy_dev, device, sizeof(float) * 3, 
        VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
    vkMapMemory(device, color_buf_mem, 0, VK_WHOLE_SIZE, 0, (void**) &p_mem_data);
    memcpy(p_mem_data, rgb, sizeof(rgb));
    vkUnmapMemory(device, color_buf_mem);

    VkWriteDescriptorSet wrt_desc_set{};
    wrt_desc_set.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    wrt_desc_set.dstSet = desc_set;
    VkDescriptorBufferInfo desc_buf_info{};
    desc_buf_info.buffer = color_buf;
    wrt_desc_set.pBufferInfo = &desc_buf_info;
    wrt_desc_set.descriptorCount = 1;
    desc_buf_info.offset = 0;
    desc_buf_info.range = VK_WHOLE_SIZE;
    wrt_desc_set.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    wrt_desc_set.dstArrayElement = 0;
    wrt_desc_set.dstBinding = 0;
    vkUpdateDescriptorSets(device, 1, &wrt_desc_set, 0, nullptr);

    VkSemaphore sem_finish_rendering = create_semaphore(device);
    VkSemaphore sem_img_available = create_semaphore(device);
    VkFence fence_acq_img = create_fence(device);
    VkFence fence_queue_submit = create_fence(device);

    vkResetFences(device, 1, &fence_acq_img);
    uint32_t available_img_idx;
    result = vkAcquireNextImageKHR(device, swpch, UINT64_MAX, sem_img_available, fence_acq_img, &available_img_idx);

    VkCommandBufferBeginInfo cmd_buf_beg_info{};
    cmd_buf_beg_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    cmd_buf_beg_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd_buf, &cmd_buf_beg_info);

        VkRenderPassBeginInfo rdps_beg_info{};
        rdps_beg_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        rdps_beg_info.renderPass = render_pass;
        rdps_beg_info.framebuffer = frame_bufs[available_img_idx];
        rdps_beg_info.renderArea.offset = {0, 0};
        rdps_beg_info.renderArea.extent = {700, 500};
        rdps_beg_info.clearValueCount = 1;
        VkClearValue clr_val{0, 0, 0, 1};
        rdps_beg_info.pClearValues = &clr_val;
        vkCmdBeginRenderPass(cmd_buf, &rdps_beg_info, VK_SUBPASS_CONTENTS_INLINE);

            vkCmdBindPipeline(cmd_buf, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
            vkCmdBindDescriptorSets(cmd_buf, VK_PIPELINE_BIND_POINT_GRAPHICS,
                pipelineLayout, 0, 1, &desc_set, 0, nullptr);
            VkViewport viewport{};
            viewport.x = 0.0f;
            viewport.y = 0.0f;
            viewport.width = 700;
            viewport.height = 500;
            viewport.minDepth = 0.0f;
            viewport.maxDepth = 1.0f;
            vkCmdSetViewport(cmd_buf, 0, 1, &viewport);
            VkRect2D scissor{};
            scissor.offset = {0, 0};
            scissor.extent.width = 700;
            scissor.extent.height = 500;
            vkCmdSetScissor(cmd_buf, 0, 1, &scissor);
            VkDeviceSize offset = 0;
            vkCmdBindVertexBuffers(cmd_buf, 0, 1, &vtx_buf, &offset);
            
            // vkCmdBindIndexBuffer(cmd_buf, idx_buf, 0, VK_INDEX_TYPE_UINT32);
            // vkCmdDrawIndexed(cmd_buf, 3, 1, 0, 0, 0);
            vkCmdDraw(cmd_buf, 3, 1, 0, 0);

        vkCmdEndRenderPass(cmd_buf);

    result = vkEndCommandBuffer(cmd_buf);

    vkResetFences(device, 1, &fence_queue_submit);
    VkSubmitInfo submit_info{};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.signalSemaphoreCount = 1;
    submit_info.pSignalSemaphores = &sem_finish_rendering;
    submit_info.waitSemaphoreCount = 1;
    submit_info.waitSemaphoreCount = 0;
    submit_info.pWaitSemaphores = &sem_img_available;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &cmd_buf;
    VkPipelineStageFlags pipeline_stage_flags = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    submit_info.pWaitDstStageMask = &pipeline_stage_flags;
    vkQueueSubmit(queue, 1, &submit_info, fence_queue_submit);

    vkQueueWaitIdle(queue);
    vkDeviceWaitIdle(device);

    VkPresentInfoKHR prst_info{};
    prst_info.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    prst_info.swapchainCount = 1;
    prst_info.pSwapchains = &swpch;
    prst_info.pImageIndices = &available_img_idx;
    prst_info.waitSemaphoreCount = 1;
    prst_info.pWaitSemaphores = &sem_finish_rendering;
    vkQueuePresentKHR(queue, &prst_info);

    getchar();
    return 0;
}
