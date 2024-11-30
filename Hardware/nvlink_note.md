# NVLink Note

## cache

* 使用 nvml 判断任意两个 cuda device 是否可以通过 nvlink 连接

    `main.cu`:

    ```cpp
    #include <nvml.h>
    #include <iostream>

    int main() 
    {
        nvmlInit();

        unsigned int dev_id = 0;
        nvmlDevice_t device;
        nvmlDeviceGetHandleByIndex(dev_id, &device);
        
        nvmlEnableState_t isEnabled;
        for (unsigned int link_id = 0; link_id < 15; ++link_id)
        {
            isEnabled = NVML_FEATURE_DISABLED;
            nvmlDeviceGetNvLinkState(device, link_id, &isEnabled);

            if (isEnabled == NVML_FEATURE_ENABLED)
            {
                printf("dev id %d, link id: %d, enabled\n", dev_id, link_id);
            }
            else
            {
                printf("dev id %d, link id: %d, not enabled\n", dev_id, link_id);
            }
        }

        nvmlShutdown();
        return 0;
    }
    ```

    compile: `nvcc -g main.cu -lnvidia-ml -o main`

    run: `./main`

    output:

    ```
    dev id 0, link id: 0, enabled
    dev id 0, link id: 1, enabled
    dev id 0, link id: 2, enabled
    dev id 0, link id: 3, enabled
    dev id 0, link id: 4, enabled
    dev id 0, link id: 5, enabled
    dev id 0, link id: 6, enabled
    dev id 0, link id: 7, enabled
    dev id 0, link id: 8, enabled
    dev id 0, link id: 9, enabled
    dev id 0, link id: 10, enabled
    dev id 0, link id: 11, enabled
    dev id 0, link id: 12, not enabled
    dev id 0, link id: 13, not enabled
    dev id 0, link id: 14, not enabled
    ```

    将 dev id 设置为 0 到 7，此时 link id 从 0 到 11 都是 enable 的。其他的配置都是 not enable。

    看起来这个只能判断 dev 是否连接到 nvlink 上。

    不清楚 nvlink 一共 12 根，每个 switch 上接 2 根 nvlink，还是每个 dev 都接 12 根 nvlink。

    猜想：判断 2 个 device 是否能通过 nvlink 互连的方式为：检查两个 dev 是否 enable 了同一个 nvlink id。如果这个猜想成立，那么很有可能 nvlink 一共 12 根，如果不是全局的 id，那么只靠 local id 无法判断 2 个 device 是否互联。

    还有一种可能性，每个 dev 都接 12 个 nvlink 到 switch 上，编号即为 0 到 11. 当前 dev 的 0 号 nvlink 只和 peer dev 的 0 号 nvlink 通信。1 号只和 1 号通信，以此类推。如果一个 dev 启用了 0 号 nvlink，另一个 dev 启用了 1 号 nvlink，这两个 nvlink 都接入到 nvswitch 上，那么这两个 dev 照样无法通信。

    注：

    1. 如果写`isEnabled = NVML_FEATURE_ENABLED;`，当实际情况是 not enabled 时，`isEnabled`不会被修改成`NVML_FEATURE_DISABLED`。

        example:

        `main.cu`:

        ```cpp
        #include <nvml.h>
        #include <iostream>

        int main() 
        {
            nvmlInit();

            unsigned int dev_id = 0;
            nvmlDevice_t device;
            nvmlDeviceGetHandleByIndex(dev_id, &device);
            
            nvmlEnableState_t isEnabled;
            for (unsigned int link_id = 0; link_id < 15; ++link_id)
            {
                // isEnabled = NVML_FEATURE_DISABLED;
                nvmlDeviceGetNvLinkState(device, link_id, &isEnabled);

                if (isEnabled == NVML_FEATURE_ENABLED)
                {
                    printf("dev id %d, link id: %d, enabled\n", dev_id, link_id);
                }
                else
                {
                    printf("dev id %d, link id: %d, not enabled\n", dev_id, link_id);
                }
            }

            nvmlShutdown();
            return 0;
        }
        ```

        output:

        ```
        dev id 0, link id: 0, enabled
        dev id 0, link id: 1, enabled
        dev id 0, link id: 2, enabled
        dev id 0, link id: 3, enabled
        dev id 0, link id: 4, enabled
        dev id 0, link id: 5, enabled
        dev id 0, link id: 6, enabled
        dev id 0, link id: 7, enabled
        dev id 0, link id: 8, enabled
        dev id 0, link id: 9, enabled
        dev id 0, link id: 10, enabled
        dev id 0, link id: 11, enabled
        dev id 0, link id: 12, enabled
        dev id 0, link id: 13, enabled
        dev id 0, link id: 14, enabled
        ```

* 224 机器上，8 个 A100，6 个 nvidia bridge，任意 2 个 A100 之间都通过 12 根 nvlink 相连，简称 NV12。每根 nvlink 提供 25 GB/s 的单向带宽，12 根一共是 12 * 25 = 300 GB/s。

    实测任意两个 a100 之间的单向通信速率差不多是 275 GB/s。

    nvlink 支持全双工，实测任意两个 a100 之间的双向带宽是 515 GB/s 左右。

## note