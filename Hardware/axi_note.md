# AXI Note

## cache

* 详细讲讲 axi burst

    ## AXI Burst 详解

    AXI Burst（突发传输）是 AXI 协议中最核心的传输机制之一，它允许一次地址握手后连续传输多个数据节拍，大幅提升总线效率。

    ---

    ## 一、基本概念

    **Burst = 一次地址相位 + 多次数据相位**

    在 AXI 中，主机发起一次传输时，只需在地址通道（AW/AR）上发送**一个起始地址**，然后在数据通道（W/R）上连续传输多个数据。这就是 Burst 传输。

    ```
    地址通道:  ──[起始地址 + 控制信息]──►
    数据通道:  ──[数据0][数据1][数据2]...[数据N]──►
    ```

    ---

    ## 二、Burst 的三个关键参数

    AXI 协议通过三个信号描述一次 Burst：

    | 信号 | 含义 | 取值范围 |
    |------|------|----------|
    | **AxLEN** | 突发长度（beat 数量） | 1~256（AXI3 为 1~16） |
    | **AxSIZE** | 每拍数据字节数 | 1,2,4,8,16,32,64,128 字节 |
    | **AxBURST** | 突发类型 | FIXED / INCR / WRAP |

    ### 1. AxLEN（长度）
    - 实际传输拍数 = **AxLEN + 1**
    - AXI3：1~16 拍
    - AXI4：1~256 拍
    - **WRAP 类型只允许 2、4、8、16 拍**

    ### 2. AxSIZE（大小）
    - 表示每拍传输的字节数，编码为 `log2(字节数)`
    - 例如 `AxSIZE=3'b010` → 4 字节/拍
    - **不能超过数据总线宽度**（如 64 位总线最大 8 字节/拍）
    - 传输必须地址对齐：起始地址必须是 AxSIZE 的整数倍

    ### 3. AxBURST（类型）

    #### ① FIXED（固定地址）
    - 每个 beat 都访问**同一地址**
    - 常用于 FIFO、外设寄存器反复读写
    - 地址不递增

    ```
    地址序列: A, A, A, A, ...
    ```

    #### ② INCR（递增）
    - 地址逐拍递增，最常用
    - 递增步长 = 每拍字节数（AxSIZE）
    - 用于普通内存访问、DMA

    ```
    起始=0x1000, size=4B, len=4
    地址: 0x1000, 0x1004, 0x1008, 0x100C, 0x1010
    ```

    #### ③ WRAP（回卷）
    - 地址递增到边界后**回卷到起始边界**
    - 长度只能是 2/4/8/16
    - 常用于 Cache Line 填充

    ```
    起始=0x1004, size=4B, len=4（wrap边界=0x1000~0x100F）
    地址: 0x1004, 0x1008, 0x100C, 0x1000
    ```

    **回卷地址计算**：  
    `lower_bound = (起始地址 / 总字节数) × 总字节数`  
    总字节数 = (AxLEN+1) × 每拍字节数

    ---

    ## 三、传输时序示例

    以 **INCR、4 拍、每拍 4 字节** 为例：

    ```
    ACLK      ─┐_┌─┐_┌─┐_┌─┐_┌─┐_┌─
    AWVALID   ───┘‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾
    AWREADY   ────────┐‾‾‾‾‾‾‾‾‾‾‾
    AWADDR    ────────< 0x1000 >───
    AWLEN     ────────<   3    >───

    WVALID    ──────────┐_┌─┐_┌─┐_┌─┐_┌─
    WREADY    ──────────┐‾‾‾‾‾‾‾‾‾‾‾‾‾‾
    WDATA     ──────────<D0><D1><D2><D3>
    WLAST     ──────────____________┌──
    ```

    关键点：
    - 地址相位只发一次
    - 数据相位连续 4 拍
    - 最后一拍 **WLAST/RLAST 拉高**标志结束

    ---

    ## 四、Burst 与总线宽度、对齐的关系

    ### 1. 窄传输（Narrow Transfer）
    当 `AxSIZE < 数据总线宽度` 时，称为窄传输：
    - 数据出现在总线的特定字节通道（byte lane）上
    - 由地址低位决定使用哪些 lane

    例如 64 位总线传 1 字节：
    ```
    WDATA[63:0]
    beat0 地址 0x0 → 数据在 [7:0]
    beat1 地址 0x1 → 数据在 [15:8]
    ...
    ```

    ### 2. 非对齐传输（Unaligned）
    - AXI 要求**起始地址对齐到 AxSIZE**
    - 第一个 beat 可以非对齐，但会占用多个 lane，后续 beat 自动对齐

    ---

    ## 五、Burst 的边界规则

    ### 4KB 边界限制
    **任何 Burst 都不能跨越 4KB 地址边界**。

    原因：
    - 4KB 通常是页大小，跨越会导致两次 TLB 查找
    - 简化从设备设计

    如果传输数据量超过 4KB，主设备必须**拆分成多个 Burst**。

    ```
    错误: 起始 0x1F00, 长度 0x200 → 跨越 0x2000 边界
    正确: 拆成 0x1F00~0x1FFF 和 0x2000~0x20FF 两次
    ```

    ---

    ## 六、读写通道的 Burst 差异

    | 通道 | 地址信号 | 数据信号 | 最后标志 |
    |------|----------|----------|----------|
    | 读 | ARADDR/ARLEN/ARSIZE/ARBURST | RDATA/RLAST | RLAST |
    | 写 | AWADDR/AWLEN/AWSIZE/AWBURST | WDATA/WLAST | WLAST |

    **写通道特有**：
    - 写数据可以**先于地址**到达（WVALID 可早于 AWVALID）
    - 写响应通道 B 用于确认写完成

    ---

    ## 七、Burst 的实际应用场景

    | 场景 | 推荐 Burst 类型 | 说明 |
    |------|----------------|------|
    | DMA 大块搬运 | INCR | 长突发效率高 |
    | Cache Line 填充 | WRAP | 保证行内对齐 |
    | FIFO 读写 | FIXED | 地址不变 |
    | 寄存器访问 | INCR（len=0） | 单拍 |
    | 视频流 | INCR | 连续帧缓冲 |

    ---

    ## 八、性能考量

    1. **长 Burst 效率高**：地址相位开销被均摊
    2. **但过长会阻塞总线**：影响其他主设备延迟
    3. **OUTSTANDING 能力**：主设备可发多个未完成 Burst 提升吞吐
    4. **4KB 拆分**：注意拆分开销

    ---

    ## 九、常见误区

    - ❌ AxLEN 是传输拍数 → ✅ 实际拍数 = AxLEN + 1
    - ❌ 任意长度都能 WRAP → ✅ WRAP 只支持 2/4/8/16
    - ❌ 可以跨 4KB → ✅ 必须拆分
    - ❌ 起始地址可任意 → ✅ 必须对齐到 AxSIZE

    ---

    如果你需要，我可以进一步展开：
    - **WRAP 地址计算的完整公式与例子**
    - **窄传输/非对齐传输的 byte lane 映射**
    - **AXI Burst 的 Verilog 实现或波形分析**
    - **AXI3 与 AXI4 在 Burst 上的差异**

    需要哪部分深入？

* 简述 AXI 协议作用、内容及用法

    AXI（Advanced eXtensible Interface）协议是ARM公司推出的高性能、高频率、高带宽的片上总线协议，广泛应用于SoC（片上系统）设计中，用于连接处理器、内存控制器、外设等IP核。

    一、作用

    * 高性能通信：支持高频率操作，提供高带宽数据传输

    * 设计复用：标准化接口便于IP核的集成和复用

    * 并行传输：支持多个未完成事务的并行处理

    * 系统可扩展：适用于从简单到复杂的各种系统

    二、核心内容

    通道架构（5个独立通道）

    * 读地址通道（AR）：传输读操作地址信息

    * 读数据通道（R）：返回读取的数据

    * 写地址通道（AW）：传输写操作地址信息

    * 写数据通道（W）：传输待写入的数据

    * 写响应通道（B）：返回写操作完成状态

    关键特性

    * 分离的地址/数据通道：地址和数据传输解耦，提高效率

    * 基于突发传输：支持1-16个数据的突发传输

    * 支持非对齐传输：通过字节选通信号实现

    * 乱序完成：支持事务的乱序完成（使用ID标识）

    * 原子操作：支持独占和锁定访问

    三种协议变体

    * AXI4：标准版本，支持突发长度最多256

    * AXI4-Lite：简化版，用于简单寄存器访问（每次1个数据）

    * AXI4-Stream：无地址流数据协议，用于高速数据流

    三、基本用法

    典型连接方式

    ```verilog
    // 主设备（Master）连接示例
    axi_master u_master (
        // 全局信号
        .ACLK    (clk),
        .ARESETn (reset_n),
        
        // 写地址通道
        .AWADDR  (awaddr),
        .AWVALID (awvalid),
        .AWREADY (awready),
        
        // 写数据通道  
        .WDATA   (wdata),
        .WVALID  (wvalid),
        .WREADY  (wready),
        
        // 写响应通道
        .BRESP   (bresp),
        .BVALID  (bvalid),
        .BREADY  (bready),
        
        // 读地址通道
        .ARADDR  (araddr),
        .ARVALID (arvalid),
        .ARREADY (arready),
        
        // 读数据通道
        .RDATA   (rdata),
        .RRESP   (rresp),
        .RVALID  (rvalid),
        .RREADY  (rready)
    );
    ```

    传输时序示例

    ```text
    // 写传输流程：
    1. 主设备在AW通道发送地址（AWVALID=1）
    2. 从设备准备好后置AWREADY=1
    3. 主设备在W通道发送数据（WVALID=1）
    4. 从设备接收数据后置WREADY=1
    5. 从设备在B通道返回响应（BVALID=1）
    6. 主设备接收响应后置BREADY=1

    // 读传输流程：
    1. 主设备在AR通道发送地址（ARVALID=1）
    2. 从设备准备好后置ARREADY=1
    3. 从设备在R通道返回数据（RVALID=1）
    4. 主设备接收数据后置RREADY=1
    ```

    设计考虑

    * 互连选择：点对点、共享总线、交叉开关或NoC

    * 性能优化：通过增加outstanding transactions提高吞吐量

    * 功耗管理：使用时钟门控和低功耗状态

    * 验证策略：使用VIP（验证IP）进行协议检查

    四、应用场景

    * 处理器与DDR控制器：高速内存访问

    * DMA控制器与外设：大数据传输

    * 硬件加速器互联：如GPU、DSP、AI加速器

    * 芯片间互联：通过SerDes扩展

    AXI协议已成为业界事实标准，在Xilinx FPGA、各种SoC设计中广泛使用，其分层架构和灵活特性使其能够适应从嵌入式系统到高性能计算的各种应用需求。

* 我是写固件的，如何使用 axi 协议？

    作为固件工程师，你使用AXI协议主要是通过CPU读写AXI外设寄存器，而不是直接实现AXI硬件接口。以下是实际工作中的使用方法：

    一、固件层面的AXI访问方式

    1. 映射为内存地址

        AXI外设的寄存器被映射到CPU的地址空间：

        ```c
        // 在头文件中定义寄存器映射
        #define PERIPH_BASE    0x40000000U
        #define UART0_BASE     (PERIPH_BASE + 0x0000)
        #define GPIO_BASE      (PERIPH_BASE + 0x1000)

        // 寄存器定义
        typedef struct {
            volatile uint32_t DATA;      // 偏移0x00 - 数据寄存器
            volatile uint32_t STATUS;    // 偏移0x04 - 状态寄存器
            volatile uint32_t CTRL;      // 偏移0x08 - 控制寄存器
            volatile uint32_t BAUD;      // 偏移0x0C - 波特率寄存器
        } UART_TypeDef;

        #define UART0          ((UART_TypeDef *)UART0_BASE)
        ```

    2. 基本读写操作

        ```c
        // 读取状态寄存器
        uint32_t status = UART0->STATUS;

        // 检查特定位（如发送缓冲区空）
        if (status & (1 << 3)) {
            // 可以发送新数据
        }

        // 写入控制寄存器（设置位）
        UART0->CTRL |= (1 << 0);  // 使能UART

        // 清除位
        UART0->CTRL &= ~(1 << 1); // 禁用中断

        // 写入数据
        UART0->DATA = 'A';
        ```

    二、实际固件开发场景

    1. 初始化外设

        ```c
        void uart_init(uint32_t baud_rate) {
            // 1. 禁用UART（写入CTRL寄存器）
            UART0->CTRL = 0x00000000;
            
            // 2. 设置波特率（写入BAUD寄存器）
            UART0->BAUD = SYSTEM_CLOCK / (16 * baud_rate);
            
            // 3. 配置数据格式（8N1）
            UART0->CTRL |= (3 << 0);  // 8位数据
            
            // 4. 使能UART
            UART0->CTRL |= (1 << 4);  // 使能发送
            UART0->CTRL |= (1 << 5);  // 使能接收
        }
        ```

    2. 轮询方式访问

        ```c
        void uart_send_byte(uint8_t data) {
            // 等待发送缓冲区空（轮询状态寄存器）
            while (!(UART0->STATUS & (1 << 3))) {
                // 空循环等待
            }
            
            // 写入数据寄存器
            UART0->DATA = data;
        }

        uint8_t uart_receive_byte(void) {
            // 检查接收缓冲区非空
            while (!(UART0->STATUS & (1 << 0))) {
                // 等待数据
            }
            
            // 读取数据寄存器
            return (uint8_t)(UART0->DATA & 0xFF);
        }
        ```

    3. 中断方式访问

        ```c
        // 中断服务程序
        void UART0_IRQHandler(void) {
            uint32_t status = UART0->STATUS;
            
            // 接收中断
            if (status & (1 << 0)) {
                uint8_t data = UART0->DATA & 0xFF;
                rx_buffer[rx_index++] = data;
            }
            
            // 发送中断
            if (status & (1 << 1)) {
                if (tx_index < tx_length) {
                    UART0->DATA = tx_buffer[tx_index++];
                } else {
                    // 禁用发送中断
                    UART0->CTRL &= ~(1 << 1);
                }
            }
        }
        ```

    三、固件工程师需要关注的重点

    1. 地址对齐

        ```c
        // 正确：32位对齐访问
        uint32_t value = *(volatile uint32_t *)0x40000000;

        // 错误：非对齐访问（可能导致异常或低效）
        uint16_t value = *(volatile uint16_t *)0x40000001;  // 地址不是2字节对齐
        ```

    2. 访问宽度

        ```c
        // AXI支持不同宽度的访问
        uint8_t  byte_data  = *(volatile uint8_t  *)addr;  // 8位访问
        uint16_t half_word  = *(volatile uint16_t *)addr;  // 16位访问  
        uint32_t word_data  = *(volatile uint32_t *)addr;  // 32位访问（最常用）
        uint64_t double_word = *(volatile uint64_t *)addr; // 64位访问
        ```

    3. 内存屏障和排序

        ```c
        // 需要确保访问顺序的场景
        void configure_device(void) {
            // 步骤1：设置参数寄存器
            DEVICE->PARAM = 0x12345678;
            
            // 内存屏障：确保上面的写入在下面的写入之前完成
            __DSB();  // 数据同步屏障
            
            // 步骤2：启动设备
            DEVICE->CTRL |= START_BIT;
            
            // 等待完成
            __DMB();  // 数据内存屏障
        }
        ```

    4. DMA操作（涉及AXI传输）

        ```c
        void start_dma_transfer(void *src, void *dst, uint32_t size) {
            // 1. 配置源地址（CPU写入DMA控制器寄存器）
            DMA->SAR = (uint32_t)src;
            
            // 2. 配置目标地址
            DMA->DAR = (uint32_t)dst;
            
            // 3. 配置传输长度
            DMA->LENGTH = size;
            
            // 4. 内存屏障：确保配置完成
            __DSB();
            
            // 5. 启动DMA（DMA控制器通过AXI总线进行数据传输）
            DMA->CTRL = DMA_ENABLE;
            
            // 6. 等待传输完成（轮询或中断）
            while (!(DMA->STATUS & DMA_DONE)) {
                // 等待
            }
        }
        ```

    四、调试和排查问题

    1. 常见问题排查

        ```c
        // 检查外设是否响应
        bool check_peripheral_present(uint32_t base_addr) {
            volatile uint32_t *id_reg = (uint32_t *)(base_addr + 0xFC);
            uint32_t id_value = *id_reg;
            
            if (id_value == 0xFFFFFFFF || id_value == 0x00000000) {
                // 可能：1) 地址错误 2) 时钟未使能 3) 复位未释放
                return false;
            }
            return true;
        }

        // 调试函数：打印寄存器值
        void debug_print_registers(uint32_t base_addr, const char *name) {
            printf("%s registers:\n", name);
            for (int i = 0; i < 16; i++) {
                uint32_t value = *(volatile uint32_t *)(base_addr + i * 4);
                printf("  0x%08X: 0x%08X\n", base_addr + i * 4, value);
            }
        }
        ```

    2. 访问错误处理

        ```c
        // 安全访问函数
        uint32_t safe_read_register(uint32_t addr) {
            volatile uint32_t *ptr = (uint32_t *)addr;
            
            // 检查地址是否在有效范围内
            if ((addr < PERIPH_BASE) || (addr > (PERIPH_BASE + 0x100000))) {
                return 0xDEADBEEF;  // 标记错误值
            }
            
            return *ptr;
        }

        void safe_write_register(uint32_t addr, uint32_t value) {
            volatile uint32_t *ptr = (uint32_t *)addr;
            
            // 对齐检查
            if (addr & 0x3) {
                printf("Error: Unaligned access to 0x%08X\n", addr);
                return;
            }
            
            *ptr = value;
        }
        ```

    五、实用工具和技巧

    1. 使用CMSIS头文件

        ```c
        #include "core_cm7.h"      // Cortex-M7内核寄存器
        #include "stm32h7xx.h"     // STM32H7外设定义

        // CMSIS提供了标准化的访问方式
        DMA1_Stream0->CR |= DMA_SxCR_EN;  // 使能DMA流

        // 系统控制（影响AXI总线）
        SCB_EnableICache();  // 启用指令缓存
        SCB_EnableDCache();  // 启用数据缓存
        ```

    2. 性能优化

        ```c
        // 批量读写（利用AXI突发传输）
        void copy_data_fast(uint32_t *dst, uint32_t *src, uint32_t words) {
            // 编译器可能优化为LDM/STM指令，产生AXI突发传输
            for (uint32_t i = 0; i < words; i++) {
                dst[i] = src[i];
            }
        }

        // 使用缓存操作
        SCB_CleanDCache_by_Addr(addr, size);    // 清理数据缓存
        SCB_InvalidateDCache_by_Addr(addr, size); // 无效化数据缓存
        ```

    总结：固件工程师的核心任务

    * 正确配置：通过读写寄存器初始化外设

    * 高效访问：使用合适的访问宽度和顺序

    * 异常处理：处理总线错误、超时等异常

    * 性能优化：利用缓存和批量传输

    * 调试排错：使用调试工具分析AXI事务

    记住：作为固件工程师，你主要与寄存器抽象打交道，而不是直接处理AXI信号线。硬件团队已经将AXI接口封装成你看到的寄存器。你需要关心的是正确的地址、时序和访问顺序。

## topics
