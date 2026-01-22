# AXI Note

## cache

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
