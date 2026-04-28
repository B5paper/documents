## cache

* 简述 gtest 的用法

    **GTest（Google Test）** 是C++的跨平台单元测试框架，核心用法可概括为**定义测试用例、编写断言、运行测试**三步，以下是极简核心流程：

    1.  **环境配置**
        - 编译GTest生成静态库（`libgtest.a`/`gtest.lib`），或直接引入源码到项目。
        - 测试代码中包含头文件：`#include <gtest/gtest.h>`。

    2.  **编写测试用例**
        - 用`TEST(TestSuiteName, TestCaseName)`宏定义测试用例，第一个参数为测试套件名，第二个为测试用例名。
        - 用**断言宏**验证逻辑，分为两类：
          - `ASSERT_*`：断言失败时立即终止当前测试用例（如`ASSERT_EQ(a, b)`验证相等）。
          - `EXPECT_*`：断言失败时继续执行，仅标记失败（如`EXPECT_NE(a, b)`验证不相等）。
        - 示例：
          ```cpp
          int Add(int a, int b) { return a + b; }
          TEST(AddTest, PositiveNumbers) {
              EXPECT_EQ(Add(2, 3), 5);
              ASSERT_EQ(Add(0, 0), 0);
          }
          ```

    3.  **运行测试**
        - 在`main`函数中调用`testing::InitGoogleTest(&argc, argv)`初始化框架，再执行`RUN_ALL_TESTS()`运行所有测试用例。
        - 示例：
          ```cpp
          int main(int argc, char **argv) {
              testing::InitGoogleTest(&argc, argv);
              return RUN_ALL_TESTS();
          }
          ```

    4.  **进阶特性（核心摘要）**
        - **测试夹具（Test Fixture）**：用`TEST_F(FixtureName, TestCaseName)`实现多个测试用例共享初始化/清理逻辑，需继承`testing::Test`。
        - **参数化测试**：通过`INSTANTIATE_TEST_SUITE_P`实现同一逻辑多组输入测试。
        - **死亡测试**：用`ASSERT_DEATH(statement, regex)`验证代码是否按预期崩溃。

    编译时需链接GTest库，运行可执行文件即可看到测试结果报告（包含成功/失败用例数、失败原因）。

* GTest和其他单元测试框架有什么区别？

    GTest 与其他 C++ 单元测试框架的核心区别在于**生态成熟度、功能丰富度、跨平台性**，以及对**复杂测试场景的原生支持**，以下是与主流框架的关键差异对比：

    1.  **与 Catch2 的区别**
        - **易用性**：Catch2 支持无 main 函数自动注册测试，语法更简洁（如 `TEST_CASE` 宏），上手成本更低；GTest 需手动编写 main 函数或使用 `gtest_main` 库。
        - **功能侧重**：Catch2 内置断言、BDD 风格测试、数据生成器，轻量且一体化；GTest 功能更细分，需配合 GMock 实现模拟测试，生态更庞大。
        - **编译速度**：Catch2 头文件仅需包含一次，编译速度快于 GTest（需链接独立库）。

    2.  **与 Boost.Test 的区别**
        - **依赖**：Boost.Test 是 Boost 库的一部分，需引入整个 Boost 生态（或单独编译）；GTest 是独立框架，无额外依赖。
        - **功能**：Boost.Test 支持更多高级特性（如内存泄漏检测、多线程测试），但配置复杂；GTest 功能更聚焦单元测试，API 设计更直观。
        - **跨平台**：两者均跨平台，但 GTest 在嵌入式、移动端的适配案例更丰富。

    3.  **与 CppUnit 的区别**
        - **设计模式**：CppUnit 基于 JUnit 风格，需继承 `TestFixture` 类并注册测试，语法繁琐；GTest 用宏定义测试用例，无需继承，更简洁。
        - **功能**：CppUnit 功能基础（仅支持核心断言和测试套件），无参数化测试、死亡测试等高级特性；GTest 原生支持复杂场景。
        - **维护状态**：CppUnit 维护不活跃，GTest 由 Google 持续更新，社区活跃。

    4.  **核心共性与差异总结**
        - **共性**：均支持断言、测试套件、测试结果报告。
        - **GTest 优势**：生态成熟（与 GMock 无缝集成）、跨平台性强、文档完善、企业级项目使用广泛。
        - **GTest 劣势**：语法相对繁琐、编译速度较慢、需单独处理模拟测试（依赖 GMock）。

    选择框架的核心依据：**小型项目优先 Catch2（轻量易用），大型项目或需模拟测试优先 GTest（生态完善），依赖 Boost 则选 Boost.Test**。

