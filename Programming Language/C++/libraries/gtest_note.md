## cache

* 详细讲讲 gtest 的用法

    Google Test (通常称为 gtest) 是一个广泛应用于 C++ 项目的单元测试框架，用于编写和运行测试用例，以验证代码的正确性。下面将详细介绍它的核心用法。

    ### 核心概念：断言 (Assertions)

    断言是 gtest 中最基本的单元，用来检查某个条件是否为真。一个测试用例（test case）就是由一系列断言构成的。

    断言主要分为两类，区别在于失败时的影响：

    *   **`EXPECT_*` 系列**：当断言失败时，会生成一个**非致命性**错误。测试函数会继续执行，因此一个测试中可以有多个 `EXPECT_*` 断言失败，并一并报告。**这是最常用的方式**，因为它能提供更全面的失败信息。
    *   **`ASSERT_*` 系列**：当断言失败时，会生成一个**致命性**错误，并**立即中止**当前的测试函数。如果后续代码在没有前置条件的情况下运行没有意义，应使用此系列。

    你还可以使用 `<<` 运算符为断言添加自定义失败信息，这对调试非常有帮助。

    ```cpp
    EXPECT_EQ(x.size(), y.size()) << "Vectors x and y are of unequal length";
    ```

    #### 常用断言宏

    | 分类 | 常用宏 | 作用 |
    | :--- | :--- | :--- |
    | **布尔值检查** | `EXPECT_TRUE(condition)`<br>`ASSERT_TRUE(condition)` | 检查条件是否为 `true`。 |
    | | `EXPECT_FALSE(condition)`<br>`ASSERT_FALSE(condition)` | 检查条件是否为 `false`。 |
    | **数值比较** | `EXPECT_EQ(expected, actual)`<br>`ASSERT_EQ(expected, actual)` | 检查 `expected` 和 `actual` 是否相等。 |
    | | `EXPECT_NE(val1, val2)`<br>`ASSERT_NE(val1, val2)` | 检查 `val1` 和 `val2` 是否不相等。 |
    | | `EXPECT_LT(val1, val2)`<br>`ASSERT_LT(val1, val2)` | 检查 `val1` 是否小于 `val2`。 |
    | | `EXPECT_LE(val1, val2)`<br>`ASSERT_LE(val1, val2)` | 检查 `val1` 是否小于等于 `val2`。 |
    | | `EXPECT_GT(val1, val2)`<br>`ASSERT_GT(val1, val2)` | 检查 `val1` 是否大于 `val2`。 |
    | | `EXPECT_GE(val1, val2)`<br>`ASSERT_GE(val1, val2)` | 检查 `val1` 是否大于等于 `val2`。 |
    | **字符串比较** | `EXPECT_STREQ(str1, str2)`<br>`ASSERT_STREQ(str1, str2)` | 检查两个 C 风格字符串 (`const char*`) 的内容是否相同。 |
    | | `EXPECT_STRNE(str1, str2)`<br>`ASSERT_STRNE(str1, str2)` | 检查两个 C 风格字符串的内容是否不同。 |
    | | `EXPECT_STRCASEEQ(str1, str2)` | 忽略大小写，检查内容是否相同。 |
    | | `EXPECT_STRCASENE(str1, str2)` | 忽略大小写，检查内容是否不同。 |
    | **浮点数比较** | `EXPECT_FLOAT_EQ(expected, actual)` | 检查两个 `float` 值是否近似相等（基于 4 个 ULPs）。 |
    | | `EXPECT_DOUBLE_EQ(expected, actual)` | 检查两个 `double` 值是否近似相等（基于 4 个 ULPs）。 |
    | | `EXPECT_NEAR(val1, val2, abs_error)` | 检查 `val1` 和 `val2` 之间的差值是否不超过用户指定的绝对误差 `abs_error`。 |
    | **异常检查** | `EXPECT_THROW(statement, exception_type)` | 检查 `statement` 是否抛出了指定类型的异常。 |
    | | `EXPECT_ANY_THROW(statement)` | 检查 `statement` 是否抛出了任何类型的异常。 |
    | | `EXPECT_NO_THROW(statement)` | 检查 `statement` 是否没有抛出任何异常。 |

    ### 编写测试：`TEST` 和 `TEST_F`

    #### 1. 基础测试：`TEST` 宏

    对于大多数独立的、不需要复杂初始化的测试，使用 `TEST` 宏即可。

    ```cpp
    #include <gtest/gtest.h>

    // 被测试函数
    int add(int a, int b) {
        return a + b;
    }

    // 测试套件名称: AddTest, 测试名称: HandlesPositiveInput
    TEST(AddTest, HandlesPositiveInput) {
        EXPECT_EQ(3, add(1, 2));
        EXPECT_EQ(5, add(2, 3));
    }

    TEST(AddTest, HandlesZeroAndNegative) {
        EXPECT_EQ(1, add(1, 0));
        EXPECT_EQ(-1, add(-2, 1));
    }
    ```
    在这个例子中，`AddTest` 是**测试套件（test suite）** 的名称，`HandlesPositiveInput` 是具体**测试（test）** 的名称。它们共同构成了测试的全名。

    #### 2. 测试夹具：`TEST_F` 宏

    当多个测试需要共享同一组数据或相同的初始化/清理代码时，可以使用**测试夹具（Test Fixture）**。这需要创建一个继承自 `testing::Test` 的类，并重写 `SetUp()` 和 `TearDown()` 方法。

    ```cpp
    #include <gtest/gtest.h>

    class MyStringTest : public ::testing::Test {
    protected:
        // 在每次测试执行前运行
        void SetUp() override {
            str = new std::string("Hello, World!");
        }

        // 在每次测试执行后运行
        void TearDown() override {
            delete str;
            str = nullptr;
        }

        std::string* str;
    };

    // 使用 TEST_F 宏，第一个参数必须是测试夹具的类名
    TEST_F(MyStringTest, LengthIsCorrect) {
        EXPECT_EQ(13, str->length());
    }

    TEST_F(MyStringTest, ContainsHello) {
        EXPECT_NE(std::string::npos, str->find("Hello"));
    }
    ```
    `SetUp()` 和 `TearDown()` 方法会在**每个** `TEST_F` 测试用例执行前后被调用，确保了测试之间的独立性。

    ### 高级特性

    #### 1. 参数化测试：`TEST_P`

    当你需要用多组不同的数据来测试同一段代码逻辑时，参数化测试可以避免重复编写相似的测试代码。你需要创建一个继承自 `testing::TestWithParam<T>` 的类，并用 `TEST_P` 定义测试。

    ```cpp
    #include <gtest/gtest.h>

    class CalculatorTest : public ::testing::TestWithParam<std::tuple<int, int, int>> {
    };

    TEST_P(CalculatorTest, HandlesAddition) {
        int a = std::get<0>(GetParam());
        int b = std::get<1>(GetParam());
        int expected_sum = std::get<2>(GetParam());
        Calculator calc;
        EXPECT_EQ(expected_sum, calc.add(a, b));
    }

    // 使用 INSTANTIATE_TEST_SUITE_P 来实例化参数集
    INSTANTIATE_TEST_SUITE_P(
        AdditionTests,
        CalculatorTest,
        testing::Values(
            std::make_tuple(1, 2, 3),
            std::make_tuple(-1, 1, 0),
            std::make_tuple(0, 0, 0)
        )
    );
    ```
    `INSTANTIATE_TEST_SUITE_P` 宏会为 `testing::Values` 中的每一组参数生成一个独立的测试用例。

    #### 2. 死亡测试：`EXPECT_DEATH`

    **死亡测试（Death Test）** 用于验证程序在特定情况下是否按照预期崩溃或退出。这通常用来检查断言 (`assert`) 或异常处理代码。

    ```cpp
    void Foo() {
        int *pInt = nullptr;
        *pInt = 42 ; // 这将导致程序崩溃
    }

    TEST(FooDeathTest, DiesOnNullPointerDereference) {
        // 期望 Foo() 函数会使程序崩溃
        EXPECT_DEATH(Foo(), "");
    }
    ```
    **重要提示**：编写死亡测试时，通常建议将测试套件（`TEST` 的第一个参数）命名为以 `DeathTest` 结尾（如 `FooDeathTest`），这样 gtest 会优先运行它们，以确保线程安全。

    #### 3. 运行控制

    *   **初始化**：在 `main` 函数中，通过 `testing::InitGoogleTest(&argc, argv);` 初始化 gtest，然后调用 `RUN_ALL_TESTS()` 运行所有测试。
    *   **重复测试**：可以使用 `--gtest_repeat` 命令行标志或设置 `GTEST_REPEAT` 环境变量来重复运行测试，这对于发现不稳定（flaky）的测试很有用。

    ### 总结

    | 特性 | 关键点 |
    | :--- | :--- |
    | **断言** | `EXPECT_*` (非致命, 推荐) 和 `ASSERT_*` (致命)。支持 `<<` 输出自定义信息。 |
    | **基本测试** | 使用 `TEST(TestSuiteName, TestName)` 宏。 |
    | **测试夹具** | 使用 `TEST_F` 宏，并创建一个继承 `testing::Test` 的类，重写 `SetUp`/`TearDown`。 |
    | **参数化测试** | 使用 `TEST_P` 和 `INSTANTIATE_TEST_SUITE_P`，继承 `testing::TestWithParam<T>`。 |
    | **死亡测试** | 使用 `EXPECT_DEATH` 等宏，测试套件名以 `DeathTest` 为后缀。 |
    | **运行** | `RUN_ALL_TESTS()` 执行所有测试，通过命令行参数或环境变量控制执行行为。 |

    掌握了这些核心用法，你就可以开始为自己的 C++ 代码编写健壮的单元测试了。如果希望了解更细节的用法，可以查阅 gtest 的官方文档。

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

