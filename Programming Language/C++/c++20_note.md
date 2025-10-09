# C++20 Note

因为目前 gcc 默认支持的版本仍是 c++ 17，所以对 c++ 20 的东西保持观望，暂时记录在这里。

## cache

* C++20，可以使用概念来约束构造函数

    ```cpp
    #include <concepts>
    #include <iostream>

    template <typename T>
    class Rational {
    public:
        // 只允许算术类型的构造函数
        Rational(T num, T denom) requires std::integral<T> || std::floating_point<T> 
            : numerator(num), denominator(denom) {
            std::cout << "Constructing Rational with arithmetic type\n";
        }
        
    private:
        T numerator;
        T denominator;
    };

    int main() {
        Rational<int> a(1, 2);      // OK
        Rational<double> b(1.0, 2.0); // OK
        // Rational<std::string> c("a", "b"); // 编译错误
    }
    ```

    output:

    ```
    Constructing Rational with arithmetic type
    Constructing Rational with arithmetic type
    ```

    也可以写成：

    ```cpp
    #include <concepts>
    #include <iostream>

    class Rational {
    public:
        // 只允许算术类型的构造函数
        template <std::integral T>
        Rational(T numerator, T denominator) {
            std::cout << "Integral Rational\n";
        }

        template <std::floating_point T>
        Rational(T numerator, T denominator) {
            std::cout << "Floating-point Rational\n";
        }
    };

    int main() {
        Rational a(1, 2);       // 调用整数版本
        Rational b(1.0, 2.0);   // 调用浮点版本
        // Rational c("1", "2"); // 编译错误
    }
    ```

## topics