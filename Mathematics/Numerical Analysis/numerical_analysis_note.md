# Numerical Analysis

## 多项式的值

计算：$p(x) = a_n x^n + a_{n-1} x^{n-1} + \cdots + a_1 x + a_0$，$a_n \neq 0$

1. 秦九韶算法

    秦九韶算法可以通过一次式的反复计算，逐步得出高次多项式的值。

    首先从前两式提取$x^{n-1}$，这样就可以得到$p(x) = (a_n x + a_{n-1}) x^{n-1} + a_{n-2} x^{n-2} + \cdots + a_1 x + a_0$，从而消去一项，按这种方法算下去，即可得到最后的结果。

    matlab 代码：

    ```matlab
    function s = get_ans(x, nums)
        %{
            x: value
            nums: (1, n + 1), 代表 [a_n, a_{n-1}, ..., a_1, a_0]
        %}

        s = 0;
        for i = 1 : length(nums)
            s = s * x + nums(i);
        end
    end
    ```

