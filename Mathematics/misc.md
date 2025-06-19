* 求三个正整数$a$, $b$, $c$，使得$\displaystyle\frac{a}{b + c} + \frac{b}{a + c} + \frac{c}{a + b} = 4$

    写了份代码，找了两组比较接近 4 的，但是找不到正好等于 4 的。

    ```cpp
    #include <stdio.h>
    #include <math.h>
    #include <unistd.h>

    // 1 / 18 + 3 / 16 + 15 / 4

    // 7, 14, 79, 3.999964277

    int main() {
        double a = 1, b = 1, c = 1;
        double N = 9999999999;
        double s = 0;
        double epsilon = 1e-4;
        double target = 4;
        size_t cnt = 0;
        for (a = 1; N - a >= epsilon; a += 1) {
            for (b = 1; N - b >= epsilon; b += 1) {
                for (c = 1; N - c >= epsilon; c += 1) {
                    ++cnt;
                    s = a / (b + c) + b / (a + c) + c / (a + b);
                    if (abs(s - target) < epsilon) {
                        printf("a: %.6f, b: %.6f, c: %.6f, s: %.6f\n",
                            a, b, c, s);
                        getchar();
                    }
                    printf("a: %.6f, b: %.6f, c: %.6f, s: %.6f\n",
                        a, b, c, s);
                    // usleep(1000 * 10);
                    if (s > target + 0.5)
                        break;
                }
                if (abs(c - 1) < epsilon && s > target + 0.5)
                    break;
            }
            if (abs(b - 1) < epsilon && s > target + 0.5)
                break;
        }
        printf("cnt: %lu\n", cnt);
        return 0;
    }
    ```