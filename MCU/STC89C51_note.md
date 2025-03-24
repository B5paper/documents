# STC 89C51 Note

## cache

* 单片机单键检测

    代码：

    ```c
    #include <reg52.h>

    sbit key_bit = P3^4;

    unsigned char press_cnt;
    unsigned char leds_out;

    void delay_10_ms()
    {
        unsigned char i, j;
        for (i = 20; i > 0; i--)
            for (j = 248; j > 0; j--)
                ;
    }

    void check_key_press()
    {
        if (key_bit == 0)
        {
            delay_10_ms();
            if (key_bit == 0)
            {
                press_cnt++;
                if (press_cnt == 8)
                {
                    press_cnt = 0;
                }
            }
            while (key_bit == 0);
        }
    }

    void shift_led()
    {
        unsigned char a, b;
        a = leds_out << press_cnt;
        b = leds_out >> (8 - press_cnt);
        P1 = a | b;
    }

    void main()
    {
        press_cnt = 0;
        leds_out = 0xfe;
        P1 = leds_out;
        while (1)
        {
            check_key_press();
            shift_led();
        }		
    }
    ```

    说明：

    * `P3`默认为高电平，按键一端连着`P3`的一个 IO 口，另一端接地，因此当按键按下时，对应的 IO 口变为低电平。

    * 按键在按下与松开的瞬间，大概有 5 ~ 10 ms 的信号不稳定时间，当我们检测到按键按下时，需要等 10 ms 再进行按键处理，防止在按下瞬间的抖动中，对同一个按键动作进行多次处理。这个过程叫做消抖。

    * `shift_led()`就是库函数`_crol_()`的写法。

* 51 单片机的开发环境

    看起来，以前的 keil 是付费的，现在的 keil 是免费的。keil 中并不携带 STC 公司芯片的信息，需要使用 STC-ISP 这个软件，把芯片模型库导入到 keil 中，keil 才会出现 STC 对应的选项。

    STC 似乎不生产 89C51，最低端的也到 89C52 了。

## note