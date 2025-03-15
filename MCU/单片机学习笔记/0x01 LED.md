# LED

## 闪烁灯

现象：点亮与单片机P1.0口相连的发光二极管，延时约0.2S，然后熄灭，再延时约0.2S，再点亮，如此循环下去。

```c
#include <reg52.h>

sbit led_1 = P1^0;

void delay02s()
{
	unsigned char i, j, k;
	for (i = 20; i > 0; i--)
		for (j = 20; j > 0; j--)
			for (k = 248; k > 0; k--);
}

void main()
{
	while (1)
	{
		led_1 = 0;
		delay02s();
		led_1 = 1;
		delay02s();	
	}
}
```

问题：

1. `sbit`是什么类型？为什么`P1^0`可以定义 P10 的 I/O 引脚？

1. `<reg52.h>`库中有什么内容？为什么要用这个库？

1. `delay02s()`为什么可以通过空 for 循环延时 200 ms？

1. 为什么`led_1 = 0`会使 led 灯点亮，而`led_1 = 1`会使 led 灯熄灭？

    猜想：P1 一共有 P10 到 P17 八个引脚，可能由 1 个字节控制。P1 的 8 个引脚都接到 74HC573 上，这个芯片可能是一个驱动或锁存器。8 个 led 灯正极统一被焊到了 VCC 上，因此只要单片机这边将引脚置 0，就可以使得 led 灯点亮。

1. 为什么烧录程序时，使用的 baud 为 115200，这个是干嘛用的，其它的值可以吗？

## 流水灯

单片机 P1 口相连的 8 个发光二极管中的一个循环移位点亮，同时蜂鸣器发出滴滴的响声。

```c
#include <reg52.h>
#include <intrins.h>

unsigned char a, b, k, j;
sbit beep = P2^3;

void delay10ms()
{
	for (a = 100; a > 0; a--)
		for (b = 225; b > 0; b--)
			;
}

void main()
{
	k = 0xfe;
	while (1)
	{
		delay10ms();
		beep = 0;
		delay10ms();
		beep = 1;
		j = _crol_(k, 1);
		k = j;
		P1 = j;
	}
}
```

说明：

1. keil 在单击编译按钮时，会自动保存修改过但未保存的代码文件。

1. `unsigned char k, j;`可以改成`char k, j;`，不影响功能。

1. 此蜂鸣器为有源蜂鸣器，当`beep = 0;`时，蜂鸣器开始响，实测频率为 2972 Hz；当`beep = 1`时，关闭蜂鸣器。

1. 只编译单个 src 文件不会生成`hex`文件，必须 build target，才会生成`hex`文件。

问题：

1. `<intrins.h>`是干嘛用的？

1. 将代码中的`unsigned char a, b;`改成`int a, b;`，会明显增加延时的时间，为什么？

1. 调研`_crol_()`函数。

1. 看起来似乎`P1`只能作为左值，不能作为右值，为什么？