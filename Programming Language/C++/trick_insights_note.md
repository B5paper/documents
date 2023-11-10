# Trick insights Note

This note is used to summarize tricks in C++ language.

* If we want to verify if two numbers are the same sign, we can use XOR: `sign_1 ^ sign_2`.

    This trick can get the product sign of two multipliers.

* the absolute value of `INT32_MIN` is greater by 1 than `INT32_MAX`

    So don't convert negative value to positive. Turn positive value to negative to avoid overflow.

* 首先让最底层的函数暴露尽量多的信息和细节，然后再在上面套一层封装，适应不同的调用需求

    比如对射线和空间三维物体交点的计算，底层的计算函数需要输出是否相交，相交的话`t`是多少，交点在哪里，相交的是哪个物体，交点处的法向量，颜色等等，然后上层函数再对这个函数进行一次封装，有些需求是只需要判断是否相交就可以，有些需求是要得到交点位置。我们使用第二层函数来对接这些需求。

* 在写多线程的代码时，由于负载不均衡是个很大的问题，所以子线程函数的颗粒需要尽可能细，细到无法再分。然后在上层再写一层调度，进行任务的分配。

* 写代码时，前面写过的代码，实现过的函数，经常会被忘记，因此写代码也需要进行 qa 记忆巩固。

* 引入一个头文件，就代表了引入一种功能。头文件与头文件之间尽量独立

* array 类似于一个数组，而 vector 类似一个指针

    array 是占用栈，而 vector 占用的是堆。