# Trick insights Note

This note is used to summarize tricks in C++ language.

* If we want to verify if two numbers are the same sign, we can use XOR: `sign_1 ^ sign_2`.

    This trick can get the product sign of two multipliers.

* the absolute value of `INT32_MIN` is greater by 1 than `INT32_MAX`

    So don't convert negative value to positive. Turn positive value to negative to avoid overflow.