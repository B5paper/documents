compile: `make`

run: `./vec_add`

output:

```
huliucheng@zjxj:~/Documents/Projects/nccl_test_2$ ./vec_add 
buf_A:
3.0, 2.0, 3.0, 1.0, 4.0, 2.0, 0.0, 3.0, 
buf_B:
1.0, 0.0, 0.0, 2.0, 1.0, 2.0, 4.0, 1.0, 
buf_C_ref:
4.0, 2.0, 3.0, 3.0, 5.0, 4.0, 4.0, 4.0, 

cubuf_A:
3.0, 2.0, 3.0, 1.0, 4.0, 2.0, 0.0, 3.0, 
cubuf_B:
1.0, 0.0, 0.0, 2.0, 1.0, 2.0, 4.0, 1.0, 
cubuf_C:
4.0, 2.0, 3.0, 3.0, 5.0, 4.0, 4.0, 4.0, 
```