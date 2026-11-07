KERNEL_DIR=/usr/src/linux-headers-6.5.0-41-generic  # 这里要和我们下载的内核版本保持一致
obj-m  +=  hlc_cdev.o
default:
	$(MAKE) -C $(KERNEL_DIR) M=$(PWD) modules

clean:
	rm -rf *.o *.ko *.mod *.mod.c *.order .*.cmd *.symvers
