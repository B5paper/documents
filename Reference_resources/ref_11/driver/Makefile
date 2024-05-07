KERNEL_DIR=/usr/src/linux-headers-6.5.0-28-generic
obj-m  +=  hello.o
default:
	$(MAKE) -C $(KERNEL_DIR) M=$(PWD) modules

clean:
	rm -f *.mod *.o *.order *.symvers *.cmd