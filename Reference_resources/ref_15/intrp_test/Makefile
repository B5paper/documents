KERNEL_DIR=/lib/modules/6.8.0-40-generic/build
obj-m  +=  intrp.o
default:
	$(MAKE) -C $(KERNEL_DIR) M=$(PWD) modules

clean:
	rm -f *.mod *.o *.order *.symvers *.cmd