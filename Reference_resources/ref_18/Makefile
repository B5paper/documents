KERN_DIR=/usr/src/linux-source-6.5.0/linux-source-6.5.0
obj-m += aaaaa.o
default: aaaaa.c
	$(MAKE) -C $(KERN_DIR) M=$(PWD) modules
clean:
	rm -f *.o *.ko .*.cmd *.mod *.mod.c *.order *.symvers