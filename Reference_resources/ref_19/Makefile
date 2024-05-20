KERN_DIR=/usr/src/linux-source-6.5.0/linux-source-6.5.0
obj-m += lnk_lst.o
default:
	$(MAKE) -C $(KERN_DIR) M=$(PWD) modules
clean:
	rm -f *.o *.ko