#include <linux/module.h>
#define INCLUDE_VERMAGIC
#include <linux/build-salt.h>
#include <linux/elfnote-lto.h>
#include <linux/export-internal.h>
#include <linux/vermagic.h>
#include <linux/compiler.h>

#ifdef CONFIG_UNWINDER_ORC
#include <asm/orc_header.h>
ORC_HEADER;
#endif

BUILD_SALT;
BUILD_LTO_INFO;

MODULE_INFO(vermagic, VERMAGIC_STRING);
MODULE_INFO(name, KBUILD_MODNAME);

__visible struct module __this_module
__section(".gnu.linkonce.this_module") = {
	.name = KBUILD_MODNAME,
	.init = init_module,
#ifdef CONFIG_MODULE_UNLOAD
	.exit = cleanup_module,
#endif
	.arch = MODULE_ARCH_INIT,
};

#ifdef CONFIG_RETPOLINE
MODULE_INFO(retpoline, "Y");
#endif



static const struct modversion_info ____versions[]
__used __section("__versions") = {
	{ 0xf4407d6b, "cdev_add" },
	{ 0x1399bb1, "class_create" },
	{ 0xd3044a78, "device_create" },
	{ 0xfd2177fc, "kernel_kobj" },
	{ 0xdffb829d, "kobject_create_and_add" },
	{ 0x533bd7c9, "sysfs_create_file_ns" },
	{ 0x92d5838e, "request_threaded_irq" },
	{ 0x6091b333, "unregister_chrdev_region" },
	{ 0x8f44466e, "cdev_del" },
	{ 0x92ce99, "class_destroy" },
	{ 0x1c0dd58c, "kobject_put" },
	{ 0xe17ecab1, "sysfs_remove_file_ns" },
	{ 0xc1514a3b, "free_irq" },
	{ 0xf7be671b, "device_destroy" },
	{ 0xbdfb6dbb, "__fentry__" },
	{ 0x122c3a7e, "_printk" },
	{ 0x5b8239ca, "__x86_return_thunk" },
	{ 0xbcab6ee6, "sscanf" },
	{ 0x3c3ff9fd, "sprintf" },
	{ 0xe3ec2f2b, "alloc_chrdev_region" },
	{ 0xa6f7a612, "cdev_init" },
	{ 0x2fa5cadd, "module_layout" },
};

MODULE_INFO(depends, "");


MODULE_INFO(srcversion, "B1506AF20883AAC6E04453A");
