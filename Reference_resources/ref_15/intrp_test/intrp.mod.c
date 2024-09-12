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
	{ 0x63597d5f, "cdev_add" },
	{ 0x1dee04a0, "class_create" },
	{ 0x714f9a77, "device_create" },
	{ 0xe279b9d0, "kernel_kobj" },
	{ 0x1077d56, "kobject_create_and_add" },
	{ 0x73fd6ecd, "sysfs_create_file_ns" },
	{ 0x92d5838e, "request_threaded_irq" },
	{ 0x6091b333, "unregister_chrdev_region" },
	{ 0xcd643cea, "cdev_del" },
	{ 0x704fc943, "class_destroy" },
	{ 0xad683c6b, "kobject_put" },
	{ 0x11ddf24f, "sysfs_remove_file_ns" },
	{ 0xc1514a3b, "free_irq" },
	{ 0xe8e5ec15, "device_destroy" },
	{ 0xbdfb6dbb, "__fentry__" },
	{ 0x122c3a7e, "_printk" },
	{ 0x5b8239ca, "__x86_return_thunk" },
	{ 0xbcab6ee6, "sscanf" },
	{ 0x3c3ff9fd, "sprintf" },
	{ 0xe3ec2f2b, "alloc_chrdev_region" },
	{ 0xf5f85346, "cdev_init" },
	{ 0x708cd699, "module_layout" },
};

MODULE_INFO(depends, "");


MODULE_INFO(srcversion, "06AE1023E9B7EFA74A841C1");
