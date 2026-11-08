#include <stdio.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <unistd.h>
#include <string.h>
                                             
#define BUF_LEN 256

#define HLC_CDEV_CMD_1 _IOW('H', 1, int)
#define HLC_CDEV_CMD_2 _IO('H', 2)

int main()
{
    char ch = 0;
    char buf[BUF_LEN] = {};
    const char *hlc_cdev_path = "/dev/hlc_dev";
    int ret;
                                             
    int fd = open(hlc_cdev_path, O_RDWR);
    if (fd == -1) {
        printf("fail to open %s\n", hlc_cdev_path);
        return -1;
    }
    printf("[OK] open /dev/hlc_cdev, fd = %d.\n", fd);
                                             
    // read
    ssize_t bytes_read; 
    bytes_read = read(fd, buf, BUF_LEN);
    if (bytes_read == -1) {
        printf("fail to read\n");
        return -1;
    }
    printf("[OK] read %ld bytes from %s.\n", bytes_read, hlc_cdev_path);
    printf("\tbuf: %s\n", buf);

    // 因为前面 read 已经将 offset 设置到末尾了，所以这里重新设置 offset
    // 如果这里不设置 seek，那么后面 write 写不进去数据
    off_t new_offset = lseek(fd, 0, SEEK_SET);
    if (new_offset < 0) {
        printf("fail to seek\n");
        return -1;
    }
    printf("[OK] lseek, new offset: %ld\n", new_offset);

    // write
    const char *test_msg = "this is a msg from test app.\n";
    ssize_t bytes_write;
    bytes_write = write(fd, test_msg, strlen(test_msg)); 
    if (bytes_write == -1) {
        printf("fail to write\n");
        return -1;
    }
    if (bytes_write == 0) {
        printf("warn: write 0 bytes!\n");
        return -1;
    }
    printf("[OK] write %ld bytes.\n", bytes_write);

    // ioctl
    ret = ioctl(fd, HLC_CDEV_CMD_2);
    if (ret == -1) {
        printf("fail to ioctl send cmd: HLC_CDEV_CMD_2\n");
        return -1;
    }
    printf("[OK] ioctl send cmd HLC_CDEV_CMD_2\n");

    ret = close(fd);
    if (ret != 0) {
        printf("fail to close fd, ret: %d\n", ret);
        return -1;
    }
    printf("[OK] close fd %d.\n", fd);

    return 0;
}
