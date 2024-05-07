#include <sys/ioctl.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/types.h>
#include <stdio.h>

#define WR_VALUE _IOW('a','a',int32_t*)
#define RD_VALUE _IOR('a','b',int32_t*)

int main()
{
    int fd = open("/dev/hlc_dev", O_RDWR);
    if (fd < 0) {
        printf("fail to open device file\n");
        return -1;
    }

    int32_t val_send = 123;
    int32_t val_recv;
    ioctl(fd, WR_VALUE, &val_send);
    printf("successfully write data by ioctl\n");

    ioctl(fd, RD_VALUE, &val_recv);
    printf("read value: %d\n", val_recv);

    close(fd);
    return 0;
}