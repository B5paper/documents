#include <stdio.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

int main()
{
    int fd = open("/dev/hlc_dev", O_RDWR);
    if (fd <= 0) {
        printf("fail to open device\n");
        return -1;
    }

    int val = 2;
    int rtv = write(fd, &val, sizeof(val));
    printf("rtv: %d\n", rtv);
    printf("wrote value 2\n");

    close(fd);
    return 0;
}