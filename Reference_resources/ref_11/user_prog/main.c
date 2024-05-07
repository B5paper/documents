#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

int main()
{
    const char *str_1 = "hello, world";
    size_t str_len = strlen(str_1);
    char *str_2;
    str_2 = malloc(str_len+1);

    int fd = open("/dev/hlc_dev", O_RDWR);
    if (fd < 0) {
        printf("fail to open device file\n");
        printf("fd: %d\n", fd);
        return -1;
    }
    printf("successfully open device file\n");
    printf("fd: %d\n", fd);

    ssize_t rtv;
    rtv = write(fd, str_1, str_len+1);
    printf("write %ld bytes\n", rtv);

    rtv = read(fd, str_2, str_len+1);
    printf("read %ld bytes\n", rtv);

    close(fd);
    printf("close device file.\n");

    printf("read from kernel driver:\n");
    printf("%s\n", str_2);
    free(str_2);
    return 0;
}