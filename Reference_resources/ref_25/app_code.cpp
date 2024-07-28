#include <iostream>
#include <unistd.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <string.h>
using namespace std;

enum IOCTL_CMDS {
    aaa
};

int main()
{
    int fd = open("/dev/hlc_dev", O_RDWR);
    if (fd <= 0)
    {
        cout << "fail to open dev file" << endl;
        return -1;
    }

    char *buf = (char*) mmap(NULL, 4096, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    printf("mapped addr: %p\n", buf);
    printf("buf: %s\n", buf);
    strcpy((char*) buf, "hello from hlc\n");
    
    int ret = ioctl(fd, aaa, NULL);
    if (ret != 0)
    {
        cout << "fail to run ioctl()..." << endl;
        return -1;
    }

    ret = munmap(buf, 4096);
    if (ret != 0)
    {
        cout << "fail to unmap" << endl;
        return -1;
    }
    close(fd);
    return 0;
}