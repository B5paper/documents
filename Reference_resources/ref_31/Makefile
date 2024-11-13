all: main main_cpu

main: main.c
	/home/hlc/Documents/Projects/openmpi-5.0.5/bin/bin/oshcc -g main.c -o main

main_cpu: main_cpu.c
	gcc -g main_cpu.c -o main_cpu

clean:
	rm -f main main_cpu