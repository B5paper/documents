default:
	make clean
	make main

main: main.cpp
	g++ -g main.cpp -L/opt/rocm-6.0.2/lib -lOpenCL -o main

clean:
	rm -f main