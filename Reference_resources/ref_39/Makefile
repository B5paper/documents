all: server client

server: server.cu
	nvcc -g server.cu -o server

client: client.cu
	nvcc -g client.cu -o client

clean:
	rm -f server client