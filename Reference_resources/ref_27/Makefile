all: server client 

server: server.c sock.o
	gcc -g server.c -libverbs sock.o -o server

client: client.c sock.o
	gcc -g client.c -libverbs sock.o -o client

sock.o: sock.h sock.c
	gcc -g -c sock.c -o sock.o

clean:
	rm -f server client sock.o
