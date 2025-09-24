CC = gcc -O3

all:	walksat makewff makequeens

walksat: walksat.c
	$(CC)  -c walksat.c
	# If linking fails, then try adding Windows Winmm.lib
	$(CC) walksat.o -lm -o walksat || $(CC) walksat.o -lm -lWinmm -o walksat

makewff: makewff.c
	$(CC)  makewff.c -lm -o makewff

makequeens: makequeens.c
	$(CC)  makequeens.c -lm -o makequeens

install: walksat makewff makequeens
	cp walksat $(HOME)/bin/
	cp makewff $(HOME)/bin/
	cp makequeens $(HOME)/bin/
	cp Scripts/* $(HOME)/bin/
	

clean:
	rm -f walksat makewff makequeens walksat.o

