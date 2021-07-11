EXEC = temp
CFLAGS = -Wall
LIBS = -lm
FILES = main.c cpu_funcs.c

all: clean $(EXEC)

temp:
	gcc -o temp $(FILES) $(LIBS) $(CFLAGS)

clean:
	rm -f $(EXEC)