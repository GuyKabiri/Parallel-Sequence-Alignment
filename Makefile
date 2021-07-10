EXEC = temp
# CFLAGS =
LIBS = -lm
FILES = main.c cpu_funcs.c

all: clean $(EXEC)

temp:
	gcc -o temp $(FILES) $(LIBS)

clean:
	rm -f $(EXEC)