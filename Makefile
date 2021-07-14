CFLAGS = -Wall -fopenmp
LIBS = -lm
C_FILES = main.c cpu_funcs.c
CU_FILES = cuda_functions.cu
O_FILES = main.o c_functions.o cuda_functions.o

build:
	mpicxx -fopenmp -c $(C_FILES) $(LIBS) $(CFLAGS) -o main.o
	mpicxx -fopenmp -c $(C_FILES) $(LIBS) $(CFLAGS) -o c_functions.o
	nvcc -I./inc -c $(CU_FILES) $(LIBS) $(CFLAGS) -o cuda_functions.o
	mpicxx -fopenmp -o mpiCudaOpemMP  $(O_FILES)  /usr/local/cuda-9.1/lib64/libcudart_static.a -ldl -lrt

all: clean build

clean:
	rm -f *.o ./mpiCudaOpemMP

run:
	mpiexec -np 2 ./mpiCudaOpemMP 5000

runOn2:
	mpiexec -np 2 -machinefile mf  -map-by  node  ./mpiCudaOpemMP 5000
