CFLAGS = -Wall
LIBS = -lm
C_FILES = main.c cpu_funcs.c
CU_FILES = cuda_funcs.cu
O_FILES = main.o c_funcs.o

build:
	mpicxx -fopenmp -c main.c -o main.o
	mpicxx -fopenmp -c cpu_funcs.c -o c_funcs.o
	# nvcc -I./inc -c $(CU_FILES) -o cuda_funcs.o
	mpicxx -fopenmp -o mpiCudaOpenMP  $(O_FILES)  #/usr/local/cuda-9.1/lib64/libcudart_static.a -ldl -lrt

clean:
	rm -f *.o ./mpiCudaOpenMP

runSeqNoCuda: 
	mpiexec -np 1 ./mpiCudaOpenMP 0

runSeqHalfCuda: 
	mpiexec -np 1 ./mpiCudaOpenMP 50

runSeqFullCuda: 
	mpiexec -np 1 ./mpiCudaOpenMP 100

run1CompParNoCuda: 
	mpiexec -np 2 ./mpiCudaOpenMP 0

run1CompParHalfCuda: 
	mpiexec -np 2 ./mpiCudaOpenMP 50

run1CompParFullCuda: 
	mpiexec -np 2 ./mpiCudaOpenMP 100

run2CompParNoCuda: 
	mpiexec -np 2 -machinefile  mf  -map-by  node  ./mpiCudaOpenMP 0

run2CompParHalfCuda: 
	mpiexec -np 2 -machinefile  mf  -map-by  node  ./mpiCudaOpenMP 50

run2CompParFullCuda: 
	mpiexec -np 2 -machinefile  mf  -map-by  node  ./mpiCudaOpenMP 100