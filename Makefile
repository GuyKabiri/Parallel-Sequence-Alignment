CFLAGS = -Wall
CUFLAGS = --compiler-options -Wall
LIBS = -lm
C_FILES = main cpu_funcs mpi_funcs
CU_FILES = cuda_funcs
O_FILES = main.o c_funcs.o mpi_funcs.o cuda_funcs.o

build:
	mpicxx -fopenmp -c $(LIBS) $(CFLAGS) main.c -o main.o
	mpicxx -fopenmp -c $(LIBS) $(CFLAGS) cpu_funcs.c -o c_funcs.o
	mpicxx -fopenmp -c $(LIBS) $(CFLAGS) mpi_funcs.c -o mpi_funcs.o
	nvcc -I./inc -c $(LIBS) $(CUFLAGS) cuda_funcs.cu -o cuda_funcs.o
	mpicxx -fopenmp -o mpiCudaOpenMP $(O_FILES) /usr/local/cuda-9.1/lib64/libcudart_static.a -ldl -lrt

clean:
	rm -f *.o ./mpiCudaOpenMP

runseq:
	mpiexec -np 1 ./mpiCudaOpenMP -100

runcuda:
	mpiexec -np 1 ./mpiCudaOpenMP 100




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