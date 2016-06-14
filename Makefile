
# $Id$

.DEFAULT: .F .F90 .c .C
.SUFFIXES: .F .F90 .c .C

SRC = $(PWD)
O = $(PWD)

F77 = mpif77
F90 = mpif90
CC = mpicc
CXX = mpicxx
NVCC = nvcc

CFLAGS = -DI64

FFLAGS = -132

NVCCINCLUDE = -I$(CUDA_ROOT)/samples/common/inc \
	-I$(MATLAB_ROOT)/extern/include

CUDAArchs= \
	-gencode arch=compute_20,code=sm_20 \
        -gencode arch=compute_30,code=sm_30 \
        -gencode arch=compute_35,code=sm_35 \
        -gencode arch=compute_37,code=sm_37 \
        -gencode arch=compute_50,code=sm_50 \
	-gencode arch=compute_52,code=sm_52 \
        -gencode arch=compute_52,code=compute_52 \
	-Xcompiler=\"-fPIC -pthread -fexceptions -m64\"

NVCCFLAGS = $(NVCCINCLUDE) $(CUDAArchs) -O3 -prec-div=true -prec-sqrt=true -rdc=true

CXXFLAGS = -std=c++0x $(NVCCINCLUDE)

Link = $(CXX)

MEXA64Files = \
	$(O)/MPI_Init.mexa64 \
	$(O)/MPI_Finalize.mexa64 \
	$(O)/PiMPI.mexa64 \
	$(O)/cudaDirect.mexa64 \
	$(O)/dstevTest.mexa64 \
	$(O)/cudaMPITest.mexa64 \
	$(O)/cudaMPIEvolution.mexa64

CUDAObjs = $(O)/cudaTest.o $(O)/cudaMPIQMMDTest.o $(O)/cudaConstMem.o \
	$(O)/testConstMem.o $(O)/cudaMem.o $(O)/evolutionMPICUDA2.o 

CUDALinkObj = $(O)/cudalink.o

OBJS = $(O)/matlabUtils.o $(O)/die.o $(O)/rmatalgo.o  $(O)/rmato.o  $(O)/indent.o \
	$(O)/out.o  $(O)/rmat.o \
	$(O)/matlabStructures.o  $(O)/matlabStructuresio.o \
	$(O)/evolutionMPICUDA.o \
	$(CUDAObjs) $(CUDALinkObj)

QMLibs = $(O)/libqmdyn.a

.DEFAULT_GOAL := $(O)/cudaMPIEvolution.mexa64

all: $(MEXA64Files)

#$(EXE) : $(OBJS)
#	$(Link) $(CXXFLAGS) -o $(EXE) $(OBJS) $(LIBS)

$(O)/%.o: %.c
	cd $(O) ; $(CC)  $(cFLAGS) -c $(SRC)/$<
$(O)/%.o: %.C
	cd $(O) ; $(CXX) $(CXXFLAGS) -c $(SRC)/$<
$(O)/%.o: %.F
	cd $(O) ; $(F77) $(FFLAGS) -c $(SRC)/$<
$(O)/%.o: %.F90
	cd $(O) ; $(F90) $(FFLAGS) -c $(SRC)/$<
$(O)/%.o: %.cu
	cd $(O) ; $(NVCC) $(NVCCFLAGS) -dc $(SRC)/$<

$(CUDALinkObj): $(CUDAObjs)
	cd $(O); $(NVCC) $(CUDAArchs) -dlink $(CUDAObjs) -o $(CUDALinkObj)

%io.C: %.h
	perl io.pl $<

$(QMLibs): $(OBJS)
	cd $(O); ar -crusv $(QMLibs) $(OBJS)

$(O)/%.mexa64: $(O)/%.o $(QMLibs)
	cd $(O); $(Link) -shared $(CXXFLAGS) -o $@ $^ $(LIBS)

clean:
	rm -f *.o *~ *.mod $(EXE) depend $(MEXA64Files) $(QMLibs) $(OBJS)

cleancuda:
	rm -rf $(CUDAObjs) $(CUDALinkObj)

depend :
	$(CXX) $(CXXFLAGS) -MM *.[cC] | perl dep.pl > $@
	$(NVCC) $(NVCCINCLUDE) -M *.cu | perl dep.pl >> $@

include depend
