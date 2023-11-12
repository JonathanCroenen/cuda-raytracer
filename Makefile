CXX = g++

NVCC = nvcc -ccbin $(CXX)

GENCODE_FLAGS = -gencode arch=compute_75,code=sm_75
DEBUG_FLAGS = -g -G
NVCC_FLAGS = -std=c++17 -O2 -MMD -MP -m64 -allow-unsupported-compiler -Xcompiler -Wall,-Wextra $(GENCODE_FLAGS)

TARGET = target

SRCDIR = ./src/
OBJDIR = ./obj/
BINDIR = ./bin/
RESULTSDIR = ./results/

INC = -I./include/

CUDA_SRC := $(shell find $(SRCDIR) -maxdepth 2 -type f -name "*.cu")
CPP_SRC := $(shell find $(SRCDIR) -maxdepth 2 -type f -name "*.cpp")

SRC := $(CUDA_SRC) $(CPP_SRC)
OBJ := $(patsubst $(SRCDIR)%, $(OBJDIR)%, $(CPP_SRC:.cpp=.o)) $(patsubst $(SRCDIR)%, $(OBJDIR)%, $(CUDA_SRC:.cu=.o))


.PHONY: clean run commands
all: default


default: $(OBJ)
	$(NVCC) $(NVCC_FLAGS) $(OBJ) -o $(BINDIR)$(TARGET) $(LIB)

debug: $(OBJ)
	NVCC_FLAGS += $(DEBUG_FLAGS)
	make default

$(OBJDIR)%.o: $(SRCDIR)%.cu
	$(NVCC) -c $(NVCC_FLAGS) $(INC) $< -o $@

$(OBJDIR)%.o: $(SRCDIR)%.cpp
	$(NVCC) -c $(NVCC_FLAGS) $(INC) $< -o $@

-include $(OBJ:.o=.d)


run: default
	$(BINDIR)$(TARGET) > $(RESULTSDIR)out.ppm

show:
	make run
	xdg-open $(RESULTSDIR)out.ppm > /dev/null 2>&1

commands:
	bear -- make

clean:
	$(RM) -f -r $(OBJDIR)* $(BINDIR)*
