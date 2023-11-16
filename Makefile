CXX = g++
NVCC = nvcc -ccbin $(CXX)

GENCODE_FLAGS = -gencode arch=compute_75,code=sm_75
DEBUG_FLAGS = -g -G
NVCC_FLAGS = -std=c++17 -O2 -MMD -MP -m64 -g $(GENCODE_FLAGS) -allow-unsupported-compiler -Xcompiler -Wall,-Wextra
TARGET = target

SRCDIR = ./src/
OBJDIR = ./obj/
BINDIR = ./bin/
RESULTSDIR = ./results/

INC = -I./include/

SRC := $(shell find $(SRCDIR) -maxdepth 2 -type f -name "*.cu")
OBJ := $(patsubst $(SRCDIR)%, $(OBJDIR)%, $(SRC:.cu=.o))


.PHONY: clean run commands
all: default

echo:
	@echo $(SRC)
	@echo $(OBJ)

default: $(OBJ)
	$(NVCC) $(NVCC_FLAGS) $(OBJ) -o $(BINDIR)$(TARGET) $(LIB)

$(OBJDIR)%.o: $(SRCDIR)%.cu
	$(NVCC) -c -dc $(NVCC_FLAGS) $(INC) $< -o $@


-include $(OBJ:.o=.d)


run: default
	$(BINDIR)$(TARGET) | ppmtojpeg > $(RESULTSDIR)out.jpg

show:
	make run
	xdg-open $(RESULTSDIR)out.jpg > /dev/null 2>&1

commands:
	bear -- make

clean:
	$(RM) -f -r $(OBJDIR)* $(BINDIR)*
