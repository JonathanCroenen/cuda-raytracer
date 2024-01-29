CXX = g++-12
NVCC = nvcc -ccbin $(CXX)

NVCC_FLAGS = -std=c++17 -O2 -MMD -MP -m64 -g -gencode arch=compute_75,code=sm_75 -Xcompiler -Wall,-Wextra 

MAKEFLAGS += -j8

LIB_DIR = lib
EXE_DIR = exe
BUILD_DIR = build

SRCS := $(shell find $(LIB_DIR) -maxdepth 3 -type f -name "*.cu")
OBJS := $(patsubst $(LIB_DIR)/%.cu, $(BUILD_DIR)/%.o, $(SRCS))

EXE_SRCS := $(shell find $(EXE_DIR) -maxdepth 3 -type f -name "*.cu")
EXES := $(patsubst $(EXE_DIR)/%.cu, $(BUILD_DIR)/$(EXE_DIR)/%, $(EXE_SRCS))

.PHONY: executables clean

executables: $(OBJS) $(EXES)

$(BUILD_DIR)/$(EXE_DIR)/%: $(EXE_DIR)/%.cu $(OBJS)
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCC_FLAGS) -I$(LIB_DIR) $^ -o $@

$(BUILD_DIR)/%.o: $(LIB_DIR)/%.cu
	@mkdir -p $(dir $@)
	$(NVCC) -c -dc $(NVCC_FLAGS) -I$(LIB_DIR) $< -o $@

-include $(OBJS:.o=.d)

clean:
	$(RM) -rf $(BUILD_DIR)/*

commands: clean
	bear -- make
