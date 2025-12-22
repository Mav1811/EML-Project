# ================================
# Compiler and flags
# ================================
CC      = g++
CFLAGS  = -O3 -Wall -Wextra  -std=c++17 -g

# ================================
# Target output and source file
# ================================
TARGET  = acl_backend
SRC     = run_onnx.cpp

# ================================
# Directories for ArmNN + ACL
# ================================
LIB_DIR = /home/orangepi/armnn/build
INC_DIR = /home/orangepi/armnn/include

# ================================
# Libraries to link for ACL backend
# You may need to add/remove libs depending on your ArmNN build
# ================================
LIBS = -larmnn -larmnnOnnxParser

# ================================
# Build rule
# ================================
all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) $(SRC) \
	-I$(INC_DIR) \
	-L$(LIB_DIR) \
	$(LIBS) \
	-o $(TARGET)

# ================================
# Cleaning
# ================================
.PHONY: clean
clean:
	rm -f $(TARGET)
