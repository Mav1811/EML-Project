CC = g++
CFLAGS = -O3 -Wall -std=c++17
TARGET = mnist_mnn_gpu
OBJ = implement_mnn.cpp
LIBRARY = /home/orangepi/MNN/build_opencl
HEADS = /home/orangepi/MNN/include

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CC) $(CFLAGS) $(OBJ) -I$(HEADS) -L$(LIBRARY) -lMNN -o $(TARGET)

.PHONY: clean
clean:
	rm -f $(TARGET)