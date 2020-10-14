# tools
CC := nvcc
RM := rm -f

# flags
CFLAGS := -ggdb
CCFLAGS := -lcublas_static -lcublasLt_static -lculibos
LDLIBS :=

# sources
SRC := src
OUT := out
src0 := main.cu
tgt0 := $(src0:.cu=)

# gmake magic
.PHONY: default all clean

#targets
default: all
all:
	$(CC) -o $(OUT)/$(tgt0) $(SRC)/$(src0)

clean:
	$(RM) $(OUT)/*

#dependencies
