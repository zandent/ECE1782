# tools
CC := nvcc
RM := rm -f

# flags
CFLAGS := -ggdb
LDFLAGS :=
LDLIBS :=

# sources
SRC := src
OUT := out
src0 := main_v0.cu
src1 := main_v1.cu
src2 := main_v2.cu
src3 := main_v3.cu
src4 := main_v4.cu
tgt0 := $(src0:.cu=)
tgt1 := $(src1:.cu=)
tgt2 := $(src2:.cu=)
tgt3 := $(src3:.cu=)
tgt4 := $(src4:.cu=)

# gmake magic
.PHONY: default all clean

#targets
default: all
all:
	$(CC) -o $(OUT)/$(tgt0) $(SRC)/$(src0)
	$(CC) -o $(OUT)/$(tgt1) $(SRC)/$(src1)
	$(CC) -o $(OUT)/$(tgt2) $(SRC)/$(src2)
	$(CC) -o $(OUT)/$(tgt3) $(SRC)/$(src3)

clean:
	$(RM) $(OUT)/*

#dependencies
