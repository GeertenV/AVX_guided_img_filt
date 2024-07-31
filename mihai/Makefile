
CFLAGS +=  -O3 -Wall -mavx2 -mfma -march=native -fomit-frame-pointer -funroll-loops 
CC = gcc 

LIB =

LIB_SOURCES = boxfilter1D.c boxfilter1D_norm.c boxfilter.c transpose.c matmul.c guidedfilter.c
LIB_OBJECTS =  $(LIB_SOURCES:.c=.o)

UNITTESTS_LIST = $(wildcard test/test_*.c)
UNITTESTS= $(UNITTESTS_LIST:.c=.test)

.SUFFIXES: .c .test

.PHONY: all check clean

all: libboxfilter.a speed_boxfilter cat_smoothing


libboxfilter.a: $(LIB_OBJECTS) $(LIB_SOURCES) 
	$(AR) rc $@ $^
	ranlib $@

check: libboxfilter.a $(UNITTESTS) 
	$(info $(UNITTESTS))
	for t in $(UNITTESTS); do ./$$t ; done 

speed_boxfilter: libboxfilter.a test/speed_boxfilter.c
	$(CC) $(CFLAGS) -g -o $@ test/speed_boxfilter.c -I. -L. -lboxfilter $(LIB) -lm; mv speed_boxfilter test;

cat_smoothing: libboxfilter.a test/cat_smoothing.c
	$(CC) $(CFLAGS) -g -o $@ test/cat_smoothing.c -I. -L. -lboxfilter $(LIB) -lm; mv cat_smoothing test;

clean:
	rm libboxfilter.a
	rm -f *.o *~ *.s
	rm -f test/*.test test/*.o test/*~
	rm -f test/speed_boxfilter
	rm -f test/cat_smoothing

.c.o:   $< 
	$(CC) $(CFLAGS) -c $< -o $@

.c.test: $< 
	$(CC) $(CFLAGS) -g -o $@ $< -I. -L. -lboxfilter $(LIB) -lm  

