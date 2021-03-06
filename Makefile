CC = nvcc
CFLAGS = -std=c++11
INCLUDES = 
LDFLAGS = -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_imgcodecs
SOURCES = main.cu
OUTF = main.exe
OBJS = main.o
OPENCV_VERSION := 3

$(OUTF): $(OBJS)
	    $(CC) $(CFLAGS) -o $(OUTF) $< $(LDFLAGS)

$(OBJS): $(SOURCES)
	    $(CC) $(CFLAGS) -c $<

rebuild: clean $(OUTF)

clean:
	    rm *.o $(OUTF)
