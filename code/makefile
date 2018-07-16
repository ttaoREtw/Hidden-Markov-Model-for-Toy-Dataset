
CC = g++
FILE_TRAIN = train.cpp
FILE_TEST = test.cpp


train: $(FILE_TRAIN) HMM.o
	$(CC) -o train.out $(FILE_TRAIN) HMM.o

test: $(FILE_TEST) HMM.o
	$(CC) -o test.out $(FILE_TEST) HMM.o

%.o: %.cpp %.h
	$(CC) -c $<

clean:
	rm *.o
	rm *.out