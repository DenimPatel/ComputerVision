CC     :=  g++
CFLAGS := -Wall -Wextra -pedantic -g `pkg-config opencv --cflags`
LIBS   := `pkg-config opencv --libs`
RM     := rm -f

.PHONY: all clean

NAME := lab4
SRCS := $(wildcard *.cpp)
OBJS := $(filter-out utility.o, $(SRCS:.cpp=.o))

all: utility.o $(OBJS) $(NAME)

objects: $(OBJS)
#
# link the .o files into the target executable 
#
$(NAME): utility.o $(OBJS)
	$(CC) $^ -o $@ $(LIBS)

#
# compile the .c file into .o files using the compiler flags
#
%.o: %.cpp
	$(CC) $(CFLAGS) -c $< -o $@ -I.

#utility is a dependency for others.
utility.o: utility.cpp
	$(CC) $(CFLAGS) -c $? -o $@ -I.

clean:
	$(RM) *.o
	$(RM) $(NAME)
