
CPP = g++
CPPFLAGS=-g -Wall
LDFLAGS=`pkg-config --cflags --libs opencv`

SRC=src
OBJ=obj
SRCS=$(wildcard $(SRC)/*.cpp)
OBJS=$(patsubst $(SRC)/*.cpp, $(OBJ)/%.o, $(SRCS))
HDRS=$(wildcard $(SRC)/*.h)
BIN=FaceNet

all: $(BIN)

$(BIN): $(OBJS) $(OBJ)
	$(CPP) $(CPPFLAGS) $(OBJS) -o $@ $(LDFLAGS)

$(OBJ)/%.0: $(SRCS)/%.cpp $(OBJ)
	$(CPP) $(CPPFLAGS) -c $< -o $@

$(OBJ):
	mkdir -p $@

clean:
	$(RM) -r $(OBJ)
	$(RM) $(BIN)
