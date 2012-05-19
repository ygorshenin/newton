CPP = mpic++
CPPFLAGS = -Wall -DNDEBUG -O2 -I$(BOOST_ROOT) -Isrc

BOOST_ROOT = /usr/local/lib
BOOST_LIBS = mpi serialization
LDFLAGS = $(addprefix $(BOOST_ROOT)/, $(addsuffix .a, $(addprefix libboost_, $(BOOST_LIBS))))

SRC_DIR = src
SRC_SUBDIRS = . math
SRC_DIRS = $(addprefix $(SRC_DIR)/, $(SRC_SUBDIRS))
SRCS = $(wildcard $(addsuffix /*.cc, $(SRC_DIRS)))
HDRS = $(wildcard $(addsuffix /*.h, $(SRC_DIRS)))

OBJ_DIR = obj
OBJ_DIRS = $(addprefix $(OBJ_DIR)/, $(SRC_SUBDIRS))
OBJS := $(patsubst $(SRC_DIR)/%, $(OBJ_DIR)/%, $(SRCS))
OBJS := $(patsubst %.cc, %.o, $(OBJS))

BIN_DIR = bin

PROGRAM = $(BIN_DIR)/newton

all: $(OBJ_DIR) $(BIN_DIR) $(PROGRAM)

$(PROGRAM): $(OBJS)
	$(CPP) $(LDFLAGS) -o $@ $^

$(OBJ_DIR):
	mkdir -p $(OBJ_DIRS)

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cc $(HDRS)
	$(CPP) $(CPPFLAGS) -c -o $@ $<

.PHONY: clean

clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR)
