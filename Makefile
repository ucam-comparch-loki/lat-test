TARGET := build/lat-test
OBJS := $(patsubst src/%.c, build/%.o, $(wildcard src/*.c))

LIBLOKI_DIR ?= /usr/groups/comparch-loki/tools/releases/libloki/current
LAT_IFC_DIR ?= /usr/groups/comparch-loki/tools/releases/lat-ifc/current
LAT_NN_DIR ?= /usr/groups/comparch-loki/tools/releases/lat-nn/current

$(TARGET): $(OBJS) | build
	loki-clang -L$(LIBLOKI_DIR)/lib -L$(LAT_IFC_DIR)/lib -L$(LAT_NN_DIR)/lib -o $@ $+ -lloki -llat-nn -llat-ifc

build/%.o: src/%.c $(wildcard src/*.h) | build
	loki-clang -O3 -I$(LIBLOKI_DIR)/include -I$(LAT_IFC_DIR)/include -I$(LAT_NN_DIR)/include -c -Werror -Wall -o $@ $<

.PHONY: test
test: $(TARGET)
	python3 src/run_tests.py

.PHONY: clean
clean:
	rm -f $(wildcard $(TARGET) *.o)
	rm -rf $(wildcard build)
	rm -f passed.txt failed.txt

build:
	mkdir $@
