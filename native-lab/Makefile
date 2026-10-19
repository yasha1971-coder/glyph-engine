CXX ?= g++
CXXFLAGS = -std=c++20 -O2 -Wall -Wextra -Wpedantic -Werror -D_FILE_OFFSET_BITS=64
LDLIBS = -lz -lcrypto
all: glyph-native
glyph-native: main.cpp
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDLIBS)
test: glyph-native
	python3 test.py ./glyph-native
sanitize:
	$(CXX) $(CXXFLAGS) -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer main.cpp -o glyph-native-san $(LDLIBS)
	python3 test.py ./glyph-native-san
