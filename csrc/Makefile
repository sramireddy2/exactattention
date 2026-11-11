CXX ?= g++
CXXFLAGS ?= -O3 -march=native -std=c++17 -fPIC -Wall -Wextra -Wno-unused-parameter
OUT ?= ../build/libexactattn_cpu.so

$(OUT): exact_attention_cpu.cpp
	mkdir -p $(dir $(OUT))
	$(CXX) $(CXXFLAGS) -shared -o $@ $< -pthread

clean:
	rm -f $(OUT)
