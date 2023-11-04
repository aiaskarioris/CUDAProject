cuda-path := /opt/cuda/bin
all: main testing

main: source
	$(cuda-path)/nvcc -o ex3 source/ex3.cu

test-tools: tools-source
	mkdir -p test
	mkdir -p tools
	g++ -o tools/gen test-source/gen.c
	g++ -o tools/check test-source/check.c
	g++ -o tools/check-test test-source/check_test.c

