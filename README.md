# Simple GPU Matrix Multiplication
This repository provides a simple intuitive self written Matrix
Multiplication, calculated on a GPU.  Its purpose is to represent a
simple stereotypical GPU matrix multiplication. The CPU code is written
in CUDA driver API and the device code in openCL. You can compile the
code with the clang compiler plus an openCL library. Moreover you need
the nvptx64--nvidiacl driver interface for the nvptx backend. You might
find http://stackoverflow.com/questions/8795114/how-to-use-clang-to-compile-opencl-to-ptx-code
helpful.
