#include <include/vmmlib/tensor.hpp>

#include <iostream>
//#include <fftw3.h>

#define VMMLIB_SAFE_ACCESSORS 0
#define VMMLIB_VERBOSE 1

using namespace vmml;

int main (int argc, char * const argv[]) {

    std::cout << "Let's create a 3-way tensor, of size 10 x 10 x 10" << std::endl;
    tensor<float> input(10,10,10);
    
    std::cout << "We initialize it as a Gaussian kernel, with sigma = 1" << std::endl;
    input.set_gaussian(1);
    
    std::cout << "This tensor is separable (i.e., has rank 1)." << std::endl;
    std::cout << "To check this, let's perform a rank-1 Tucker decomposition and see that it approximates exactly our input" << std::endl;
    
    std::cout << "The decomposition consists of a core and several factor matrices, which must be declared" << std::endl;
    tensor<float> core(1,1,1); // The Tucker core will have 1 element
    tensor<float> U1(10,1), U2(10,1), U3(10,1); // The 3 factor matrices, each with onlyZ1 column
    
    std::cout << "We need to initialize the factor matrices. The basis functions for the DCT are a good choice in general." << std::endl;
    U1.set_dct();
    U2.set_dct();
    U3.set_dct();
    
    std::cout << "Perform the Tucker decomposition" << std::endl;
    input.tucker_decomposition(core,U1,U2,U3);
    
    std::cout << "We just obtained a rank-1 decomposition. Let's reconstruct it" << std::endl;
    tensor<float> reconstruction = core.ttm(U1,U2,U3); // Tensor-times-matrix multiplication
    
    std::cout << "The difference between original and reconstruction (Frobenius norm) is close to 0: norm of the approximation error = " << (input-reconstruction).frobenius_norm() << std::endl;
    
    std::cout << "We have performed lossless compression. Size comparison: original size = " << input.get_size() << "; compressed size = " << core.get_size() + U1.get_size() + U2.get_size() + U3.get_size() << std::endl;
}
