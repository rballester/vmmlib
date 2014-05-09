#include <include/vmmlib/tensor_includes.hpp>

#include <iostream>
//#include <fftw3.h>

#define VMMLIB_SAFE_ACCESSORS 0
#define VMMLIB_VERBOSE 1

using namespace vmml;

int main (int argc, char * const argv[]) {

    // Let's create a 3-way tensor, of size 10 x 10 x 10
    tensor<float> input(10,10,10);
    
    // We initialize it as a Gaussian kernel, with sigma = 1
    input.set_gaussian(1);
    
    // This tensor is separable (i.e., has rank 1). 
    // To check this, let's perform a rank-1 Tucker decomposition and see that
    // approximates exactly our input
    
    // The decomposition consists of a core and several factor matrices, which
    // we declare here:
    tensor<float> core(1,1,1); // The Tucker core will have 1 element
    tensor<float> U1(10,1), U2(10,1), U3(10,1); // The 3 factor matrices
    
    // We need to initialize the factor matrices. The basis functions for the DCT
    // are a good choice in general:
    U1.set_dct();
    U2.set_dct();
    U3.set_dct();
    
    // Tucker decomposition
    input.tucker_decomposition(core,U1,U2,U3);
    
    // We just obtained a rank-1 decomposition. Let's reconstruct it:
    tensor<float> reconstruction = core.ttm(U1,U2,U3); // Tensor-times-matrix multiplication
    
    // Print the difference between original and reconstruction (Frobenius norm);
    // it's practically 0:
    std::cout << "Norm of the approximation error: " << (input-reconstruction).frobenius_norm() << std::endl;
    
    // We have performed lossless compression. Size comparison:
    std::cout << "Original size: " << input.get_size() << "; compressed size: " <<
            core.get_size() + U1.get_size() + U2.get_size() + U3.get_size() << std::endl;
}
