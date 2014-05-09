//#include <vmmlib/t3_hooi.hpp>
//#include <vmmlib/t3_converter.hpp>
//#include <vmmlib/tensor_mmapper.hpp>

#include <include/vmmlib/tensor.hpp>
#include <include/vmmlib/tensor_initializers.hpp>
#include <include/vmmlib/tensor_accessors.hpp>
#include <include/vmmlib/tensor_scalar_operations.hpp>
#include <include/vmmlib/tensor_transformations.hpp>
#include <include/vmmlib/tensor_tensor_operations.hpp>
#include <include/vmmlib/tensor_io.hpp>
#include <include/vmmlib/tensor_properties.hpp>

#include <iostream>
#include <fftw3.h>

#define VMMLIB_SAFE_ACCESSORS 0
#define VMMLIB_VERBOSE 1

using namespace vmml;

int main (int argc, char * const argv[]) {

    // Let's create a 3-way tensor, of size 5 x 5 x 5
    tensor<float> input(5,5,5);
    
    // We initialize it as a Gaussian kernel, with sigma = 1
    input.set_gaussian(1);
    
    // This tensor is separable (i.e., has rank 1). 
    // To check this, let's perform a rank-1 Tucker decomposition
    
    // The decomposition consists of a core and several factor matrices, which
    // we declare here:
    tensor<float> core(1,1,1); // The Tucker core will have 1 element
    tensor<float> U1(5,1), U2(5,1), U3(5,1); // The 3 factor matrices
    
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
    std::cout << "Norm of the error: " << input.frobenius_norm(reconstruction) << std::endl;
    
}
