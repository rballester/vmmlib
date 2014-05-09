/* 
 * File:   tensor_initializers.hpp
 * Author: rballester
 *
 * Created on May 8, 2014, 12:00 PM
 */

#ifndef TENSOR_INITIALIZERS_HPP
#define	TENSOR_INITIALIZERS_HPP

#include "tensor.hpp"

namespace vmml
{

    /**
 * ... text ...
 */

// Fill with zeros
template< typename T>
void tensor<T>::set_zero()
{
    memset(array,0,size*sizeof(T));
}

// Set all values to be a given constant
template< typename T>
void tensor<T>::set_constant(T constant)
{
    for( size_t counter = 0; counter < size; ++counter )
    {
        array[counter] = constant;
    }
}

// Fill with random numbers between -1 and 1
template< typename T>
void tensor<T>::set_random( int seed = -1 )
{
    if ( seed >= 0 )
        srand( seed );

    double fillValue = 0.0f;
    for( size_t counter = 0; counter < size; ++counter )
    {
        fillValue = rand()/double(RAND_MAX);
        array[counter] = static_cast< double >( fillValue ); // TODO type
    }
}

// In a matrix, set its columns as the DCT basis functions
template< typename T>
void tensor<T>::set_dct()
{
    assert(n_dims == 2);

    #pragma omp parallel for
    for( size_t row = 0; row < d[0]; ++row )
    {
        #pragma omp parallel for
        for( size_t col = 0; col < d[1]; ++col )
        {
            double c = sqrt(2.0/d[0]);
            if (col == 0)
                c = sqrt(1.0/d[0]);

            at(row,col) = c*cos((2.0*row+1)*col*M_PI/(2.0*d[0]));
        }
    }
}

// Copy the memory from a given buffer
template< typename T>
void tensor<T>::set_memory(const T* memory) // TODO: make it cast from any type
{
    std::copy(memory, memory + size, array);
//        memcpy(array,memory,size*sizeof(T));
}

// Set the values to form an N-dimensional gaussian bell
template< typename T>
void tensor<T>::set_gaussian(double sigma1, double sigma2, double sigma3) // GENERIC (1-3D)
{
    if (sigma2 <= 0) sigma2 = sigma1;
    if (sigma3 <= 0) sigma3 = sigma1;
    assert(sigma1 > 0);

    double center_row = (d[0]-1)/2.0;
    double center_col = (d[1]-1)/2.0;
    double center_slice = (d[2]-1)/2.0;

    #pragma omp parallel for
    for (size_t slice = 0; slice < d[2]; ++slice)
    {
        #pragma omp parallel for
        for (size_t row = 0; row < d[0]; ++row)
        {
            #pragma omp parallel for
            for (size_t col = 0; col < d[1]; ++col)
            {
                at(row, col, slice) = exp(
                        -(row-center_row)*(row-center_row)/(2.0*sigma1*sigma1)
                        -(col-center_col)*(col-center_col)/(2.0*sigma2*sigma2)
                        -(slice-center_slice)*(slice-center_slice)/(2.0*sigma3*sigma3)
                        );
            }
        }
    }
    *this /= sum();
}

// Approximation of the N-dimensional Laplacian operator
template< typename T>
void tensor<T>::set_laplacian() // GENERIC (1-3D). After http://en.wikipedia.org/wiki/Discrete_Laplace_operator#Implementation_in_Image_Processing
{
    for (size_t i = 0; i < n_dims; ++i)
    {
        assert(d[i] == 3);
    }

    for (size_t slice = 0; slice < d[2]; ++slice)
    {
        for (size_t row = 0; row < d[0]; ++row)
        {
            for (size_t col = 0; col < d[1]; ++col)
            {
                size_t n_central = int(row == 1) + int(col == 1) + int(slice == 1);

                if (n_central == n_dims) at(row,col,slice) = -2*int(n_dims);
                else if (n_central == n_dims-1) at(row,col,slice) = 1;
                else at(row,col,slice) = 0;
            }
        }
    }
}

}

#endif	/* TENSOR_INITIALIZERS_HPP */

