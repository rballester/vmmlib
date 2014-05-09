/* 
 * File:   tensor_properties.hpp
 * Author: rballester
 *
 * Created on May 9, 2014, 7:43 PM
 */

#ifndef TENSOR_PROPERTIES_HPP
#define	TENSOR_PROPERTIES_HPP

#include "tensor.hpp"

namespace vmml
{
    
template< typename T>
bool tensor<T>::is_symmetric() const // NON-GENERIC (2D)
{
    assert(n_dims == 2);
    assert(d[0] == d[1]);

    for (size_t col = 0; col < d[1]; ++col) {
        for (size_t row = col; row < d[0]; ++row) {
            if (at(row,col) != at(col,row))
                return false;
        }
    }
    return true;
}

// Usual L2 norm
template< typename T>
double tensor<T>::frobenius_norm() const
{
    double norm = 0.0f;
    for( size_t counter = 0; counter < size; ++counter )
    {
        norm += array[counter]*array[counter];
    }
    return sqrt(norm);
}

// Frobenius norm of the difference with another tensor
template< typename T>
double tensor<T>::frobenius_norm(const tensor<T>& other) const // GENERIC (1-3D)
{
    assert(d[0] == other.d[0]);
    assert(d[1] == other.d[1]);
    assert(d[2] == other.d[2]);

    double norm = 0.0f;
    for( size_t counter = 0; counter < size; ++counter )
    {
        norm += (array[counter] - other.array[counter])*(array[counter] - other.array[counter]);
    }
    return sqrt(norm);
}

// L1 norm
template< typename T>
double tensor<T>::manhattan_norm() const
{
    T norm = 0;
    for( size_t counter = 0; counter < size; ++counter )
    {
        T val = array[counter];
        if (val < 0) val *= -1;
        norm += val;
    }
    return norm;
}

// Relative error (between 0 and 1) of the given tensor compared with this one
template< typename T>
double tensor<T>::relative_error(const tensor<T>& other) const
{
    return frobenius_norm(other)/frobenius_norm();
}

// Peak Signal-to-Noise Ratio
template< typename T>
double tensor<T>::psnr(const tensor<T>& other, double max_amplitude) const // GENERIC (1-3D)
{
    assert(d[0] == other.d[0]);
    assert(d[1] == other.d[1]);
    assert(d[2] == other.d[2]);

    double mse = 0.0f;
    for (size_t counter = 0; counter < size; ++counter)
    {
        mse += (array[counter] - other.array[counter])*(array[counter] - other.array[counter]);
    }
    mse = mse/size;

    return 10*log10(max_amplitude*max_amplitude/mse);
}

template< typename T>
T tensor<T>::sum() const
{
    T result = 0;
    for (size_t counter = 0; counter < size; ++counter) {
        result += array[counter];
    }
    return result;
}

template< typename T>
T tensor<T>::maximum() const
{
    T result = std::numeric_limits<T>::min();
    for (size_t counter = 0; counter < size; ++counter) {
        result = std::max<float>(result,array[counter]);
    }
    return result;
}

template< typename T>
T tensor<T>::minimum() const
{
    T result = std::numeric_limits<T>::max();
    for (size_t counter = 0; counter < size; ++counter) {
        result = std::min<float>(result,array[counter]);
    }
    return result;
}

template< typename T>
double tensor<T>::mean() const
{
    return sum()/size;
}

template< typename T>
double tensor<T>::variance() const
{
    double val = 0.0;
    double sum_val = 0.0;
    double mean_val = mean();
    for (size_t counter = 0; counter < size; ++counter) {
        val = array[counter] - mean_val;
        val *= val;
        sum_val += val;
    }

    return double(sum_val/(size - 1));
}

template< typename T>
double tensor<T>::stdev() const
{
    return sqrt(variance());
}

}

#endif	/* TENSOR_PROPERTIES_HPP */

