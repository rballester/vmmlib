/* 
 * File:   tensor_scalar_operations.hpp
 * Author: rballester
 *
 * Created on May 9, 2014, 6:54 PM
 */

#ifndef TENSOR_SCALAR_OPERATIONS_HPP
#define	TENSOR_SCALAR_OPERATIONS_HPP

#include "tensor.hpp"

namespace vmml
{
    
// Checks if the tensor is equal (up to some tolerance) to other tensor
template< typename T>
const bool tensor<T>::equals(const tensor<T> & other, T tol) // GENERIC (1-3D)
{
    assert(n_dims == other.n_dims);
    assert(d[0] == other.d[0]);
    assert(d[1] == other.d[1]);
    assert(d[2] == other.d[2]);

    for (size_t counter = 0; counter < size; ++counter)
    {
        if (abs(array[counter] - other.array[counter]) > tol)
            return false;
    }
    return true;
}

template< typename T>
const tensor<T>& tensor<T>::operator=( const tensor<T>& other) // TODO Can it be shortened using the copy constructor?
{
    assert(this != &other);
    n_dims = other.n_dims;
    d[0] = other.d[0];
    d[1] = other.d[1];
    d[2] = other.d[2];
    size = other.size;
    delete[] array;
    array = new T[other.size];
    memcpy(array, other.array, size*sizeof(T));
    return *this;
}

template< typename T>
void tensor<T>::operator+=( const tensor<T>& other ) // GENERIC (1-3D)
{
    assert(n_dims == other.n_dims);
    assert(d[0] == other.d[0] and d[1] == other.d[1] and d[2] == other.d[2]);

    #pragma omp parallel for
    for (size_t counter = 0; counter < size; ++counter)
    {
        array[counter] += other.array[counter];
    }
}

template< typename T>
void tensor<T>::operator-=( const tensor<T>& other ) // GENERIC (1-3D)
{
    assert(n_dims == other.n_dims);
    assert(d[0] == other.d[0] and d[1] == other.d[1] and d[2] == other.d[2]);

    #pragma omp parallel for
    for (size_t counter = 0; counter < size; ++counter)
    {
        array[counter] -= other.array[counter];
    }
}

template< typename T>
tensor<T> tensor<T>::operator*(T scalar)
{
    tensor result(*this);

    #pragma omp parallel for
    for (size_t counter = 0; counter < size; ++counter)
    {
        result.array[counter] = array[counter]*scalar;
    }
    return result;
}

template< typename T>
void tensor<T>::operator*=(T scalar)
{
    #pragma omp parallel for
    for (size_t counter = 0; counter < size; ++counter)
    {
        array[counter] *= scalar;
    }
}

template< typename T>
tensor<T> tensor<T>::operator/(T scalar)
{
    tensor result(*this);

    #pragma omp parallel for
    for (size_t counter = 0; counter < size; ++counter)
    {
        result.array[counter] = array[counter]/scalar;
    }
    return result;
}

template< typename T>
void tensor<T>::operator/=(T scalar)
{
    #pragma omp parallel for
    for (size_t counter = 0; counter < size; ++counter)
    {
        array[counter] /= scalar;
    }
}

template< typename T>
void tensor<T>::power(double exponent)
{
    #pragma omp parallel for
    for (size_t counter = 0; counter < size; ++counter)
    {
        if (exponent < 1)
            assert(array[counter] >= -FLT_EPSILON);     // TODO: make epsilon dependent on type of array

        array[counter] = pow(array[counter],exponent);
    }
}
    
}

#endif	/* TENSOR_SCALAR_OPERATIONS_HPP */

