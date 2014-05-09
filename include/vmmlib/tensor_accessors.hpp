/* 
 * File:   tensor_accessors.hpp
 * Author: rballester
 *
 * Created on May 9, 2014, 6:35 PM
 */

#ifndef TENSOR_ACCESSORS_HPP
#define	TENSOR_ACCESSORS_HPP

#include "tensor.hpp"

namespace vmml
{
 
template< typename T>
size_t tensor<T>::get_n_dims() const
{
    return n_dims;
}

template< typename T>
size_t tensor<T>::get_dim(size_t dim) const
{
    assert(dim < n_dims);

    return d[dim];
}

template< typename T>
size_t tensor<T>::get_size() const
{
    return size;
}

template< typename T>
T* tensor<T>::get_array()
{
    return array;
}

// Retrieve a subset of the tensor
template< typename T>
void tensor<T>::get_sub_tensor(tensor<T>& result, size_t row_offset, size_t col_offset, size_t slice_offset) const // GENERIC (1-3D)
{
    assert(row_offset >= 0 and row_offset + result.d[0] <= d[0]);
    assert(col_offset >= 0 and col_offset + result.d[1] <= d[1]);
    assert(slice_offset >= 0 and slice_offset + result.d[2] <= d[2]);

    #pragma omp parallel for
    for (size_t slice = 0; slice < result.d[2]; ++slice)
    {
        #pragma omp parallel for
        for (size_t row = 0; row < result.d[0]; ++row)
        {
            #pragma omp parallel for
            for (size_t col = 0; col < result.d[1]; ++col)
            {
                result.at(row, col, slice)
                        = at(row_offset + row, col_offset + col, slice_offset + slice);
            }
        }
    }
}

// In this variant, copying regions outside the domain does not cause an error; they are just set to 0 instead. Returns 1 if something was copied, 0 otherwise
template< typename T>
bool tensor<T>::get_sub_tensor_general(tensor<T>& result, int row_offset, int col_offset, int slice_offset) const // GENERIC (1-3D)
{
    result.set_zero();

    int src_lower_i = row_offset, src_upper_i = row_offset+result.d[0], dst_lower_i = 0, dst_upper_i = result.d[0];
    if (src_lower_i < 0) {
        dst_lower_i -= src_lower_i;
        src_lower_i = 0;
    }
    if (src_upper_i > int(d[0])) {
        dst_upper_i -= (src_upper_i - d[0]);
        src_upper_i = d[0];
    }

    int src_lower_j = col_offset, src_upper_j = col_offset+result.d[1], dst_lower_j = 0, dst_upper_j = result.d[1];
    if (src_lower_j < 0) {
        dst_lower_j -= src_lower_j;
        src_lower_j = 0;
    }
    if (src_upper_j > int(d[1])) {
        dst_upper_j -= (src_upper_j - d[1]);
        src_upper_j = d[1];
    }

    int src_lower_k = slice_offset, src_upper_k = slice_offset+result.d[2], dst_lower_k = 0, dst_upper_k = result.d[2];
    if (src_lower_k < 0) {
        dst_lower_k -= src_lower_k;
        src_lower_k = 0;
    }
    if (src_upper_k > int(d[2])) {
        dst_upper_k -= (src_upper_k - d[2]);
        src_upper_k = d[2];
    }

    #pragma omp parallel for
    for (int slice = dst_lower_k; slice < dst_upper_k; ++slice)
    {
        #pragma omp parallel for
        for (int row = dst_lower_i; row < dst_upper_i; ++row)
        {
            #pragma omp parallel for
            for (int col = dst_lower_j; col < dst_upper_j; ++col)
            {
                result.at(row,col,slice) = at(row - dst_lower_i + src_lower_i, col - dst_lower_j + src_lower_j, slice - dst_lower_k + src_lower_k);
            }
        }
    }
    return (dst_lower_i < dst_upper_i and dst_lower_j < dst_upper_j and dst_lower_k < dst_upper_k);
}

// Sets a region to be the contents of a given tensor
template< typename T>
void tensor<T>::set_sub_tensor(const tensor<T>& data, size_t row_offset, size_t col_offset, size_t slice_offset) // GENERIC (1-3D)
{
    assert(row_offset >= 0 and row_offset + data.d[0] <= d[0]);
    assert(col_offset >= 0 and col_offset + data.d[1] <= d[1]);
    assert(slice_offset >= 0 and slice_offset + data.d[2] <= d[2]);

    #pragma omp parallel for
    for (size_t slice = 0; slice < data.d[2]; ++slice)
    {
        #pragma omp parallel for
        for (size_t row = 0; row < data.d[0]; ++row)
        {
            #pragma omp parallel for
            for (size_t col = 0; col < data.d[1]; ++col)
            {
                at(row_offset + row, col_offset + col, slice_offset + slice)
                    = data.at(row, col, slice);
            }
        }
    }
}

}

#endif	/* TENSOR_ACCESSORS_HPP */

