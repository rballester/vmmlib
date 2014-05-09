/* 
 * File:   tensor_tensor_operations.hpp
 * Author: rballester
 *
 * Created on May 9, 2014, 7:12 PM
 */

#ifndef TENSOR_TENSOR_OPERATIONS_HPP
#define	TENSOR_TENSOR_OPERATIONS_HPP

#include "tensor.hpp"

#include <cblas.h>

namespace vmml
{
    
struct sgemm_params // Used for BLAS operations (matrix-matrix product)
{
    CBLAS_ORDER     order;
    CBLAS_TRANSPOSE trans_a;
    CBLAS_TRANSPOSE trans_b;
    integer  		m;
    integer 		n;
    integer 		k;
    float_t			alpha;
    float_t*        a;
    integer        lda; //leading dimension of input array matrix left
    float_t*        b;
    integer         ldb; //leading dimension of input array matrix right
    float_t			beta;
    float_t*        c;
    integer        ldc; //leading dimension of output array matrix right
};
    
// Tensor-tensor convolution. The result has the same size as the original
template< typename T>
tensor<T> tensor<T>::convolve( const tensor& kernel ) const // GENERIC (1-3D)
{
    tensor<float> result(*this); // TODO unnecessary copy

    #pragma omp parallel for
    for( int slice = 0; slice < int(d[2]); ++slice )
    {
        #pragma omp parallel for
        for( int row = 0; row < int(d[0]); ++row )
        {
            #pragma omp parallel for
            for( int col = 0; col < int(d[1]); ++col )
            {
                T sum_ = 0;

                for( int k = 0; k < int(kernel.d[2]); ++k )
                {
                    int src_slice = slice - int(kernel.d[2]) / 2 + k;

                    for( int i = 0; i < int(kernel.d[0]); ++i )
                    {
                        int src_row = row - int(kernel.d[0]) / 2 + i;

                        for( int j = 0; j < int(kernel.d[1]); ++j )
                        {
                            int src_col = col - int(kernel.d[1]) / 2 + j;

                            if( src_row >= 0 and src_row < int(d[0]) and src_col >= 0 and src_col < int(d[1]) and src_slice >= 0 and src_slice < int(d[2])) {
                                sum_ += at( src_row, src_col, src_slice ) * kernel.at( i, j, k);
                            }
                        }
                    }
                }
                result.at( row, col, slice) = sum_;
            }
        }
    }
    return result;
}

// Regular matrix times matrix product. Also vectors are supported
template< typename T>
tensor<T> tensor<T>::mtm(const tensor<T>& factor) const
{
    assert(n_dims <= 2);
    assert(factor.n_dims <= 2);
    assert(d[1] == factor.d[0]);

    tensor<T> result(d[0],factor.d[1]);

    sgemm_params p;

    p.order      = CblasColMajor; //
    p.trans_a    = CblasNoTrans;
    p.trans_b    = CblasNoTrans;
    p.m          = d[0];
    p.n          = factor.d[1];
    p.k          = d[1];
    p.alpha      = 1;
    p.a          = 0;
    p.lda        = d[0];
    p.b          = 0;
    p.ldb        = d[1]; //no transpose
    p.beta       = 0;
    p.c          = 0;
    p.ldc        = d[0];

    // blas needs non-const data
    tensor<T> A_copy(*this);
    tensor<T> B_copy(factor);

    p.a         = A_copy.get_array();
    p.b         = B_copy.get_array();
    p.c         = result.get_array();

    cblas_sgemm(p.order,p.trans_a,p.trans_b,p.m,p.n,p.k,p.alpha,p.a,p.lda,p.b,p.ldb,p.beta,p.c,p.ldc);

    return result;
}

// Returns A*A^T
template< typename T>
tensor<T> tensor<T>::covariance() const
{
    assert(n_dims == 2);

    tensor<T> result(d[0],d[0]);

    sgemm_params p;

    p.order      = CblasColMajor; //
    p.trans_a    = CblasNoTrans;
    p.trans_b    = CblasTrans;
    p.m          = d[0];
    p.n          = d[0];
    p.k          = d[1];
    p.alpha      = 1;
    p.a          = 0;
    p.lda        = d[0];
    p.b          = 0;
    p.ldb        = d[0];
    p.beta       = 0;
    p.c          = 0;
    p.ldc        = d[0];

    // blas needs non-const data
    tensor<T> A_copy(*this);

    p.a         = A_copy.get_array();
    p.b         = A_copy.get_array();
    p.c         = result.get_array();

    cblas_sgemm(p.order,p.trans_a,p.trans_b,p.m,p.n,p.k,p.alpha,p.a,p.lda,p.b,p.ldb,p.beta,p.c,p.ldc);

    return result;
}

// Reconstructs a tensor from its CP (aka canonical, aka CANDECOMP/PARAFAC) components
template< typename T>
void tensor<T>::reconstruct_cp(const tensor<T>& lambdas, const tensor<T>& U1, const tensor<T>& U2, const tensor<T>& U3) // NON-GENERIC
{
    assert(n_dims == 3);
    assert(lambdas.n_dims == 1);
    assert(U1.n_dims == 2);
    assert(U2.n_dims == 2);
    assert(U3.n_dims == 2);
    assert(lambdas.d[0] == U1.d[1]);
    assert(lambdas.d[0] == U2.d[1]);
    assert(lambdas.d[0] == U3.d[1]);
    assert(d[0] == U1.d[0] and d[1] == U2.d[0] and d[2] == U3.d[0]);

    set_zero();

    for (size_t lambda = 0; lambda < lambdas.d[0]; ++lambda) {
        #pragma omp parallel for
        for (size_t slice = 0; slice < d[2]; ++slice) {
            #pragma omp parallel for
            for (size_t row = 0; row < d[0]; ++row) {
                #pragma omp parallel for
                for (size_t col = 0; col < d[1]; ++col) {
                    at(row,col,slice) += lambdas.at(lambda)*U1.at(row,lambda)*U2.at(col,lambda)*U3.at(slice,lambda);
                }
            }
        }
    }
}

// Full tensor-times-matrix, along every mode
template< typename T>
tensor<T> tensor<T>::ttm(const tensor<T>& matrix1, const tensor<T>& matrix2, const tensor<T>& matrix3) const // NON-GENERIC
{
    return ttm1(matrix1).ttm2(matrix2).ttm3(matrix3);
}

template< typename T>
tensor<T> tensor<T>::ttm1(const tensor<T>& matrix) const // NON-GENERIC
{
    assert(n_dims == 3);
    assert(matrix.n_dims == 2);
    assert(matrix.d[1] == d[0]);

    tensor result(matrix.d[0],d[1],d[2]);

    #pragma omp parallel for
    for (size_t slice = 0; slice < d[2]; ++slice)
    {
        tensor<float> factor(d[0],d[1],1);
        get_sub_tensor(factor,0,0,slice);
        factor.unembed(2);
        tensor<float> product = matrix.mtm(factor);
        product.embed(2);
        result.set_sub_tensor(product,0,0,slice);
    }
    return result;
}

template< typename T>
tensor<T> tensor<T>::ttm2(const tensor& matrix) const // NON-GENERIC
{
    assert(n_dims == 3);
    assert(matrix.n_dims == 2);
    assert(matrix.d[1] == d[1]);

    tensor result(d[0],matrix.d[0],d[2]);

    #pragma omp parallel for
    for (size_t row = 0; row < d[0]; ++row)
    {            
        tensor<float> factor(1,d[1],d[2]);
        get_sub_tensor(factor,row,0,0);
        factor.unembed(0);
        tensor<float> product = matrix.mtm(factor);
        product.embed(0);
        result.set_sub_tensor(product,row,0,0);
    }
    return result;
}

template< typename T>
tensor<T> tensor<T>::ttm3(const tensor& matrix) const // NON-GENERIC
{
    assert(n_dims == 3);
    assert(matrix.n_dims == 2);
    assert(matrix.d[1] == d[2]);

    tensor result(d[0],d[1],matrix.d[0]);

    #pragma omp parallel for
    for (size_t col = 0; col < d[1]; ++col)
    {
        tensor<float> factor(d[0],1,d[2]);
        get_sub_tensor(factor,0,col,0);
        factor.unembed(1);
        factor = factor.transpose();
        tensor<float> product = matrix.mtm(factor);
        product = product.transpose();
        product.embed(1);
        result.set_sub_tensor(product,0,col,0);
    }
    return result;
}
    
}

#endif	/* TENSOR_TENSOR_OPERATIONS_HPP */

