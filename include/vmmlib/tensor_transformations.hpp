/* 
 * File:   tensor_transformations.hpp
 * Author: rballester
 *
 * Created on May 9, 2014, 7:02 PM
 */

#ifndef TENSOR_TRANSFORMATIONS_HPP
#define	TENSOR_TRANSFORMATIONS_HPP

#include "tensor.hpp"

#include <fftw3.h>
#include "lapack/detail/f2c.h"
#include "lapack/detail/clapack.h"

#undef min // Hack to undo the effects of an evil header (http://stackoverflow.com/questions/518517/macro-max-requires-2-arguments-but-only-1-given)
#undef max

namespace vmml
{

struct svd_params // Used for LAPACK operations on singular vectors
{
    char            jobu;
    char            jobvt;
    integer         m;
    integer         n;
    float_t*        a;
    integer         lda;
    float_t*        s;
    float_t*        u;
    integer         ldu;
    float_t*        vt;
    integer         ldvt;
    float_t*        work;
    integer         lwork;
    integer         info;
};

struct eigs_params // Used for LAPACK operations on eigenvectors
{
    char            jobz;
    char            range;
    char            uplo;
    integer         n;
    float_t*        a;
    integer         lda; //leading dimension of input array
    float_t*        vl;
    float_t*        vu;
    integer         il;
    integer         iu;
    float_t         abstol;
    integer         m; //number of found eigenvalues
    float_t*        w; //first m eigenvalues
    float_t*        z; //first m eigenvectors
    integer         ldz; //leading dimension of z
    float_t*        work;
    integer         lwork;
    integer*        iwork;
    integer*        ifail;
    integer         info;
};

// Matrix transposition
template< typename T>
tensor<T> tensor<T>::transpose() const
{
    assert(n_dims == 2);

    tensor<T> result(d[1],d[0]);
    #pragma omp parallel for
    for( size_t row = 0; row < d[0]; ++row )
    {
        #pragma omp parallel for
        for( size_t col = 0; col < d[1]; ++col )
        {
            result.at(col,row) = at(row,col);
        }
    }
    return result;
}

// Removes redundant dimensions. E.g. a tensor of size 1 x 4 x 1 x 6 will become 4 x 6
template< typename T>
void tensor<T>::squeeze()
{
    size_t d_tmp[n_dims];
    for (size_t i = 0; i < n_dims; ++i)
        d_tmp[i] = d[i];
    int counter = 0;
    for (size_t i = 0; i < n_dims; ++i) {
        if (d_tmp[i] > 1) {
            d[counter] = d_tmp[i];
            ++counter;
        }
    }

    for (size_t i = counter; i < n_dims; ++i)
        d[i] = 1;

    n_dims = counter;
}

// Embeds a tensor of dimension d into R^{d+1}. pos indicates in which position the new coordinate goes
template< typename T>
void tensor<T>::embed(size_t pos) // NON-GENERIC (2D)
{
    assert(n_dims == 2);
    assert(pos <= n_dims);

    n_dims++;

    if (pos == 0) {
        d[2] = d[1]; d[1] = d[0]; d[0] = 1;
    }
    else if (pos == 1) {
        d[2] = d[1]; d[1] = 1;
    }
}

// Removes the pos-th dimension of a tensor, assuming it has size 1 along it
template< typename T>
void tensor<T>::unembed(size_t pos) // NON-GENERIC (3D)
{
    assert(n_dims == 3);
    assert(pos < n_dims);
    assert(d[pos] == 1);

    n_dims--;

    if (pos == 0) {
        d[0] = d[1]; d[1] = d[2]; d[2] = 1;
    }
    else if (pos == 1) {
        d[1] = d[2]; d[2] = 1;
    }
}

// Downsamples (by integer factors)
template< typename T>
tensor<T> tensor<T>::downsample(size_t factor1, size_t factor2, size_t factor3) const // GENERIC (1-3D)
{
    // TODO Provisional version. the factors shouldn't be necessarily a divisor of the dimension sizes
    assert(d[0]%factor1 == 0);
    assert(d[1]%factor2 == 0);
    assert(d[2]%factor3 == 0);

    tensor<T> result(d[0]/factor1,d[1]/factor2,d[2]/factor3);
    result.n_dims = n_dims;
    size_t size_reduction_factor = factor1*factor2*factor3;

    #pragma omp parallel for
    for (size_t dst_slice = 0; dst_slice < d[2]/factor3; ++dst_slice) {
        #pragma omp parallel for
        for (size_t dst_row = 0; dst_row < d[0]/factor1; ++dst_row) {
            #pragma omp parallel for
            for (size_t dst_col = 0; dst_col < d[1]/factor2; ++dst_col) {

                T sum = 0;
                for (size_t src_slice = dst_slice*factor3; src_slice < (dst_slice+1)*factor3; ++src_slice) {
                    for (size_t src_row = dst_row*factor1; src_row < (dst_row+1)*factor1; ++src_row) {
                        for (size_t src_col = dst_col*factor2; src_col < (dst_col+1)*factor2; ++src_col) {
                            sum += at(src_row,src_col,src_slice);
                        }
                    }
                }
                result.at(dst_row,dst_col,dst_slice) = sum/size_reduction_factor;
            }
        }
    }
//        tensor<float> gaussian(n*10); // Gaussian version. TODO: probably slower than strictly necessary!
//        gaussian.set_gaussian(n);
//        tensor<float> copy(*this);
//        copy = copy.convolve(gaussian);
//        for (size_t col = 0; col < d[1]; ++col) {
//            for (size_t bucket = 0; bucket < n_buckets; ++bucket) {
//                result.at(bucket,col) = copy.at(int(bucket*n+n/2),col);
//            }
//        }
    return result;
}

// Changes the tensor size. Truncates (or pads with zeros) as necessary to fit the new dimensions
template< typename T>
tensor<T> tensor<T>::resize(size_t d0, size_t d1, size_t d2) const // GENERIC (1-3D)
{
    assert(d0 > 0 and d1 > 0 and d2 > 0);

    tensor<T> result(d0,d1,d2);
    get_sub_tensor_general(result,0,0,0);
    return result;
}

// Approximation of the norm of the gradient, after http://en.wikipedia.org/wiki/Sobel_operator
template< typename T>
tensor<T> tensor<T>::sobel_transformation() // NON-GENERIC
{
    assert(n_dims == 3);

    tensor<float> result(d[0],d[1],d[2]);
    result.set_zero();

    tensor<float> h(3), h_prima(3);
    h.at(0) = 1; h.at(1) = 2; h.at(2) = 1;
    h_prima.at(0) = 1; h_prima.at(1) = 0; h_prima.at(2) = -1;

    tensor<float> partial_sum(d[0],d[1],d[2]);
    tensor<float> sobel(3,3,3);
    tensor<float> lambdas(1);
    lambdas.at(0) = 1;

    tensor<float> U1(3,1), U2(3,1), U3(3,1);

    U1.set_sub_tensor(h_prima,0,0); U2.set_sub_tensor(h,0,0); U3.set_sub_tensor(h,0,0);
    sobel.reconstruct_cp(lambdas,U1,U2,U3);
    partial_sum = convolve(sobel);
    partial_sum.power(2);
    result += partial_sum;

    U1.set_sub_tensor(h,0,0); U2.set_sub_tensor(h_prima,0,0); U3.set_sub_tensor(h,0,0);
    sobel.reconstruct_cp(lambdas,U1,U2,U3);
    partial_sum = convolve(sobel);
    partial_sum.power(2);
    result += partial_sum;

    U1.set_sub_tensor(h,0,0); U2.set_sub_tensor(h,0,0); U3.set_sub_tensor(h_prima,0,0);
    sobel.reconstruct_cp(lambdas,U1,U2,U3);
    partial_sum = convolve(sobel);
    partial_sum.power(2);
    result += partial_sum;

    result.power(0.5);
    return result;
}

// Columnwise 1D DCT. See http://octave.1599824.n4.nabble.com/How-to-use-FFTW3-DCT-IDCT-to-match-that-of-Octave-td4633005.html
// TODO only works for float
template< typename T>
void tensor<T>::dct() // GENERIC (1-3D)
{
    float *in = (float*)fftwf_malloc(sizeof(float) * d[0]);    
    float *out = (float*)fftwf_malloc(sizeof(float) * d[0]);

    /* create plan */
    fftwf_plan p = fftwf_plan_r2r_1d(d[0], in, out, FFTW_REDFT10, FFTW_MEASURE);

    for (size_t slice = 0; slice < d[2]; ++slice) {
        for (size_t col = 0; col < d[1]; ++col) {

            // Copy into buffer
            for (size_t row = 0; row < d[0]; ++row) {
                in[row] = at(row,col,slice);
            }

            /* execute plan */
            fftwf_execute(p);

            // Retrieve buffer contents, rescaling
            at(0,col,slice) = out[0]/4;
            for (size_t row = 1; row < d[0]; ++row) {
                at(row,col,slice) = out[row]*sqrt(2)/4;
            }
        }
    }

    /* free resources */
    fftwf_destroy_plan(p);
    fftwf_free(in);
    fftwf_free(out);
}

// Column-wise inverse DCT
// TODO only works for float
template< typename T>
void tensor<T>::idct() // GENERIC (1-3D)
{
    float *in = (float*)fftwf_malloc(sizeof(float) * d[0]);    
    float *out = (float*)fftwf_malloc(sizeof(float) * d[0]);

    /* create plan */
    fftwf_plan p = fftwf_plan_r2r_1d(d[0], in, out, FFTW_REDFT01, FFTW_ESTIMATE);

    for (size_t slice = 0; slice < d[2]; ++slice) {
        for (size_t col = 0; col < d[1]; ++col) {

            // Copy into buffer, scaling
            in[0] = at(0,col,slice)*4;
            for (size_t row = 1; row < d[0]; ++row) {
                in[row] = at(row,col,slice)/(sqrt(2)/4);
            }

            /* execute plan */
            fftwf_execute(p);

            // Retrieve buffer contents, rescaling
            for (size_t row = 0; row < d[0]; ++row) {
                at(row,col,slice) = out[row]/(2*d[0]);
            }
        }
    }

    fftwf_destroy_plan(p);
    fftwf_free(in);
    fftwf_free(out);
}

// Returns the Summed Area Table (every point is the integral on the rectangular region defined by that point and the origin). Used e.g. to quickly compute histograms
template< typename T>
tensor<T> tensor<T>::summed_area_table() const // GENERIC (1-3D)
{
    assert(n_dims == 3);

    tensor<float> result;
    result.init(n_dims,d[0],d[1],d[2]);
    result.set_zero(); // TODO needed?

    for (size_t slice = 0; slice < d[2]; ++slice) {
        for (size_t col = 0; col < d[1]; ++col) {
            for (size_t row = 0; row < d[0]; ++row) {
                result.at(row,col,slice) = at(row,col,slice) + result.at_general(row-1,col,slice) + result.at_general(row,col-1,slice) + result.at_general(row,col,slice-1) - result.at_general(row-1,col-1,slice) - result.at_general(row-1,col,slice-1) - result.at_general(row,col-1,slice-1) + result.at_general(row-1,col-1,slice-1);
            }
        }
    }
    return result;
}

// Full Singular Value Decomposition of a matrix
template< typename T>
bool tensor<T>::svd(tensor<float>& U, tensor<float>& S, tensor<float>& Vt) const // TODO: right now, only for T = float
{
    assert(n_dims == 2);
    assert(U.d[0] == d[0] and U.d[1] == std::min<float>(d[0],d[1]));
    assert(S.d[0] == std::min<float>(d[0],d[1]));
    assert(Vt.d[0] == std::min<float>(d[0],d[1]) and Vt.d[1] == d[1]);

    svd_params p;

    // Workspace query (used to know the necessary buffer size)
    p.jobu      = 'S';
    p.jobvt     = 'S';
    p.m         = d[0];
    p.n         = d[1];
    p.a         = NULL;
    p.lda       = d[0];
    p.s         = NULL;
    p.u         = NULL;
    p.ldu       = d[0];
    p.vt        = NULL;
    p.ldvt      = std::min<float>(d[0],d[1]);
    p.work      = new float_t;
    p.lwork     = -1;
    p.info      = 0;

    sgesvd_(&p.jobu,&p.jobvt,&p.m,&p.n,p.a,&p.lda,p.s,p.u,&p.ldu,p.vt,&p.ldvt,p.work,&p.lwork,&p.info);
    p.lwork = static_cast<integer>( p.work[0] );
    delete p.work;
    p.work = new float_t[ p.lwork ];

    // Real query
    p.a = new T[size]; // Lapack destroys the contents of the input matrix
    memcpy(p.a,array,size*sizeof(T));
    p.u = U.array;
    p.s = S.array;
    p.vt = Vt.array;

    sgesvd_(&p.jobu,&p.jobvt,&p.m,&p.n,p.a,&p.lda,p.s,p.u,&p.ldu,p.vt,&p.ldvt,p.work,&p.lwork,&p.info);

    delete[] p.a;
    delete[] p.work;

    return p.info == 0;
}

// Returns true if the computation went OK, false otherwise
template< typename T>
bool tensor<T>::left_singular_vectors(tensor<float>& U) const // TODO: right now, only for T = float
{
    assert(n_dims == 2);
    assert(U.d[0] == d[0] and U.d[1] == std::min<float>(d[0],d[1])); // TODO: do it for a variable number of columns of U

    svd_params p;

    // Workspace query (used to know the necessary buffer size)
    p.jobu      = 'S';
    p.jobvt     = 'N';
    p.m         = d[0];
    p.n         = d[1];
    p.a         = NULL;
    p.lda       = d[0];
    p.s         = NULL;
    p.u         = NULL;
    p.ldu       = d[0];
    p.vt        = NULL;
    p.ldvt      = 1;
    p.work      = new float_t;
    p.lwork     = -1;
    p.info      = 0;

    sgesvd_(&p.jobu,&p.jobvt,&p.m,&p.n,p.a,&p.lda,p.s,p.u,&p.ldu,p.vt,&p.ldvt,p.work,&p.lwork,&p.info);
    p.lwork = static_cast<integer>( p.work[0] );

    delete p.work;
    p.work = new float_t[ p.lwork ];

    // Real query
    p.a = new T[size]; // Lapack destroys the contents of the input matrix
    memcpy(p.a,array,size*sizeof(T));
    p.s = new T[std::min(d[0],d[1])];
    p.u = U.array;

    sgesvd_(&p.jobu,&p.jobvt,&p.m,&p.n,p.a,&p.lda,p.s,p.u,&p.ldu,p.vt,&p.ldvt,p.work,&p.lwork,&p.info);
    delete[] p.a;
    delete[] p.s;
    delete[] p.work;

    return p.info == 0;
}

// Set the tensor to its left singular vectors
template< typename T>
bool tensor<T>::reorthogonalize()
{
    tensor<T> copy(*this);
    bool result = left_singular_vectors(copy);
    *this = copy;
    return result;
}

// Sets U to the leading eigenvectors of the current matrix
template< typename T>
bool tensor<T>::symmetric_eigenvectors(tensor<float>& U) const // TODO: only works for T = float
{
    assert(n_dims == 2);
    assert(is_symmetric());
    assert(d[0] == U.d[0]);
    assert(d[1] >= U.d[1]);

    // (1) get all eigenvalues and eigenvectors
//        evectors_type* all_eigvectors = new evectors_type;
//        evalues_type* all_eigvalues = new evalues_type;	

    eigs_params p;

    // Workspace query (used to know the necessary buffer size)
    p.jobz      = 'V'; // Compute eigenvalues and eigenvectors.
    p.range     = 'I'; // the U.d[1] most significant eigenvectors will be found
    p.uplo      = 'U'; // Upper triangle of A is stored; or Lower triangle of A is stored.
    p.n         = d[0];
    p.a         = NULL; // Empty (workspace query)
    p.lda       = d[0];
    p.vl        = 0; //Not referenced if RANGE = 'A' or 'I'.
    p.vu        = 0; //Not referenced if RANGE = 'A' or 'I'.	 
    p.il        = d[1]-U.d[1]+1; //Not referenced if RANGE = 'A' or 'V'.
    p.iu        = d[1]; //Not referenced if RANGE = 'A' or 'V'.	
    p.abstol    = 0.000001; //lie in an interval [a,b] of width less than or equal to ABSTOL + EPS *   max( |a|,|b| )
    p.m         = U.d[1]; //The total number of eigenvalues found.  0 <= M <= N.
    p.w         = NULL; // Empty (workspace query)
    p.z         = NULL; // Empty (workspace query)
    p.ldz       = d[0]; // The leading dimension of the array Z.  LDZ >= 1, and if JOBZ = 'V', LDZ >= max(1,N).
    p.work      = new float_t;
    //FIXME: check if correct datatype
    p.iwork     = new integer[5*d[0]]; //[5*N]; // INTEGER array, dimension (5*N)
    p.ifail     = new integer[d[0]]; //[N];
    p.lwork     = -1; //8N

    ssyevx_(&p.jobz,&p.range,&p.uplo,&p.n,p.a,&p.lda,p.vl,p.vu,&p.il,&p.iu,&p.abstol,&p.m,p.w,p.z,&p.ldz,p.work,&p.lwork,p.iwork,p.ifail,&p.info);

    p.lwork = static_cast< integer >( p.work[0] );
    delete p.work;

    p.work = new float_t[ p.lwork ];

    // Real query        
    p.a = new T[size]; // Input matrix
    memcpy(p.a,array,size*sizeof(T));
    p.w = new T[size]; // First m eigenvalues
    p.z = U.array; // First m eigenvectors

    ssyevx_(&p.jobz,&p.range,&p.uplo,&p.n,p.a,&p.lda,p.vl,p.vu,&p.il,&p.iu,&p.abstol,&p.m,p.w,p.z,&p.ldz,p.work,&p.lwork,p.iwork,p.ifail,&p.info);

    // The eigenvectors are sorted in ascending order. We change the order now
//        tensor<T> tmp(d[0]);
//        U.get_sub_tensor(tmp,0,0);
    for (size_t col = 0; col < U.d[1]/2; ++col) {
        for (size_t row = 0; row < U.d[0]; ++row) {
            T tmp = U.at(row,col);
            U.at(row,col) = U.at(row,U.d[1]-col-1);
            U.at(row,U.d[1]-col-1) = tmp;
        }
    }

    delete p.work;
    delete p.iwork;
    delete p.ifail;
    delete p.a;
    delete p.w;

    return p.info == 0;
}

template< typename T>
const void tensor<T>::tucker_decomposition(tensor<T>& core, tensor<T>& U1, tensor<T>& U2, tensor<T>& U3, size_t max_iters, double tol) const // NON-GENERIC (3D)
{
    assert(n_dims == 3);
    assert(core.n_dims == 3);
    assert(U1.n_dims == 2);
    assert(U2.n_dims == 2);
    assert(U3.n_dims == 2);
    assert(d[0] == U1.d[0]);
    assert(d[1] == U2.d[0]);
    assert(d[2] == U3.d[0]);
    assert(U1.d[0] >= U1.d[1]);
    assert(U2.d[0] >= U2.d[1]);
    assert(U3.d[0] >= U3.d[1]);
    assert(core.d[0] == U1.d[1]);
    assert(core.d[1] == U2.d[1]);
    assert(core.d[2] == U3.d[1]);

    double error_old = 0;
    double error = std::numeric_limits<double>::max();

    // TODO preallocate already here? Or immediately before they are needed?
    tensor<float> unfolding1(d[0],core.d[1]*core.d[2]);
    tensor<float> unfolding2(d[1],core.d[0]*core.d[2]);
    tensor<float> unfolding3(d[2],core.d[0]*core.d[1]);

    double target_frobenius_norm = frobenius_norm();

    for (size_t iteration = 0; iteration < max_iters and abs(error_old - error) > tol; ++iteration) {

        error_old = error;

        tensor<T> projection = ttm2(U2.transpose()).ttm3(U3.transpose());
        for (size_t slice = 0; slice < projection.d[2]; ++slice) {
            for (size_t col = 0; col < projection.d[1]; ++col) {
                for (size_t row = 0; row < projection.d[0]; ++row) {
                    unfolding1.at(row,slice*projection.d[1] + col) = projection.at(row,col,slice);
                }
            }
        }
        unfolding1.covariance().symmetric_eigenvectors(U1);

        projection = ttm1(U1.transpose()).ttm3(U3.transpose());
        for (size_t slice = 0; slice < projection.d[2]; ++slice) {
            for (size_t col = 0; col < projection.d[1]; ++col) {
                for (size_t row = 0; row < projection.d[0]; ++row) {
                    unfolding2.at(col,row*projection.d[2] + slice) = projection.at(row,col,slice);
                }
            }
        }
        unfolding2.covariance().symmetric_eigenvectors(U2);

        projection = ttm1(U1.transpose()).ttm2(U2.transpose());
        for (size_t slice = 0; slice < projection.d[2]; ++slice) {
            for (size_t col = 0; col < projection.d[1]; ++col) {
                for (size_t row = 0; row < projection.d[0]; ++row) {
                    unfolding3.at(slice,col*projection.d[0] + row) = projection.at(row,col,slice);
                }
            }
        }
        unfolding3.covariance().symmetric_eigenvectors(U3);

        core = projection.ttm3(U3.transpose());
        double achieved_frobenius_norm = core.frobenius_norm();
        error = sqrt(target_frobenius_norm*target_frobenius_norm - achieved_frobenius_norm*achieved_frobenius_norm)/target_frobenius_norm;
#ifdef VMMLIB_VERBOSE
        std::cout << "Iteration = " << iteration << ", error = " << error << std::endl;
#endif
    }
}
    
}

#endif	/* TENSOR_TRANSFORMATIONS_HPP */
