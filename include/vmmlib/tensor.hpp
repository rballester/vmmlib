/* 
 * Author: rballester
 *
 * Created on March 14, 2014, 10:43 AM
 */

// TODO: make asserts optional
// TODO: switch memcpy's to std::copy
// TODO: split in several files
// TODO: make a function out of the unfolding
// TODO: make linear traversals (array[counter]) during scalar operations with OpenMP

#ifndef TENSOR_HPP
#define TENSOR_HPP

#pragma GCC optimize 3      // TEST

#include <vector>
#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <cassert>
#include <cfloat>
#include <cmath>
#include <iostream>
#include <cstring>
#include <sys/mman.h>
#include <unistd.h>
#include <fcntl.h>
#include <algorithm>
#include <limits>

#include <include/vmmlib/lapack/detail/f2c.h>
#include <include/vmmlib/lapack/detail/clapack.h>
#include <cblas.h>
#include <fftw3.h>

#include <omp.h>
#undef min // Hack to undo the effects of some evil header (http://stackoverflow.com/questions/518517/macro-max-requires-2-arguments-but-only-1-given)
#undef max

#define SAFE_ACCESSORS 0

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

template< typename T = float >
class tensor
{
private:

    size_t n_dims;
    size_t d[3];
    size_t size;
    T* array;
    
    void* mmapped_data;
    int fd;
    
    void init(size_t n_dims_, size_t d0, size_t d1, size_t d2) // Only for internal constructor use
    {
        n_dims = n_dims_;
        d[0] = d0;
        d[1] = d1;
        d[2] = d2;
        size = d0*d1*d2;
        array = new T[size];
    }
    
public:

    tensor()
    {
        init(0,0,0,0);
    }
    
    tensor(size_t d0)
    {   
        init(1,d0,1,1);
    }
    
    tensor(size_t d0, size_t d1)
    {   
        init(2,d0,d1,1);
    }
    
    tensor(size_t d0, size_t d1, size_t d2)
    {   
        init(3,d0,d1,d2);
    }
    
    tensor(const tensor<T>& other) { // GENERIC (1-3D)
        
        n_dims = other.get_n_dims();
        
        for (int i = 0; i < n_dims; i++)
            d[i] = other.get_dim(i);
        for (int i = n_dims; i < 3; i++)
            d[i] = 1;

        size = other.get_size();
        array = new T[other.get_size()];

        T* other_array = other.get_array();
        for( size_t counter = 0; counter < size; ++counter ) {
//            array[counter] = static_cast<T> (other_array[counter]);
            array[counter] = other_array[counter];
        }
    }
    
    ~tensor() {
        delete[] array;
    }

/* Initializers ***************************************************************/
    
    // Fill with zeros
    void set_zero()
    {
        memset(array,0,size*sizeof(T));
    }
    
    // Fill with random numbers between -1 and 1
    void set_random( int seed = -1 )
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
    void set_dct()
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
    void set_memory(const T* memory) // TODO: make it cast from any type
    {
        std::copy(memory, memory + size, array);
//        memcpy(array,memory,size*sizeof(T));
    }
    
    // Set the values to form an N-dimensional gaussian bell
    void set_gaussian(double sigma1, double sigma2 = 0, double sigma3 = 0) // GENERIC (1-3D)
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
    void set_laplacian() // GENERIC (1-3D). After http://en.wikipedia.org/wiki/Discrete_Laplace_operator#Implementation_in_Image_Processing
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
    
/******************************************************************************/

/* Accessors ******************************************************************/
    
    size_t get_n_dims() const
    {
        return n_dims;
    }
    
    size_t get_dim(size_t dim) const
    {
        assert(dim < n_dims);
        
        return d[dim];
    }
    
    size_t get_size() const
    {
        return size;
    }
    
    T* get_array() const
    {
        return array;
    }
    
    inline T& at( size_t row_index, size_t col_index = 0, size_t slice_index = 0 ) // GENERIC (1-3D)
    {
#if SAFE_ACCESSORS
        assert(row_index >= 0 and row_index < d[0]);
        assert(col_index >= 0 and col_index < d[1]);
        assert(slice_index >= 0 and slice_index < d[2]);
#endif
        
        return array[ slice_index*d[0]*d[1] + col_index*d[0] + row_index ];
    }
    
    const inline T& at( size_t row_index, size_t col_index = 0, size_t slice_index = 0 ) const // GENERIC (1-3D)
    {
#if SAFE_ACCESSORS
        assert(row_index >= 0 and row_index < d[0]);
        assert(col_index >= 0 and col_index < d[1]);
        assert(slice_index >= 0 and slice_index < d[2]);
#endif
        
        return array[ slice_index*d[0]*d[1] + col_index*d[0] + row_index ];
    }
    
    // Accessing an index out of bounds gives a 0 result;
    const inline T at_general( size_t row_index, size_t col_index = 0, size_t slice_index = 0 ) const // GENERIC (1-3D)
    {
        if (row_index >= 0 and row_index < d[0] and col_index >= 0 and col_index < d[1] and slice_index >= 0 and slice_index < d[2])       
            return array[ slice_index*d[0]*d[1] + col_index*d[0] + row_index ];
        return 0;
    }
    
    // Retrieve a subset of the tensor
    void get_sub_tensor(tensor<T>& result, size_t row_offset, size_t col_offset = 0, size_t slice_offset = 0) const // GENERIC (1-3D)
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
    bool get_sub_tensor_general(tensor<T>& result, int row_offset, int col_offset = 0, int slice_offset = 0) const // GENERIC (1-3D)
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
    void set_sub_tensor(const tensor<T>& data, size_t row_offset, size_t col_offset = 0, size_t slice_offset = 0) // GENERIC (1-3D)
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
/******************************************************************************/
    
/* Scalar operations (i.e. operating element by element) **********************/

    // Checks if the tensor is equal (up to some tolerance) to other tensor
    const bool equals(const tensor<T> & other, T tol) // GENERIC (1-3D)
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
    
    const tensor<T>& operator=( const T scalar ) // TODO Can it be shortened using the copy constructor?
    {
        for( size_t counter = 0; counter < size; ++counter )
        {
            array[counter] = scalar;
        }
        return *this;
    }
    
    const tensor<T>& operator=( const tensor<T>& other) // TODO Can it be shortened using the copy constructor?
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
    
    inline tensor<T> operator+( const tensor<T>& other ) const  // GENERIC (1-3D)
    {
        tensor<T> result( *this );
        result += other;
        return result;
    }

    void operator+=( const tensor<T>& other ) // GENERIC (1-3D)
    {
        assert(n_dims == other.n_dims);
        assert(d[0] == other.d[0] and d[1] == other.d[1] and d[2] == other.d[2]);
        
        #pragma omp parallel for
        for (size_t counter = 0; counter < size; ++counter)
        {
            array[counter] += other.array[counter];
        }
    }
    
    inline tensor<T> operator-( const tensor<T>& other ) const  // GENERIC (1-3D)
    {
        tensor<T> result( *this );
        result -= other;
        return result;
    }

    void operator-=( const tensor<T>& other ) // GENERIC (1-3D)
    {
        assert(n_dims == other.n_dims);
        assert(d[0] == other.d[0] and d[1] == other.d[1] and d[2] == other.d[2]);
        
        #pragma omp parallel for
        for (size_t counter = 0; counter < size; ++counter)
        {
            array[counter] -= other.array[counter];
        }
    }

    tensor<T> operator*(T scalar)
    {
        tensor result(*this);

        #pragma omp parallel for
        for (size_t counter = 0; counter < size; ++counter)
        {
            result.array[counter] = array[counter]*scalar;
        }
        return result;
    }
    
    void operator*=(T scalar)
    {
        #pragma omp parallel for
        for (size_t counter = 0; counter < size; ++counter)
        {
            array[counter] *= scalar;
        }
    }
    
    tensor<T> operator/(T scalar)
    {
        tensor result(*this);

        #pragma omp parallel for
        for (size_t counter = 0; counter < size; ++counter)
        {
            result.array[counter] = array[counter]/scalar;
        }
        return result;
    }
    
    void operator/=(T scalar)
    {
        #pragma omp parallel for
        for (size_t counter = 0; counter < size; ++counter)
        {
            array[counter] /= scalar;
        }
    }
    
    void power(double exponent)
    {
        #pragma omp parallel for
        for (size_t counter = 0; counter < size; ++counter)
        {
            if (exponent < 1)
                assert(array[counter] >= -FLT_EPSILON);     // TODO: make epsilon dependent on type of array

            array[counter] = pow(array[counter],exponent);
        }
    }
   
    void absolute_value()
    {
        #pragma omp parallel for
        for (size_t counter = 0; counter < size; ++counter)
        {
            array[counter] = abs(array[counter]);
        }
    }
    
    void round_values()
    {
        #pragma omp parallel for
        for (size_t counter = 0; counter < size; ++counter)
        {
            array[counter] = round(array[counter]);
        }
    }
    
    // Linearly maps the values in the first range (values outside are treated like the closest interval end) into the second range
    void map_histogram(T source_left, T source_right, T target_left, T target_right)
    {
        for (size_t counter = 0; counter < size; ++counter)
        {
            array[counter] = std::max<float>(array[counter],source_left);
            array[counter] = std::min<float>(array[counter],source_right);
            array[counter] = (array[counter]-source_left)/float(source_right-source_left)*(target_right-target_left) + target_left;
        }
    }
    
/******************************************************************************/
    
/* Transformations ************************************************************/
    
    // Matrix transposition
    tensor<T> transpose() const
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
    void squeeze()
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
    void embed(size_t pos) // NON-GENERIC (2D)
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
    void unembed(size_t pos) // NON-GENERIC (3D)
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
    tensor<T> downsample(size_t factor1, size_t factor2 = 1, size_t factor3 = 1) const // GENERIC (1-3D)
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
    tensor<T> resize(size_t d0, size_t d1 = 1, size_t d2 = 1) const // GENERIC (1-3D)
    {
        assert(d0 > 0 and d1 > 0 and d2 > 0);
        
        tensor<T> result(d0,d1,d2);
        get_sub_tensor_general(result,0,0,0);
        return result;
    }
    
    // Approximation of the norm of the gradient, after http://en.wikipedia.org/wiki/Sobel_operator
    tensor<T> sobel_transformation() // NON-GENERIC
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
    void dct() // GENERIC (1-3D)
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
    void idct() // GENERIC (1-3D)
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
    
    // GENERIC (1-3D)
    template< typename TT >
    void cast_from(tensor<TT>& other) {
        
        for( size_t counter = 0; counter < size; ++counter ) {
            array[counter] = static_cast<T> (other.get_array()[counter]);
        }
    }
//        VMML_TEMPLATE_STRING
//    template< typename TT >
//    void
//    VMML_TEMPLATE_CLASSNAME::cast_from(const tensor3< I1, I2, I3, TT >& other) {
//#if 0
//        typedef tensor3< I1, I2, I3, TT > t3_tt_type;
//        typedef typename t3_tt_type::const_iterator tt_const_iterator;
//
//        iterator it = begin(), it_end = end();
//        tt_const_iterator other_it = other.begin();
//        for (; it != it_end; ++it, ++other_it) {
//            *it = static_cast<T> (*other_it);
//        }
//#else
//#pragma omp parallel for
//        for (long slice_idx = 0; slice_idx < (long) I3; ++slice_idx) {
//#pragma omp parallel for
//            for (long row_index = 0; row_index < (long) I1; ++row_index) {
//#pragma omp parallel for
//                for (long col_index = 0; col_index < (long) I2; ++col_index) {
//                    at(row_index, col_index, slice_idx) = static_cast<T> (other.at(row_index, col_index, slice_idx));
//                }
//            }
//        }
//
//#endif
//    }
    
    // Returns the Summed Area Table (every point is the integral on the rectangular region defined by that point and the origin). Used e.g. to quickly compute histograms
    tensor<T> summed_area_table() const // GENERIC (1-3D)
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
    bool svd(tensor<float>& U, tensor<float>& S, tensor<float>& Vt) const // TODO: right now, only for T = float
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
    bool left_singular_vectors(tensor<float>& U) const // TODO: right now, only for T = float
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
    bool reorthogonalize()
    {
        tensor<T> copy(*this);
        bool result = left_singular_vectors(copy);
        *this = copy;
        return result;
    }
    
    // Sets U to the leading eigenvectors of the current matrix
    bool symmetric_eigenvectors(tensor<float>& U) const // TODO: only works for T = float
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
    
    const void tucker_decomposition(tensor<T>& core, tensor<T>& U1, tensor<T>& U2, tensor<T>& U3, size_t max_iters = 3, double tol = 1e-3) const // NON-GENERIC (3D)
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
        
        double error_old = std::numeric_limits<double>::max();
        double error = 0;
        
        // TODO preallocate like this?
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
            error = sqrt(std::max<double>(0,target_frobenius_norm*target_frobenius_norm - achieved_frobenius_norm*achieved_frobenius_norm))/target_frobenius_norm;
            std::cout << "Iteration = " << iteration+1 << ", error = " << error << std::endl;
        }
    }
/******************************************************************************/

/* Tensor-tensor operations ***************************************************/

    // Tensor-tensor convolution. The result has the same size as the original
    tensor<T> convolve( const tensor& kernel ) const // GENERIC (1-3D)
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
    tensor<T> mtm(const tensor<T>& factor) const
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
    tensor<T> covariance() const
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
    void reconstruct_cp(const tensor<T>& lambdas, const tensor<T>& U1, const tensor<T>& U2, const tensor<T>& U3) // NON-GENERIC
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
    tensor<T> ttm(const tensor<T>& matrix1, const tensor<T>& matrix2, const tensor<T>& matrix3) const // NON-GENERIC
    {
        return ttm1(matrix1).ttm2(matrix2).ttm3(matrix3);
    }

    tensor<T> ttm1(const tensor<T>& matrix) const // NON-GENERIC
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
    
    tensor ttm2(const tensor& matrix) const // NON-GENERIC
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
    
    tensor ttm3(const tensor& matrix) const // NON-GENERIC
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
/******************************************************************************/

/* I/O ************************************************************************/
    
    friend std::ostream& operator << ( std::ostream& os, const tensor<T>& t ) // NON-GENERIC
    {
        os << "Tensor of size ";
        for (size_t dim = 0; dim < t.n_dims; ++dim) {
            if (dim > 0) os << " x ";
            os << t.d[dim];
        }
        os << std::endl;
        for( size_t slice_index = 0; slice_index < t.d[2]; ++slice_index )
        {
            os << "------ Slice " << slice_index << "------" << std::endl;
            for( size_t row_index = 0; row_index < t.d[0]; ++row_index )
            {
                os << "(";
                for( size_t col_index = 0; col_index < t.d[1]; ++col_index )
                {
                    os << t.array[ slice_index*t.d[0]*t.d[1] + col_index*t.d[0] + row_index ];
                    if (col_index + 1 < t.d[1] )
                        os << ", ";
                }
                os << ")" << std::endl;
            }
            os << std::endl;
        }
        return os;
    }
    
    // Sets up the tensor for reading operations on a memory mapped file, in order to optimize treatment of large data. close_mmap() should be called after the tensor is not needed
    void open_mmap(const std::string& file) // TODO: only tested in Linux
    {
        fd = open( file.c_str(), O_RDONLY, 0 );
        assert(fd != -1);

        off_t offset = 0;

        mmapped_data = (void*)mmap( 0, size*sizeof(T), PROT_READ, MAP_FILE | MAP_SHARED, fd, offset );
        assert(mmapped_data != MAP_FAILED);
        array = reinterpret_cast<T*> (mmapped_data);
    }
    
    void close_mmap() // TODO: only tested in Linux
    {
        munmap( mmapped_data, size*sizeof(T) );
        ::close( fd );
        array = 0;
    }
    
    // Read from a file in disk, with elements in column-major order
    void read_from_raw(const std::string& file)
    {
        std::ifstream input_file(file.c_str(), std::ios::binary | std::ios::ate);
        assert(input_file.is_open());
        size_t file_size = input_file.tellg();
        input_file.close();
        assert(file_size == size*sizeof(T));
        input_file.open(file.c_str(), std::ios::in);
        
        size_t max_file_len = 2147483648u - sizeof (T);
        size_t len_data = sizeof (T) * size;
        size_t len_read = 0;
        char* data = new char[ len_data ];

        T* it = array;
        while (len_data > 0) {
            len_read = (len_data % max_file_len) > 0 ? len_data % max_file_len : len_data;
            len_data -= len_read;
            input_file.read(data, len_read);

            T* T_ptr = (T*)&(data[0]);
            size_t elements_read = 0;
            for (; elements_read < size && (len_read > 0); ++it, len_read -= sizeof (T)) {
                *it = *T_ptr;
                ++T_ptr;
                ++elements_read;
            }
        }

        delete[] data;
        input_file.close();
    }
    
    void write_to_raw(const std::string& file) const
    {

        std::ofstream output_file;
        output_file.open(file.c_str());
        assert(output_file.is_open());
        output_file.write(reinterpret_cast<char*>(array),size*sizeof(T));
        output_file.close();
    }
    
    // Dump the contents in a file in Comma Separated Values format, for easy human interpretation
    void write_to_csv(const std::string& file) const // NON-GENERIC
    {
        assert(n_dims == 2);
        
        std::ofstream output_file;
        output_file.open( file.c_str() );
        assert(output_file.is_open());
        for (size_t i = 0; i < d[0]; ++i) {
            for (size_t j = 0; j < d[1]; ++j) {
                if (j > 0)
                    output_file << ",";
                output_file << at(i,j);
            }
            output_file << std::endl;
        }
        output_file.close();
    }
    
    // Display a short fingerprint of the tensor
    void debug() const
    {
        std::cerr << "Size: ";
        for (size_t dim = 0; dim < n_dims; ++dim) {
            if (dim > 0) std::cerr << " x ";
            std::cerr << d[dim]; 
        }
        std::cerr << " | First elements: ";
        for (size_t i = 0; i < std::min<size_t>(3,size); ++i) {
            if (i > 0)
                std::cerr << ", ";
            std::cerr << array[i];
        }
        std::cerr << "... | Sum = " << sum() << ", L2 norm = " << frobenius_norm();
        std::cerr << " | Max = " << maximum() << ", min = " << minimum() << std::endl;
    }
/******************************************************************************/

/* Tensor properties **********************************************************/
    
    bool is_symmetric() const // NON-GENERIC (2D)
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
    double frobenius_norm() const
    {
        double norm = 0.0f;
        for( size_t counter = 0; counter < size; ++counter )
        {
            norm += array[counter]*array[counter];
        }
        return sqrt(norm);
    }

    // Frobenius norm of the difference with another tensor
    double frobenius_norm(const tensor<T>& other) const // GENERIC (1-3D)
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
    double manhattan_norm() const
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
    double relative_error(const tensor<T>& other) const
    {
        return frobenius_norm(other)/frobenius_norm();
    }

    T sum() const
    {
        T result = 0;
        for (size_t counter = 0; counter < size; ++counter) {
            result += array[counter];
        }
        return result;
    }
    
    T maximum() const
    {
        T result = std::numeric_limits<T>::min();
        for (size_t counter = 0; counter < size; ++counter) {
            result = std::max<float>(result,array[counter]);
        }
        return result;
    }

    T minimum() const
    {
        T result = std::numeric_limits<T>::max();
        for (size_t counter = 0; counter < size; ++counter) {
            result = std::min<float>(result,array[counter]);
        }
        return result;
    }

    double mean() const
    {
        return sum()/size;
    }
    
    double variance() const
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

    double stdev() const
    {
        return sqrt(variance());
    }
    
    // Returns matrix with one row per bin and 3 columns: bin start value, bin end value and bin count
    tensor<float> histogram(int n_bins, size_t step = 1) const
    {
        tensor<float> result(n_bins,3);
        result.set_zero();
        tensor<size_t> bins(n_bins);
        bins.set_zero();        
                
        T minVal = minimum();
        T maxVal = maximum();
        
        for (int i = 0; i < n_bins; ++i) {
            result.at(i,0) = i/float(n_bins)*(maxVal-minVal) + minVal;
            result.at(i,1) = (i+1)/float(n_bins)*(maxVal-minVal) + minVal;
        }
        
        for (size_t counter = 0; counter < size; counter += step) {
            bins.at(round( (array[counter]-minVal)/(maxVal-minVal)*(n_bins-1)))++;
        }

        for (int i = 0; i < n_bins; ++i) {
            result.at(i,2) = float(bins.at(i)); // TODO do a proper cast!
        }
        
        return result;
    }
    
/******************************************************************************/

};

}

#endif
