/* 
 * Author: rballester
 *
 * Created on March 14, 2014, 10:43 AM
 */

// TODO: make asserts optional
// TODO: switch memcpy's to std::copy
// TODO: make a function out of the unfolding
// TODO: make scalar operations inline?

#ifndef TENSOR_HPP
#define TENSOR_HPP

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

#include <omp.h>

namespace vmml
{

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
    
    tensor(const tensor<T>& other) // GENERIC (1-3D)
    {
        n_dims = other.n_dims;
        d[0] = other.d[0];
        d[1] = other.d[1];
        d[2] = other.d[2];
        size = other.size;
        array = new T[other.size];
        memcpy(array, other.array, size*sizeof(T));
    }
    
    ~tensor() {
        delete[] array;
    }

/* Initializers ***************************************************************/
    
    void set_zero();
    void set_constant(T constant);
    void set_random( int seed );
    void set_dct();
    void set_memory(const T* memory);
    void set_gaussian(double sigma1, double sigma2 = 0, double sigma3 = 0);
    void set_laplacian();

/******************************************************************************/

/* Accessors ******************************************************************/
    
    size_t get_n_dims() const;
    size_t get_dim(size_t dim) const;
    size_t get_size() const;
    T* get_array();
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
    const inline T at_general( size_t row_index, size_t col_index = 0, size_t slice_index = 0 ) const // GENERIC (1-3D)
    {
        if (row_index >= 0 and row_index < d[0] and col_index >= 0 and col_index < d[1] and slice_index >= 0 and slice_index < d[2])       
            return array[ slice_index*d[0]*d[1] + col_index*d[0] + row_index ];
        return 0;
    }
    void get_sub_tensor(tensor<T>& result, size_t row_offset, size_t col_offset = 0, size_t slice_offset = 0) const; // GENERIC (1-3D)
    bool get_sub_tensor_general(tensor<T>& result, int row_offset, int col_offset = 0, int slice_offset = 0) const; // GENERIC (1-3D)
    void set_sub_tensor(const tensor<T>& data, size_t row_offset, size_t col_offset = 0, size_t slice_offset = 0); // GENERIC (1-3D)

/******************************************************************************/
    
/* Scalar operations (i.e. operating element by element) **********************/

    // Checks if the tensor is equal (up to some tolerance) to other tensor
    const bool equals(const tensor<T> & other, T tol); // GENERIC (1-3D)   
    const tensor<T>& operator=( const tensor<T>& other); // TODO Can it be shortened using the copy constructor?
    inline tensor<T> operator+( const tensor<T>& other ) const  // GENERIC (1-3D)
    {
        tensor<T> result( *this );
        result += other;
        return result;
    }
    void operator+=( const tensor<T>& other ); // GENERIC (1-3D)    
    inline tensor<T> operator-( const tensor<T>& other ) const  // GENERIC (1-3D)
    {
        tensor<T> result( *this );
        result -= other;
        return result;
    }
    void operator-=( const tensor<T>& other ); // GENERIC (1-3D)
    tensor<T> operator*(T scalar);    
    void operator*=(T scalar);    
    tensor<T> operator/(T scalar);    
    void operator/=(T scalar);    
    void power(double exponent);

/******************************************************************************/
    
/* Transformations ************************************************************/
    
    tensor<T> transpose() const;
    void squeeze();
    void embed(size_t pos); // NON-GENERIC (2D)
    void unembed(size_t pos); // NON-GENERIC (3D)
    tensor<T> downsample(size_t factor1, size_t factor2 = 1, size_t factor3 = 1) const; // GENERIC (1-3D)
    tensor<T> resize(size_t d0, size_t d1 = 1, size_t d2 = 1) const; // GENERIC (1-3D)
    tensor<T> sobel_transformation(); // NON-GENERIC
    void dct(); // GENERIC (1-3D)
    void idct(); // GENERIC (1-3D)
    tensor<T> summed_area_table() const; // GENERIC (1-3D)
    bool svd(tensor<float>& U, tensor<float>& S, tensor<float>& Vt) const; // TODO: right now, only for T = float
    bool left_singular_vectors(tensor<float>& U) const; // TODO: right now, only for T = float
    bool reorthogonalize();
    bool symmetric_eigenvectors(tensor<float>& U) const; // TODO: only works for T = float    
    const void tucker_decomposition(tensor<T>& core, tensor<T>& U1, tensor<T>& U2, tensor<T>& U3, size_t max_iters = 3, double tol = 1e-3) const; // NON-GENERIC (3D)
    
/******************************************************************************/

/* Tensor-tensor operations ***************************************************/

    tensor<T> convolve( const tensor& kernel ) const; // GENERIC (1-3D)
    tensor<T> mtm(const tensor<T>& factor) const;
    tensor<T> covariance() const;
    void reconstruct_cp(const tensor<T>& lambdas, const tensor<T>& U1, const tensor<T>& U2, const tensor<T>& U3); // NON-GENERIC
    tensor<T> ttm(const tensor<T>& matrix1, const tensor<T>& matrix2, const tensor<T>& matrix3) const; // NON-GENERIC
    tensor<T> ttm1(const tensor<T>& matrix) const; // NON-GENERIC    
    tensor<T> ttm2(const tensor& matrix) const; // NON-GENERIC    
    tensor<T> ttm3(const tensor& matrix) const; // NON-GENERIC

/******************************************************************************/

/* I/O ************************************************************************/
    
    friend std::ostream& operator<< ( std::ostream& os, const tensor<T>& t ) // NON-GENERIC
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
    void open_mmap(const std::string& file); // TODO: only tested in Linux    
    void close_mmap(); // TODO: only tested in Linux
    void read_from_raw(const std::string& file);    
    void write_to_raw(const std::string& file) const;
    void write_to_csv(const std::string& file) const; // NON-GENERIC    
    void debug() const;

/******************************************************************************/

/* Tensor properties **********************************************************/
    
    bool is_symmetric() const; // NON-GENERIC (2D)
    double frobenius_norm() const;
    double frobenius_norm(const tensor<T>& other) const; // GENERIC (1-3D)
    double manhattan_norm() const;
    double relative_error(const tensor<T>& other) const;
    double psnr(const tensor<T>& other, double max_amplitude) const; // GENERIC (1-3D)
    T sum() const;    
    T maximum() const;
    T minimum() const;
    double mean() const;
    double variance() const;
    double stdev() const;
    
/******************************************************************************/

};

}

#endif
