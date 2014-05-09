/* 
 * File:   tensor_io.hpp
 * Author: rballester
 *
 * Created on May 9, 2014, 7:17 PM
 */

#ifndef TENSOR_IO_HPP
#define	TENSOR_IO_HPP

#include "tensor.hpp"

namespace vmml
{

// Sets up the tensor for reading operations on a memory mapped file, in order to optimize treatment of large data. close_mmap() should be called after the tensor is not needed
template< typename T>
void tensor<T>::open_mmap(const std::string& file) // TODO: only tested in Linux
{
    fd = open( file.c_str(), O_RDONLY, 0 );
    assert(fd != -1);

    off_t offset = 0;

    mmapped_data = (void*)mmap( 0, size*sizeof(T), PROT_READ, MAP_FILE | MAP_SHARED, fd, offset );
    assert(mmapped_data != MAP_FAILED);
    array = reinterpret_cast<T*> (mmapped_data);
}

template< typename T>
void tensor<T>::close_mmap() // TODO: only tested in Linux
{
    munmap( mmapped_data, size*sizeof(T) );
    ::close( fd );
    array = 0;
}

// Read from a file in disk, with elements in column-major order
template< typename T>
void tensor<T>::read_from_raw(const std::string& file)
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

template< typename T>
void tensor<T>::write_to_raw(const std::string& file) const
{

    std::ofstream output_file;
    output_file.open(file.c_str());
    assert(output_file.is_open());
    output_file.write(reinterpret_cast<char*>(array),size*sizeof(T));
    output_file.close();
}

// Dump the contents in a file in Comma Separated Values format, for easy human interpretation
template< typename T>
void tensor<T>::write_to_csv(const std::string& file) const // NON-GENERIC
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
template< typename T>
void tensor<T>::debug() const
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

}

#endif	/* TENSOR_IO_HPP */

