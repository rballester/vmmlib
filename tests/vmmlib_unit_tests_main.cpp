#include <iostream>

#include "unit_test_globals.hpp"

#include "vector_test.hpp"
#include "matrix_test.hpp"
#include "lowpass_filter_test.hpp"
#include "frustum_test.hpp"
#include "intersection_test.hpp"
#include "quaternion_test.hpp"
#include "qr_decomposition_test.hpp"
#include "svd_test.hpp"
#include "util_test.hpp"

#ifdef VMMLIB_USE_LAPACK
#  include "lapack_linear_least_squares_test.hpp"
#  include "lapack_gaussian_elimination_test.hpp"
#  include "lapack_svd_test.hpp"
#  include "lapack_sym_eigs_test.hpp"
#  include "cp3_tensor_test.hpp"
#  include "qtucker3_tensor_test.hpp"
#  include "t3_hooi_test.hpp"
#  include "t3_hopm_test.hpp"
#  include "t3_hosvd_test.hpp"
#  include "t3_ihopm_test.hpp"
#  include "t3_ihooi_test.hpp"
#  include "t3_ttm_test.hpp"
#  include "tensor3_iterator_test.hpp"
#  include "tensor3_test.hpp"
#  include "tucker3_exporter_importer_test.hpp"
#  include "tucker3_tensor_test.hpp"
#  include "matrix_pseudoinverse_test.hpp"
#  include "tensor4_test.hpp"
#  include "t4_hooi_test.hpp"
#  include "t4_converter_test.hpp"
#  include "t4_ttm_test.hpp"
#endif

#ifdef VMMLIB_USE_BLAS
#  include "blas_daxpy_test.hpp"
#  include "blas_dgemm_test.hpp"
#  include "blas_dot_test.hpp"
#endif

void run_and_log( vmml::unit_test& test )
{
    test.run();
    std::cout << test << std::endl;
}

int main( int, const char** )
{

    vmml::vector_test vector_test_;
    run_and_log( vector_test_ );

    vmml::matrix_test matrix_test_;
    run_and_log( matrix_test_ );

    vmml::lowpass_filter_test lowpass_filter_test_;
    run_and_log( lowpass_filter_test_ );

    vmml::frustum_test frustum_test_;
    run_and_log( frustum_test_ );

    vmml::intersection_test intersection_test_;
    run_and_log( intersection_test_ );

    vmml::quaternion_test quaternion_test_;
    run_and_log( quaternion_test_ );

	vmml::qr_decomposition_test qr_test_;
	run_and_log( qr_test_ );

	vmml::svd_test svd_test_;
	run_and_log( svd_test_ );

    vmml::util_test util_test_;
    run_and_log( util_test_ );

#ifdef VMMLIB_USE_LAPACK
    vmml::lapack_svd_test lapack_svd_test_;
    run_and_log( lapack_svd_test_ );

    vmml::lapack_linear_least_squares_test lapack_llsq_test;
    run_and_log( lapack_llsq_test );

    vmml::lapack_gaussian_elimination_test lapack_ge_test;
    run_and_log( lapack_ge_test );

	vmml::lapack_sym_eigs_test lapack_sym_eigs_test_;
    run_and_log( lapack_sym_eigs_test_ );
#endif

#ifdef VMMLIB_USE_BLAS
    vmml::blas_dgemm_test blas_mm;
    run_and_log( blas_mm );

    vmml::blas_dot_test blas_vvi;
    run_and_log( blas_vvi );

	vmml::blas_daxpy_test b_daxpy;
    run_and_log( b_daxpy );
#endif

#ifdef VMMLIB_USE_BLAS 
#ifdef VMMLIB_USE_LAPACK
    vmml::t3_hosvd_test t3hosvd;
    run_and_log( t3hosvd );

    vmml::t3_ihooi_test t3ihooi;
    run_and_log( t3ihooi );

    vmml::t3_hopm_test t3hopm;
    run_and_log( t3hopm );

	vmml::t3_ihopm_test t3ihopm;
    run_and_log( t3ihopm );

    vmml::t3_hooi_test t3hooi;
    run_and_log( t3hooi );

    vmml::matrix_pseudoinverse_test m_pinv;
    run_and_log( m_pinv );

    vmml::cp3_tensor_test cp3t;
    run_and_log( cp3t );

    vmml::qtucker3_tensor_test tt3tq;
    run_and_log( tt3tq );
    
      vmml::tensor3_test t3t;
    run_and_log( t3t );

    vmml::tensor3_iterator_test t3it;
    run_and_log( t3it );

    vmml::t3_ttm_test t3ttm;
    run_and_log( t3ttm );

    vmml::tucker3_tensor_test tt3t;
    run_and_log( tt3t );

	vmml::tucker3_exporter_importer_test tt3ei;
    run_and_log( tt3ei );

	vmml::tensor4_test t4t;
    run_and_log( t4t );

    vmml::t4_hooi_test t4hooi;
    run_and_log( t4hooi );

    vmml::t4_ttm_test t4ttm;
    run_and_log( t4ttm );

	vmml::t4_converter_test t4ct;
    run_and_log( t4ct );
#endif
#endif

    std::cout << vmml::unit_test_globals::get_instance() << std::endl;

}
