//#include <vmmlib/t3_hooi.hpp>
//#include <vmmlib/t3_converter.hpp>
//#include <vmmlib/tensor_mmapper.hpp>

#include <include/vmmlib/tensor.hpp>
#include <include/vmmlib/tensor_initializers.hpp>

#include <iostream>
#include <fftw3.h>

#define VMMLIB_SAFE_ACCESSORS 0
#define VMMLIB_VERBOSE 1

using namespace vmml;

int main (int argc, char * const argv[]) {

    // Let's create a 3-way tensor, of size 5 x 5 x 5
    tensor<float> input(5,5,5);
    
    // We initialize it as a Gaussian kernel, with sigma = 1
    input.set_gaussian(1);
    
    // This tensor is separable (i.e., has rank 1). 
    // To check this, let's perform a rank-1 Tucker decomposition
    
    // The decomposition consists of a core and several factor matrices, which
    // we declare here:
    tensor<float> core(1,1,1); // The Tucker core will have 1 element
    tensor<float> U1(5,1), U2(5,1), U3(5,1); // The 3 factor matrices
    
    // We need to initialize the factor matrices. The basis functions for the DCT
    // are a good choice in general:
    U1.set_dct();
    U2.set_dct();
    U3.set_dct();
    
    // Tucker decomposition
    input.tucker_decomposition(core,U1,U2,U3);
    
    // We just obtained a rank-1 decomposition. Let's reconstruct it:
    tensor<float> reconstruction = core.ttm(U1,U2,U3); // Tensor-times-matrix multiplication
    
    // Print the difference between original and reconstruction (Frobenius norm);
    // it's practically 0:
    std::cout << "Norm of the error: " << input.frobenius_norm(reconstruction) << std::endl;
    
//    tensor<float> gaussian(5,5);
//    gaussian.set_gaussian(1);
//    
//    tensor<float> laplacian(3,3);
//    laplacian.set_zero();
//    laplacian.set_laplacian();
//    
//    std::cout << laplacian << std::endl;
    
//    tensor<float> U1(6,1,3);
//    U1.set_random();
//    std::cout << U1 << std::endl;
//    U1.squeeze();
//    std::cout << U1 << std::endl;
//    std::cout << U1.get_dim(0) << " " << U1.get_dim(1) << " " << U1.get_dim(2) << " " << U1.get_size() << std::endl;
    
    
    
    // TTM with BLAS vs. without
//    tensor3<256, 256, 256, float> core_old;
////    core_old.set
//    matrix<256,256,float> factor_old;
//    factor_old.set_random();
//    tensor3<256, 256, 256, float> result_old;
//    clock_t start = clock();
//    t3_ttm::multiply_lateral_fwd(core_old,factor_old,result_old);
//    clock_t end = clock();
//    std::cerr << double (end-start) / CLOCKS_PER_SEC * 1000.0 << std::endl;
//    
//    tensor<float> core(256,256,256);
//    core.set_memory(core_old.get_array_ptr());
//    tensor<float> factor(256,256);
//    factor.set_memory(factor_old.begin());
//    start = clock();
//    tensor<float> result = core.ttm1(factor);
//    end = clock();
//    std::cerr << double (end-start) / CLOCKS_PER_SEC * 1000.0 << std::endl;
    
    // Matrix-matrix multiplication
//    tensor<float> A(3,4), B(4,2);
//    A.set_dct();
//    B.set_dct();
//    std::cout << A << std::endl;
//    std::cout << B << std::endl;
//    tensor<float> C = A.mtm(B);
//    std::cout << C << std::endl;
    
//    tensor<float> U(6,2), S(2), Vt(2,2);
//    U1.svd(U,S,Vt);
//    std::cout << U << std::endl;
//    std::cout << S << std::endl;
//    std::cout << Vt << std::endl;
    
//    tensor<float> LoG(5,5);
//    LoG = gaussian.convolve(laplacian);
    
//    tensor<float> test_downsampled = test.downsample(2);
//    std::cout << test_downsampled << std::endl;
//    tensor<float> window_trad(140,32);
//    window_trad.read_from_raw("/tmp/window_trad.raw");
//    tensor<float> tmp(140,32);
//    window_trad.left_singular_vectors(tmp);
//    tmp.write_to_raw("/tmp/window_re.raw");
//    
//    
//    lapack_svd< 140, 32, float > svd;
//    vector<32,float> lambdas;
//    matrix<140,32,float> window_trad_vmmlib;
//    for(int i = 0; i < 140; ++i) {
//        for(int j = 0; j < 32; ++j)
//            window_trad_vmmlib.at(i,j) = window_trad.at(i,j);
//    }
//    svd.compute_and_overwrite_input(window_trad_vmmlib, lambdas);
//    window_trad_vmmlib.write_to_raw("/tmp/","window_vmmlib.raw");
            
//    window_trad.close_mmap();
//    tensor<float> U1(4,2);
//    U1.set_dct();
////    U2.set_dct();
////    U3.set_dct();
////    U1.reorthogonalize();
//    
//    std::cout << U1 << std::endl;
//    tensor<float> blah(20,2);
//    bool sth_was_copied = U1.get_sub_tensor_general(blah,-10,0);
//    std::cout << sth_was_copied << std::endl;
//    std::cout << blah << std::endl;
    
//    // Downsampling check
//    tensor<float> U1(6,4);
//    U1.set_dct();
//    std::cout << U1 << std::endl;
//    U1 = U1.downsample(2);
//    std::cout << U1 << std::endl;
    
//// Compute a full resolution Sobel transformation
//    tensor<float> hazelnut(512,512,512);
//    hazelnut.read_from_raw("/home/rballester/datasets/survey_testdata/hnut512_comp.raw");
//    hazelnut /= hazelnut.maximum();
//    std::cout << "Norm before: " << hazelnut.frobenius_norm() << std::endl;
//    hazelnut = hazelnut.sobel_transformation();
//    std::cout << "Norm after: " << hazelnut.frobenius_norm() << std::endl;
//    hazelnut.write_to_raw("/home/rballester/datasets/survey_testdata/hnut512_sobel.raw");
  
    // Check correctness of tree_builder result
//    tensor<float> U1(608,32), U2(608,32), U3(608,32);
//    tensor<float> core(32,32,32);
//    U1.read_from_raw("/home/rballester/vmmlibIntern/vmmlib2/output/hnut512/L0_0_U1.raw");
//    U2.read_from_raw("/home/rballester/vmmlibIntern/vmmlib2/output/hnut512/L0_0_U2.raw");
//    U3.read_from_raw("/home/rballester/vmmlibIntern/vmmlib2/output/hnut512/L0_0_U3.raw");
//    U1 = U1.downsample(8);
//    U2 = U2.downsample(8);
//    U3 = U3.downsample(8);
//    core.read_from_raw("/home/rballester/vmmlibIntern/vmmlib2/output/hnut512/L0_0_0_0_core.raw");
//    tensor<float> reco = core.ttm(U1,U2,U3);
//    reco.write_to_raw("/tmp/reco.raw");
    
//    float values[2];
//    values[0] = -1;
//    values[1] = -10;
//    tensor<float> blah(1,2);
//    blah.set_memory(values);
//    std::cout << blah << std::endl;
    
    // TTM correctness
//    tensor<float> U1(2,6);
//    U1.set_dct();
////    U3.set_dct();
//
//    tensor<float> X(6,6,6);
//    X.read_from_raw("/home/rballester/tree_builder/example.raw");
//    std::cout << X << std::endl;
//    tensor<float> X2 = X.ttm3(U1);
//    std::cout << X2 << std::endl;
    
    
//    X.write_to_raw("/home/rballester/tree_builder/example2.raw");
    
//    tensor<float> core = X.ttm(U1.transpose(),U2.transpose(),U3.transpose());

//    tensor<float> subtensor(3,3);
//    X.get_sub_tensor(subtensor,0,1,1);
//    std::cout << subtensor << std::endl;
//    std::cout << U1 << std::endl;
//    tensor<float> llsv(6,2);
//    llsv.set_zero();
//    U1.reorthogonalize();
//    std::cout << X  << std::endl;
//    X.close_mmap();
    
    // FFTW tests
//    int N = 10;
//    
//    fftw_complex *in, *out;
//    fftw_plan p;
////    ...
//    in = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N);
//    out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N);
//    p = fftw_plan_dft_1d(N, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
////    ...
//    fftw_execute(p); /* repeat as needed */
////    ...
//    fftw_destroy_plan(p);
//    fftw_free(in); fftw_free(out);
    
//    int i;
//    double *in;
//    double *in2;
//    int n = 100;
//    int nc;
//    fftw_complex *out;
//    fftw_plan plan_backward;
//    fftw_plan plan_forward;
//    unsigned int seed = 123456789;
//
//    printf ( "\n" );
//    printf ( "TEST02\n" );
//    printf ( "  Demonstrate FFTW3 on a single vector of real data.\n" );
//    printf ( "\n" );
//    printf ( "  Transform data to FFT coefficients.\n" );
//    printf ( "  Backtransform FFT coefficients to recover data.\n" );
//    printf ( "  Compare recovered data to original data.\n" );
//    /*
//    Set up an array to hold the data, and assign the data.
//    */
//    in = fftw_malloc ( sizeof ( double ) * n );
//
//    srand ( seed );
//
//    for ( i = 0; i < n; i++ )
//    {
//      in[i] = rand()/double(RAND_MAX);
//    }
//
//    printf ( "\n" );
//    printf ( "  Input Data:\n" );
//    printf ( "\n" );
//
//    for ( i = 0; i < n; i++ )
//    {
//      printf ( "  %4d  %12f\n", i, in[i] );
//    }
//    /*
//    Set up an array to hold the transformed data,
//    get a "plan", and execute the plan to transform the IN data to
//    the OUT FFT coefficients.
//    */
//    nc = ( n / 2 ) + 1;
//
//    out = fftw_malloc ( sizeof ( fftw_complex ) * nc );
//
//    plan_forward = fftw_plan_dft_r2c_1d ( n, in, out, FFTW_ESTIMATE );
//
//    fftw_execute ( plan_forward );
//
//    printf ( "\n" );
//    printf ( "  Output FFT Coefficients:\n" );
//    printf ( "\n" );
//
//    for ( i = 0; i < nc; i++ )
//    {
//      printf ( "  %4d  %12f  %12f\n", i, out[i][0], out[i][1] );
//    }
//    /*
//    Set up an arrray to hold the backtransformed data IN2,
//    get a "plan", and execute the plan to backtransform the OUT
//    FFT coefficients to IN2.
//    */
//    in2 = fftw_malloc ( sizeof ( double ) * n );
//
//    plan_backward = fftw_plan_dft_c2r_1d ( n, out, in2, FFTW_ESTIMATE );
//
//    fftw_execute ( plan_backward );
//
//    printf ( "\n" );
//    printf ( "  Recovered input data divided by N:\n" );
//    printf ( "\n" );
//
//    for ( i = 0; i < n; i++ )
//    {
//      printf ( "  %4d  %12f\n", i, in2[i] / ( double ) ( n ) );
//    }
//    /*
//    Release the memory associated with the plans.
//    */
//    fftw_destroy_plan ( plan_forward );
//    fftw_destroy_plan ( plan_backward );
//
//    fftw_free ( in );
//    fftw_free ( in2 );
//    fftw_free ( out );
    
//    int N = 4;
//    int RANK = 1;
//    double *in, *out;
//	in = fftw_malloc(sizeof(double) * N);
//	out = fftw_malloc(sizeof(double) * N);
//
//	/* create plan */
//	fftw_plan p;
//	fftw_r2r_kind kind[1] = {FFTW_REDFT10};
//	int dimentions[1] = {N}; /* dimentions */
//	p = fftw_plan_r2r(RANK, dimentions, in, out, kind, FFTW_ESTIMATE);
//
//	/* matrix data collection */
//	printf("enter a %d vector separated by space and/or new line\n", N);
//	unsigned char n;
//	for (n = 0; n < N; n++){ /* rows and cols in C order*/
////		fscanf(stdin, "%lf", &in[n]);
//        std::cin >> in[n];
//	}
//
//	/* display the input matrix for fun */
//	for (n = 0; n < N; n++){ /* rows and cols in C order*/
//		printf("%.2lf ", in[n]);
////		if ((n+1) % N1 == 0) printf("\n");
//	}
//
//	/* execute plan */
//	fftw_execute(p);
//
//    out[0] /= 4;
//    for (n = 1; n < N; n++){ /* rows and cols in C order*/
//		out[n] *= sqrt(2)/4;
////		if ((n+1) % N1 == 0) printf("\n");
//	}
//    
//	/* display dft matrix */
//	printf("dct:\n");
//	for (n = 0; n < N; n++){ /* rows and cols in C order*/
//		printf("%.5lf ", out[n]);
////		if ((n+1) % N1 == 0) printf("\n");
//	}
//
//    out[0] *= 4;
//    for (n = 1; n < N; n++){ /* rows and cols in C order*/
//		out[n] /= (sqrt(2)/4);
////		if ((n+1) % N1 == 0) printf("\n");
//	}
//    
//	/* inverse dct */
//	printf("idct:\n");
//	kind[0] = FFTW_REDFT01;
//	p = fftw_plan_r2r(RANK, dimentions, out, in, kind, FFTW_ESTIMATE);
//	fftw_execute(p);
//    
//    for (n = 0; n < N; n++){ /* rows and cols in C order*/
//		in[n] /= 2*N;
////		if ((n+1) % N1 == 0) printf("\n");
//	}
//    
//	for (n = 0; n < N; n++){ /* rows and cols in C order*/
//		printf("%.5lf ", in[n]);
////		if ((n+1) % N1 == 0) printf("\n");
//	}
//
//	/* free resources */
//	fftw_destroy_plan(p);
//	fftw_free(in);
//	fftw_free(out);

//    tensor<float> blah(6,6);
////    vector.at(0) = 1;
////    vector.at(1) = 2;
////    vector.at(2) = 3;
////    vector.at(3) = 4;
//    blah.set_dct();
//    tensor<float> tmp = blah;
//    std::cout << blah << std::endl;
//    blah.dct();
//    
//    int offset = blah.get_dim(0)/2;
//    tensor<float> blank(offset,2);
//    blank.set_zero();
////    blah.set_sub_tensor(blank,blah.get_dim(0)-offset,0);
//    std::cout << blah << std::endl;
//    
//    blah.idct();
//    std::cout << blah << std::endl;
//    
//    tensor<float> diff = blah-tmp;
//    diff.debug();
    
    // CP tests
    
//    tensor<float> lambdas(100);
//    lambdas.set_random(1);
//    tensor<float> U1(200,100), U2(200,100), U3(200,100);
//    U1.set_dct();
//    U2.set_dct();
//    U3.set_dct();
//    tensor<float> reco(200,200,200);
//    reco.reconstruct_cp(lambdas,U1,U2,U3);
//    reco.debug();
    
    // Convolution tests
//    tensor<float> F(6080,32);
//    F.set_random(1);
//    tensor<float> G(100);
//    G.set_random(1);
//    F.convolve(G).debug();
    
    // get/set_sub_tensor tests
//    tensor<float> blank(300,300,300);
//    blank.set_zero();
//    tensor<float> patch(300,300,300);
//    patch.set_random(1);
//    patch.get_sub_tensor_general(blank,0,0,0);
//    blank.debug();
    
    // set_gaussian tests
//    tensor<float> gaussian(500,500,500);
//    gaussian.set_gaussian(100);
//    gaussian.downsample(2,2,2);
//    gaussian.debug();
    
    // summed_area_table tests
//    tensor<float> test(4,4);
//    test.set_dct();
//    std::cerr << test << std::endl;
//    tensor<float> sat = test.summed_area_table();
//    std::cerr << sat << std::endl;
    
    // Covariance tests
//    tensor<float> U1(6,4);
//    U1.set_dct();
//    std::cout << U1 << std::endl;
//    U1 = U1.covariance();
//    std::cout << U1 << std::endl;
    
    // Eigenvector tests
//    tensor<float> U1(6,4);
//    U1.set_dct();
//    U1 = U1.covariance();
//    std::cout << U1 << std::endl;
//    tensor<float> eigs(6,6);
//    (U1.symmetric_eigenvectors(eigs));
//    tensor<float> one_eigenvector(6,1);
//    eigs.get_sub_tensor(one_eigenvector,0,2);
//    std::cout << "One: " << std::endl << one_eigenvector << std::endl;
//    std::cout << "Product: " << std::endl << U1.mtm(one_eigenvector) << std::endl;
//    std::cout << eigs << std::endl;
//    assert(U1.mtm(one_eigenvector).equals(one_eigenvector,1e-5));
    
//    Tucker tests
//    tensor<float> A(3,3,3);
//    A.set_laplacian();
//    A.write_to_raw("/tmp/A.raw");
//    tensor<float> B(1,1,1);
//    tensor<float> U1(3,1), U2(3,1), U3(3,1);
//    U1.set_dct();
//    U2.set_dct();
//    U3.set_dct();
//    A.tucker_decomposition(B,U1,U2,U3);
//    B.debug();
//    U1.debug();
//    U2.debug();
//    U3.debug();
//    std::cout << "Error: " << A.relative_error(B.ttm(U1,U2,U3)) << std::endl;
}

