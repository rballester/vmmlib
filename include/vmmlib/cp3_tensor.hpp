/* 
 * VMMLib - Tensor Classes
 *  
 * @author Susanne Suter
 * @author Jonas Boesch
 *
 * The cp3 tensor class is consists of three basis matrices u1-u3 and R lambda values for a given rank-R approximation
 * CP stands for Candecomp/Parafac (1970)
 * - Carroll, J. and Chang, Jih-Jie: Analysis of Individual Differences in Multidimensional Scaling via an N-way generalization of ``Eckart--Young'' decompositions,
 * Psychometrika 35: 283–319, 1970
 * - R. A. Harshman: Foundations of the PARAFAC procedure: Models and conditions for an 'explanatory' multi-modal factor analysis,
 * UCLA Working Papers in Phonetics, Vol. 16, No. 1., 1970
 * - De Lathauwer L., De Moor B., Vandewalle J., ``On the Best rank-1 and Rank-$(R_1, R_2, ..., R_N)$ Approximation and Applications of Higher-Order Tensors'', 
 * SIAM J. Matrix Anal. Appl., vol. 21, no. 4, Apr. 2000, pp. 1324-1342.
 * - T. G. Kolda and B. W. Bader. Tensor Decompositions and Applications. 
 * SIAM Review, Volume 51, Number 3, Pages 455-500, September 2009.
 * 
 */

#ifndef __VMML__CP3_TENSOR__HPP__
#define __VMML__CP3_TENSOR__HPP__

#include <vmmlib/tensor3.hpp>
#include <vmmlib/tensor3_iterator.hpp>
#include <vmmlib/lapack_svd.hpp>
#include <vmmlib/matrix_pseudoinverse.hpp>

//TODO allocate data with new

namespace vmml
{
	
	template< size_t I1, size_t I2, size_t I3, size_t R, typename T_value = float, typename T_coeff = float >
	class cp3_tensor
	{
	public:    		
		typedef float T_internal;	
		typedef double T_svd;

		typedef tensor3< I1, I2, I3, T_value > t3_type;
		typedef typename t3_type::iterator t3_iterator;
		typedef typename t3_type::const_iterator t3_const_iterator;
		
		typedef tensor3< I1, I2, I3, T_internal > t3_comp_type;
		
		typedef tensor3< I1, I2, I3, T_coeff > t3_coeff_type;
		typedef typename t3_coeff_type::iterator t3_coeff_iterator;
		typedef typename t3_coeff_type::const_iterator t3_coeff_const_iterator;
		
		typedef matrix< I1, R, T_coeff > u1_type;
		typedef typename u1_type::iterator u1_iterator;
		typedef typename u1_type::const_iterator u1_const_iterator;
		
		typedef matrix< I2, R, T_coeff > u2_type;
		typedef typename u2_type::iterator u2_iterator;
		typedef typename u2_type::const_iterator u2_const_iterator;
		
		typedef matrix< I3, R, T_coeff > u3_type;
		typedef typename u3_type::iterator u3_iterator;
		typedef typename u3_type::const_iterator u3_const_iterator;
		
		typedef matrix< I1, R, T_internal > u1_comp_type;
		typedef matrix< I2, R, T_internal > u2_comp_type;
		typedef matrix< I3, R, T_internal > u3_comp_type;
		
		typedef vector< R, T_internal > lambda_comp_type;
		typedef vector< R, T_coeff > lambda_type;
		
		typedef matrix< I1, I2*I3, T_internal > mode1_matricization_type;
		typedef matrix< I2, I1*I3, T_internal > mode2_matricization_type;
		typedef matrix< I3, I1*I2, T_internal > mode3_matricization_type;		

		cp3_tensor(  u1_type& U1, u2_type& U2, u3_type& U3, lambda_type& lambdas_ );
		cp3_tensor();
		~cp3_tensor();
		
		void get_lambdas( lambda_type& data_ ) const { data_  = *_lambdas; } ;
		void get_u1( u1_type& U1 ) const { U1 = *_u1; } ;
		void get_u2( u2_type& U2 ) const { U2 = *_u2; } ;
		void get_u3( u3_type& U3 ) const { U3 = *_u3; } ;
		
		void set_core( const lambda_type& lambdas_ )  { _lambdas = lambda_type( lambdas_ ); _lambdas_comp.cast_from( _lambdas ); } ;
		void set_u1( u1_type& U1 ) { *_u1 = U1; _u1_comp->cast_from( U1 ); } ;
		void set_u2( u2_type& U2 ) { *_u2 = U2; _u1_comp->cast_from( U2 ); } ;
		void set_u3( u3_type& U3 ) { *_u3 = U3; _u1_comp->cast_from( U3 ); } ;
		
		void set_lambda_comp( lambda_comp_type& lambdas_ )  { _lambdas_comp = lambda_comp_type( lambdas_ ); _lambdas.cast_from( _lambdas_comp ); } ;
		void set_u1_comp( u1_comp_type& U1 ) { *_u1_comp = U1; _u1->cast_from( U1 ); } ;
		void set_u2_comp( u2_comp_type& U2 ) { *_u2_comp = U2; _u1->cast_from( U2 ); } ;
		void set_u3_comp( u3_comp_type& U3 ) { *_u3_comp = U3; _u1->cast_from( U3 ); } ;
		
		void get_lambda_comp( lambda_comp_type& data_ ) const { data_ = _lambdas_comp; } ;
		void get_u1_comp( u1_comp_type& U1 ) const { U1 = *_u1_comp; } ;
		void get_u2_comp( u2_comp_type& U2 ) const { U2 = *_u2_comp; } ;
		void get_u3_comp( u3_comp_type& U3 ) const { U3 = *_u3_comp; } ;
		
		void export_to( std::vector< T_coeff >& data_ ) const;
		void import_from( std::vector< T_coeff >& data_ );	
		
		void reconstruct( t3_type& data_ ) const;
		void decompose( const t3_type& data_, const size_t max_iterations_ = 100 ); 
		void cp_als( const t3_type& data_, const size_t max_iterations_ = 100 );
		//higher-order power method (lathauwer et al., 2000b)
		void hopm( const t3_type& data_, const size_t max_iterations_ = 100 );
		
		template< size_t M, size_t N, typename T >
		void get_svd_u( const matrix< M, N, T >& data_, matrix< M, R, T_internal >& u_ );

		template< size_t J1, size_t J2, size_t J3, typename T >
		void hosvd_mode2(  const tensor3<J1, J2, J3, T >& data_ );
		template< size_t J1, size_t J2, size_t J3, typename T >
		void hosvd_mode3( const tensor3<J1, J2, J3, T >& data_ );
		
		void optimize_mode1( const t3_comp_type& data_ );
		void optimize_mode2( const t3_comp_type& data_ );		
		void optimize_mode3( const t3_comp_type& data_ );
		
	protected:
		cp3_tensor( const cp3_tensor< R, I1, I1, I1, T_value, T_coeff >& other ) {};
		cp3_tensor< R, I1, I1, I1, T_value, T_coeff > operator=( const cp3_tensor< R, I1, I1, I1, T_value, T_coeff >& other ) { return *this; };

        void cast_members();
        void cast_comp_members();
		
	private:
		lambda_type* _lambdas ;
		u1_type* _u1 ;
		u2_type* _u2 ;
		u3_type* _u3 ;
		
        lambda_comp_type* _lambdas_comp ;
        u1_comp_type* _u1_comp ;
        u2_comp_type* _u2_comp ;
        u3_comp_type* _u3_comp ;
		
	}; // class cp3_tensor
	
	
#define VMML_TEMPLATE_STRING    	template< size_t I1, size_t I2, size_t I3, size_t R, typename T_value, typename T_coeff >
#define VMML_TEMPLATE_CLASSNAME     cp3_tensor< I1, I2, I3, R, T_value, T_coeff >
	
	
VMML_TEMPLATE_STRING
VMML_TEMPLATE_CLASSNAME::cp3_tensor( u1_type& U1, u2_type& U2, u3_type& U3, lambda_type& lambdas_ )
{
	set_lambdas(lambdas_);
	set_u1( U1);
	set_u2( U2);
	set_u3( U3);
}

VMML_TEMPLATE_STRING
VMML_TEMPLATE_CLASSNAME::cp3_tensor()
{
	_lambdas = new vector< R, T_coeff>(); 
	_lambdas->set( 0 );
	_u1 = new u1_type(); _u1->zero();
	_u2 = new u2_type(); _u2->zero();
	_u3 = new u3_type(); _u3->zero();
	_lambdas_comp = new vector< R, T_internal>; 
	_lambdas_comp->set( 0 );
	_u1_comp = new u1_comp_type; _u1_comp->zero();
	_u2_comp = new u2_comp_type; _u2_comp->zero();
	_u3_comp = new u3_comp_type; _u3_comp->zero();
}
	
VMML_TEMPLATE_STRING
VMML_TEMPLATE_CLASSNAME::~cp3_tensor()
{
	delete _u1;
	delete _u2;
	delete _u3;
	delete _lambdas;
	delete _u1_comp;
	delete _u2_comp;
	delete _u3_comp;
	delete _lambdas_comp;
}
	
	
VMML_TEMPLATE_STRING
void
VMML_TEMPLATE_CLASSNAME::cast_members()
{
	_u1->cast_from( *_u1_comp );
	_u2->cast_from( *_u2_comp );
	_u3->cast_from( *_u3_comp );	
	_lambdas->cast_from( *_lambdas_comp );
}

VMML_TEMPLATE_STRING
void
VMML_TEMPLATE_CLASSNAME::cast_comp_members()
{
	_u1_comp->cast_from( *_u1 );
	_u2_comp->cast_from( *_u2 );
	_u3_comp->cast_from( *_u3 );	
	_lambdas_comp->cast_from( _lambdas );
}
	
	
VMML_TEMPLATE_STRING
void 
VMML_TEMPLATE_CLASSNAME::reconstruct( t3_type& data_ ) const
{
	//FIXME: check data types
    t3_comp_type data;
    data.cast_from( data_ );
	
	typedef matrix< R, I1, T_internal > u1_comp_t_type;
	typedef matrix< R, I2, T_internal > u2_comp_t_type;
	typedef matrix< R, I3, T_internal > u3_comp_t_type;
	
	u1_comp_t_type* u1_t = new u1_comp_t_type;
	u2_comp_t_type* u2_t = new u2_comp_t_type;
	u3_comp_t_type* u3_t = new u3_comp_t_type;
	typedef 	matrix<  R, I2 * I3, T_internal > m_temp_type;
	m_temp_type* temp =  new m_temp_type; 
	
	*u1_t = transpose(*_u1_comp);
	*u2_t = transpose(*_u2_comp);
	*u3_t = transpose(*_u3_comp);
	
    data.reconstruct_CP( *_lambdas_comp, *u1_t, *u2_t, *u3_t, *temp );
	
	delete temp;
	delete u1_t;
	delete u2_t;
	delete u3_t;
 
     //convert reconstructed data, which is in type T_internal (double, float) to T_value (uint8 or uint16)
    if( (sizeof(T_value) == 1) || (sizeof(T_value) == 2) ){
	data_.float_t_to_uint_t( data );
    } else {
	   data_.cast_from( data );
    }
}


VMML_TEMPLATE_STRING
void 
VMML_TEMPLATE_CLASSNAME::decompose( const t3_type& data_, const size_t max_iterations_  )
{
	cp_als( data_, max_iterations_ );
}

VMML_TEMPLATE_STRING
void 
VMML_TEMPLATE_CLASSNAME::cp_als( const t3_type& data_, const size_t max_iterations_  )
{
	hopm( data_, max_iterations_ );
}

VMML_TEMPLATE_STRING
void 
VMML_TEMPLATE_CLASSNAME::hopm( const t3_type& data_, const size_t max_iterations_ )
{
	t3_comp_type data;
	data.cast_from( data_ );
	t3_type approximated_data;
	t3_type residual_data;
	residual_data.zero();
	
	double approx_norm = 0;
	double max_f_norm = data.frobenius_norm();
	double normresidual  = 0;
	double fit = 0;
	if (max_f_norm == 0 )
		fit = 1;
	double fitchange = 1;
	double fitold = fit;
	double fitchange_tolerance = 1.0e-4;
	
	//intialize u1-u3
	//hosvd_mode1( data_, _u1 ); inital guess not needed for u1 since it will be computed in the first optimization step
	hosvd_mode2( data_ );
	hosvd_mode3( data_ );

#if CP_LOG
	std::cout << "CP ALS: HOPM (for tensor3) " << std::endl;
#endif	
	
	size_t i = 0;
	//size_t max_iterations = 100;
	while( (fitchange >= fitchange_tolerance) && ( i < max_iterations_ ) ) //do until converges
	{
		fitold = fit;
		optimize_mode1( data );
		optimize_mode2( data );
		optimize_mode3( data );
		
		//Reconstruct cptensor and measure norm of approximation
		reconstruct( approximated_data ); //FIX reconstruction
		approx_norm = approximated_data.frobenius_norm();
		residual_data = data_ - approximated_data;
		normresidual = residual_data.frobenius_norm();
		fit = 1 - ( normresidual / max_f_norm ); 
		fitchange = fabs(fitold - fit);
		
#if CP_LOG
		std::cout << "iteration '" << i << "', fit: " << fit 
		<< ", fitdelta: " << fitchange 
		<< ", frobenius norm: " << approx_norm << std::endl;		
#endif
		++i;
	} // end ALS

 	cast_members();
}

	
	
VMML_TEMPLATE_STRING
template< size_t J1, size_t J2, size_t J3, typename T >
void 
VMML_TEMPLATE_CLASSNAME::hosvd_mode2( const tensor3<J1, J2, J3, T >& data_ )
{
	typedef matrix< J2, J1*J3, T > unfolded_matrix_type;
	unfolded_matrix_type* u = new unfolded_matrix_type; // -> u1
	data_.frontal_unfolding_bwd( *u );
	
	get_svd_u( *u, *_u2_comp );
	
	delete u;
}
	
		
VMML_TEMPLATE_STRING
template< size_t J1, size_t J2, size_t J3, typename T >
void 
VMML_TEMPLATE_CLASSNAME::hosvd_mode3( const tensor3<J1, J2, J3, T >& data_  )
{
	typedef matrix< J3, J1*J2, T > unfolded_matrix_type;
	unfolded_matrix_type* u = new unfolded_matrix_type; // -> u1
	data_.horizontal_unfolding_bwd( *u );
	
	get_svd_u( *u, *_u3_comp );
	
	delete u;
}
	

VMML_TEMPLATE_STRING
void 
VMML_TEMPLATE_CLASSNAME::optimize_mode1( const t3_comp_type& data_ )
{	
	mode1_matricization_type* unfolding = new mode1_matricization_type; // -> u1
	//data_.horizontal_unfolding_bwd( *unfolding ); //lathauwer
	data_.frontal_unfolding_fwd( *unfolding ); //lathauwer
	
	typedef matrix< I2*I3, R, T_internal > krp_matrix_type;
	krp_matrix_type* u1_krp  = new krp_matrix_type;
	//*u1_krp = _u2_comp->khatri_rao_product( *_u3_comp );	
	*u1_krp = _u3_comp->khatri_rao_product( *_u2_comp );	
	u1_comp_type* u_new = new u1_comp_type;
	u_new->multiply( *unfolding, *u1_krp );
	
	//std::cout << "khatri-rao:\n" << *u1_krp << std::endl;
	//std::cout << "unfolding:\n" << *unfolding << std::endl;
	//std::cout << "after khatri-rao mult(u_new):\n" << *u_new << std::endl;
	
	typedef matrix< R, R , T_internal > m_r2_type;
	m_r2_type* u2_r = new m_r2_type;
	m_r2_type* u3_r = new m_r2_type;
	
	u2_r->multiply( transpose(*_u2_comp), *_u2_comp );
	u3_r->multiply( transpose(*_u3_comp), *_u3_comp );
	u2_r->multiply_piecewise( *u3_r );
	
	m_r2_type* pinv_t = new m_r2_type;
	compute_pseudoinverse< m_r2_type > compute_pinv;
	compute_pinv( *u2_r, *pinv_t );
	
	_u1_comp->multiply( *u_new, transpose(*pinv_t) );
	//std::cout << "u1_comp new (before normalization):\n" << *_u1_comp << std::endl;

	*u_new = *_u1_comp;
	u_new->multiply_piecewise( *u_new ); //2 norm
	u_new->columnwise_sum( *_lambdas_comp );
	_lambdas_comp->sqrt_elementwise();
	lambda_comp_type* tmp = new lambda_comp_type;
	//std::cout << "lambdas:\n" << *_lambdas_comp << std::endl;
	*tmp = *_lambdas_comp;
	tmp->reciprocal();
	u1_comp_type* diag_lambdas = new u1_comp_type;
	diag_lambdas->diag( *tmp );
	//FIXME: compute max norm for higher iterations
	
	
	*u_new = *_u1_comp;
	_u1_comp->multiply( *u_new, *diag_lambdas ); 
	//std::cout << "diag_lambdas_rec: " << *diag_lambdas << std::endl;
	//std::cout << "u_new: " << *u_new << std::endl;
	//std::cout << "normalized u1: " << *_u1_comp << std::endl;

	delete unfolding;
	delete u1_krp;
	delete u2_r;
	delete u3_r;
	delete pinv_t;
	delete u_new;
	delete diag_lambdas;
	delete tmp;
}


VMML_TEMPLATE_STRING
void 
VMML_TEMPLATE_CLASSNAME::optimize_mode2( const t3_comp_type& data_ )
{
	mode2_matricization_type* unfolding = new mode2_matricization_type; // -> u2
	//data_.frontal_unfolding_bwd( *unfolding ); //lathauwer
	data_.frontal_unfolding_bwd( *unfolding );
	
	typedef matrix< I1*I3, R, T_internal > krp_matrix_type;
	krp_matrix_type* u2_krp  = new krp_matrix_type;
	*u2_krp = _u3_comp->khatri_rao_product( *_u1_comp );	
	u2_comp_type* u_new = new u2_comp_type;
	u_new->multiply( *unfolding, *u2_krp );
	
	//std::cout << "khatri-rao:\n" << *u2_krp << std::endl;
	//std::cout << "unfolding:\n" << *unfolding << std::endl;
	//std::cout << "after khatri-rao mult(u_new):\n" << *u_new << std::endl;
	
	typedef matrix< R, R , T_internal > m_r2_type;
	m_r2_type* u1_r = new m_r2_type;
	m_r2_type* u3_r = new m_r2_type;
	
	u1_r->multiply( transpose(*_u1_comp), *_u1_comp );
	u3_r->multiply( transpose(*_u3_comp), *_u3_comp );
	u1_r->multiply_piecewise( *u3_r );
	
	m_r2_type* pinv_t = new m_r2_type;
	compute_pseudoinverse< m_r2_type > compute_pinv;
	compute_pinv( *u1_r, *pinv_t );
	
	_u2_comp->multiply( *u_new, transpose(*pinv_t) );
	//std::cout << "u2_comp new (before normalization):\n" << *_u2_comp << std::endl;
	
	//normalize with lambdas
	*u_new = *_u2_comp;
	u_new->multiply_piecewise( *u_new ); //2 norm
	u_new->columnwise_sum( *_lambdas_comp );
	_lambdas_comp->sqrt_elementwise();
	lambda_comp_type* tmp = new lambda_comp_type;
	*tmp = *_lambdas_comp;
	tmp->reciprocal();
	u2_comp_type* diag_lambdas = new u2_comp_type;
	diag_lambdas->diag( *tmp );
	
	*u_new = *_u2_comp;
	_u2_comp->multiply( *u_new, *diag_lambdas );
	//std::cout << "diag_lambdas_rec: " << *diag_lambdas << std::endl;
	//std::cout << "u_new: " << *u_new << std::endl;
	//std::cout << "normalized u2: " << *_u2_comp << std::endl;
	
	delete unfolding;
	delete u2_krp;
	delete u1_r;
	delete u3_r;
	delete pinv_t;
	delete u_new;
	delete diag_lambdas;
	delete tmp;
}	


VMML_TEMPLATE_STRING
void  
VMML_TEMPLATE_CLASSNAME::optimize_mode3( const t3_comp_type& data_ )
{
	mode3_matricization_type* unfolding = new mode3_matricization_type; //-> u3
	//data_.horizontal_unfolding_bwd( *unfolding );//lathauwer
	data_.lateral_unfolding_fwd( *unfolding );
	
	typedef matrix< I1*I2, R, T_internal > krp_matrix_type;
	krp_matrix_type* u3_krp  = new krp_matrix_type;
	*u3_krp = _u2_comp->khatri_rao_product( *_u1_comp );	
	u3_comp_type* u_new = new u3_comp_type;
	u_new->multiply( *unfolding, *u3_krp );
	
	//std::cout << "khatri-rao:\n" << *u3_krp << std::endl;
	//std::cout << "unfolding:\n" << *unfolding << std::endl;
	//std::cout << "after khatri-rao mult(u_new):\n" << *u_new << std::endl;
	
	typedef matrix< R, R , T_internal > m_r2_type;
	m_r2_type* u1_r = new m_r2_type;
	m_r2_type* u2_r = new m_r2_type;
	
	u1_r->multiply( transpose(*_u1_comp), *_u1_comp );
	u2_r->multiply( transpose(*_u2_comp), *_u2_comp );
	u1_r->multiply_piecewise( *u2_r );
	//std::cout << "u1_r:\n" << *u1_r << std::endl;
	
	m_r2_type* pinv_t = new m_r2_type;
	compute_pseudoinverse< m_r2_type > compute_pinv;
	compute_pinv( *u1_r, *pinv_t );
	
	//std::cout << "u_new:\n" << *u_new << std::endl;
	_u3_comp->multiply( *u_new, transpose(*pinv_t) );
	//std::cout << "u3_comp new (before normalization):\n" << *_u3_comp << std::endl;
	
	//normalize with lambdas
	*u_new = *_u3_comp;
	u_new->multiply_piecewise( *u_new ); //2 norm
	u_new->columnwise_sum( *_lambdas_comp );
	_lambdas_comp->sqrt_elementwise();
	lambda_comp_type* tmp = new lambda_comp_type;
	*tmp = *_lambdas_comp;
	tmp->reciprocal();
	u3_comp_type* diag_lambdas = new u3_comp_type;
	diag_lambdas->diag( *tmp );
	
	*u_new = *_u3_comp;
	_u3_comp->multiply( *u_new, *diag_lambdas );
	//std::cout << "diag_lambdas_rec: " << *diag_lambdas << std::endl;
	//std::cout << "u_new: " << *u_new << std::endl;
	//std::cout << "normalized u2: " << *_u3_comp << std::endl;
	delete unfolding;
	delete u3_krp;
	delete u1_r;
	delete u2_r;
	delete pinv_t;
	delete u_new;	
	delete diag_lambdas;
	delete tmp;
}

	
VMML_TEMPLATE_STRING
void
VMML_TEMPLATE_CLASSNAME::export_to( std::vector< T_coeff >& data_ ) const
{
	u1_const_iterator  it = _u1.begin(),
	it_end = _u1.end();
	for( ; it != it_end; ++it )
	{
		data_.push_back( *it );
	}
	
	u2_const_iterator  u2_it = _u2.begin(),
	u2_it_end = _u2.end();
	for( ; u2_it != u2_it_end; ++u2_it )
	{
		data_.push_back( *u2_it );
	}
	
	u3_const_iterator  u3_it = _u3.begin(),
	u3_it_end = _u3.end();
	for( ; u3_it != u3_it_end; ++u3_it )
	{
		data_.push_back( *u3_it );
	}
	
	//TODO: iterate over lambdas
}


VMML_TEMPLATE_STRING
void
VMML_TEMPLATE_CLASSNAME::import_from( std::vector< T_coeff >& data_ )
{
	size_t i = 0; //iterator over data_
	
	u1_iterator  it = _u1.begin(),
	it_end = _u1.end();
	for( ; it != it_end; ++it, ++i )
	{
		*it = data_.at(i);
	}
	
	u2_iterator  u2_it = _u2.begin(),
	u2_it_end = _u2.end();
	for( ; u2_it != u2_it_end; ++u2_it, ++i )
	{
		*u2_it = data_.at(i);
	}
	
	u3_iterator  u3_it = _u3.begin(),
	u3_it_end = _u3.end();
	for( ; u3_it != u3_it_end; ++u3_it, ++i )
	{
		*u3_it = data_.at(i);
	}
	
	//TODO: import lambdas
	
}	


	
VMML_TEMPLATE_STRING
template< size_t M, size_t N, typename T >
void 
VMML_TEMPLATE_CLASSNAME::get_svd_u( const matrix< M, N, T >& data_, matrix< M, R, T_internal >& u_ )
{
	typedef	matrix< M, N, T_svd > svd_type;
	typedef	matrix< M, N, T_coeff > coeff_type;
	typedef	matrix< M, N, T_internal > internal_type;
	typedef vector< N, T_svd > lambdas_type;
	
	svd_type* u_double = new svd_type; 
	u_double->cast_from( data_ );
	
	coeff_type* u_quant = new coeff_type; 
	internal_type* u_internal = new internal_type; 
	
	lambdas_type* lambdas  = new lambdas_type;
	lapack_svd< M, N, T_svd >* svd = new lapack_svd<  M, N, T_svd >();
	if( svd->compute_and_overwrite_input( *u_double, *lambdas )) {
#if FIXME
		if( _is_quantify_coeff ){
			T_internal min_value = 0; T_internal max_value = 0;
			u_internal->cast_from( *u_double );
			u_internal->quantize( *u_quant, min_value, max_value );
			u_quant->dequantize( *u_internal, min_value, max_value );
		} else if ( sizeof( T_internal ) != 4 ){
			u_internal->cast_from( *u_double );
		} else {
			*u_internal = *u_double;
		}
#else
		*u_internal = *u_double;		
#endif
		
		u_internal->get_sub_matrix( u_ );
		
	} else {
		u_.zero();
	}
	
	delete lambdas;
	delete svd;
	delete u_double;
	delete u_quant;
	delete u_internal;
}	

	
		
#undef VMML_TEMPLATE_STRING
#undef VMML_TEMPLATE_CLASSNAME
		
	} // namespace vmml
	
#endif
		
