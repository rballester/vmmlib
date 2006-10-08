#ifndef _vmml_AxisAlignedBoundingBox_H_
#define _vmml_AxisAlignedBoundingBox_H_

#include "VMMLib.h"
#include "Sphere.h"

namespace vmml
{

template< typename T >
class AxisAlignedBoundingBox
{
public:
    AxisAlignedBoundingBox();
    AxisAlignedBoundingBox( const Vector3< T >& pMin, const Vector3< T >& pMax );
    AxisAlignedBoundingBox( const Sphere< T >& sphere );
    AxisAlignedBoundingBox( T cx, T cy, T cz, T size );
    
    bool isIn( const Vector3< T >& pos );
    bool isIn( const Sphere< T >& sphere );

    inline void set( const Vector3< T >& pMin, const Vector3< T >& pMax );
	inline void setMin( const Vector3< T >& pMin );
	inline void setMax( const Vector3< T >& pMax );
    inline const Vector3< T >& getMin() const;
    inline const Vector3< T >& getMax() const;

protected:
	Vector3< T > _min;
	Vector3< T > _max;
    
};

template< typename T >
AxisAlignedBoundingBox< T >::AxisAlignedBoundingBox()
{}

template< typename T >
AxisAlignedBoundingBox< T >::AxisAlignedBoundingBox( const Vector3< T >& pMin, const Vector3< T >& pMax )
    : _min( pMin )
    , _max( pMax )
{}

template< typename T >
AxisAlignedBoundingBox< T >::AxisAlignedBoundingBox( const Sphere< T >& sphere )
{
    _max = _min = sphere.getCenter();
    _max += sphere.getRadius();
    _min -= sphere.getRadius();
}

template< typename T >
AxisAlignedBoundingBox< T >::AxisAlignedBoundingBox( T cx, T cy, T cz, T size )
{
    _max = _min = Vector3f( cx, cy, cz );
    _max += size;
    _min -= size;
}

template< typename T >
bool AxisAlignedBoundingBox< T >::isIn( const Sphere< T >& sphere )
{
    Vector3< T > sv ( sphere.getCenter() );
    sv += sphere.getRadius();
    if ( sv.x > _max.x || sv.y > _max.y || sv.z > _max.z )
        return false;
    sv -= sphere.getRadius() * 2.0f;
    if ( sv.x < _min.x || sv.y < _min.y || sv.z < _min.z )
        return false;
    return true;
}

template< typename T >
bool AxisAlignedBoundingBox< T >::isIn( const Vector3< T >& pos )
{
    if ( pos.x > _max.x || pos.y > _max.y || pos.z > _max.z 
            || pos.x < _min.x || pos.y < _min.y || pos.z < _min.z )
    {
        return false;
    }
    return true;
}


template< typename T >
inline void AxisAlignedBoundingBox< T >
    ::set( const Vector3< T >& pMin, const Vector3< T >& pMax )
{
    _min = pMin;
    _max = pMax;
}

template< typename T >
inline void AxisAlignedBoundingBox< T >
    ::setMin( const Vector3< T >& pMin ) 
{ 
    _min = pMin; 
}

template< typename T >
inline void AxisAlignedBoundingBox< T >
    ::setMax( const Vector3< T >& pMax ) 
{ 
    _max = pMax; 
}

template< typename T >
inline const Vector3< T >& AxisAlignedBoundingBox< T >
    ::getMin() const 
{   
    return _min; 
}

template< typename T >
inline const Vector3< T >& AxisAlignedBoundingBox< T >
    ::getMax() const
{ 
    return _max; 
};


typedef AxisAlignedBoundingBox< float > Aabbf;

}; //namespace vmml

#endif