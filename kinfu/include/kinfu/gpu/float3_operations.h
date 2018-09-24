/*
 * float3_operations.h
 *
 *  Created on: Jun 1, 2018
 *      Author: hamit
 */

#ifndef KINFU_INCLUDE_KINFU_GPU_FLOAT3_OPERATIONS_H_
#define KINFU_INCLUDE_KINFU_GPU_FLOAT3_OPERATIONS_H_
#include <vector_types.h>
#include <vector_functions.h>
#include <iostream>

  namespace kinfu
  {
    namespace gpu
    {
      inline float
      dot(const float3& v1, const float3& v2)
      {
        return v1.x * v2.x + v1.y*v2.y + v1.z*v2.z;
      }

      inline float3&
      operator+=(float3 & vec, const float& v)
      {
        vec.x += v;  vec.y += v;  vec.z += v; return vec;
      }

      inline float3&
      operator+=(float3& vec, const float3& v)
      {
        vec.x += v.x;  vec.y += v.y;  vec.z += v.z; return vec;
      }

      inline float3
      operator+(const float3& v1, const float3& v2)
      {
        return make_float3 (v1.x + v2.x, v1.y + v2.y, v1.z + v2.z);
      }

      inline float3&
      operator*=(float3& vec, const float& v)
      {
        vec.x *= v;  vec.y *= v;  vec.z *= v; return vec;
      }

      inline float3&
      operator-=(float3& vec, const float& v)
      {
        vec.x -= v;  vec.y -= v;  vec.z -= v; return vec;
      }

      inline float3&
      operator-=(float3& vec, const float3& v)
      {
        vec.x -= v.x;  vec.y -= v.y;  vec.z -= v.z; return vec;
      }

      inline float3
      operator-(const float3& v1, const float3& v2)
      {
        return make_float3 (v1.x - v2.x, v1.y - v2.y, v1.z - v2.z);
      }

      inline float3
      operator-(const float3& v1)
      {
        return make_float3 (-v1.x, -v1.y, -v1.z);
      }

      inline float3
      operator-(float3& v1)
      {
        v1.x = -v1.x; v1.y = -v1.y; v1.z = -v1.z; return v1;
      }

      inline float3
      operator*(const float3& v1, const float& v)
      {
        return make_float3 (v1.x * v, v1.y * v, v1.z * v);
      }

      inline float
      norm(const float3& v)
      {
        return sqrt (dot (v, v));
      }

      inline std::ostream&
      operator << (std::ostream& os, const float3& v1)
      {
        os << "[" << v1.x << ", " << v1.y <<  ", " << v1.z<< "]";
        return (os);
      }

      /*inline float3
      normalized(const float3& v)
      {
        return v * rsqrt(dot(v, v));
      }*/

      inline float3
      cross(const float3& v1, const float3& v2)
      {
        return make_float3 (v1.y * v2.z - v1.z * v2.y, v1.z * v2.x - v1.x * v2.z, v1.x * v2.y - v1.y * v2.x);
      }
    }
  }




#endif /* KINFU_INCLUDE_KINFU_GPU_FLOAT3_OPERATIONS_H_ */
