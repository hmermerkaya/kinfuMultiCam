/*
 * device.hpp
 *
 *  Created on: May 31, 2018
 *      Author: hamit
 */

#ifndef KINFU_SRC_CUDA_DEVICE_HPP_
#define KINFU_SRC_CUDA_DEVICE_HPP_


#include "utils.hpp" //temporary reimplementing to release kinfu without pcl_gpu_utils

#include "../internal.h"

#include "pointer_shift.cu" // contains primitive needed by all cuda functions dealing with rolling tsdf buffer
#include <cuda_runtime.h>
#include <cuda.h>
//#include <gpu/float3_operations.h>
using namespace kinfu::gpu;
  namespace kinfu
  {
    namespace device
    {
      #define INV_DIV 3.051850947599719e-5f

      __device__ __forceinline__ void
      pack_tsdf (float tsdf, int weight, short2& value)
      {
        int fixedp = max (-DIVISOR, min (DIVISOR, __float2int_rz (tsdf * DIVISOR)));
        //int fixedp = __float2int_rz(tsdf * DIVISOR);
        value = make_short2 (fixedp, weight);
      }

      __device__ __forceinline__ void
      unpack_tsdf (short2 value, float& tsdf, int& weight)
      {
        weight = value.y;
        tsdf = __int2float_rn (value.x) / DIVISOR;   //*/ * INV_DIV;
      }

      __device__ __forceinline__ float
      unpack_tsdf (short2 value)
      {
        return static_cast<float>(value.x) / DIVISOR;    //*/ * INV_DIV;
      }


      __device__ __forceinline__ float3
      operator* (const Mat33& m, const float3& vec)
      {
        return make_float3 ((float)dot (m.data[0], vec), (float)dot (m.data[1], vec), (float)dot (m.data[2], vec));
      }


      ////////////////////////////////////////////////////////////////////////////////////////
      ///// Prefix Scan utility

      enum ScanKind { exclusive, inclusive };

      template<ScanKind Kind, class T>
      __device__ __forceinline__ T
      scan_warp ( volatile T *ptr, const unsigned int idx = threadIdx.x )
      {
        const unsigned int lane = idx & 31;       // index of thread in warp (0..31)

        if (lane >=  1) ptr[idx] = ptr[idx -  1] + ptr[idx];
        if (lane >=  2) ptr[idx] = ptr[idx -  2] + ptr[idx];
        if (lane >=  4) ptr[idx] = ptr[idx -  4] + ptr[idx];
        if (lane >=  8) ptr[idx] = ptr[idx -  8] + ptr[idx];
        if (lane >= 16) ptr[idx] = ptr[idx - 16] + ptr[idx];

        if (Kind == inclusive)
          return ptr[idx];
        else
          return (lane > 0) ? ptr[idx - 1] : 0;
      }
    }
  }




#endif /* KINFU_SRC_CUDA_DEVICE_HPP_ */
