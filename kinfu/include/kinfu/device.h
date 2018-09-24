/*
 * device.h
 *
 *  Created on: May 31, 2018
 *      Author: hamit
 */

#ifndef KINFU_INCLUDE_KINFU_DEVICE_H_
#define KINFU_INCLUDE_KINFU_DEVICE_H_


#ifdef __JETBRAINS_IDE__
#define __CUDACC__ 1
#define __host__
#define __device__
#define __global__
#define __forceinline__
#define __shared__
inline void __syncthreads() {}
inline void __threadfence_block() {}
template<class T> inline T __clz(const T val) { return val; }
struct __cuda_fake_struct { int x; int y; int z;};
extern __cuda_fake_struct blockDim;
extern __cuda_fake_struct threadIdx;
extern __cuda_fake_struct blockIdx;
#endif



#include <gpu/device_array.h>
#include <iostream> // used by operator
//#include <pcl/gpu/kinfu_large_scale/tsdf_buffer.h>
#include <cuda_runtime.h>

// using namespace pcl::gpu;


namespace kinfu
{
	namespace device
	{
		///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		// Types
		typedef unsigned short ushort;
		typedef DeviceArray2D<float> MapArr;
		typedef DeviceArray2D<ushort> DepthMap;
		typedef float4 PointType;

		//TSDF fixed point divisor (if old format is enabled)
		const int DIVISOR = 32767;     // SHRT_MAX;

		//RGB images resolution kinect_v1
		// const float  HEIGHT = 480.0f;
		// const float  WIDTH = 640.0f;

		//RGB images resolution kinect_v2
		const float  HEIGHT = 424.0f;
		const float  WIDTH = 512.0f;

		//Should be multiple of 32
		enum { VOLUME_X = 256, VOLUME_Y = 256, VOLUME_Z = 256 };


		//Temporary constant (until we make it automatic) that holds the Kinect's focal length kinect_v1
		//const float FOCAL_LENGTH = 575.816f;

		// kinect_v2
		const float FOCAL_LENGTH = 364.963f;

		const float VOLUME_SIZE = 3.0f; // physical size represented by the TSDF volume. In meters
		const float DISTANCE_THRESHOLD = 1.5f; // when the camera target point is farther than DISTANCE_THRESHOLD from the current cube's center, shifting occurs. In meters
		const int SNAPSHOT_RATE = 45; // every 45 frames an RGB snapshot will be saved. -et parameter is needed when calling Kinfu Large Scale in command line.


		/** \brief Camera intrinsics structure
		*/
		struct Intr
		{
		float fx, fy, cx, cy;
		Intr () {}
		Intr (float fx_, float fy_, float cx_, float cy_) : fx (fx_), fy (fy_), cx (cx_), cy (cy_) {}

		Intr operator () (int level_index) const
		{
		  int div = 1 << level_index;
		  return (Intr (fx / div, fy / div, cx / div, cy / div));
		}

		friend inline std::ostream&
		operator << (std::ostream& os, const Intr& intr)
		{
		  os << "([f = " << intr.fx << ", " << intr.fy << "] [cp = " << intr.cx << ", " << intr.cy << "])";
		  return (os);
		}
		};

		/** \brief 3x3 Matrix for device code
		*/
		struct Mat33
		{
		float3 data[3];
		};
	}
}


#endif /* KINFU_INCLUDE_KINFU_DEVICE_H_ */
