/*
 * texture_mapping.cpp
 *
 *  Created on: Aug 30, 2018
 *      Author: hamit
 */

#include "device.hpp"
//#include <gpu/texture_mappingGPU.h>



namespace kinfu {

	namespace device {

		__global__ void
		transformCloudGPUKernel(const PtrSz<float3> cloudIn, PtrSz<float3> cloudOut, const Mat33 Rmat, const float3 tvec ){

           int pointid= threadIdx.x + blockIdx.x * blockDim.x;

           float3 point =  cloudIn[pointid];

           if (!isnan(point.x) && !isnan(point.y) && !isnan(point.z)) {

        	   float3 tmppoint =  Rmat*point + tvec;
        	   cloudOut[pointid] = tmppoint;

           }







		}


		void transformCloudGPU(const  DeviceArray<float3> &cloudIn, DeviceArray<float3> &cloudOut,const Mat33& Rmat, const float3& tvec ){

			cloudOut.create(cloudIn.size());
			dim3 block (512);
			dim3 grid (divUp (cloudIn.size(), block.x));

			transformCloudGPUKernel<<<grid, block>>>(cloudIn, cloudOut, Rmat, tvec );
			cudaSafeCall ( cudaGetLastError () );
			cudaSafeCall (cudaDeviceSynchronize ());

		}




	}



}

namespace kinfu {

	namespace device {

		struct TextureCam {

			float cx, cy, focal_length_x, focal_length_y;
			const float3* points;
			mutable float2* projs;
			const float qnan = numeric_limits<float>::quiet_NaN ();

			__device__ __forceinline__ void
			operator () () const {

				int id= threadIdx.x + blockIdx.x * blockDim.x;

				float uv1X = ((focal_length_x * (points[id*3].x / points[id*3].z) + cx) / 2*cx); //horizontal
				float uv1Y = 1 -  ((focal_length_y * (points[id*3].y / points[id*3].z ) + cy) / 2*cy); //vertical
				bool uv1 = uv1X >= 0.0 && uv1X <= 1.0 && uv1Y  >= 0.0 && uv1Y  <= 1.0;


				float uv2X = ((focal_length_x * (points[id*3+1].x / points[id*3+1].z) + cx) / 2*cx); //horizontal
				float uv2Y = 1 -  ((focal_length_y * (points[id*3+1].y / points[id*3+1].z ) + cy) / 2*cy); //vertical
				bool uv2 = uv2X >= 0.0 && uv2X <= 1.0 && uv2Y  >= 0.0 && uv2Y  <= 1.0;

				float uv3X = ((focal_length_x * (points[id*3 +2].x / points[id*3+2].z) + cx) / 2*cx); //horizontal
				float uv3Y = 1 -  ((focal_length_y * (points[id*3+2].y / points[id*3+2].z ) + cy) / 2*cy); //vertical
				bool uv3 = uv3X >= 0.0 && uv3X <= 1.0 && uv3Y  >= 0.0 && uv3Y  <= 1.0;

				if ( uv1 && uv2 && uv3) {

					projs[id*3] = make_float2(uv1X, uv1Y);
					projs[id*3+1] = make_float2(uv2X, uv2Y);
					projs[id*3+2] = make_float2(uv3X, uv3Y);

				} else {
					projs[id*3] = make_float2(qnan, qnan);
					projs[id*3+1] = make_float2(qnan, qnan);
					projs[id*3+2] = make_float2(qnan, qnan);

				}











//				if ( uv1 ) {
//					float uv2X = ((focal_length_x * (points[id*3+1].x / points[id*3+1].z) + cx) / 2*cx); //horizontal
//					float uv2Y = 1-  ((focal_length_y * (points[id*3+1].y / points[id*3+1].z ) + cy) / 2*cy); //vertical
//					bool uv2 = uv2X >= 0.0 && uv2X <= 1.0 && uv2Y  >= 0.0 && uv2Y  <= 1.0;
//
//					if ( uv2 ) {
//
//						float uv3X = ((focal_length_x * (points[id*3 +2].x / points[id*3+2].z) + cx) / 2*cx); //horizontal
//						float uv3Y = 1-  ((focal_length_y * (points[id*3+2].y / points[id*3+2].z ) + cy) / 2*cy); //vertical
//						bool uv3 = uv3X >= 0.0 && uv3X <= 1.0 && uv3Y  >= 0.0 && uv3Y  <= 1.0;
//
//						if ( uv3 ) {
//
//
//
//						}
//
//					}
//
//
//
//
//
//				}

			}



		};


	__global__ void
	areFacesProjectedKernel(const TextureCam cam) {
		cam();
	}


	void
	areFacesProjected(const DeviceArray<float3> &cloudIn, DeviceArray<float2> &projections, float height, float width, float focal_length) {


		projections.create(cloudIn.size());

		TextureCam cam;

		cam.cy = height/2;
		cam.cx = width/2;
		cam.focal_length_x = focal_length;
		cam.focal_length_y = focal_length;

		cam.projs = projections.ptr();
		cam.points = cloudIn.ptr();


		dim3 block (512);
		dim3 grid (divUp (cloudIn.size()/3, block.x));
		areFacesProjectedKernel<<<grid, block >>>(cam);
		cudaSafeCall ( cudaGetLastError () );
		cudaSafeCall (cudaDeviceSynchronize ());





	}




	}
}






