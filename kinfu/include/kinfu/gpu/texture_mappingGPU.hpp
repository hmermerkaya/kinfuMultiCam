/*
 * texture_mappingGPU.hpp
 *
 *  Created on: Aug 30, 2018
 *      Author: hamit
 */

#ifndef KINFU_INCLUDE_KINFU_GPU_TEXTURE_MAPPINGGPU_HPP_
#define KINFU_INCLUDE_KINFU_GPU_TEXTURE_MAPPINGGPU_HPP_
#include <gpu/texture_mappingGPU.h>
#include "../src/internal.h"

template<typename PointInT> void
kinfu::TextureMappingGPU<PointInT>::textureMeshwithMultipleCameras (kinfu::gpu::DeviceArray< pcl::PointXYZ /*kinfu::device::PointType*/> &meshIn,  const kinfu::texture_mappingGPU::CameraVector &cameras)
{


kinfu::gpu::DeviceArray<float3> mesh_cloud((float3 *)meshIn.ptr(), (size_t)meshIn.size()) ; ///*pcl::PointXYZ*/ > pointsArray (meshIn.ptr(), meshIn.size());
	//std::vector<float3> vecfloat3;
	//pointsArray.download(vecfloat3);
	for (int current_cam = 0; current_cam < static_cast<int> (cameras.size ()); ++current_cam)
	  {
		Eigen::Matrix3f Rmat = cameras[current_cam].pose.inverse().linear();
		Eigen::Vector3f rvec = cameras[current_cam].pose.inverse().translation();


		 kinfu::device::Mat33 rotation_out = device_cast<kinfu::device::Mat33> (Rmat);
		 float3 translation_out = device_cast<float3>(rvec);

		 kinfu::gpu::DeviceArray<float3> cam_cloud;
		 kinfu::device::transformCloudGPU(mesh_cloud, cam_cloud, rotation_out, translation_out  );

		 kinfu::gpu::DeviceArray<float2> projections;

		 float height = cameras[current_cam].height;
		 float width = cameras[current_cam].width;
		 float focal_length = cameras[current_cam].focal_length;

		 kinfu::device::areFacesProjected(cam_cloud, projections, height, width, focal_length);



		 std::vector<float3> vecfloat3;
		 cam_cloud.download(vecfloat3);
		 int j=2;
	  }
	//int i=vecfloat3.size();
	//int j=2;
}


#endif /* KINFU_INCLUDE_KINFU_GPU_TEXTURE_MAPPINGGPU_HPP_ */
