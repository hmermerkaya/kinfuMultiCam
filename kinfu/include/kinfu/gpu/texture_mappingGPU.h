/*
 * texture_mappingGPU.h
 *
 *  Created on: Aug 30, 2018
 *      Author: hamit
 */

#ifndef KINFU_INCLUDE_KINFU_GPU_TEXTURE_MAPPINGGPU_H_
#define KINFU_INCLUDE_KINFU_GPU_TEXTURE_MAPPINGGPU_H_
#include <pcl/common/transforms.h>

#include <gpu/device_array.h>
#include <device.h>
#include <TextureMesh.h>


namespace kinfu {
	namespace texture_mappingGPU {


	 struct Camera
	    {
	      Camera () : pose (), focal_length (), focal_length_w (-1), focal_length_h (-1),
	        center_w (-1), center_h (-1), height (), width (), texture_file () {}
	      Eigen::Affine3f pose;
	      double focal_length;
	      double focal_length_w;  // optional
	      double focal_length_h;  // optinoal
	      double center_w;  // optional
	      double center_h;  // optional
	      double height;
	      double width;
	      std::string texture_file;

	      EIGEN_MAKE_ALIGNED_OPERATOR_NEW
	    };

	    /** \brief Structure that links a uv coordinate to its 3D point and face.
	      */
	    struct UvIndex
	    {
	      UvIndex () : idx_cloud (), idx_face () {}
	      int idx_cloud; // Index of the PointXYZ in the camera's cloud
	      int idx_face; // Face corresponding to that projection
	    };

	    typedef std::vector<Camera, Eigen::aligned_allocator<Camera> > CameraVector;
	}

	template<typename PointInT>

		class TextureMappingGPU {



		public:
			typedef kinfu::texture_mappingGPU::Camera Camera;

			TextureMappingGPU() :
			tex_files_ (), tex_material_ ()
			{

			}
			~TextureMappingGPU(){

			}

		template<class D, class Matx> D&
		      device_cast (Matx& matx)
		      {
		        return (*reinterpret_cast<D*>(matx.data ()));
		      }
		void textureMeshwithMultipleCameras ( kinfu::gpu::DeviceArray< pcl::PointXYZ /* kinfu::device::PointType*/> &meshIn,  const kinfu::texture_mappingGPU::CameraVector &cameras);
		private:
			std::vector<std::string> tex_files_;

			      /** \brief list of texture materials */
			TexMaterial tex_material_;
		};


}

#include <gpu/texture_mappingGPU.hpp>
#endif /* KINFU_INCLUDE_KINFU_GPU_TEXTURE_MAPPINGGPU_H_ */
