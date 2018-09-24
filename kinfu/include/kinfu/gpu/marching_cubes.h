/*
 * marching_cubes.h
 *
 *  Created on: May 31, 2018
 *      Author: hamit
 */
//#pragma once
#ifndef KINFU_INCLUDE_KINFU_GPU_MARCHING_CUBES_H_
#define KINFU_INCLUDE_KINFU_GPU_MARCHING_CUBES_H_

#include <pcl/pcl_macros.h>
#include <gpu/device_array.h>
#include <Eigen/Core>
//#include <boost/graph/buffer_concepts.hpp>
#include <pcl/point_types.h>


  namespace kinfu
  {
	namespace gpu
	{
	  class TsdfVolume;

	  /** \brief MarchingCubes implements MarchingCubes functionality for TSDF volume on GPU
		* \author Anatoly Baskeheev, Itseez Ltd, (myname.mysurname@mycompany.com)
		*/
	  class PCL_EXPORTS MarchingCubes
	  {
	  public:

		/** \brief Default size for triangles buffer */
		enum
		{
		  POINTS_PER_TRIANGLE = 3,
		  DEFAULT_TRIANGLES_BUFFER_SIZE = 2 * 1000 * 1000 * POINTS_PER_TRIANGLE * 2
		};

		/** \brief Point type. */
		typedef pcl::PointXYZ PointType;

		/** \brief Smart pointer. */
		typedef boost::shared_ptr<MarchingCubes> Ptr;

		/** \brief Default constructor */
		MarchingCubes();

		/** \brief Destructor */
		~MarchingCubes();

		/** \brief Runs marching cubes triangulation.
			* \param[in] tsdf
			* \param[in] triangles_buffer Buffer for triangles. Its size determines max extracted triangles. If empty, it will be allocated with default size will be used.
			* \return Array with triangles. Each 3 consequent points belong to a single triangle. The returned array points to 'triangles_buffer' data.
			*/
		DeviceArray<PointType>
		run(const TsdfVolume& tsdf, DeviceArray<PointType>& triangles_buffer);

	  private:
		/** \brief Edge table for marching cubes  */
		DeviceArray<int> edgeTable_;

		/** \brief Number of vertices table for marching cubes  */
		DeviceArray<int> numVertsTable_;

		/** \brief Triangles table for marching cubes  */
		DeviceArray<int> triTable_;

		/** \brief Temporary buffer used by marching cubes (first row stores occupied voxel id, second number of vertices, third points offsets */
		DeviceArray2D<int> occupied_voxels_buffer_;
	  };
	}
  }




#endif /* KINFU_INCLUDE_KINFU_GPU_MARCHING_CUBES_H_ */
