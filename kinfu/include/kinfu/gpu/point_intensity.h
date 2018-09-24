/*
 * point_intensity.h
 *
 *  Created on: May 31, 2018
 *      Author: hamit
 */

#ifndef KINFU_INCLUDE_KINFU_GPU_POINT_INTENSITY_H_
#define KINFU_INCLUDE_KINFU_GPU_POINT_INTENSITY_H_



#include <iostream>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>

struct EIGEN_ALIGN16 PointIntensity
{

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
  union
  {
    struct
    {
      float intensity;
    };
    float data[4];
  };
};

POINT_CLOUD_REGISTER_POINT_STRUCT( PointIntensity, (float, intensity, intensity) )



#endif /* KINFU_INCLUDE_KINFU_GPU_POINT_INTENSITY_H_ */
