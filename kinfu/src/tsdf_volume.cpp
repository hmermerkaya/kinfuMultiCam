/*
 * tsdf_volume.cpp
 *
 *  Created on: May 31, 2018
 *      Author: hamit
 */

#include <gpu/tsdf_volume.h>
#include "internal.h"
#include <algorithm>
#include <Eigen/Core>

#include <iostream>

using namespace pcl;
//using namespace kinfu::device;
using namespace Eigen;
using kinfu::device::device_cast;

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

kinfu::gpu::TsdfVolume::TsdfVolume(const Vector3i& resolution) : resolution_(resolution), volume_host_ (new std::vector<float>), weights_host_ (new std::vector<short>)
{
  int volume_x = resolution_(0);
  int volume_y = resolution_(1);
  int volume_z = resolution_(2);

  volume_.create (volume_y * volume_z, volume_x);

  const Vector3f default_volume_size = Vector3f::Constant (3.f); //meters
  const float    default_tranc_dist  = 0.03f; //meters

  setSize(default_volume_size);
  setTsdfTruncDist(default_tranc_dist);

  reset();

}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void
kinfu::gpu::TsdfVolume::setSize(const Vector3f& size)
{
  size_ = size;
  setTsdfTruncDist(tranc_dist_);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void
kinfu::gpu::TsdfVolume::setTsdfTruncDist (float distance)
{
  float cx = size_(0) / resolution_(0);
  float cy = size_(1) / resolution_(1);
  float cz = size_(2) / resolution_(2);

  tranc_dist_ = std::max (distance, 2.1f * std::max (cx, std::max (cy, cz)));
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

kinfu::gpu::DeviceArray2D<int>
kinfu::gpu::TsdfVolume::data() const
{
  return volume_;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

const Eigen::Vector3f&
kinfu::gpu::TsdfVolume::getSize() const
{
    return size_;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

const Eigen::Vector3i&
kinfu::gpu::TsdfVolume::getResolution() const
{
  return resolution_;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

const Eigen::Vector3f
kinfu::gpu::TsdfVolume::getVoxelSize() const
{
	 return size_.array () / resolution_.array().cast<float>();
}



///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

float
kinfu::gpu::TsdfVolume::getTsdfTruncDist () const
{
  return tranc_dist_;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void
kinfu::gpu::TsdfVolume::reset()
{
 kinfu::device::initVolume(volume_);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void
kinfu::gpu::TsdfVolume::fetchCloudHost (PointCloud<PointXYZI>& cloud, bool connected26) const
{
  PointCloud<PointXYZ>::Ptr cloud_ptr_ = PointCloud<PointXYZ>::Ptr (new PointCloud<PointXYZ>);
  PointCloud<PointIntensity>::Ptr cloud_i_ptr_ = PointCloud<PointIntensity>::Ptr (new PointCloud<PointIntensity>);
  fetchCloudHost(*cloud_ptr_);
  pcl::concatenateFields (*cloud_ptr_, *cloud_i_ptr_, cloud);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void
kinfu::gpu::TsdfVolume::fetchCloudHost (PointCloud<PointType>& cloud, bool connected26) const
{
  int volume_x = resolution_(0);
  int volume_y = resolution_(1);
  int volume_z = resolution_(2);

  int cols;
  std::vector<int> volume_host;
  volume_.download (volume_host, cols);

  cloud.points.clear ();
  cloud.points.reserve (10000);

  const int DIVISOR = kinfu::device::DIVISOR; // SHRT_MAX;

#define FETCH(x, y, z) volume_host[(x) + (y) * volume_x + (z) * volume_y * volume_x]

  Array3f cell_size = getVoxelSize();

  for (int x = 1; x < volume_x-1; ++x)
  {
    for (int y = 1; y < volume_y-1; ++y)
    {
      for (int z = 0; z < volume_z-1; ++z)
      {
        int tmp = FETCH (x, y, z);
        int W = reinterpret_cast<short2*>(&tmp)->y;
        int F = reinterpret_cast<short2*>(&tmp)->x;

        if (W == 0 || F == DIVISOR)
          continue;

        Vector3f V = ((Array3f(x, y, z) + 0.5f) * cell_size).matrix ();

        if (connected26)
        {
          int dz = 1;
          for (int dy = -1; dy < 2; ++dy)
            for (int dx = -1; dx < 2; ++dx)
            {
              int tmp = FETCH (x+dx, y+dy, z+dz);

              int Wn = reinterpret_cast<short2*>(&tmp)->y;
              int Fn = reinterpret_cast<short2*>(&tmp)->x;
              if (Wn == 0 || Fn == DIVISOR)
                continue;

              if ((F > 0 && Fn < 0) || (F < 0 && Fn > 0))
              {
                Vector3f Vn = ((Array3f (x+dx, y+dy, z+dz) + 0.5f) * cell_size).matrix ();
                Vector3f point = (V * abs (Fn) + Vn * abs (F)) / (abs (F) + abs (Fn));

                pcl::PointXYZ xyz;
                xyz.x = point (0);
                xyz.y = point (1);
                xyz.z = point (2);

                cloud.points.push_back (xyz);
              }
            }
          dz = 0;
          for (int dy = 0; dy < 2; ++dy)
            for (int dx = -1; dx < dy * 2; ++dx)
            {
              int tmp = FETCH (x+dx, y+dy, z+dz);

              int Wn = reinterpret_cast<short2*>(&tmp)->y;
              int Fn = reinterpret_cast<short2*>(&tmp)->x;
              if (Wn == 0 || Fn == DIVISOR)
                continue;

              if ((F > 0 && Fn < 0) || (F < 0 && Fn > 0))
              {
                Vector3f Vn = ((Array3f (x+dx, y+dy, z+dz) + 0.5f) * cell_size).matrix ();
                Vector3f point = (V * abs(Fn) + Vn * abs(F))/(abs(F) + abs (Fn));

                pcl::PointXYZ xyz;
                xyz.x = point (0);
                xyz.y = point (1);
                xyz.z = point (2);

                cloud.points.push_back (xyz);
              }
            }
        }
        else /* if (connected26) */
        {
          for (int i = 0; i < 3; ++i)
          {
            int ds[] = {0, 0, 0};
            ds[i] = 1;

            int dx = ds[0];
            int dy = ds[1];
            int dz = ds[2];

            int tmp = FETCH (x+dx, y+dy, z+dz);

            int Wn = reinterpret_cast<short2*>(&tmp)->y;
            int Fn = reinterpret_cast<short2*>(&tmp)->x;
            if (Wn == 0 || Fn == DIVISOR)
              continue;

            if ((F > 0 && Fn < 0) || (F < 0 && Fn > 0))
            {
              Vector3f Vn = ((Array3f (x+dx, y+dy, z+dz) + 0.5f) * cell_size).matrix ();
              Vector3f point = (V * abs (Fn) + Vn * abs (F)) / (abs (F) + abs (Fn));

              pcl::PointXYZ xyz;
              xyz.x = point (0);
              xyz.y = point (1);
              xyz.z = point (2);

              cloud.points.push_back (xyz);
            }
          }
        } /* if (connected26) */
      }
    }
  }
#undef FETCH
  cloud.width  = (int)cloud.points.size ();
  cloud.height = 1;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//kinfu::gpu::DeviceArray<kinfu::gpu::TsdfVolume::PointType>
//kinfu::gpu::TsdfVolume::fetchCloud (DeviceArray<PointType>& cloud_buffer) const
//{
//  if (cloud_buffer.empty ())
//    cloud_buffer.create (DEFAULT_CLOUD_BUFFER_SIZE);
//
//  float3 device_volume_size = device_cast<const float3> (size_);
//  size_t size =kinfu::device::extractCloud (volume_, device_volume_size, cloud_buffer);
//  return (DeviceArray<PointType> (cloud_buffer.ptr (), size));
//}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//void
//kinfu::gpu::TsdfVolume::fetchNormals (const DeviceArray<PointType>& cloud, DeviceArray<PointType>& normals) const
//{
//  normals.create (cloud.size ());
//  const float3 device_volume_size = device_cast<const float3> (size_);
// kinfu::device::extractNormals (volume_, device_volume_size, cloud, (kinfu::device::PointType*)normals.ptr ());
//}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//void
//kinfu::gpu::TsdfVolume::pushSlice (PointCloud<PointXYZI>::Ptr existing_data_cloud, const kinfu::gpu::tsdf_buffer* buffer) const
//{
//  size_t gpu_array_size = existing_data_cloud->points.size ();
//
//  if(gpu_array_size == 0)
//  {
//    //std::cout << "[KinfuTracker](pushSlice) Existing data cloud has no points\n";//CREATE AS PCL MESSAGE
//    return;
//  }
//
//  const pcl::PointXYZI *first_point_ptr = &(existing_data_cloud->points[0]);
//
//  kinfu::device::DeviceArray<pcl::PointXYZI> cloud_gpu;
//  cloud_gpu.upload (first_point_ptr, gpu_array_size);
//
//  DeviceArray<float4>& cloud_cast = (DeviceArray<float4>&) cloud_gpu;
//  //volume().pushCloudAsSlice (cloud_cast, &buffer_);
// kinfu::device::pushCloudAsSliceGPU (volume_, cloud_cast, buffer);
//}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//size_t
//kinfu::gpu::TsdfVolume::fetchSliceAsCloud (DeviceArray<PointType>& cloud_buffer_xyz, DeviceArray<float>& cloud_buffer_intensity, const kinfu::gpu::tsdf_buffer* buffer, int shiftX, int shiftY, int shiftZ ) const
//{
//  if (cloud_buffer_xyz.empty ())
//    cloud_buffer_xyz.create (DEFAULT_CLOUD_BUFFER_SIZE/2);
//
//  if (cloud_buffer_intensity.empty ()) {
//    cloud_buffer_intensity.create (DEFAULT_CLOUD_BUFFER_SIZE/2);
//  }
//
//  float3 device_volume_size = device_cast<const float3> (size_);
//
//  size_t size =kinfu::device::extractSliceAsCloud (volume_, device_volume_size, buffer, shiftX, shiftY, shiftZ, cloud_buffer_xyz, cloud_buffer_intensity);
//
//  std::cout << " SIZE IS " << size << std::endl;
//
//  return (size);
//}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//void
//kinfu::gpu::TsdfVolume::fetchNormals (const DeviceArray<PointType>& cloud, DeviceArray<NormalType>& normals) const
//{
//  normals.create (cloud.size ());
//  const float3 device_volume_size = device_cast<const float3> (size_);
// kinfu::device::extractNormals(volume_, device_volume_size, cloud, (kinfu::device::float8*)normals.ptr ());
//}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void
kinfu::gpu::TsdfVolume::convertToTsdfCloud ( pcl::PointCloud<pcl::PointXYZI>::Ptr &cloud,
                                                    const unsigned step) const
{
  int sx = (header_.resolution(0));
  int sy = header_.resolution(1);
  int sz = header_.resolution(2);

  const int cloud_size = static_cast<int> (header_.getVolumeSize() / (step*step*step));

  cloud->clear();
  cloud->reserve (std::min (cloud_size/10, 500000));

  int volume_idx = 0, cloud_idx = 0;
  // #pragma omp parallel for // if used, increment over idx not possible! use index calculation
  for (int z = 0; z < sz; z+=step)
    for (int y = 0; y < sy; y+=step)
      for (int x = 0; x < sx; x+=step, ++cloud_idx)
      {
        volume_idx = sx*sy*z + sx*y + x;
        // pcl::PointXYZI &point = cloud->points[cloud_idx];

        if (weights_host_->at(volume_idx) == 0 || volume_host_->at(volume_idx) > 0.98 ) continue;
      //  if (weights_host_->at(volume_idx) == 0) continue;

        pcl::PointXYZI point;
        point.x = x; point.y = y; point.z = z;//*64;
        point.intensity = volume_host_->at(volume_idx);
        cloud->push_back (point);
      }
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void
kinfu::gpu::TsdfVolume::downloadTsdf (std::vector<float>& tsdf) const
{
  tsdf.resize (volume_.cols() * volume_.rows());
  volume_.download(&tsdf[0], volume_.cols() * sizeof(int));

#pragma omp parallel for
  for(int i = 0; i < (int) tsdf.size(); ++i)
  {
    float tmp = reinterpret_cast<short2*>(&tsdf[i])->x;
    tsdf[i] = tmp/kinfu::device::DIVISOR;
  }
}

void
kinfu::gpu::TsdfVolume::downloadTsdfLocal () const
{
  kinfu::gpu::TsdfVolume::downloadTsdf (*volume_host_);
}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void
kinfu::gpu::TsdfVolume::downloadTsdfAndWeights (std::vector<float>& tsdf, std::vector<short>& weights) const
{
  int volumeSize = volume_.cols() * volume_.rows();
  tsdf.resize (volumeSize);
  weights.resize (volumeSize);
  volume_.download(&tsdf[0], volume_.cols() * sizeof(int));

  #pragma omp parallel for
  for(int i = 0; i < (int) tsdf.size(); ++i)
  {
    short2 elem = *reinterpret_cast<short2*>(&tsdf[i]);
    tsdf[i] = (float)(elem.x)/kinfu::device::DIVISOR;
    weights[i] = (short)(elem.y);
  }
}


void
kinfu::gpu::TsdfVolume::downloadTsdfAndWeightsLocal () const
{
  kinfu::gpu::TsdfVolume::downloadTsdfAndWeights (*volume_host_, *weights_host_);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

bool
kinfu::gpu::TsdfVolume::save (const std::string &filename, bool binary) const
{
  pcl::console::print_info ("Saving TSDF volume to "); pcl::console::print_value ("%s ... ", filename.c_str());
  std::cout << std::flush;

  std::ofstream file (filename.c_str(), binary ? std::ios_base::binary : std::ios_base::out);

  if (file.is_open())
  {
    if (binary)
    {
      // HEADER
      // write resolution and size of volume
      file.write ((char*) &header_, sizeof (Header));
      /* file.write ((char*) &header_.resolution, sizeof(Eigen::Vector3i));
      file.write ((char*) &header_.volume_size, sizeof(Eigen::Vector3f));
      // write  element size
      int volume_element_size = sizeof(VolumeT);
      file.write ((char*) &volume_element_size, sizeof(int));
      int weights_element_size = sizeof(WeightT);
      file.write ((char*) &weights_element_size, sizeof(int)); */

      // DATA
      // write data
      file.write ((char*) &(volume_host_->at(0)), volume_host_->size()*sizeof(float));
      file.write ((char*) &(weights_host_->at(0)), weights_host_->size()*sizeof(short));
    }
    else
    {
      // write resolution and size of volume and element size
      file << header_.resolution(0) << " " << header_.resolution(1) << " " << header_.resolution(2) << std::endl;
      file << header_.volume_size(0) << " " << header_.volume_size(1) << " " << header_.volume_size(2) << std::endl;
      file << sizeof (float) << " " << sizeof(short) << std::endl;

      // write data
      for (std::vector<float>::const_iterator iter = volume_host_->begin(); iter != volume_host_->end(); ++iter)
        file << *iter << std::endl;
    }

    file.close();
  }
  else
  {
    pcl::console::print_error ("[saveTsdfVolume] Error: Couldn't open file %s.\n", filename.c_str());
    return false;
  }

  pcl::console::print_info ("done [%d voxels]\n", this->size ());

  return true;
}


bool
kinfu::gpu::TsdfVolume::load (const std::string &filename, bool binary)
{
  pcl::console::print_info ("Loading TSDF volume from "); pcl::console::print_value ("%s ... ", filename.c_str());
  std::cout << std::flush;

  std::ifstream file (filename.c_str());

  if (file.is_open())
  {
    if (binary)
    {
      // read HEADER
      file.read ((char*) &header_, sizeof (Header));
      /* file.read (&header_.resolution, sizeof(Eigen::Array3i));
      file.read (&header_.volume_size, sizeof(Eigen::Vector3f));
      file.read (&header_.volume_element_size, sizeof(int));
      file.read (&header_.weights_element_size, sizeof(int)); */

      // check if element size fits to data
      if (header_.volume_element_size != sizeof(float))
      {
        pcl::console::print_error ("[TSDFVolume::load] Error: Given volume element size (%d) doesn't fit data (%d)", sizeof(float), header_.volume_element_size);
        return false;
      }
      if ( header_.weights_element_size != sizeof(short))
      {
        pcl::console::print_error ("[TSDFVolume::load] Error: Given weights element size (%d) doesn't fit data (%d)", sizeof(short), header_.weights_element_size);
        return false;
      }

      // read DATA
      int num_elements = header_.getVolumeSize();
      volume_host_->resize (num_elements);
      weights_host_->resize (num_elements);
      file.read ((char*) &(*volume_host_)[0], num_elements * sizeof(float));
      file.read ((char*) &(*weights_host_)[0], num_elements * sizeof(short));
    }
    else
    {
      pcl::console::print_error ("[TSDFVolume::load] Error: ASCII loading not implemented.\n");
    }

    file.close ();
  }
  else
  {
    pcl::console::print_error ("[TSDFVolume::load] Error: Cloudn't read file %s.\n", filename.c_str());
    return false;
  }

  const Eigen::Vector3i &res = this->gridResolution();
  pcl::console::print_info ("done [%d voxels, res %dx%dx%d]\n", this->size(), res[0], res[1], res[2]);

  return true;
}



