/*
 * TextureMesh.h
 *
 *  Created on: Jun 28, 2018
 *      Author: hamit
 */

#ifndef KINFU_INCLUDE_TEXTUREMESH_H_
#define KINFU_INCLUDE_TEXTUREMESH_H_

#include <Eigen/Core>
#include <string>
#include <pcl/PCLPointCloud2.h>
#include <pcl/Vertices.h>

namespace kinfu
{
  /** \author Khai Tran */
  struct TexMaterial
  {
    TexMaterial () : tex_name (), tex_file (), tex_Ka (), tex_Kd (), tex_Ks (), tex_d (), tex_Ns (), tex_illum () {}

    struct RGB
    {
      float r;
      float g;
      float b;
    }; //RGB

    /** \brief Texture name. */
    std::string tex_name;

    /** \brief Texture file. */
    std::string tex_file;

    /** \brief Defines the ambient color of the material to be (r,g,b). */
    RGB         tex_Ka;

    /** \brief Defines the diffuse color of the material to be (r,g,b). */
    RGB         tex_Kd;

    /** \brief Defines the specular color of the material to be (r,g,b). This color shows up in highlights. */
    RGB         tex_Ks;

    /** \brief Defines the transparency of the material to be alpha. */
    float       tex_d;

    /** \brief Defines the shininess of the material to be s. */
    float       tex_Ns;

    /** \brief Denotes the illumination model used by the material.
      *
      * illum = 1 indicates a flat material with no specular highlights, so the value of Ks is not used.
      * illum = 2 denotes the presence of specular highlights, and so a specification for Ks is required.
      */
    int         tex_illum;

    pcl::PointXY maxUVCoord;
    pcl::PointXY minUVCoord;



  }; // TexMaterial

  /** \author Khai Tran */
  struct TextureMesh
  {
    TextureMesh () :
      cloud (), tex_polygons (), tex_coordinates (), tex_materials () {}

    pcl::PCLPointCloud2  cloud;
    pcl::PCLHeader  header;


    std::vector<std::vector<pcl::Vertices> >    tex_polygons;     // polygon which is mapped with specific texture defined in TexMaterial
    std::vector<std::vector<Eigen::Vector2f, Eigen::aligned_allocator<Eigen::Vector2f> > > tex_coordinates;  // UV coordinates
    std::vector<kinfu::TexMaterial>               tex_materials;    // define texture material

    public:
      typedef boost::shared_ptr<kinfu::TextureMesh> Ptr;
      typedef boost::shared_ptr<kinfu::TextureMesh const> ConstPtr;
   }; // struct TextureMesh

   typedef boost::shared_ptr<kinfu::TextureMesh> TextureMeshPtr;
   typedef boost::shared_ptr<kinfu::TextureMesh const> TextureMeshConstPtr;
} // namespace pcl




#endif /* KINFU_INCLUDE_TEXTUREMESH_H_ */
