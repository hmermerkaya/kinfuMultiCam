#include <iostream>

#include <chrono>
// extra headers for writing out ply file
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/console/print.h>
#include <pcl/console/parse.h>
#include <pcl/console/time.h>
#include <pcl/io/ply_io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/registration/transforms.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/common/transforms.h>
#include <pcl/io/vtk_lib_io.h>
#include <pcl/io/obj_io.h>

#include <boost/thread/condition.hpp>
#include <boost/circular_buffer.hpp>
#include <boost/make_shared.hpp>

#include <opencv2/opencv.hpp>
#include <thread>
#include <memory>
#include <pcl/common/time.h>

#include <boost/asio.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/date_time/local_time/local_time.hpp>

//#include "PCDBuffer.hpp"
#include <k2g.h>
#include <gpu/initialization.h>
#include <gpu/kernel_containers.h>
#include <gpu/pixel_rgb.h>
#include <gpu/kinfu.h>
#include <gpu/marching_cubes.h>
#include <device.h>
#include <unistd.h>
//#include <texture_mapping.h>
#include <gpu/texture_mappingGPU.h>
#include <opencv2/opencv.hpp>



//#include <texture_mapping.hpp>
using namespace std;
using namespace pcl;
using namespace pcl::console;
namespace pc = pcl::console;
using namespace boost::gregorian;
using namespace boost::local_time;
using namespace boost::posix_time;




bool is_done=false;
boost::mutex io_mutex;

boost::shared_ptr<pcl::PolygonMesh> convertToMesh(const kinfu::gpu::DeviceArray<PointXYZ>& triangles)
{
  if (triangles.empty())
          return boost::shared_ptr<pcl::PolygonMesh>();

  pcl::PointCloud<pcl::PointXYZ> cloud;
  cloud.width  = (int)triangles.size();
  cloud.height = 1;
  triangles.download(cloud.points);

  boost::shared_ptr<pcl::PolygonMesh> mesh_ptr( new pcl::PolygonMesh() );
  pcl::toPCLPointCloud2(cloud, mesh_ptr->cloud);

  mesh_ptr->polygons.resize (triangles.size() / 3);
  for (size_t i = 0; i < mesh_ptr->polygons.size (); ++i)
  {
    pcl::Vertices v;
    v.vertices.push_back(i*3+0);
    v.vertices.push_back(i*3+2);
    v.vertices.push_back(i*3+1);
    mesh_ptr->polygons[i] = v;
  }
  return mesh_ptr;
}

void readTransformsBinary(const std::string &file,  std::vector<Eigen::Matrix4f> &transforms) {

        size_t sz ;
        fstream FILE(file,ios::binary|ios::in);
        //while (binary_file.good()) {
        FILE.read(reinterpret_cast<char*>(&sz), sizeof(size_t));
       // std::cout<<"sz "<<sz<<std::endl;
        transforms.resize(sz);
        FILE.read(reinterpret_cast<char*>(&transforms[0]), sz * sizeof(transforms[0]));
        FILE.close();

}


template <typename PointT>
struct KinFuLSApp
{

	enum { PCD_BIN = 1, PCD_ASCII = 2, PLY = 3, MESH_PLY = 7, MESH_VTK = 8 };
	KinFuLSApp( float vsz=3.0f, float shiftDistance=0.3f) : exit_ (false), scan_ (false), scan_mesh_(false), scan_volume_ (false), independent_camera_ (false),
															focal_length_(-1.f), was_lost_(false), time_ms_ (0)
	{
		//Init Kinfu Tracker
		Eigen::Vector3f volume_size = Eigen::Vector3f::Constant (vsz/*meters*/);

		PCL_WARN ("--- CURRENT SETTINGS ---\n");
		PCL_INFO ("Volume size is set to %.2f meters\n", vsz);
		PCL_INFO ("Volume will shift when the camera target point is farther than %.2f meters from the volume center\n", shiftDistance);
		PCL_INFO ("The target point is located at [0, 0, %.2f] in camera coordinates\n", 0.6*vsz);
		PCL_WARN ("------------------------\n");

		// warning message if shifting distance is abnormally big compared to volume size
		if(shiftDistance > 2.5 * vsz)
			PCL_WARN ("WARNING Shifting distance (%.2f) is very large compared to the volume size (%.2f).\nYou can modify it using --shifting_distance.\n", shiftDistance, vsz);

		kinfu_ = new kinfu::gpu::KinfuTracker (volume_size, shiftDistance);
        std::vector<std::string> files{"/home/hamit/kinfuMultiCam/bin/cam1.txt"};//, "/home/hamit/kinfuMultiCam/bin/cam2.txt"};
       // std::vector<Eigen::Matrix<float,4,4,Eigen::RowMajor>,Eigen::aligned_allocator<Eigen::Vector4f>> tmpglb=convert2GlobalTransformations(files);
		kinfu_->setRMatAndTVec(convert2GlobalTransformations(files));

		loadCamsIntrinsics();

		Eigen::Matrix3f R = Eigen::Matrix3f::Identity ();   // * AngleAxisf( pcl::deg2rad(-30.f), Vector3f::UnitX());
		Eigen::Vector3f t = /*Vector3f (1.5, 1.5,0);*/  volume_size * 0.5f - Eigen::Vector3f (0, 0, volume_size (2) / 2 * 1.2f);

		Eigen::Affine3f pose = Eigen::Translation3f (t) * Eigen::AngleAxisf (R);

		kinfu_->setInitialCameraPose (pose);
		kinfu_->volume().setTsdfTruncDist (0.030f/*meters*/);
		// kinfu_->setIcpCorespFilteringParams (0.1f/*meters*/, sin ( pcl::deg2rad(20.f) ));
		//kinfu_->setDepthTruncationForICP(3.f/*meters*/);
		kinfu_->setCameraMovementThreshold(0.001f);

		//Init KinFuLSApp
		tsdf_cloud_ptr_ = pcl::PointCloud<pcl::PointXYZI>::Ptr (new pcl::PointCloud<pcl::PointXYZI>);
		// image_view_.raycaster_ptr_ = RayCaster::Ptr( new RayCaster(kinfu_->rows (), kinfu_->cols ()) );

		//  scene_cloud_view_.cloud_viewer_.registerKeyboardCallback (keyboard_callback, (void*)this);
		//   image_view_.viewerScene_.registerKeyboardCallback (keyboard_callback, (void*)this);
		//   image_view_.viewerDepth_.registerKeyboardCallback (keyboard_callback, (void*)this);

		//   scene_cloud_view_.toggleCube(volume_size);
		//   frame_counter_ = 0;
		//   enable_texture_extraction_ = false;



		//kinect_v2
		float height = 424.0f;
		float width = 512.0f;

		Eigen::Matrix3f Rid = Eigen::Matrix3f::Identity ();   // * AngleAxisf( pcl::deg2rad(-30.f), Vector3f::UnitX());
		Eigen::Vector3f T = Eigen::Vector3f (0, 0, -volume_size(0)*1.5f);
		delta_lost_pose_ = Eigen::Translation3f (T) * Eigen::AngleAxisf (Rid);

	}

	~KinFuLSApp()
	{
//    if (evaluation_ptr_)
//      evaluation_ptr_->saveAllPoses(*kinfu_);
	}
	std::ifstream& GotoLine(std::ifstream& file, unsigned int num)
	{
	  file.seekg (std::ios::beg);
	  for(int i=0; i < num - 1; ++i)
	  {
	    file.ignore (std::numeric_limits<std::streamsize>::max (),'\n');
	  }
	  return (file);
	}

	/** \brief Helper function that reads a camera file outputed by Kinfu */
	bool readCamPoseFile(std::string filename, kinfu::TextureMappingGPU<pcl::PointXYZ>::Camera &cam)
	{
	  ifstream myReadFile;
	  myReadFile.open(filename.c_str (), ios::in);
	  if(!myReadFile.is_open ())
	  {
	    PCL_ERROR ("Error opening file %d\n", filename.c_str ());
	    return false;
	  }
	  myReadFile.seekg(ios::beg);

	  char current_line[1024];
	  double val;

	  // go to line 2 to read translations
	  GotoLine(myReadFile, 2);
	  myReadFile >> val; cam.pose (0,3)=val; //TX
	  myReadFile >> val; cam.pose (1,3)=val; //TY
	  myReadFile >> val; cam.pose (2,3)=val; //TZ

	  // go to line 7 to read rotations
	  GotoLine(myReadFile, 7);

	  myReadFile >> val; cam.pose (0,0)=val;
	  myReadFile >> val; cam.pose (0,1)=val;
	  myReadFile >> val; cam.pose (0,2)=val;

	  myReadFile >> val; cam.pose (1,0)=val;
	  myReadFile >> val; cam.pose (1,1)=val;
	  myReadFile >> val; cam.pose (1,2)=val;

	  myReadFile >> val; cam.pose (2,0)=val;
	  myReadFile >> val; cam.pose (2,1)=val;
	  myReadFile >> val; cam.pose (2,2)=val;

	  cam.pose (3,0) = 0.0;
	  cam.pose (3,1) = 0.0;
	  cam.pose (3,2) = 0.0;
	  cam.pose (3,3) = 1.0; //Scale

	  // go to line 12 to read camera focal length and size
	  GotoLine (myReadFile, 12);
	  myReadFile >> val; cam.focal_length=val;
	  myReadFile >> val; cam.height=val;
	  myReadFile >> val; cam.width=val;

	//  std::cout<<"cam pose 1 "<<cam.pose.translation()<<std::endl;
	  // close file
	  myReadFile.close ();

	  return true;

	}
	void loadCamsIntrinsics(){
		 const boost::filesystem::path base_dir ("/home/hamit/kinfuMultiCam/bin/CamsDir");
		  std::string extension (".txt");
		  int cpt_cam = 0;
		  for (boost::filesystem::directory_iterator it (base_dir); it != boost::filesystem::directory_iterator (); ++it)
		  {
			if(boost::filesystem::is_regular_file (it->status ()) && boost::filesystem::extension (it->path ()) == extension)
			{
			  kinfu::TextureMappingGPU<pcl::PointXYZ>::Camera cam;
			  readCamPoseFile(it->path ().string (), cam);
			  cam.texture_file = boost::filesystem::basename (it->path ()) + ".jpg";
			  std::cout<<"campose: "<< cam.pose.linear()<<" cam.texture_file "<<cam.texture_file<<std::endl;
			  cams.push_back (cam);
			  cpt_cam++ ;
			}
		  }
		  PCL_INFO ("\tLoaded %d textures.\n", cams.size ());
		  PCL_INFO ("...Done.\n");

	}

	int
	saveOBJFile (const std::string &file_name,
	             const kinfu::TextureMesh &tex_mesh, unsigned precision)
	{
	  if (tex_mesh.cloud.data.empty ())
	  {
	    PCL_ERROR ("[pcl::io::saveOBJFile] Input point cloud has no data!\n");
	    return (-1);
	  }

	  // Open file
	  std::ofstream fs;
	  fs.precision (precision);
	  fs.open (file_name.c_str ());

	  // Define material file
	  std::string mtl_file_name = file_name.substr (0, file_name.find_last_of (".")) + ".mtl";
	  // Strip path for "mtllib" command
	  std::string mtl_file_name_nopath = mtl_file_name;
	  mtl_file_name_nopath.erase (0, mtl_file_name.find_last_of ('/') + 1);

	  /* Write 3D information */
	  // number of points
	  int nr_points  = tex_mesh.cloud.width * tex_mesh.cloud.height;
	  int point_size = tex_mesh.cloud.data.size () / nr_points;

	  // mesh size
	  int nr_meshes = tex_mesh.tex_polygons.size ();
	  std::cout<<"nr meshes: "<<nr_meshes<<std::endl;
	  // number of faces for header
	  int nr_faces = 0;
	  for (int m = 0; m < nr_meshes; ++m)
	    nr_faces += tex_mesh.tex_polygons[m].size ();

	  // Write the header information
	  fs << "####" << std::endl;
	  fs << "# OBJ dataFile simple version. File name: " << file_name << std::endl;
	  fs << "# Vertices: " << nr_points << std::endl;
	  fs << "# Faces: " <<nr_faces << std::endl;
	  fs << "# Material information:" << std::endl;
	  fs << "mtllib " << mtl_file_name_nopath << std::endl;
	  fs << "####" << std::endl;

	  // Write vertex coordinates
	  fs << "# Vertices" << std::endl;
	  for (int i = 0; i < nr_points; ++i)
	  {
	    int xyz = 0;
	    // "v" just be written one
	    bool v_written = false;
	    for (size_t d = 0; d < tex_mesh.cloud.fields.size (); ++d)
	    {
	      int count = tex_mesh.cloud.fields[d].count;
	      if (count == 0)
	        count = 1;          // we simply cannot tolerate 0 counts (coming from older converter code)
	      int c = 0;
	      // adding vertex
	      if ((tex_mesh.cloud.fields[d].datatype == pcl::PCLPointField::FLOAT32) && (
	                tex_mesh.cloud.fields[d].name == "x" ||
	                tex_mesh.cloud.fields[d].name == "y" ||
	                tex_mesh.cloud.fields[d].name == "z"))
	      {
	        if (!v_written)
	        {
	            // write vertices beginning with v
	            fs << "v ";
	            v_written = true;
	        }
	        float value;
	        memcpy (&value, &tex_mesh.cloud.data[i * point_size + tex_mesh.cloud.fields[d].offset + c * sizeof (float)], sizeof (float));
	        fs << value;
	        if (++xyz == 3)
	            break;
	        fs << " ";
	      }
	    }
	    if (xyz != 3)
	    {
	      PCL_ERROR ("[pcl::io::saveOBJFile] Input point cloud has no XYZ data!\n");
	      return (-2);
	    }
	    fs << std::endl;
	  }
	  fs << "# "<< nr_points <<" vertices" << std::endl;

	  // Write vertex normals
//	  for (int i = 0; i < nr_points; ++i)
//	  {
//	    int xyz = 0;
//	    // "vn" just be written one
//	    bool v_written = false;
//	    for (size_t d = 0; d < tex_mesh.cloud.fields.size (); ++d)
//	    {
//	      int count = tex_mesh.cloud.fields[d].count;
//	      if (count == 0)
//	      count = 1;          // we simply cannot tolerate 0 counts (coming from older converter code)
//	      int c = 0;
//	      // adding vertex
//	      if ((tex_mesh.cloud.fields[d].datatype == pcl::PCLPointField::FLOAT32) && (
//	      tex_mesh.cloud.fields[d].name == "normal_x" ||
//	      tex_mesh.cloud.fields[d].name == "normal_y" ||
//	      tex_mesh.cloud.fields[d].name == "normal_z"))
//	      {
//	        if (!v_written)
//	        {
//	          // write vertices beginning with vn
//	          fs << "vn ";
//	          v_written = true;
//	        }
//	        float value;
//	        memcpy (&value, &tex_mesh.cloud.data[i * point_size + tex_mesh.cloud.fields[d].offset + c * sizeof (float)], sizeof (float));
//	        fs << value;
//	        if (++xyz == 3)
//	          break;
//	        fs << " ";
//	      }
//	    }
//	    if (xyz != 3)
//	    {
//	    PCL_ERROR ("[pcl::io::saveOBJFile] Input point cloud has no normals!\n");
//	    return (-2);
//	    }
//	    fs << std::endl;
//	  }
	  // Write vertex texture with "vt" (adding latter)

	  for (int m = 0; m < nr_meshes; ++m)
	  {
	    if(tex_mesh.tex_coordinates.size() == 0)
	      continue;

	    PCL_INFO ("%d vertex textures in submesh %d\n", tex_mesh.tex_coordinates[m].size (), m);
	    fs << "# " << tex_mesh.tex_coordinates[m].size() << " vertex textures in submesh " << m <<  std::endl;
	    for (size_t i = 0; i < tex_mesh.tex_coordinates[m].size (); ++i)
	    {
	      fs << "vt ";
	      fs <<  tex_mesh.tex_coordinates[m][i][0] << " " << tex_mesh.tex_coordinates[m][i][1] << std::endl;
	    }
	  }

	  int f_idx = 0;

	  // int idx_vt =0;
	  PCL_INFO ("Writting faces...\n");
	  for (int m = 0; m < nr_meshes-1; ++m)
	  {
	    if (m > 0)
	      f_idx += tex_mesh.tex_polygons[m-1].size ();

	    if(tex_mesh.tex_materials.size() !=0)
	    {
	      fs << "# The material will be used for mesh " << m << std::endl;
	      //TODO pbl here with multi texture and unseen faces
	      fs << "usemtl " <<  tex_mesh.tex_materials[m].tex_name << std::endl;
	      fs << "# Faces" << std::endl;
	    }
	    for (size_t i = 0; i < tex_mesh.tex_polygons[m].size(); ++i)
	    {
	      // Write faces with "f"
	      fs << "f";
	      size_t j = 0;
	      // There's one UV per vertex per face, i.e., the same vertex can have
	      // different UV depending on the face.
	      for (j = 0; j < tex_mesh.tex_polygons[m][i].vertices.size (); ++j)
	      {
	        unsigned int idx = tex_mesh.tex_polygons[m][i].vertices[j] + 1;
	        fs << " " << idx
	        << "/" << 3*(i+f_idx) +j+1;
	      //  << "/" << idx; // vertex index in obj file format starting with 1
	      }
	      fs << std::endl;
	    }
	    PCL_INFO ("%d faces in mesh %d \n", tex_mesh.tex_polygons[m].size () , m);
	    fs << "# "<< tex_mesh.tex_polygons[m].size() << " faces in mesh " << m << std::endl;
	  }
	  fs << "# End of File";

	  // Close obj file
	  PCL_INFO ("Closing obj file\n");
	  fs.close ();

	  /* Write material defination for OBJ file*/
	  // Open file
	  PCL_INFO ("Writing material files\n");
	  //dont do it if no material to write
	  if(tex_mesh.tex_materials.size() ==0)
	    return (0);

	  std::ofstream m_fs;
	  m_fs.precision (precision);
	  m_fs.open (mtl_file_name.c_str ());

	  // default
	  m_fs << "#" << std::endl;
	  m_fs << "# Wavefront material file" << std::endl;
	  m_fs << "#" << std::endl;
	  for(int m = 0; m < nr_meshes-1; ++m)
	  {
	    m_fs << "newmtl " << tex_mesh.tex_materials[m].tex_name << std::endl;
	    m_fs << "Ka "<< tex_mesh.tex_materials[m].tex_Ka.r << " " << tex_mesh.tex_materials[m].tex_Ka.g << " " << tex_mesh.tex_materials[m].tex_Ka.b << std::endl; // defines the ambient color of the material to be (r,g,b).
	    m_fs << "Kd "<< tex_mesh.tex_materials[m].tex_Kd.r << " " << tex_mesh.tex_materials[m].tex_Kd.g << " " << tex_mesh.tex_materials[m].tex_Kd.b << std::endl; // defines the diffuse color of the material to be (r,g,b).
	    m_fs << "Ks "<< tex_mesh.tex_materials[m].tex_Ks.r << " " << tex_mesh.tex_materials[m].tex_Ks.g << " " << tex_mesh.tex_materials[m].tex_Ks.b << std::endl; // defines the specular color of the material to be (r,g,b). This color shows up in highlights.
	    m_fs << "d " << tex_mesh.tex_materials[m].tex_d << std::endl; // defines the transparency of the material to be alpha.
	    m_fs << "Ns "<< tex_mesh.tex_materials[m].tex_Ns  << std::endl; // defines the shininess of the material to be s.
	    m_fs << "illum "<< tex_mesh.tex_materials[m].tex_illum << std::endl; // denotes the illumination model used by the material.
	    // illum = 1 indicates a flat material with no specular highlights, so the value of Ks is not used.
	    // illum = 2 denotes the presence of specular highlights, and so a specification for Ks is required.
	    m_fs << "map_Kd " << tex_mesh.tex_materials[m].tex_file << std::endl;

	    m_fs << "###" << std::endl;
	  }
	  m_fs.close ();
	  return (0);
	}


	cv::Mat mergeIMGFiles(const std::vector<kinfu::TexMaterial> &texMatrls,
			std::vector<std::vector<Eigen::Vector2f, Eigen::aligned_allocator<Eigen::Vector2f> > >  &tex_coords){

		std::vector<cv::Mat> imgFiles;

		size_t nfiles= texMatrls.size()-1;

		int height= (int)roundf(sqrt(nfiles));
		int width = height == int(sqrt(nfiles))? height + 1: height ;
		cv::Size size (width, height);

		cv::Mat mergedImgFile(540, 960, CV_8UC3, cv::Scalar(0,0,0));

		for (const auto &it: texMatrls){
			size_t found=it.tex_file.find("occluded");
			if (found!=std::string::npos) continue;
			cv::Mat tmpmat = cv::imread(it.tex_file, CV_LOAD_IMAGE_COLOR);
			cv::Rect myROI(it.minUVCoord.x*tmpmat.cols, it.minUVCoord.y*tmpmat.rows, (it.maxUVCoord.x-it.minUVCoord.x)*tmpmat.cols,
						(it.maxUVCoord.y-it.minUVCoord.y)*tmpmat.rows);
			tmpmat(myROI).copyTo(tmpmat);

			float fx = (float)mergedImgFile.cols/size.width/tmpmat.cols;
			float fy = (float)mergedImgFile.rows/size.height/tmpmat.rows;

			cv::resize(tmpmat, tmpmat, cv::Size(), fx, fy);
			imgFiles.push_back(tmpmat);


		}




		for (int i= 0; i<size.height; ++i){
			for (int j= 0; j<size.width; ++j) {

				 if (j+i*size.width < imgFiles.size()) {

//					 float fx=mergedImgFile.cols/size.width/imgFiles.at(j+i*size.width).cols;
//					 float fy=mergedImgFile.cols/size.height/imgFiles.at(j+i*size.width).rows;


					 int cols = imgFiles.at(j+i*size.width).cols;
				 	 int rows = imgFiles.at(j+i*size.width).rows;
					 imgFiles.at(j+i*size.width).copyTo(mergedImgFile(cv::Rect(j*cols, i*rows, cols, rows)));

					 float xmin= texMatrls.at((j+i*size.width)).minUVCoord.x;
					 float xmax= texMatrls.at((j+i*size.width)).maxUVCoord.x;
					 float ymin= texMatrls.at((j+i*size.width)).minUVCoord.y;
					 float ymax= texMatrls.at((j+i*size.width)).maxUVCoord.y;
					 for (int k=0;k<tex_coords.at(j+i*size.width).size(); k++){
						 tex_coords[j+i*size.width][k][0] = (tex_coords[j+i*size.width][k][0]-xmin)/(xmax-xmin);
						 tex_coords[j+i*size.width][k][0] = tex_coords[j+i*size.width][k][0]*(1.0f/size.width) + (j/size.width);

						 tex_coords[j+i*size.width][k][1] = (tex_coords[j+i*size.width][k][1] -ymin)/(ymax-ymin);
						 tex_coords[j+i*size.width][k][1] = tex_coords[j+i*size.width][k][1]*(1.0f/size.height) + (i/size.height);

					 }



				 }

			}

		}



		return std::move(mergedImgFile);







	}


	template <typename T>
	std::vector<T> flatten(const std::vector<std::vector<T>>& v) {
	  std::size_t total_size = 0;
	  for (const auto& sub : v)
		  total_size += sub.size(); // I wish there was a transform_accumulate
	  std::vector<T> result;
	  result.reserve(total_size);
	  for (const auto& sub : v)
		  result.insert(result.end(), sub.begin(), sub.end());
	  return result;
	}


	int
	saveOBJFile2 (const std::string &file_name,
		              kinfu::TextureMesh &tex_mesh, unsigned precision)
	{
	  if (tex_mesh.cloud.data.empty ())
	  {
		PCL_ERROR ("[pcl::io::saveOBJFile] Input point cloud has no data!\n");
		return (-1);
	  }

	  // Open file
	  std::ofstream fs;
	  fs.precision (precision);
	  fs.open (file_name.c_str ());

	  // Define material file
	  std::string mtl_file_name = file_name.substr (0, file_name.find_last_of (".")) + ".mtl";
	  // Strip path for "mtllib" command
	  std::string mtl_file_name_nopath = mtl_file_name;
	  mtl_file_name_nopath.erase (0, mtl_file_name.find_last_of ('/') + 1);

	  /* Write 3D information */
	  // number of points
	  int nr_points  = tex_mesh.cloud.width * tex_mesh.cloud.height;
	  int point_size = tex_mesh.cloud.data.size () / nr_points;


	  // mesh size
	  tex_mesh.tex_polygons[0]=flatten(tex_mesh.tex_polygons);
	  tex_mesh.tex_polygons.erase(tex_mesh.tex_polygons.begin()+1, tex_mesh.tex_polygons.end());

	  int nr_meshes = tex_mesh.tex_polygons.size ();
	 // std::cout<<"nr meshes: "<<nr_meshes<<std::endl;
	  // number of faces for header
	  int nr_faces = 0;
	  for (int m = 0; m < nr_meshes; ++m)
		nr_faces += tex_mesh.tex_polygons[m].size ();

	  // Write the header information
	  fs << "####" << std::endl;
	  fs << "# OBJ dataFile simple version. File name: " << file_name << std::endl;
	  fs << "# Vertices: " << nr_points << std::endl;
	  fs << "# Faces: " <<nr_faces << std::endl;
	  fs << "# Material information:" << std::endl;
	  fs << "mtllib " << mtl_file_name_nopath << std::endl;
	  fs << "####" << std::endl;

	  // Write vertex coordinates
	  fs << "# Vertices" << std::endl;
	  for (int i = 0; i < nr_points; ++i)
	  {
		int xyz = 0;
		// "v" just be written one
		bool v_written = false;
		for (size_t d = 0; d < tex_mesh.cloud.fields.size (); ++d)
		{
		  int count = tex_mesh.cloud.fields[d].count;
		  if (count == 0)
			count = 1;          // we simply cannot tolerate 0 counts (coming from older converter code)
		  int c = 0;
		  // adding vertex
		  if ((tex_mesh.cloud.fields[d].datatype == pcl::PCLPointField::FLOAT32) && (
					tex_mesh.cloud.fields[d].name == "x" ||
					tex_mesh.cloud.fields[d].name == "y" ||
					tex_mesh.cloud.fields[d].name == "z"))
		  {
			if (!v_written)
			{
				// write vertices beginning with v
				fs << "v ";
				v_written = true;
			}
			float value;
			memcpy (&value, &tex_mesh.cloud.data[i * point_size + tex_mesh.cloud.fields[d].offset + c * sizeof (float)], sizeof (float));
			fs << value;
			if (++xyz == 3)
				break;
			fs << " ";
		  }
		}
		if (xyz != 3)
		{
		  PCL_ERROR ("[pcl::io::saveOBJFile] Input point cloud has no XYZ data!\n");
		  return (-2);
		}
		fs << std::endl;
	  }
	  fs << "# "<< nr_points <<" vertices" << std::endl;

	  // Write vertex normals
//	  for (int i = 0; i < nr_points; ++i)
//	  {
//	    int xyz = 0;
//	    // "vn" just be written one
//	    bool v_written = false;
//	    for (size_t d = 0; d < tex_mesh.cloud.fields.size (); ++d)
//	    {
//	      int count = tex_mesh.cloud.fields[d].count;
//	      if (count == 0)
//	      count = 1;          // we simply cannot tolerate 0 counts (coming from older converter code)
//	      int c = 0;
//	      // adding vertex
//	      if ((tex_mesh.cloud.fields[d].datatype == pcl::PCLPointField::FLOAT32) && (
//	      tex_mesh.cloud.fields[d].name == "normal_x" ||
//	      tex_mesh.cloud.fields[d].name == "normal_y" ||
//	      tex_mesh.cloud.fields[d].name == "normal_z"))
//	      {
//	        if (!v_written)
//	        {
//	          // write vertices beginning with vn
//	          fs << "vn ";
//	          v_written = true;
//	        }
//	        float value;
//	        memcpy (&value, &tex_mesh.cloud.data[i * point_size + tex_mesh.cloud.fields[d].offset + c * sizeof (float)], sizeof (float));
//	        fs << value;
//	        if (++xyz == 3)
//	          break;
//	        fs << " ";
//	      }
//	    }
//	    if (xyz != 3)
//	    {
//	    PCL_ERROR ("[pcl::io::saveOBJFile] Input point cloud has no normals!\n");
//	    return (-2);
//	    }
//	    fs << std::endl;
//	  }
	  // Write vertex texture with "vt" (adding latter)


	  cv::Mat mergedIMG = mergeIMGFiles(tex_mesh.tex_materials, tex_mesh.tex_coordinates);
	  cv::imwrite("mergedImg.jpg", mergedIMG);

	  for (int m = 0; m < tex_mesh.tex_coordinates.size();++m)
	  // for (int m = 0; m < nr_meshes; ++m)
	  {
//		if(tex_mesh.tex_coordinates.size() == 0)
//		  continue;

		PCL_INFO ("%d vertex textures in submesh %d\n", tex_mesh.tex_coordinates[m].size (), m);
		fs << "# " << tex_mesh.tex_coordinates[m].size() << " vertex textures in submesh " << m <<  std::endl;
		for (size_t i = 0; i < tex_mesh.tex_coordinates[m].size (); ++i)
		{
		  fs << "vt ";
		  fs <<  tex_mesh.tex_coordinates[m][i][0] << " " << tex_mesh.tex_coordinates[m][i][1] << std::endl;
		}
	  }





	  int f_idx = 0;

	  // int idx_vt =0;

	  PCL_INFO ("Writting faces...\n");
	  for (int m = 0; m < nr_meshes; ++m)
	  {
		if (m > 0)
		  f_idx += tex_mesh.tex_polygons[m-1].size ();

		if(tex_mesh.tex_materials.size() !=0)
		{
		  fs << "# The material will be used for mesh " << m << std::endl;
		  //TODO pbl here with multi texture and unseen faces
		  fs << "usemtl " <<  "MainMaterial"<< std::endl;
		  fs << "# Faces" << std::endl;
		}
		for (size_t i = 0; i < tex_mesh.tex_polygons[m].size(); ++i)
		{
		  // Write faces with "f"
		  fs << "f";
		  size_t j = 0;
		  // There's one UV per vertex per face, i.e., the same vertex can have
		  // different UV depending on the face.
		  for (j = 0; j < tex_mesh.tex_polygons[m][i].vertices.size (); ++j)
		  {
			unsigned int idx = tex_mesh.tex_polygons[m][i].vertices[j] + 1;
			fs << " " << idx
			<< "/" << 3*(i+f_idx) +j+1;
		  //  << "/" << idx; // vertex index in obj file format starting with 1
		  }
		  fs << std::endl;
		}
		PCL_INFO ("%d faces in mesh %d \n", tex_mesh.tex_polygons[m].size () , m);
		fs << "# "<< tex_mesh.tex_polygons[m].size() << " faces in mesh " << m << std::endl;
	  }
	  fs << "# End of File";

	  // Close obj file
	  PCL_INFO ("Closing obj file\n");
	  fs.close ();

	  /* Write material defination for OBJ file*/
	  // Open file
	  PCL_INFO ("Writing material files\n");
	  //dont do it if no material to write
	  if(tex_mesh.tex_materials.size() ==0)
		return (0);

	  std::ofstream m_fs;
	  m_fs.precision (precision);
	  m_fs.open (mtl_file_name.c_str ());

	  // default
	  m_fs << "#" << std::endl;
	  m_fs << "# Wavefront material file" << std::endl;
	  m_fs << "#" << std::endl;
	  for(int m = 0; m < nr_meshes; ++m)
	  {
		m_fs << "newmtl " << "MainMaterial" << std::endl;
		m_fs << "Ka "<< tex_mesh.tex_materials[m].tex_Ka.r << " " << tex_mesh.tex_materials[m].tex_Ka.g << " " << tex_mesh.tex_materials[m].tex_Ka.b << std::endl; // defines the ambient color of the material to be (r,g,b).
		m_fs << "Kd "<< tex_mesh.tex_materials[m].tex_Kd.r << " " << tex_mesh.tex_materials[m].tex_Kd.g << " " << tex_mesh.tex_materials[m].tex_Kd.b << std::endl; // defines the diffuse color of the material to be (r,g,b).
		m_fs << "Ks "<< tex_mesh.tex_materials[m].tex_Ks.r << " " << tex_mesh.tex_materials[m].tex_Ks.g << " " << tex_mesh.tex_materials[m].tex_Ks.b << std::endl; // defines the specular color of the material to be (r,g,b). This color shows up in highlights.
		m_fs << "d " << tex_mesh.tex_materials[m].tex_d << std::endl; // defines the transparency of the material to be alpha.
		m_fs << "Ns "<< tex_mesh.tex_materials[m].tex_Ns  << std::endl; // defines the shininess of the material to be s.
		m_fs << "illum "<< tex_mesh.tex_materials[m].tex_illum << std::endl; // denotes the illumination model used by the material.
		// illum = 1 indicates a flat material with no specular highlights, so the value of Ks is not used.
		// illum = 2 denotes the presence of specular highlights, and so a specification for Ks is required.
		m_fs << "map_Kd " << "mergedImg.jpg" << std::endl;

		m_fs << "###" << std::endl;
	  }
	  m_fs.close ();
	  return (0);
	}



	 void
	 readTransformFromText(const std::string &file, Eigen::Matrix<float, 4, 4, Eigen::RowMajor> &transform )
	 {




	        std::ifstream infile(file.c_str());
	        std::stringstream buffer;

	        buffer << infile.rdbuf();
	        float temp;
	        std::vector<float> matrixElements;
	        while (buffer >> temp ) {
	        matrixElements.push_back(temp);
	      //  std::cout<<"temp: "<<temp<<"\n";

	        }

	       transform= Eigen::Map<Eigen::Matrix<float,4,4,Eigen::RowMajor> >(matrixElements.data());
          // transform=Eigen::Map<Eigen::Matrix4f >(matrixElements.data());



	      infile.close();



	}

	 std::vector<Eigen::Matrix<float,4,4,Eigen::RowMajor>, Eigen::aligned_allocator<Eigen::Matrix<float,4,4,Eigen::RowMajor>>>
	 convert2GlobalTransformations(const std::vector<std::string> &files )
	 {
		 std::vector<Eigen::Matrix<float,4,4,Eigen::RowMajor>, Eigen::aligned_allocator<Eigen::Matrix<float,4,4,Eigen::RowMajor>>> globalTrasformation(files.size());
		 unsigned i=0;
		 for (auto file: files ){
			 readTransformFromText(file, globalTrasformation[i] );
			 if (i!=0) globalTrasformation[i]=globalTrasformation[i-1]*globalTrasformation[i];
			i++;
		 }
		 return std::move(globalTrasformation);
	 }



	void
	readTransformsBinary(const std::string &file,  std::vector<Eigen::Matrix4f> &transforms)
	{

	        size_t sz ;
	        fstream FILE(file,ios::binary|ios::in);
	        //while (binary_file.good()) {
	        FILE.read(reinterpret_cast<char*>(&sz), sizeof(size_t));
	       // std::cout<<"sz "<<sz<<std::endl;
	        transforms.resize(sz);
	        FILE.read(reinterpret_cast<char*>(&transforms[0]), sz * sizeof(transforms[0]));
	        FILE.close();

	}


	void
	toggleColorIntegration()
	{

		cout << "Color integration: " << (integrate_colors_ ? "On" : "Off ( requires registration mode )") << endl;
	}

	void
	toggleIndependentCamera()
	{
		independent_camera_ = !independent_camera_;
		cout << "Camera mode: " << (independent_camera_ ?  "Independent" : "Bound to Kinect pose") << endl;
	}



	void execute(const kinfu::gpu::PtrStepSz<const unsigned short>& depth, const kinfu::gpu::PtrStepSz<const kinfu::gpu::PixelRGB>& rgb24)
	{
		bool has_image = false;
		frame_counter_++;


//		if (has_data)		{
			depth_device_.upload (depth.data, depth.step, depth.rows, depth.cols);

			{
				//   SampledScopeTime fps(time_ms_);

				//run kinfu algorithm

				has_image = (*kinfu_) (depth_device_);

			//	cout << "has_image " << has_image<< endl;
			}

			//  image_view_.showDepth (depth_);

			//image_view_.showGeneratedDepth(kinfu_, kinfu_->getCameraPose());

	//	}

//		if (scan_  ) //if scan mode is OR and ICP just lost itself => show current volume as point cloud
//		{
//			scan_ = false;
//			//scene_cloud_view_.show (*kinfu_, integrate_colors_); // this triggers out of memory errors, so I comment it out for now (Raph)
//
//			if (scan_volume_)
//			{
//				cout << "Downloading TSDF volume from device ... " << flush;
//				// kinfu_->volume().downloadTsdfAndWeighs (tsdf_volume_.volumeWriteable (), tsdf_volume_.weightsWriteable ());
//				kinfu_->volume ().downloadTsdfAndWeightsLocal ();
//				// tsdf_volume_.setHeader (Eigen::Vector3i (pcl::device::kinfuLS::VOLUME_X, pcl::device::kinfuLS::VOLUME_Y, pcl::device::kinfuLS::VOLUME_Z), kinfu_->volume().getSize ());
//				kinfu_->volume ().setHeader (Eigen::Vector3i (kinfu::device::VOLUME_X, kinfu::device::VOLUME_Y, kinfu::device::VOLUME_Z), kinfu_->volume().getSize ());
//				// cout << "done [" << tsdf_volume_.size () << " voxels]" << endl << endl;
//				cout << "done [" << kinfu_->volume ().size () << " voxels]" << endl << endl;
//
//				cout << "Converting volume to TSDF cloud ... " << flush;
//				// tsdf_volume_.convertToTsdfCloud (tsdf_cloud_ptr_);
//				kinfu_->volume ().convertToTsdfCloud (tsdf_cloud_ptr_);
//				// cout << "done [" << tsdf_cloud_ptr_->size () << " points]" << endl << endl;
//				cout << "done [" << kinfu_->volume ().size () << " points]" << endl << endl;
//			}
//			else
//				cout << "[!] tsdf volume download is disabled" << endl << endl;
//		}


//    if (scan_mesh_)
//    {
//      scan_mesh_ = false;
//      scene_cloud_view_.showMesh(*kinfu_, integrate_colors_);
//    }
//
//    if (has_image)
//    {
//      Eigen::Affine3f viewer_pose = getViewerPose(scene_cloud_view_.cloud_viewer_);
//      image_view_.showScene (*kinfu_, rgb24, registration_, independent_camera_ ? &viewer_pose : 0);
//    }

//    if (current_frame_cloud_view_)
//      current_frame_cloud_view_->show (*kinfu_);

		// if ICP is lost, we show the world from a farther view point
//    if(kinfu_->icpIsLost())
//    {
//      setViewerPose (scene_cloud_view_.cloud_viewer_, kinfu_->getCameraPose () * delta_lost_pose_);
//    }
//    else if (!independent_camera_)
//      setViewerPose (scene_cloud_view_.cloud_viewer_, kinfu_->getCameraPose ());

//    if (enable_texture_extraction_ && !kinfu_->icpIsLost ()) {
//      if ( (frame_counter_  % snapshot_rate_) == 0 )   // Should be defined as a parameter. Done.
//      {
//        screenshot_manager_.saveImage (kinfu_->getCameraPose (), rgb24);
//      }
//    }
//
//    // display ICP state
//    scene_cloud_view_.displayICPState (*kinfu_, was_lost_);

	}


	void sourceKinfu(const pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr & DC3)
	{
		{
		//	std::cout << "Giving colors1\n";

		//	std::cout << "width "<<width<<"\n";
			//  boost::this_thread::sleep (boost::posix_time::millisec (5));
		//	boost::mutex::scoped_try_lock lock(data_ready_mutex_);
			//  boost::unique_lock<boost::mutex> lock(data_ready_mutex_);
			//std::cout << lock << std::endl; //causes compile errors
//			if (exit_ || !lock)
//				return;

			int width  = DC3->width;
			int height = DC3->height;
			depth_.cols = width;
			depth_.rows = height;
			depth_.step = depth_.cols * depth_.elemSize();
			source_depth_data_.resize(depth_.cols * depth_.rows);

//			rgb24_.cols = width;
//			rgb24_.rows = height;
//			rgb24_.step = rgb24_.cols * rgb24_.elemSize();
//			source_image_data_.resize(rgb24_.cols * rgb24_.rows);

//			unsigned char *rgb    = (unsigned char *)  &source_image_data_[0];
			unsigned short *depth = (unsigned short *) &source_depth_data_[0];

			//std::cout << "Giving colors3\n";
			for (int i=0; i<width*height; i++) {
				PointXYZRGB pt = DC3->at(i);
//				rgb[3*i +0] = pt.r;
//				rgb[3*i +1] = pt.g;
//				rgb[3*i +2] = pt.b;
				depth[i]    = pt.z/0.001;
			}
			//std::cout << "Giving colors4\n";
//			rgb24_.data = &source_image_data_[0];
			depth_.data = &source_depth_data_[0];
		}
	//	has_data_=true;
	//	data_ready_cond_.notify_one();
	}

	void
	startMainLoop (void)
	{

		{
		//	boost::unique_lock<boost::mutex> lock(data_ready_mutex_);



			// cout << "has_data" << has_data_<< endl;
		//	boost::thread_group threads;
            kinfu_->resetGlobaltime();

			for (auto & cl: capture_) {

		//		threads.create_thread(boost::bind(&KinFuLSApp::sourceKinfu, this , cl) );
			//	std::cout<<"clouds point : "<<cl->points[100000]<<std::endl;

				sourceKinfu(cl);
				//continue;
			//	bool has_data = data_ready_cond_.timed_wait(lock, boost::posix_time::millisec(100), [this]() { return has_data_ == true; });
				// has_data_=false;
				// while (!has_data_) data_ready_cond_.wait(lock);
				// bool has_data = has_data_;
				// cout << "depth rgb24" << depth_<< " "<< rgb24_<< endl;
				try { this->execute(depth_, rgb24_);
				// std::cout << "depth " << depth_.data<< " "<< std:: endl;

				}
				catch (const std::bad_alloc & /*e*/) {
					cout << "Bad alloc" << endl;
					//  break;
				}
				catch (const std::exception & /*e*/) {
					cout << "Exception" << endl;
					//	  break;
				}


			}
		//	return;

			 if (!marching_cubes_)
			    marching_cubes_ = kinfu::gpu::MarchingCubes::Ptr( new kinfu::gpu::MarchingCubes() );

			    kinfu::gpu::DeviceArray<PointXYZ> triangles_device = marching_cubes_->run(kinfu_->volume(), triangles_buffer_device_);
			    mesh_ptr_ = convertToMesh(triangles_device);
			 Eigen::Matrix<float, 4, 4> temp_matr;
			   temp_matr<< 1, 0, 0, 1.5,
			               0,  1,  0, 1.5,
			               0,  0, 1, -0.3,
			               0, 0, 0, 1;
//			   plyWriter.write ("frame-test.ply", *capture_[0], true, false);
//			   pcl::transformPointCloud(*capture_[0], *capture_[0], temp_matr );
//			    plyWriter.write ("frame-test00.ply", *capture_[0], true, false);
////			    cv::imwrite("0.jpg", *colors_[0] );
//			   pcl::PolygonMesh triangles;
//			   pcl::io::loadPolygonFilePLY("frame-test01.ply", triangles);


			//    cout << "Saving mesh to to 'mesh.ply'... " << flush;
		      pcl::io::savePLYFile("mesh.ply", *mesh_ptr_);

			 //   pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
			  //  pcl::fromPCLPointCloud2(mesh_ptr_->cloud, *cloud);
//			    kinfu::TextureMesh mesh;
//			    mesh.cloud = std::move(triangles.cloud);
//			    std::vector< pcl::Vertices> polygon_1;
//
//				 // push faces into the texturemesh object
//				polygon_1.resize (triangles.polygons.size ());
//				for(size_t i =0; i < triangles.polygons.size (); ++i)
//				{
//				   polygon_1[i] = triangles.polygons[i];
//				}
//			   	 mesh.tex_polygons.push_back(polygon_1);

			    kinfu::TextureMesh mesh;
			    mesh.cloud = std::move(mesh_ptr_->cloud);
			    std::vector< pcl::Vertices> polygon_1;

			     // push faces into the texturemesh object
			    polygon_1.resize (mesh_ptr_->polygons.size ());
			    for(size_t i =0; i < mesh_ptr_->polygons.size (); ++i)
			    {
			       polygon_1[i] = mesh_ptr_->polygons[i];
			    }
			    mesh.tex_polygons.push_back(polygon_1);


			    mesh.tex_materials.resize (cams.size () + 1);
			     for(int i = 0 ; i <= cams.size() ; ++i)
			     {
			       kinfu::TexMaterial mesh_material;
			       mesh_material.tex_Ka.r = 0.2f;
			       mesh_material.tex_Ka.g = 0.2f;
			       mesh_material.tex_Ka.b = 0.2f;

			       mesh_material.tex_Kd.r = 0.8f;
			       mesh_material.tex_Kd.g = 0.8f;
			       mesh_material.tex_Kd.b = 0.8f;

			       mesh_material.tex_Ks.r = 1.0f;
			       mesh_material.tex_Ks.g = 1.0f;
			       mesh_material.tex_Ks.b = 1.0f;

			       mesh_material.tex_d = 1.0f;
			       mesh_material.tex_Ns = 75.0f;
			       mesh_material.tex_illum = 2;

			       std::stringstream tex_name;
			       tex_name << "material_" << i;
			       tex_name >> mesh_material.tex_name;

			       if(i < cams.size ())
			         mesh_material.tex_file = cams[i].texture_file;
			       else
			         mesh_material.tex_file = "occluded.jpg";

			       mesh.tex_materials[i] = mesh_material;
			     }

			    //cv::resize(*colors_[0],*colors_[0],  cv::Size(), 0.5, 0.4 );
			    cv::imwrite("0.jpg", *colors_[0] );
			//    cv::imwrite("1.jpg", *colors_[1] );
			   // *colors_[0]=cv::imwrite("0.jpg", *colors_[0] );
			   // kinfu::TextureMapping<pcl::PointXYZ> tm; // TextureMapping object that will perform the sort
			   // tm.setColorsMap(colorsMap_);
			   // tm.openFileForTitting("coordDepthandRgb.txt");
			  //  tm.textureMeshwithMultipleCameras(mesh, cams);
			 //   tm.textureMeshwithMultipleCameras(mesh, cams);
			  //  this->saveOBJFile ("textured_mesh.obj", mesh, 5);



			    kinfu::TextureMappingGPU<kinfu::device::PointType> tmGPU;
			    tmGPU.textureMeshwithMultipleCameras(triangles_device, cams);


		}



	}

	void setSource( const std::vector<boost::shared_ptr<PointCloud<PointT>>> &source, const std::vector<boost::shared_ptr<cv::Mat> > &colorsMap, const std::vector<boost::shared_ptr<cv::Mat> > &colors){

		colors_= colors;
		colorsMap_=colorsMap;
		capture_= source;
	}

	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


	bool exit_;
	bool scan_;
	bool scan_mesh_;
	bool scan_volume_;

	bool independent_camera_;
	int frame_counter_;

	bool registration_;
	bool integrate_colors_;
	bool pcd_source_;
	float focal_length_;
	PLYWriter plyWriter;
	kinfu::gpu::KinfuTracker *kinfu_;


	kinfu::gpu::KinfuTracker::DepthMap depth_device_;

	pcl::PointCloud<pcl::PointXYZI>::Ptr tsdf_cloud_ptr_;
	std::vector<boost::shared_ptr<PointCloud<PointT>>> capture_;
   std::vector<boost::shared_ptr<cv::Mat> > colors_;
   std::vector<boost::shared_ptr<cv::Mat> > colorsMap_;

	boost::recursive_mutex data_ready_mutex2_;
	boost::mutex data_ready_mutex_;
	boost::condition_variable data_ready_cond_;
	bool has_data_=false;
	std::vector<kinfu::gpu::PixelRGB> source_image_data_;
	std::vector<unsigned short> source_depth_data_;
	kinfu::gpu::PtrStepSz<const unsigned short> depth_;
	kinfu::gpu::PtrStepSz<const kinfu::gpu::PixelRGB> rgb24_;

	kinfu::gpu::MarchingCubes::Ptr marching_cubes_;

	kinfu::gpu::DeviceArray<PointXYZ> triangles_buffer_device_;
	boost::shared_ptr<pcl::PolygonMesh> mesh_ptr_;

	Eigen::Affine3f delta_lost_pose_;
	kinfu::texture_mappingGPU::CameraVector cams;

	bool was_lost_;

	int time_ms_;
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

};

struct K2G_generator{
public:
  K2G_generator(Processor freenectprocessor, bool mirroring, char **argv): freenectprocessor_(freenectprocessor), mirroring_(mirroring), argv_(argv),n_(0){}
    K2G * operator ()(){return new K2G(freenectprocessor_, mirroring_, argv_[n_++ ]);}
private:
  unsigned int n_;
  Processor freenectprocessor_;
  bool mirroring_;
  char **argv_;
};
//////////////////////////////////////////////////////////////////////////////////////////
template <typename PointT>
class PCDBuffer
{
  public:
    PCDBuffer () {}

    bool
   // pushBack (const std::vector<typename PointCloud<PointT>::Ptr>  &, const std::vector<boost::shared_ptr<cv::Mat> > &); // thread-save wrapper for push_back() method of ciruclar_buffer

  //   pushBack ( const std::pair< boost::shared_ptr<std::vector< PointCloud<PointT>>>,  boost::shared_ptr<std::vector<cv::Mat>> > & ) ;

	pushBack(const std::vector<std::tuple<boost::shared_ptr<PointCloud<PointT>>, boost::shared_ptr<cv::Mat>,  boost::shared_ptr<cv::Mat> > >   &);

//	pushBack(const std::vector<std::pair<boost::shared_ptr<PointCloud<PointT>>, boost::shared_ptr<cv::Mat> > >   &);

   // std::pair< boost::shared_ptr<std::vector< PointCloud<PointT>>>,  boost::shared_ptr<std::vector<cv::Mat>> >
    std::vector<std::tuple<boost::shared_ptr<PointCloud<PointT>>, boost::shared_ptr<cv::Mat>, boost::shared_ptr<cv::Mat> > >
    //std::vector<std::pair<boost::shared_ptr<PointCloud<PointT>>, boost::shared_ptr<cv::Mat> > >
   //  std::pair<std::vector<typename PointCloud<PointT>::Ptr>, std::vector<boost::shared_ptr<cv::Mat>>>
// const  std::vector< typename PointCloud<PointT>::Ptr >
    getFront (); // thread-save wrapper for front() method of ciruclar_buffer

    inline bool
    isFull ()
    {
      boost::mutex::scoped_lock buff_lock (bmutex_);
      return (buffer_.full ());
    }

    inline bool
    isEmpty ()
    {
      boost::mutex::scoped_lock buff_lock (bmutex_);
      return (buffer_.empty ());
    }

    inline int
    getSize ()
    {
      boost::mutex::scoped_lock buff_lock (bmutex_);
      return (int (buffer_.size ()));
    }

    inline int
    getCapacity ()
    {
      return (int (buffer_.capacity ()));
    }

    inline void
    setCapacity (int buff_size)
    {
      boost::mutex::scoped_lock buff_lock (bmutex_);
      buffer_.set_capacity (buff_size);
    }

  private:
    PCDBuffer (const PCDBuffer&); // Disabled copy constructor
    PCDBuffer& operator = (const PCDBuffer&); // Disabled assignment operator

    boost::mutex bmutex_;
    boost::condition_variable buff_empty_;
  //  boost::circular_buffer< std::vector<typename PointCloud<PointT>::Ptr> >  buffer_;
    //boost::circular_buffer<std::pair<std::vector<typename PointCloud<PointT>::Ptr>, std::vector<boost::shared_ptr<cv::Mat>>> > buffer_;
  //  boost::circular_buffer<  std::pair<  boost::shared_ptr<std::vector< PointCloud<PointT>>>,  boost::shared_ptr<std::vector<cv::Mat>> > > buffer_;

    boost::circular_buffer<std::vector<std::tuple<boost::shared_ptr<PointCloud<PointT>>, boost::shared_ptr<cv::Mat>, boost::shared_ptr<cv::Mat> > >  >  buffer_;

  //  boost::circular_buffer<std::vector<std::pair<boost::shared_ptr<PointCloud<PointT>>, boost::shared_ptr<cv::Mat> > >  >  buffer_;
    //boost::circular_buffer<std::pair<std::vector<typename PointCloud<PointT>::Ptr, std::vector<boost::shared_ptr<cv::Mat>  >  buffer_;

  //  boost::circular_buffer<  std::vector<typename PointCloud<PointT>::Ptr> >  bufferColors_;

};

//////////////////////////////////////////////////////////////////////////////////////////
template <typename PointT> bool
PCDBuffer<PointT>::pushBack (
//const   std::vector<std::pair<boost::shared_ptr<PointCloud<PointT>>, boost::shared_ptr<cv::Mat> > > & cloudColors
const   std::vector<std::tuple<boost::shared_ptr<PointCloud<PointT>>, boost::shared_ptr<cv::Mat>, boost::shared_ptr<cv::Mat> > > & cloudColors


	 //const std::pair<  boost::shared_ptr<std::vector< PointCloud<PointT>>>,  boost::shared_ptr<std::vector<cv::Mat>> >
/* const std::vector<typename PointCloud<PointT>::Ptr>  & clouds, const std::vector<boost::shared_ptr<cv::Mat> > &colors*/ )
{
  bool retVal = false;
  {
    boost::mutex::scoped_lock buff_lock (bmutex_);
    if (!buffer_.full ())
      retVal = true;
     // buffer_.push_back (std::make_pair(clouds,colors));
      buffer_.push_back (cloudColors);
      //buffer_.second.push_back (colors);

  }
  buff_empty_.notify_one ();
  return (retVal);
}

//////////////////////////////////////////////////////////////////////////////////////////
template <typename PointT>
 //std::pair<  boost::shared_ptr<std::vector< PointCloud<PointT>>>,  boost::shared_ptr<std::vector<cv::Mat>> >
std::vector<std::tuple<boost::shared_ptr<PointCloud<PointT>>, boost::shared_ptr<cv::Mat>, boost::shared_ptr<cv::Mat> > >
//std::vector<std::pair<boost::shared_ptr<PointCloud<PointT>>, boost::shared_ptr<cv::Mat> > >
// std::pair<std::vector<typename PointCloud<PointT>::Ptr>, std::vector<boost::shared_ptr<cv::Mat>>>
//const  std::vector<typename PointCloud<PointT>::Ptr >
PCDBuffer<PointT>::getFront ()
{
	std::vector<std::tuple<boost::shared_ptr<PointCloud<PointT>>, boost::shared_ptr<cv::Mat>, boost::shared_ptr<cv::Mat> > >  cloudColors;
 //  std::vector<std::pair<boost::shared_ptr<PointCloud<PointT>>, boost::shared_ptr<cv::Mat> > >  cloudColors;
 // std::pair<  boost::shared_ptr<std::vector< PointCloud<PointT>>>,  boost::shared_ptr<std::vector<cv::Mat>> > cloudColors;
 // std::pair<std::vector<typename PointCloud<PointT>::Ptr>, std::vector<boost::shared_ptr<cv::Mat>>> cloudColors;
 // std::vector< typename PointCloud<PointT>::Ptr > cloud;
  {
    boost::mutex::scoped_lock buff_lock (bmutex_);
    while (buffer_.empty ())
    {
      if (is_done)
        break;
      {
        boost::mutex::scoped_lock io_lock (io_mutex);
        //cerr << "No data in buffer_ yet or buffer is empty." << endl;
      }
      buff_empty_.wait (buff_lock);
    }
    //cloud = buffer_.front ();
    cloudColors=buffer_.front();
    buffer_.pop_front ();
  }
  return (cloudColors);
}

#define FPS_CALC(_WHAT_, buff) \
do \
{ \
    static unsigned count = 0;\
    static double last = getTime ();\
    double now = getTime (); \
    ++count; \
    std::cout<<"buff.getSize () "<<buff.getSize () <<std::endl;\
    if (now - last >= 1.0) \
    { \
      cerr << "Average framerate("<< _WHAT_ << "): " << double(count)/double(now - last) << " Hz. Queue size: " << buff.getSize () << "\n"; \
      count = 0; \
      last = now; \
    } \
}while(false)





template <typename PointT>
class Producer
{
  private:

//    std::mutex mtx;
//
//    std::condition_variable cv;
    bool ready = false;
    int cnt=0;
    ///////////////////////////////////////////////////////////////////////////////////////





    ///////////////////////////////////////////////////////////////////////////////////////
//    void threadPointCloud (K2G* kinect, boost::shared_ptr<pcl::PointCloud<pcl::PointXYZRGB>> &pointCloud , boost::shared_ptr<cv::Mat> &color, Eigen::Matrix4f transform, unsigned i) {
//       //boost::mutex::scoped_lock buff_lock (io_mutex);
//       std::this_thread::sleep_until(nextTime);
//
//        startTime[i]=std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
//
//        pointCloud= thresholdDepth (pointCloud, 0.4, 2.3);
//
//        endTime[i]=std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
//
//
//
//        pcl::transformPointCloud (*pointCloud, *pointCloud, transform);
//
//        //std::cout<<"time stamp: " <<std::chrono::high_resolution_clock::to_time_t(std::chrono::high_resolution_clock::now())<<std:endl;
//
//    }

    void
    grabAndSend ()
    {
        Processor freenectprocessor = CUDA;
     // freenectprocessor = static_cast<Processor>(atoi(argv[1]));

        int kinect2_count=1;

        bool mirroring=true;
        std::vector<K2G *> kinects(kinect2_count);

        clouds.resize(kinect2_count);
        colors.resize(kinect2_count);
        colorsMap.resize(kinect2_count);
        std::vector<boost::shared_ptr<pcl::PointCloud<pcl::PointXYZRGB>>> clouds_1(kinect2_count);

        boost::shared_ptr<pcl::PointCloud<pcl::PointXYZRGB>> allPointClouds(new pcl::PointCloud<pcl::PointXYZRGB>);

     //   Eigen::Matrix4f GlobalTransform = Eigen::Matrix4f::Identity ();

      //char *arg[]={"011054343347","021401644747","003726334247","291296634347"};
   //// char *arg[]={ "011054343347", "291296634347","003726334247","021401644747"};
   //  char *arg[]= {"021401644747", "291296634347" , "011054343347", "003726334247"};
    //    char *arg[]= { "291296634347" , "011054343347", "003726334247"};
 // char *arg[]={ "021401644747","003726334247","011054343347"};
  // char *arg[]={ "021401644747", "011054343347", "003726334247"};

      //  char *arg[]={ "021401644747","291296634347","003726334247"};
     //    char *arg[]={ (char *)"021401644747", (char *)"291296634347"};
        char *arg[]={  (char *)"291296634347"};
      //  char *arg[]={(char *)"021401644747"};
        // char *arg[]={"003726334247"};
        // char *arg[]={"011054343347"};
        K2G_generator kinect_generator(freenectprocessor, mirroring, arg);
        std::generate(kinects.begin(), kinects.end(), kinect_generator);

        for(size_t i = 0; i < kinect2_count; ++i)
        {
           //  if (i==1) continue;

            clouds[i] = kinects[i]->getCloud();
            colors[i].reset(new cv::Mat(  ));
            colorsMap[i].reset(new cv::Mat(  ));

//            clouds[i]->sensor_orientation_.w() = 0.0;
//            clouds[i]->sensor_orientation_.x() = 1.0;
//            clouds[i]->sensor_orientation_.y() = 0.0;
//            clouds[i]->sensor_orientation_.z() = 0.0;
            kinects[i]->printParameters();


        }

      // std::vector<Eigen::Matrix4f> Transforms;


//      if (is_file_exist("output.transforms")  ) {
//
//       readTransformsBinary("output.transforms",Transforms);
//
//
//
//
//      }  else return ;

//      std::chrono::high_resolution_clock::time_point tnow, tpost,timePoint ;
//      nextTime = std::chrono::high_resolution_clock::now();
//      timePoint+=std::chrono::milliseconds(ttime);
//      timePoint+=std::chrono::milliseconds(2000);
//     // std::chrono::time_point<std::chrono::milliseconds> timePoint;
//      nextTime+=std::chrono::milliseconds(3000);


     // ptime time_t_epoch(date(1970,1,1));
      //posixNexTime = microsec_clock::local_time();
      posixNexTime = boost::asio::time_traits<boost::posix_time::ptime>::now();
      //diff = now - time_t_epoch;
    //  posixNexTime+=time_duration(milliseconds(3000));
      int say=0;
      int sumendTimeDeltas = 0,sumstartTimeDeltas = 0;
      for(int i=0;i<3;i++){startTime[i]=0;endTime[i]=0;}

        while (true)
        {

        //  if( say == 200) break ;

            say++;




            boost::asio::io_service io;
            //  boost::asio::deadline_timer t1(io, boost::posix_time::milliseconds(15));
            //   boost::asio::deadline_timer t2(io, boost::posix_time::milliseconds(15));
            //  boost::asio::deadline_timer t3(io, boost::posix_time::milliseconds(30));

            boost::asio::deadline_timer t1(io);
        //   boost::asio::deadline_timer t2(io);
           // boost::asio::deadline_timer t3(io);
            posixNexTime+=time_duration(milliseconds(80));
            t1.expires_at(posixNexTime);
      //      t2.expires_at(posixNexTime);
           // t3.expires_at(posixNexTime);

                   //  t1.async_wait(boost::bind(&Producer::threadPointCloud,this, kinects[0], boost::ref(clouds[0]), Eigen::Matrix4f::Identity ()));
            t1.async_wait([&] (const boost::system::error_code&){

                startTime[0]=std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();

                colors[0].reset(new cv::Mat(  ));   ;//clouds[0].reset();

               // kinects[0]->getColorwithCloud(colors[0], clouds[0]);
                kinects[0]->getColorwithCloudwithColorMap(colors[0], colorsMap[0],  clouds[0], true);
//                clouds[0]->sensor_orientation_.w() = 0.0;
//				clouds[0]->sensor_orientation_.x() = 1.0;
//				clouds[0]->sensor_orientation_.y() = 0.0;
//				clouds[0]->sensor_orientation_.z() = 0.0;

                endTime[0]=std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();


              });



//           t2.async_wait([&](const boost::system::error_code&){
//
//               startTime[1]=std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
//
//               colors[1].reset(new cv::Mat(  ));
//
//               kinects[1]->getColorwithCloudwithColorMap(colors[1], colorsMap[1],  clouds[1], true);
//             //  kinects[1]->getColorwithCloud(colors[1], clouds[1]);
//
//               endTime[1]=std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
//
//
//           });

//
//
//
//            t3.async_wait([&](const boost::system::error_code&){
//
//                startTime[2]=std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
//
//
//                colors[2].reset(new cv::Mat(  ));
//
//                kinects[2]->getColorwithCloud(colors[2], clouds[2]);
//
//                endTime[2]=std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
//
//
//          });

         std::thread ta([&](){io.run();});
       //  std::thread tb([&](){io.run();});
       //  std::thread tc([&](){io.run();});



         ta.join(); //tb.join();//tc.join();



         //std::cout<<"start delta 1 2 3: " <<abs(start[0]- start[1])<<" "<< abs(start[0]- start[2]) <<" "<<abs(start[1]- start[2]) <<std::endl;
         sumstartTimeDeltas=sumstartTimeDeltas + (1.f/1)*sqrt((startTime[0]- startTime[1])*(startTime[0]- startTime[1])+ (startTime[0]- startTime[2])*(startTime[0]- startTime[2]) + (startTime[1]- startTime[2])*(startTime[1]- startTime[2])) ;
         sumendTimeDeltas=sumendTimeDeltas + (1.f/1)*sqrt((endTime[0]- endTime[1])*(endTime[0]- endTime[1])+ (endTime[0]-endTime[2])*(endTime[0]- endTime[2]) + (endTime[1]- endTime[2])*(endTime[1]- endTime[2])) ;





//          std::vector<std::pair<boost::shared_ptr<PointCloud<PointT>>, boost::shared_ptr<cv::Mat> > > colorClouds;
//
//          for(int i=0;i<clouds.size(); i++){
//
//          colorClouds.push_back(std::make_pair(boost::make_shared<PointCloud<PointT>>(*clouds[i]), boost::make_shared<cv::Mat>( *colors[i])));
//          }

		 std::vector<std::tuple<boost::shared_ptr<PointCloud<PointT>>, boost::shared_ptr<cv::Mat>, boost::shared_ptr<cv::Mat> > > colorClouds;

		 for(int i=0;i<clouds.size(); i++)
			colorClouds.push_back(std::make_tuple(boost::make_shared<PointCloud<PointT>>(*clouds[i]), boost::make_shared<cv::Mat>( *colorsMap[i]), boost::make_shared<cv::Mat>( *colors[i])));




          if (!buf_.pushBack ( colorClouds) )
            {
              {
                boost::mutex::scoped_lock io_lock (io_mutex);
                print_warn ("Warning! Buffer was full, overwriting data!\n");
              }
            }






        FPS_CALC ("cloud callback.", buf_);


        cnt++;
        if (is_done || cnt>0)
          break;



      //  boost::this_thread::sleep (boost::posix_time::milliseconds (60));
      }
        std::cout<<"Average start Delta: "<<float(sumstartTimeDeltas)/say<<std::endl;
        std::cout<<"Average end Delta: "<<float(sumendTimeDeltas)/say<<std::endl;


    for(auto & k : kinects)
    k->shutDown();

    }

  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Producer (PCDBuffer<PointT> &buf)
      : buf_ (buf)
    //    depth_mode_ (depth_mode)
    {
      thread_.reset (new boost::thread (boost::bind (&Producer::grabAndSend, this)));
    }

    ///////////////////////////////////////////////////////////////////////////////////////
    void
    stop ()
    {
      thread_->join ();
      boost::mutex::scoped_lock io_lock (io_mutex);
      print_highlight ("Producer done.\n");
    }

    private:
        PCDBuffer<PointT> &buf_;
        //  openni_wrapper::OpenNIDevice::DepthMode depth_mode_;
        boost::shared_ptr<boost::thread> thread_;

        /*   boost::shared_ptr<std::vector<pcl::PointCloud<pcl::PointXYZRGB>>> clouds;
        boost::shared_ptr<std::vector<cv::Mat>> colors;*/
        std::chrono::high_resolution_clock::time_point  nextTime;
        ptime posixNexTime;
        long startTime[3],endTime[3];
        std::vector< boost::shared_ptr<pcl::PointCloud<pcl::PointXYZRGB>>> clouds;
        std::vector<boost::shared_ptr<cv::Mat>> colors;
        std::vector<boost::shared_ptr<cv::Mat>> colorsMap;

};



template <typename PointT>
class Consumer
{
  private:
    ///////////////////////////////////////////////////////////////////////////////////////
    void
	writeToDisk (  std::vector<std::tuple<boost::shared_ptr<PointCloud<PointT>>, boost::shared_ptr<cv::Mat>, boost::shared_ptr<cv::Mat> > >   &&colorClouds ) {
        std::string time = boost::posix_time::to_iso_string (boost::posix_time::microsec_clock::local_time ());
        uint8_t j=0;
    	typedef  std::vector<std::tuple<boost::shared_ptr<PointCloud<PointT>>, boost::shared_ptr<cv::Mat>, boost::shared_ptr<cv::Mat> > > colorMapCloudstuples;
    	for (typename colorMapCloudstuples::const_iterator i = colorClouds.begin(); i != colorClouds.end(); ++i) {
    		   plyWriter.write ("frame-" + time + "_" + std::to_string(j)+".ply", *(get<0>(*i)), true, false);
    		   cv::imwrite("frame-" + time + "_" + std::to_string(j) + ".jpg", *(get<2>(*i)) );
    		   boost::shared_ptr<cv::Mat> tmp= get<1>(*i);

		  for(int k=0;k<tmp->rows;k++ )
        	for(int l=0;l<tmp->cols;l++ )
    		std::cout<<"color map: "<<k<<" th. row "<<l<<" th. coloumn : "<<tmp->at<cv::Vec2f>(k, l)[0]<<" "<<tmp->at<cv::Vec2f>(k, l)[1]<<std::endl;
    		   j++;
    	}


    	 FPS_CALC ("cloud write.", buf_);
    }

	void writeToDisk (  std::vector<std::pair<boost::shared_ptr<PointCloud<PointT>>, boost::shared_ptr<cv::Mat> > >   &&colorClouds )
    {
      std::string time = boost::posix_time::to_iso_string (boost::posix_time::microsec_clock::local_time ());
      allPointClouds.reset(new pcl::PointCloud<pcl::PointXYZRGB>);
      for (int i=0;i<colorClouds.size();++i) {
     // stringstream ss;
      //ss << "frame-" << time <<"_"<<i<< ".ply";
    //   colorClouds.first[i] = thresholdDepth (colorClouds.first[i], 0.4, 2.3);
    //   std::cout<<Transforms.at(1)<<std::endl;

     //  colorClouds[i].first= thresholdDepth (colorClouds[i].first, 0.4, 2.);

      // colorClouds[i].first=removeOutliers(colorClouds[i].first, 0.03, 50);
    //    colorClouds[i].first=downsample(colorClouds[i].first, 0.005);

    //    if (i!=0) pcl::transformPointCloud (*colorClouds[i].first, *colorClouds[i].first, Transforms.at(i-1) );

      //*allPointClouds+=*colorClouds[i].first;
      plyWriter.write ("frame-" + time + "_" + std::to_string(i)+".ply", *(colorClouds[i].first), true, false);
      cv::imwrite("frame-" + time + "_" + std::to_string(i) + ".jpg", *(colorClouds[i].second) );
        //std::vector<int> compression_params;


      //writer_.writeBinaryCompressed (ss.str (), *clouds[i]);
     }

    //writer_.writeBinaryCompressed ("frame-" + time + ".pcd", *allPointClouds);
    // plyWriter.write ("frame-" + time + ".ply", *allPointClouds, true, false);

      FPS_CALC ("cloud write.", buf_);
    }
//    void send2KinfuApp(std::vector<std::pair<boost::shared_ptr<PointCloud<PointT>>, boost::shared_ptr<cv::Mat> > >   &&colorClouds){
//         std::vector<boost::shared_ptr<PointCloud<PointT>> >   clouds;
//         std::vector<boost::shared_ptr<cv::Mat> > colors;
//    	 for (auto & cl:colorClouds) {
//    	 //	std::cout<<"clouds point : "<<cl.first->points[2000]<<std::endl;
//    	 	clouds.push_back(cl.first);
//    	 	colors.push_back(cl.second);
//
//    	 }
//      app->setSource(std::move(clouds), std::move(colors));
//
//	  pcl::console::setVerbosityLevel(pcl::console::L_VERBOSE);
//	  try { app->startMainLoop (); }
//	  catch (const pcl::PCLException& /*e*/) { cout << "PCLException" << endl; }
//	  catch (const std::bad_alloc& /*e*/) { cout << "Bad alloc" << endl; }
//	  catch (const std::exception& /*e*/) { cout << "Exception" << endl; }
//
//      FPS_CALC ("cloud write.", buf_);
//    }


    void send2KinfuApp(std::vector<std::tuple<boost::shared_ptr<PointCloud<PointT>>, boost::shared_ptr<cv::Mat>, boost::shared_ptr<cv::Mat> > >   &&colorClouds){
          std::vector<boost::shared_ptr<PointCloud<PointT>> >   clouds;
          std::vector<boost::shared_ptr<cv::Mat> > colors;
          std::vector<boost::shared_ptr<cv::Mat> > colorsMap;

     	 for (auto & cl:colorClouds) {
     	 //	std::cout<<"clouds point : "<<cl.first->points[2000]<<std::endl;
     	 	clouds.push_back(std::get<0>(cl));
     	 	colorsMap.push_back(std::get<1>(cl));
     	 	colors.push_back(std::get<2>(cl));

     	 }
       app->setSource(std::move(clouds), std::move(colorsMap), std::move(colors));

 	  pcl::console::setVerbosityLevel(pcl::console::L_VERBOSE);
 	  try { app->startMainLoop (); }
 	  catch (const pcl::PCLException& /*e*/) { cout << "PCLException" << endl; }
 	  catch (const std::bad_alloc& /*e*/) { cout << "Bad alloc" << endl; }
 	  catch (const std::exception& /*e*/) { cout << "Exception" << endl; }

       FPS_CALC ("cloud write.", buf_);
     }
//     void
//    writePolyMeshToDisk (const  pcl::PolygonMesh & mesh)
//    {
//      stringstream ss;
//      std::string time = boost::posix_time::to_iso_string (boost::posix_time::microsec_clock::local_time ());
//      ss << "frame-" << time << ".vtk";
//      pcl::io::saveVTKFile (ss.str(), mesh);
//     // FPS_CALC ("cloud write.", buf_);
//    }




    ///////////////////////////////////////////////////////////////////////////////////////
    // Consumer thread function
    void
    receiveAndProcess ()
    {

//
//      if (is_file_exist("output.transforms")  ) {
//
//       readTransformsBinary("output.transforms",Transforms);
//
//
//
//
//      }  else return ;



      while (true)
      {
        if (is_done)   break;
     //  writeToDisk (buf_.getFront ());
       send2KinfuApp (buf_.getFront());
      }

      {
        boost::mutex::scoped_lock io_lock (io_mutex);
        print_info ("Writing remaing %ld clouds in the buffer to disk...\n", buf_.getSize ());
      }
      while (!buf_.isEmpty ()) {

    	send2KinfuApp (buf_.getFront());
       //writeToDisk (buf_.getFront ());
      }
    }

  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Consumer (PCDBuffer<PointT> &buf)
      : buf_ (buf)
    {
      app=boost::shared_ptr<KinFuLSApp<pcl::PointXYZRGB>>(new KinFuLSApp<pcl::PointXYZRGB>());
     // std::cout<<"hello "<<std::endl;
      thread_.reset (new boost::thread (boost::bind (&Consumer::receiveAndProcess, this)));
    }

    ///////////////////////////////////////////////////////////////////////////////////////
    void
    stop ()
    {
      thread_->join ();
      boost::mutex::scoped_lock io_lock (io_mutex);
      print_highlight ("Consumer done.\n");
    }

  private:
    PCDBuffer<PointT> &buf_;
    boost::shared_ptr<boost::thread> thread_;
    PCDWriter writer_;
    PLYWriter plyWriter;
  //  std::vector<Eigen::Matrix4f> Transforms;

    boost::shared_ptr<pcl::PointCloud<pcl::PointXYZRGB>> allPointClouds;//(new pcl::PointCloud<pcl::PointXYZRGB>);
    boost::shared_ptr<KinFuLSApp<pcl::PointXYZRGB>> app;

};

#if defined(__linux__) || defined (TARGET_OS_MAC)

// Get the available memory size on Linux/BSD systems

size_t
getTotalSystemMemory ()
{
  uint64_t memory = std::numeric_limits<size_t>::max ();

#ifdef _SC_AVPHYS_PAGES
  uint64_t pages = sysconf (_SC_AVPHYS_PAGES);
  uint64_t page_size = sysconf (_SC_PAGE_SIZE);

  memory = pages * page_size;

#elif defined(HAVE_SYSCTL) && defined(HW_PHYSMEM)
  // This works on *bsd and darwin.
  unsigned int physmem;
  size_t len = sizeof physmem;
  static int mib[2] = { CTL_HW, HW_PHYSMEM };

  if (sysctl (mib, ARRAY_SIZE (mib), &physmem, &len, NULL, 0) == 0 && len == sizeof (physmem))
  {
    memory = physmem;
  }
#endif

  if (memory > uint64_t (std::numeric_limits<size_t>::max ()))
  {
    memory = std::numeric_limits<size_t>::max ();
  }

  print_info ("Total available memory size: %lluMB.\n", memory / 1048576ull);
  return size_t (memory);
}

const size_t BUFFER_SIZE = size_t (getTotalSystemMemory () / (512 * 424 * sizeof (pcl::PointXYZRGBA)));
#else

const size_t BUFFER_SIZE = 400;
#endif

void
ctrlC (int)
{
  boost::mutex::scoped_lock io_lock (io_mutex);
  print_info ("\nCtrl-C detected, exit condition set to true.\n");
  is_done = true;
}




int main (int argc, char* argv[])
{

     if (pc::find_switch (argc, argv, "--help") || pc::find_switch (argc, argv, "-h"))  std::cout<<"Hello"<<std::endl;
        //return print_cli_help ();


    int device = 0;
    pc::parse_argument (argc, argv, "-gpu", device);


    pc::parse_argument (argc, argv, "-gpu", device);
    kinfu::gpu::setDevice(device);

    kinfu::gpu::printShortCudaDeviceInfo (device);



    int buff_size = BUFFER_SIZE;



    PCDBuffer<PointXYZRGB> buf;
    buf.setCapacity (buff_size);
    Producer<PointXYZRGB> producer (buf);
    boost::this_thread::sleep (boost::posix_time::seconds (1));
    Consumer<PointXYZRGB> consumer (buf);

    signal (SIGINT, ctrlC);
    producer.stop ();
    consumer.stop ();






}
