/*
 * Software License Agreement (BSD License)
 *
 *  Point Cloud Library (PCL) - www.pointclouds.org
 *  Copyright (c) 2012, Sudarshan Srinivasan <sudarshan85@gmail.com>
 *  Copyright (c) 2012-, Open Perception, Inc.
 * 
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 * 
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 */

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
//#include <pcl/io/openni_grabber.h>
#include <boost/thread/condition.hpp>
#include <boost/circular_buffer.hpp>
#include <csignal>
#include <limits>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>

#include <pcl/console/print.h>
#include <pcl/console/parse.h>
#include <pcl/common/time.h> //fps calculations
#include <pcl/registration/transforms.h>
#include <pcl/io/vtk_io.h>
#include <opencv2/opencv.hpp>

#include "k2g.h"
#include <pcl/visualization/cloud_viewer.h>
#include <chrono>
#include <thread>             // std::thread
#include <condition_variable> // std::condition_variable
#include <mutex>              // std::mutex, std::unique_lock

#include <pcl/filters/passthrough.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/segmentation/extract_clusters.h>

#include <boost/asio.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/date_time/local_time/local_time.hpp>

#include <tuple>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/xml_parser.hpp>
//#include <algorithm>


using namespace std;
using namespace pcl;
using namespace pcl::console;

using namespace boost::gregorian; 
using namespace boost::local_time;
using namespace boost::posix_time;

bool is_done = false;
boost::mutex io_mutex;
long ttime;
//std::chrono::milliseconds waitTime;
std::chrono::high_resolution_clock::time_point  nextTime;
ptime posixNexTime;
long startTime[3],endTime[3];
std::vector<Eigen::Matrix4f>  Calibs, GlobalCalibTransforms;

boost::shared_ptr<pcl::PointCloud<pcl::PointXYZRGB>>

downsample (const boost::shared_ptr<pcl::PointCloud<pcl::PointXYZRGB>>
 & input, float leaf_size)
{
  pcl::VoxelGrid<pcl::PointXYZRGB> voxel_grid;
  voxel_grid.setInputCloud (input);
  voxel_grid.setLeafSize (leaf_size, leaf_size, leaf_size);
  boost::shared_ptr<pcl::PointCloud<pcl::PointXYZRGB>>
 downsampled (new pcl::PointCloud<pcl::PointXYZRGB>);
  voxel_grid.filter (*downsampled);

  return (downsampled);
}



boost::shared_ptr<pcl::PointCloud<pcl::PointXYZRGB>>
thresholdDepth (const boost::shared_ptr<pcl::PointCloud<pcl::PointXYZRGB>> & input, float min_depth, float max_depth)
{
 // std::vector<int> mapping;
  // pcl::removeNaNFromPointCloud(*input, *input, mapping);

  pcl::PassThrough<pcl::PointXYZRGB> pass_through;
  pass_through.setInputCloud (input);
  pass_through.setFilterFieldName ("z");
  pass_through.setKeepOrganized (true); 
  pass_through.setFilterLimits (min_depth, max_depth);
  boost::shared_ptr<pcl::PointCloud<pcl::PointXYZRGB>> thresholded (new pcl::PointCloud<pcl::PointXYZRGB>);
  pass_through.filter (*thresholded);

  return (thresholded);
}



boost::shared_ptr<pcl::PointCloud<pcl::PointXYZRGB>>
removeOutliers (const boost::shared_ptr<pcl::PointCloud<pcl::PointXYZRGB>>  & input, float radius, int min_neighbors)
{
  pcl::RadiusOutlierRemoval<pcl::PointXYZRGB> radius_outlier_removal;
  radius_outlier_removal.setInputCloud (input);
  radius_outlier_removal.setRadiusSearch (radius);
  radius_outlier_removal.setMinNeighborsInRadius (min_neighbors);
  boost::shared_ptr<pcl::PointCloud<pcl::PointXYZRGB>> inliers (new pcl::PointCloud<pcl::PointXYZRGB>);
  radius_outlier_removal.filter (*inliers);

  return (inliers);
}

void
clusterObjects (const boost::shared_ptr<pcl::PointCloud<pcl::PointXYZRGB>>  & input, 
                float cluster_tolerance, int min_cluster_size, int max_cluster_size,
                std::vector<pcl::PointIndices> & cluster_indices_out)
{  
  pcl::EuclideanClusterExtraction<pcl::PointXYZRGB> ec;
  ec.setClusterTolerance (cluster_tolerance);
  ec.setMinClusterSize (min_cluster_size);
  ec.setMaxClusterSize (max_cluster_size);

  ec.setInputCloud (input);
  ec.extract (cluster_indices_out);
}


bool is_file_exist(const char *fileName)
{
    std::ifstream infile(fileName);
    return infile.good();
}

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

#if defined(__linux__) || defined (TARGET_OS_MAC)
#include <unistd.h>
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


 void readTransformFromText(const std::string &file, Eigen::Matrix4f &transform ){



   
       std::ifstream infile(file.c_str());
        std::stringstream buffer;

        buffer << infile.rdbuf();
        float temp;
        std::vector<float> matrixElements;
        while (buffer >> temp) {
        matrixElements.push_back(temp);
      //  std::cout<<"temp: "<<temp<<"\n";

        }

         transform= Eigen::Map<Eigen::Matrix<float,4,4,Eigen::RowMajor> >(matrixElements.data());




      infile.close();



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


//////////////////////////////////////////////////////////////////////////////////////////
template <typename PointT>
class PCDBuffer
{
  public:
    PCDBuffer () {}

    bool 
   // pushBack (const std::vector<typename PointCloud<PointT>::Ptr>  &, const std::vector<boost::shared_ptr<cv::Mat> > &); // thread-save wrapper for push_back() method of ciruclar_buffer

  //   pushBack ( const std::pair< boost::shared_ptr<std::vector< PointCloud<PointT>>>,  boost::shared_ptr<std::vector<cv::Mat>> > & ) ;

    pushBack(const std::vector<std::pair<boost::shared_ptr<PointCloud<PointT>>, boost::shared_ptr<cv::Mat> > >   &);
   
   // std::pair< boost::shared_ptr<std::vector< PointCloud<PointT>>>,  boost::shared_ptr<std::vector<cv::Mat>> >
      std::vector<std::pair<boost::shared_ptr<PointCloud<PointT>>, boost::shared_ptr<cv::Mat> > >  
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

 boost::circular_buffer<  std::vector<std::pair<boost::shared_ptr<PointCloud<PointT>>, boost::shared_ptr<cv::Mat> > >  >  buffer_;
    //boost::circular_buffer<std::pair<std::vector<typename PointCloud<PointT>::Ptr, std::vector<boost::shared_ptr<cv::Mat>  >  buffer_;

  //  boost::circular_buffer<  std::vector<typename PointCloud<PointT>::Ptr> >  bufferColors_;

};

//////////////////////////////////////////////////////////////////////////////////////////
template <typename PointT> bool 
PCDBuffer<PointT>::pushBack (
const   std::vector<std::pair<boost::shared_ptr<PointCloud<PointT>>, boost::shared_ptr<cv::Mat> > > 
  //const std::pair<  boost::shared_ptr<std::vector< PointCloud<PointT>>>,  boost::shared_ptr<std::vector<cv::Mat>> >
    & cloudColors
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
  std::vector<std::pair<boost::shared_ptr<PointCloud<PointT>>, boost::shared_ptr<cv::Mat> > >  
// std::pair<std::vector<typename PointCloud<PointT>::Ptr>, std::vector<boost::shared_ptr<cv::Mat>>>
//const  std::vector<typename PointCloud<PointT>::Ptr > 
PCDBuffer<PointT>::getFront ()
{
    std::vector<std::pair<boost::shared_ptr<PointCloud<PointT>>, boost::shared_ptr<cv::Mat> > >  cloudColors;
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

//////////////////////////////////////////////////////////////////////////////////////////
// Producer thread class
template <typename PointT>
class Producer
{
  private:

    std::mutex mtx;

    std::condition_variable cv;
    bool ready = false;
    ///////////////////////////////////////////////////////////////////////////////////////
   /* void 
    grabberCallBack (const typename PointCloud<PointT>::ConstPtr& cloud)
    {
      if (!buf_.pushBack (cloud))
      {
        {
          boost::mutex::scoped_lock io_lock (io_mutex);
          print_warn ("Warning! Buffer was full, overwriting data!\n");
        }
      }
      FPS_CALC ("cloud callback.", buf_);
    }*/
     //(new PointCloud<PointXYZRGB>());
    void go() {
     std::unique_lock<std::mutex> lck(mtx);
     ready = true;
     cv.notify_all();
     }





    ///////////////////////////////////////////////////////////////////////////////////////
    void threadPointCloud (K2G* kinect, boost::shared_ptr<pcl::PointCloud<pcl::PointXYZRGB>> &pointCloud , boost::shared_ptr<cv::Mat> &color, Eigen::Matrix4f transform, unsigned i) {
       //boost::mutex::scoped_lock buff_lock (io_mutex);
       std::this_thread::sleep_until(nextTime);

     /*   std::unique_lock<std::mutex> lck(mtx);
        while (!ready) cv.wait(lck);
        lck.unlock();*/


   /*   boost::asio::io_service io;
      boost::asio::deadline_timer timer(io);
      timer.expires_at(posixNexTime);*/
    //   std::cout << "time stamp before " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count()  << std::endl;
        startTime[i]=std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();

        // std::cout<<"step 1"<<std::endl;
     //   kinect->getColorwithCloud( pointCloud);

       // pointCloud=kinect->getCloud();
        pointCloud= thresholdDepth (pointCloud, 0.4, 2.3);
       // cv::Mat depth;
       // kinect->get(*color,depth);
       // color = kinect->getColor();
       //  kinect->getColor(*color);

      //  std::cout<<"step 2"<<std::endl;
        endTime[i]=std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();

      //  std::cout<<"time stamp: " <<std::chrono::high_resolution_clock::to_time_t(std::chrono::high_resolution_clock::now())<<std::endl;
     //   std::cout << "time stamp: " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count()  << std::endl;
      //pointCloud.reset (new pcl::PointCloud<pcl::PointXYZRGB> (*kinect->getCloud()));



   
       // pointCloud=downsample(pointCloud,0.004);
     
        pcl::transformPointCloud (*pointCloud, *pointCloud, transform);

        //std::cout<<"time stamp: " <<std::chrono::high_resolution_clock::to_time_t(std::chrono::high_resolution_clock::now())<<std:endl;

    }

    void 
    grabAndSend ()
    {


     

      Processor freenectprocessor = CUDA;
     // freenectprocessor = static_cast<Processor>(atoi(argv[1]));

      int kinect2_count=2;

       bool mirroring=true;
      std::vector<K2G *> kinects(kinect2_count);
      // std::vector<boost::shared_ptr<pcl::PointCloud<pcl::PointXYZRGB>>> clouds(kinect2_count);
    //  std::vector<boost::shared_ptr<cv::Mat> > colors(kinect2_count);
      clouds.resize(kinect2_count);
      colors.resize(kinect2_count);
      std::vector<boost::shared_ptr<pcl::PointCloud<pcl::PointXYZRGB>>> clouds_1(kinect2_count);

      boost::shared_ptr<pcl::PointCloud<pcl::PointXYZRGB>> allPointClouds(new pcl::PointCloud<pcl::PointXYZRGB>);
     // std::vector<pcl::PointCloud<pcl::PointXYZRGB>> clouds(kinect2_count);
    //  std::vector<cv::Mat> colors(kinect2_count);

     

     // boost::shared_ptr<pcl::PointCloud<pcl::PointXYZRGB>> allPointClouds(new pcl::PointCloud<pcl::PointXYZRGB>);

      //std::map<boost::shared_ptr<pcl::PointCloud<pcl::PointXYZRGB>>,boost::shared_ptr<pcl::PointCloud<pcl::PointXYZRGB>>> cloudPairs;
      //cloudPairs.insert ( std::pair<char,int>('a',100) );

     //pcl::PointCloud<pcl::PointXYZRGB>::Ptr result (new pcl::PointCloud<pcl::PointXYZRGB> ), source, target,targetToSrc;
      Eigen::Matrix4f GlobalTransform = Eigen::Matrix4f::Identity ();

      //char *arg[]={"011054343347","021401644747","003726334247","291296634347"};
   //// char *arg[]={ "011054343347", "291296634347","003726334247","021401644747"};
   //  char *arg[]= {"021401644747", "291296634347" , "011054343347", "003726334247"};
    //    char *arg[]= { "291296634347" , "011054343347", "003726334247"};
 // char *arg[]={ "021401644747","003726334247","011054343347"};
  // char *arg[]={ "021401644747", "011054343347", "003726334247"};

     char *arg[]={ (char *)"021401644747",  (char *)"291296634347"};
    //"003726334247"};
    //    char *arg[]={"021401644747","003726334247"};
   // char *arg[]={"021401644747"};
   // char *arg[]={"003726334247"};
  // char *arg[]={"011054343347"};
      K2G_generator kinect_generator(freenectprocessor, mirroring, arg);
      std::generate(kinects.begin(), kinects.end(), kinect_generator);

      for(size_t i = 0; i < kinect2_count; ++i)
        {
           //  if (i==1) continue;

          clouds[i] = kinects[i]->getCloud();
          colors[i].reset(new cv::Mat(  ));
         // colors[i] = std::make_shared<cv::Mat>(cv::Mat( cv::Mat::zeros(1080, 1920, CV_32F) )); 
        //  clouds[i]=thresholdDepth (clouds_1[i], 1.3, 2.1);
        //  clouds[i] = kinects[i]->getCloud();
         // colors[i]=kinects[i]->getColor();
        // if ( i>0) pcl::transformPointCloud (*clouds[i], *clouds[i], Transforms.at(i-1) );
          clouds[i]->sensor_orientation_.w() = 0.0;
          clouds[i]->sensor_orientation_.x() = 1.0;
          clouds[i]->sensor_orientation_.y() = 0.0;
          clouds[i]->sensor_orientation_.z() = 0.0; 
          //pcl::copyPointCloud(*clouds[i],*allPointClouds);
         // *allPointClouds+=*clouds[i];
          
        }
        
       //  std::cout<<"step 0"<<std::endl;


    /*  if (!buf_.pushBack (allPointClouds))
      {
        {
          boost::mutex::scoped_lock io_lock (io_mutex);
          print_warn ("Warning! Buffer was full, overwriting data!\n");
        }
      }*/

     // FPS_CALC ("cloud callback.", buf_);


    //  OpenNIGrabber* grabber = new OpenNIGrabber ();
     // grabber->getDevice ()->setDepthOutputFormat (depth_mode_);

   //   Grabber* interface = grabber;
   //   boost::function<void (const typename PointCloud<PointT>::ConstPtr&)> f = boost::bind (&Producer::grabberCallBack, this, _1);
   //   interface->registerCallback (f);
    //  interface->start ();
   
       std::vector<Eigen::Matrix4f> Transforms;

    
      if (is_file_exist("output.transforms")  ) {

       readTransformsBinary("output.transforms",Transforms);




      }  else return ;



      std::chrono::high_resolution_clock::time_point tnow, tpost,timePoint ;
      nextTime = std::chrono::high_resolution_clock::now();
      timePoint+=std::chrono::milliseconds(ttime);
      timePoint+=std::chrono::milliseconds(2000);
     // std::chrono::time_point<std::chrono::milliseconds> timePoint;
      nextTime+=std::chrono::milliseconds(3000);


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
      //  allPointClouds.reset(new pcl::PointCloud<pcl::PointXYZRGB>);  
        
         
        //  tnow = std::chrono::high_resolution_clock::now();

         
        /*boost::thread_group threads;

         nextTime+=std::chrono::milliseconds(70);


         for(size_t i = 0; i < kinect2_count; ++i) {
          
           colors[i].reset(new cv::Mat( cv::Mat::zeros(1080, 1920, CV_8UC4) )); 
           // colors[i].reset(new cv::Mat(  )); 
           
            clouds[i].reset(new pcl::PointCloud<pcl::PointXYZRGB>);
           //  if (i==1) continue;
             if (i>0)  threads.create_thread(boost::bind(&Producer::threadPointCloud,this, kinects[i], boost::ref(clouds[i]), boost::ref(colors[i]), Transforms.at(i-1),i));

             else threads.create_thread(boost::bind(&Producer::threadPointCloud,this, kinects[i], boost::ref(clouds[i]), boost::ref(colors[i]), Eigen::Matrix4f::Identity (),i));

         }

        //  threads.interrupt_all();
       //  std::this_thread::sleep_for(std::chrono::milliseconds(50));
        //  go();
          threads.join_all(); 


              //std::cout<<"start delta 1 2 3: " <<abs(start[0]- start[1])<<" "<< abs(start[0]- start[2]) <<" "<<abs(start[1]- start[2]) <<std::endl;
       //  sumStartDeltas=sumStartDeltas+ abs(start[0]- start[1])+ abs(start[0]- start[2]) + abs(start[1]- start[2]) ;
         sumstartTimeDeltas=sumstartTimeDeltas+ (1.f/3)*sqrt((startTime[0]- startTime[1])*(startTime[0]- startTime[1])+ (startTime[0]- startTime[2])*(startTime[0]- startTime[2]) + (startTime[1]- startTime[2])*(startTime[1]- startTime[2])) ;
        // sumstartTimeDeltas=sumstartTimeDeltas + (1.f/1)*sqrt((startTime[0]- startTime[1])*(startTime[0]- startTime[1])) ;

      //   std::cout<<"Counter: "<<say<<"\nAverage start Delta: "<<float(sumstartTimeDeltas)/say<<std::endl;
         //sumEndDeltas=sumEndDeltas + abs(end[0]- end[1])+ abs(end[0]- end[2]) + abs(end[1]- end[2]) ;
         sumendTimeDeltas=sumendTimeDeltas+ (1.f/3)*sqrt((endTime[0]- endTime[1])*(endTime[0]- endTime[1])+ (endTime[0]-endTime[2])*(endTime[0]- endTime[2]) + (endTime[1]- endTime[2])*(endTime[1]- endTime[2])) ;
       //  sumendTimeDeltas=sumendTimeDeltas + (1.f/1)*sqrt((endTime[0]- endTime[1])*(endTime[0]- endTime[1])) ;

       // std::cout<<"Average end Delta: "<<float(sumendTimeDeltas)/say<<std::endl;

*/

/*          for(size_t i = 0; i < kinect2_count; ++i) { 
                  //  if (i==1) continue;
                *allPointClouds+=*clouds[i];


        }
*/




         boost::asio::io_service io;
       //  boost::asio::deadline_timer t1(io, boost::posix_time::milliseconds(15));
      //   boost::asio::deadline_timer t2(io, boost::posix_time::milliseconds(15));
       //  boost::asio::deadline_timer t3(io, boost::posix_time::milliseconds(30));
         
         boost::asio::deadline_timer t1(io);
         boost::asio::deadline_timer t2(io);
        // boost::asio::deadline_timer t3(io);
         posixNexTime+=time_duration(milliseconds(80));
         t1.expires_at(posixNexTime);
         t2.expires_at(posixNexTime);
       //  t3.expires_at(posixNexTime);

       //  t1.async_wait(boost::bind(&Producer::threadPointCloud,this, kinects[0], boost::ref(clouds[0]), Eigen::Matrix4f::Identity ()));         
         t1.async_wait([&] (const boost::system::error_code&){
           
            startTime[0]=std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
    //         std::cout << "time stamp before 1: " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count()  << std::endl;
             //clouds[0] = kinects[0]->updateCloud(clouds[0]);
           //  clouds[0] = kinects[0]->getCloud();
             // boost::shared_ptr< cv::Mat> color;
             colors[0].reset(new cv::Mat(  ));   ;//clouds[0].reset();
            //   colors[0]= kinects[0]->getColorwithCloud( clouds[0]);

             //  clouds[0].reset(new pcl::PointCloud<pcl::PointXYZRGB>);
              //kinects[0]->getColorwithCloud(&((*colors.get()).at(0)),&((*clouds.get()).at(0))) ;
               kinects[0]->getColorwithCloud(colors[0], clouds[0]);

            //  clouds[0]= thresholdDepth (clouds[0], 0.4, 2.3);

                //  clouds[0]  = thresholdDepth (clouds[0], 0.4, 2.3);
               //   pcl::transformPointCloud (*clouds[0], *clouds[0], Eigen::Matrix4f::Identity () );

              //  colors[0].reset(new cv::Mat(*kinects[0]->getColorwithCloud( clouds[0])));

            //colors[0]=boost::make_shared< cv::Mat> (*kinects[0]->getColorwithCloud( clouds[0]));
             //  clouds[0]  = thresholdDepth (clouds[0], 0.4, 2.3);
            // colors[0]= kinects[0]->getColor();
           //  kinects[0]->getColor(colors[0]);
             endTime[0]=std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();

           // clouds[0]=kinects[0]->getCloud();
             //std::cout << "time stamp 1: " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count()  << std::endl;
  
        // clouds[0] = kinects[0]->updateCloud(clouds[0]);
            //pcl::transformPointCloud (*clouds[0], *clouds[0], Eigen::Matrix4f::Identity () );
          });

        
         //t2.async_wait(boost::bind(&Producer::threadPointCloud,this, kinects[1], boost::ref(clouds[1]), Transforms.at(0) ));   

          t2.async_wait([&](const boost::system::error_code&){
             
             startTime[1]=std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();

  //           std::cout << "time stamp before 2: " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count()  << std::endl;
            // clouds[1]=kinects[1]->getCloud();
            // clouds[1] = kinects[1]->updateCloud(clouds[1]);
             // clouds[1] = kinects[1]->getCloud();
               // colors[1].reset(new cv::Mat(*kinects[1]->getColorwithCloud( clouds[1])));boost::shared_ptr< cv::Mat> color;
             //   colors[1].reset();//clouds[1].reset();
             colors[1].reset(new cv::Mat(  ));
            // colors[1]=kinects[1]->getColorwithCloud( clouds[1]);
             //  clouds[1].reset(new pcl::PointCloud<pcl::PointXYZRGB>);
             kinects[1]->getColorwithCloud(colors[1], clouds[1]);
            //kinects[1]->getColorwithCloud(colors[1], clouds[1]);
             //  colors[1].reset(new cv::Mat(*kinects[1]->getColorwithCloud( clouds[1])));
//std::cout << "time stamp before 22: " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count()  << std::endl;
         
            //  clouds[1]  = thresholdDepth (clouds[1], 0.4, 2.3);
              // pcl::transformPointCloud (*clouds[1], *clouds[1], Transforms.at(0) );
           // colors[1]=boost::make_shared< cv::Mat> (*kinects[1]->getColorwithCloud( clouds[1]));
             //colors[1]= kinects[1]->getColor();
            // kinects[1]->getColor(colors[1]);
           //  std::cout << "time stamp 2: " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count()  << std::endl;
             endTime[1]=std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();

             //  clouds[1]=kinects[1]->getCloud();
             //pcl::transformPointCloud (*clouds[1], *clouds[1], Transforms.at(0) );

          });


         //t2.async_wait(boost::bind(&Producer::threadPointCloud,this, kinects[1], boost::ref(clouds[1]), Transforms.at(0) ));   

//           t3.async_wait([&](const boost::system::error_code&){
            
//              startTime[2]=std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();

//       //       std::cout << "time stamp before 3: " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count()  << std::endl;
             
// //             clouds[2] = kinects[2]->updateCloud(clouds[2]);
//             // clouds[2] = kinects[2]->getCloud();
//            //  boost::shared_ptr< cv::Mat> color;
//               //  colors[2].reset();//clouds[2].reset();
//              //   colors[2]= kinects[2]->getColorwithCloud( clouds[2]);
//                //   clouds[2].reset(new pcl::PointCloud<pcl::PointXYZRGB>);
//                  colors[2].reset(new cv::Mat(  ));

//                  kinects[2]->getColorwithCloud(colors[2], clouds[2]);
//                 //colors[2].reset(new cv::Mat(*kinects[2]->getColorwithCloud( clouds[2])));
//                 // colors[2]=
//             //  colors[2]=boost::make_shared< cv::Mat> (*kinects[2]->getColorwithCloud( clouds[2]));
//             //   clouds[2]  = thresholdDepth (clouds[2], 0.4, 2.3);
//             //   pcl::transformPointCloud (*clouds[2], *clouds[2], Transforms.at(1) );
//            // // colors[2]= kinects[2]->getColor();
//             // kinects[2]->getColor(colors[2]);
             
//              endTime[2]=std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();

//              //std::cout << "time stamp 3: " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count()  << std::endl;

//              //  clouds[1]=kinects[1]->getCloud();
//             //  pcl::transformPointCloud (*clouds[2], *clouds[2], Transforms.at(1) );
//           }); 

         std::thread ta([&](){io.run();});
         std::thread tb([&](){io.run();});
        // std::thread tc([&](){io.run();});



         ta.join();tb.join();//tc.join();

     

         //std::cout<<"start delta 1 2 3: " <<abs(start[0]- start[1])<<" "<< abs(start[0]- start[2]) <<" "<<abs(start[1]- start[2]) <<std::endl;
         sumstartTimeDeltas=sumstartTimeDeltas + (1.f/1)*sqrt((startTime[0]- startTime[1])*(startTime[0]- startTime[1])+ (startTime[0]- startTime[2])*(startTime[0]- startTime[2]) + (startTime[1]- startTime[2])*(startTime[1]- startTime[2])) ;
        // sumstartTimeDeltas=sumstartTimeDeltas + (1.f/1)*sqrt((startTime[0]- startTime[1])*(startTime[0]- startTime[1])) ;

         //std::cout<<"Average start Delta: "<<float(sumstartTimeDeltas)/say<<std::endl;
         sumendTimeDeltas=sumendTimeDeltas + (1.f/1)*sqrt((endTime[0]- endTime[1])*(endTime[0]- endTime[1])+ (endTime[0]-endTime[2])*(endTime[0]- endTime[2]) + (endTime[1]- endTime[2])*(endTime[1]- endTime[2])) ;
//         sumendTimeDeltas=sumendTimeDeltas + (1.f/1)*sqrt((endTime[0]- endTime[1])*(endTime[0]- endTime[1])) ;

       // std::cout<<"Average start Delta: "<<float(sumendTimeDeltas)/say<<std::endl;

     /*  for(size_t i = 0; i < kinect2_count; ++i) {
           
           *allPointClouds+=*clouds[i];
       }
       */

       /* timePoint+=std::chrono::milliseconds(70);
       // std:this_thread::sleep_until(timePoint);

        for(size_t i = 0; i < kinect2_count; ++i) { 
        std::cout << "<<< time stamp  before " << i<<" "<<std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count()  << std::endl;
 
        clouds[i] = kinects[i]->updateCloud(clouds[i]); 
        std::cout << "<<< time stamp  " <<  i<<" "<<std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count()  << std::endl;

       // clouds[i]= thresholdDepth (clouds_1[i], 0.4, 2.5);
       // clouds[i]=removeOutliers(clouds[i],0.05,100);
      //  clouds[i] = kinects[i]->updateCloud(clouds[i]); 

          if ( i>0) pcl::transformPointCloud (*clouds[i], *clouds[i], Transforms.at(i-1) );

         // *allPointClouds+=*clouds[i];

          }

     for(size_t i = 0; i < kinect2_count; ++i) *allPointClouds+=*clouds[i];*/


     // tpost = std::chrono::high_resolution_clock::now();
    //  std::cout << "delta " << std::chrono::duration_cast<std::chrono::duration<float>>(tpost - tnow).count() * 1000 << std::endl;

    //  k2g.get(color, depth, cloud);

     
/*
         std::vector<boost::shared_ptr<pcl::PointCloud<pcl::PointXYZRGB>>> newclouds;
         newclouds.reserve(clouds.size());

          std::vector<boost::shared_ptr<cv::Mat>> newcolors;
          newcolors.reserve(colors.size());


          for (const auto& b : colors)
             newcolors.push_back(boost::make_shared<cv::Mat>(*b));
          for (const auto& e : clouds)
          newclouds.push_back(boost::make_shared<pcl::PointCloud<pcl::PointXYZRGB>>(*e));*/

      //   std::pair< const boost::shared_ptr<std::vector< PointCloud<PointT>>>, const boost::shared_ptr<std::vector<cv::Mat>> > 

// std::vector<boost::shared_ptr<pcl::PointCloud<pcl::PointXYZRGB>>>
     //   std::cout<<"step 0"<<std::endl;
/*
          boost::shared_ptr<std::vector<PointCloud<PointT>>> newclouds(new std::vector<PointCloud<PointT>>());
         //         std::cout<<"step 1"<<std::endl;

          //std::make_shared <std::vector<PointCloud<PointT>>>(clouds.size()));
          newclouds->reserve(clouds.size());
        //  newclouds.reset(new std::vector<PointCloud<PointT>>());
          for (const auto& e : clouds)  newclouds->push_back(*e);
     //   std::cout<<"step 1"<<std::endl;

          boost::shared_ptr<std::vector<cv::Mat>> newcolors(new std::vector<cv::Mat>());
          newcolors->reserve(colors.size());
          for (const auto& e : colors)
            newcolors->push_back(e);
        // std::cout<<"step 2"<<std::endl;*/


          std::vector<std::pair<boost::shared_ptr<PointCloud<PointT>>, boost::shared_ptr<cv::Mat> > > colorClouds;

          for(int i=0;i<clouds.size(); i++){

           // colorClouds.push_back(std::make_pair(clouds[i],colors[i]));
          colorClouds.push_back(std::make_pair(boost::make_shared<PointCloud<PointT>>(*clouds[i]),boost::make_shared<cv::Mat>( *colors[i])));
          }

          if (!buf_.pushBack ( colorClouds) )
            {
              {
                boost::mutex::scoped_lock io_lock (io_mutex);
                print_warn ("Warning! Buffer was full, overwriting data!\n");
              }
            }
        FPS_CALC ("cloud callback.", buf_);

     
      
      // Showing only color since depth is float and needs conversion
     // cv::imshow("color", color);
     // int c = cv::waitKey(1);
        if (is_done)
          break;



      //  boost::this_thread::sleep (boost::posix_time::milliseconds (60));
      }
        std::cout<<"Average start Delta: "<<float(sumstartTimeDeltas)/say<<std::endl;
        std::cout<<"Average end Delta: "<<float(sumendTimeDeltas)/say<<std::endl;

   
    for(auto & k : kinects)
    k->shutDown();

    }

  public:
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


   std::vector< boost::shared_ptr<pcl::PointCloud<pcl::PointXYZRGB>>> clouds;
   std::vector<boost::shared_ptr<cv::Mat>> colors;

};

//////////////////////////////////////////////////////////////////////////////////////////
// Consumer thread class
template <typename PointT>
class Consumer
{
  private:
    ///////////////////////////////////////////////////////////////////////////////////////
    void 
    writeToDisk (  std::vector<std::pair<boost::shared_ptr<PointCloud<PointT>>, boost::shared_ptr<cv::Mat> > >   &&colorClouds
      /*const  std::vector< typename PointCloud<PointT>::Ptr >  &clouds*/ )
    {
      std::string time = boost::posix_time::to_iso_string (boost::posix_time::microsec_clock::local_time ());
      allPointClouds.reset(new pcl::PointCloud<pcl::PointXYZRGB>);
      for (int i=0;i<colorClouds.size();++i) {
     // stringstream ss;
      //ss << "frame-" << time <<"_"<<i<< ".ply";
    //   colorClouds.first[i] = thresholdDepth (colorClouds.first[i], 0.4, 2.3);
    //   std::cout<<Transforms.at(1)<<std::endl;
   
       colorClouds[i].first= thresholdDepth (colorClouds[i].first, 0.4, 2.);
      // colorClouds[i].first=removeOutliers(colorClouds[i].first, 0.03, 50);
    //    colorClouds[i].first=downsample(colorClouds[i].first, 0.005); 
        if (i!=0) pcl::transformPointCloud (*colorClouds[i].first, *colorClouds[i].first, Transforms.at(i-1) );

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


     void 
    writePolyMeshToDisk (const  pcl::PolygonMesh & mesh)
    {
      stringstream ss;
      std::string time = boost::posix_time::to_iso_string (boost::posix_time::microsec_clock::local_time ());
      ss << "frame-" << time << ".vtk";
      pcl::io::saveVTKFile (ss.str(), mesh);
     // FPS_CALC ("cloud write.", buf_);
    }




    ///////////////////////////////////////////////////////////////////////////////////////
    // Consumer thread function
    void 
    receiveAndProcess ()
    {


      if (is_file_exist("output.transforms")  ) {

       readTransformsBinary("output.transforms",Transforms);




      }  else return ;



      while (true)
      {
        if (is_done)   break;
        writeToDisk (buf_.getFront ());
      }

      {
        boost::mutex::scoped_lock io_lock (io_mutex);
        print_info ("Writing remaing %ld clouds in the buffer to disk...\n", buf_.getSize ());
      }
      while (!buf_.isEmpty ()) {


        writeToDisk (buf_.getFront ());
      }
    }

  public:
    Consumer (PCDBuffer<PointT> &buf)
      : buf_ (buf)
    {
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
    std::vector<Eigen::Matrix4f> Transforms;

    boost::shared_ptr<pcl::PointCloud<pcl::PointXYZRGB>> allPointClouds;//(new pcl::PointCloud<pcl::PointXYZRGB>);


};

//////////////////////////////////////////////////////////////////////////////////////////
void 
ctrlC (int)
{
  boost::mutex::scoped_lock io_lock (io_mutex);
  print_info ("\nCtrl-C detected, exit condition set to true.\n");
  is_done = true;
}

//////////////////////////////////////////////////////////////////////////////////////////
void
printHelp (int default_buff_size, int, char **argv)
{
  using pcl::console::print_error;
  using pcl::console::print_info;

  print_error ("Syntax is: %s ((<device_id> | <path-to-oni-file>) [-xyz] [-shift] [-buf X]  | -l [<device_id>] | -h | --help)]\n", argv [0]);
  print_info ("%s -h | --help : shows this help\n", argv [0]);
  print_info ("%s -xyz : save only XYZ data, even if the device is RGB capable\n", argv [0]);
  print_info ("%s -shift : use OpenNI shift values rather than 12-bit depth\n", argv [0]);
  print_info ("%s -buf X ; use a buffer size of X frames (default: ", argv [0]);
  print_value ("%d", default_buff_size); print_info (")\n");
  print_info ("%s -l : list all available devices\n", argv [0]);
  print_info ("%s -l <device-id> :list all available modes for specified device\n", argv [0]);
  print_info ("\t\t<device_id> may be \"#1\", \"#2\", ... for the first, second etc device in the list\n");
#ifndef _WIN32
  print_info ("\t\t                   bus@address for the device connected to a specific usb-bus / address combination\n");
  print_info ("\t\t                   <serial-number>\n");
#endif
  print_info ("\n\nexamples:\n");
  print_info ("%s \"#1\"\n", argv [0]);
  print_info ("\t\t uses the first device.\n");
  print_info ("%s  \"./temp/test.oni\"\n", argv [0]);
  print_info ("\t\t uses the oni-player device to play back oni file given by path.\n");
  print_info ("%s -l\n", argv [0]);
  print_info ("\t\t list all available devices.\n");
  print_info ("%s -l \"#2\"\n", argv [0]);
  print_info ("\t\t list all available modes for the second device.\n");
  #ifndef _WIN32
  print_info ("%s A00361800903049A\n", argv [0]);
  print_info ("\t\t uses the device with the serial number \'A00361800903049A\'.\n");
  print_info ("%s 1@16\n", argv [0]);
  print_info ("\t\t uses the device on address 16 at USB bus 1.\n");
  #endif
}

//////////////////////////////////////////////////////////////////////////////////////////


int
main (int argc, char** argv)
{
//    std::cout << "<<< time stamp 2 : " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count()  << std::endl;

// if (parse_argument (argc, argv, "-time", ttime) != -1)
 //   print_highlight ("time is %d .\n", ttime);
 //  ttime=stol(argv[1]);
 //cout<<"argv[1] "<<ttime<<endl;
   

  //   std::cout << "<<< time stamp 2 "<< i <<" : " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count()  << std::endl;

  print_highlight ("PCL OpenNI Recorder for saving buffered PCD (binary compressed to disk). See %s -h for options.\n", argv[0]);

  std::string device_id ("");
  int buff_size = BUFFER_SIZE;



  if (parse_argument (argc, argv, "-buf", buff_size) != -1)
    print_highlight ("Setting buffer size to %d frames.\n", buff_size);
  else
    print_highlight ("Using default buffer size of %d frames.\n", buff_size);

  print_highlight ("Starting the producer and consumer threads... Press Ctrl+C to end\n");

    print_highlight ("PointXYZRGBA enabled.\n");




    
   // if (is_file_exist("output.globalcalibtrans")){


   //    readTransformsBinary("output.globalcalibtrans",GlobalCalibTransforms);




   // }  else return 0;



   //  using boost::property_tree::ptree;
   //  ptree pt;

   //  read_xml("cameras.xml", pt, boost::property_tree::xml_parser::trim_whitespace );

   

   //  //  std::cout<<"Transform "<<i<< "\n"<<Transforms.at(i)<<std::endl;

   //    for( auto &v : pt.get_child("document.chunk.cameras"))
   //      {

   //         for (int i=0;i<GlobalCalibTransforms.size();i++)  {
            


   //        // std::string name;
   //       //  ptree sub_pt;
   //        // std::tie(name, sub_pt) = v;
   //         std::string temp = v.second.get<std::string>("<xmlattr>.id");
   //         if (std::stoi(temp)!=i) continue; 

   //         std::cout << v.second.get<std::string>("transform")<< "\n temp " <<temp<< std::endl;
   //         std::stringstream iss;
   //         iss<<GlobalCalibTransforms.at(i);

   //         temp=iss.str();
   //         std::replace(temp.begin(), temp.end(), '\n',' ');
   //        // temp.erase(std::remove(temp.begin(), temp.end(), '\n'), temp.end());
   //         v.second.put<std::string>("transform",temp);

   //       }

   //        //  std::cout <<v.first.data() <<" "<< v.second.data() << std::endl;
   //      // }
   //     }  
   //   std::locale newlocale1("en_GB.UTF-8");
   //   auto settings = boost::property_tree::xml_writer_make_settings<std::string>('\t', 1);
   //  //bpt::xml_parser::xml_writer_settings<char> xmlstyle(' ',4);

   //  write_xml("camerastest.xml", pt,  newlocale1, settings);



    PCDBuffer<PointXYZRGB> buf;
    buf.setCapacity (buff_size);
    Producer<PointXYZRGB> producer (buf);
    boost::this_thread::sleep (boost::posix_time::seconds (2));
    Consumer<PointXYZRGB> consumer (buf);

    signal (SIGINT, ctrlC);
    producer.stop ();
    consumer.stop ();

   // producer.stop ();

  
  /*
  {
    print_highlight ("PointXYZ enabled.\n");
    PCDBuffer<PointXYZ> buf;
    buf.setCapacity (buff_size);
    Producer<PointXYZ> producer (buf, depth_mode);
    boost::this_thread::sleep (boost::posix_time::seconds (2));
    Consumer<PointXYZ> consumer (buf);

    signal (SIGINT, ctrlC);
    producer.stop ();
    consumer.stop ();
  }*/
  return (0);
}
