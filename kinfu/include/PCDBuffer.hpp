#ifndef PCDBUFFER_HPP_INCLUDED
#define PCDBUFFER_HPP_INCLUDED
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <boost/thread/condition.hpp>
#include <boost/circular_buffer.hpp>
#include <opencv2/opencv.hpp>

#include <pcl/common/time.h>
#include <pcl/console/parse.h>

using namespace std;
using namespace pcl;
using namespace pcl::console;


bool is_done=false;

boost::mutex io_mutex;
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

    boost::circular_buffer<std::vector<std::pair<boost::shared_ptr<PointCloud<PointT>>, boost::shared_ptr<cv::Mat> > >  >  buffer_;
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





#endif // PCDBUFFER_HPP_INCLUDED
