#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <visualization_msgs/MarkerArray.h>

#include <iostream>
#include <boost/thread/thread.hpp>
#include <Eigen/Dense>

#include <eigen_conversions/eigen_msg.h>
#include <pcl_conversions/pcl_conversions.h>
#include <geometry_msgs/PoseStamped.h>

#include <pcl/ModelCoefficients.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl/conversions.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/crop_box.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <pcl/common/geometry.h>
#include <pcl/common/time.h>
#include <pcl/common/transforms.h>
#include <pcl/features/normal_3d.h>

#include <pcl_ros/transforms.h>
#include <pcl/registration/icp.h>

#include <pcl/filters/radius_outlier_removal.h>

using namespace std;

visualization_msgs::MarkerArray line_list;
visualization_msgs::Marker lines;
visualization_msgs::Marker box_points;
visualization_msgs::Marker min_point;

ros::Publisher pose_pub;
ros::Publisher bbox_pub;
ros::Publisher eastbox_pub;
ros::Publisher westbox_pub;
ros::Publisher cloud_pub;
ros::Publisher midbox_pub;
ros::Publisher mideastbox_pub;
ros::Publisher midwestbox_pub;
ros::Publisher minpoint_pub;
ros::Publisher cropbox_pub;
ros::Publisher cropcloud_pub;

float object_size_threshold_min_;
float object_size_threshold_max_;
int plane_normal_threshold_;
bool use_plane_normal_threshold_;
float roi_min_x_, roi_max_x_;
float roi_min_y_, roi_max_y_;
float roi_min_z_, roi_max_z_;
float euclidean_cluster_tolerance_;

typedef pcl::PointXYZ PCLPoint; //A point structure representing Euclidean xyz coordinates.
typedef pcl::PointCloud<PCLPoint> PointCloud;  //typedef pcl::PointCloud <PointT >::ConstPtr
typedef PointCloud::Ptr PointCloudPtr;

 //SOR filter  It computes first the average distance of each point to its neighbors (considering k nearest neighbors for each - k is the first parameter). Then it rejects the points that are farther than the average distance plus a number of times the standard deviation (second parameter).
void SORfilter(const PointCloudPtr input, PointCloudPtr output) {
  pcl::StatisticalOutlierRemoval<PCLPoint> sor;
  sor.setInputCloud (input);
  sor.setMeanK (50);
  sor.setStddevMulThresh (1);
  sor.filter (*output);

  return;
}

void RORfilter(const PointCloudPtr input, PointCloudPtr output_ror) {
   pcl::RadiusOutlierRemoval<pcl::PointXYZ> ror;
   // build the filter
   ror.setInputCloud(input);
   ror.setRadiusSearch(0.8);
   ror.setMinNeighborsInRadius (8);
   // apply filter
   ror.filter (*output_ror);
   return;

 }


void objectSegmentation(const PointCloudPtr input_cloud, PointCloudPtr& object_cloud) {

  int scene_cloud_point_thresh_ = 200;
  int object_cloud_point_thresh_ = 50;

    PointCloudPtr cloud_plane(new PointCloud);
      PointCloudPtr cloud_f(new PointCloud);

  PointCloudPtr cloud_filtered (new PointCloud);
  pcl::copyPointCloud(*input_cloud, *cloud_filtered);
  std::cerr << "scene_cloud INSIDE segmentation: " << cloud_filtered->points.size()<<std::endl;

  pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients); //model coefficients are stored internally.
  pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
  pcl::SACSegmentation<PCLPoint> seg;   // represents the Nodelet segmentation class for Sample Consensus methods and models, in the sense that it just creates a Nodelet wrapper for generic-purpose SAC-based segmentation
  seg.setOptimizeCoefficients (true);
  seg.setModelType (pcl::SACMODEL_PLANE); //http://docs.pointclouds.org/trunk/model__types_8h_source.html
  seg.setMethodType (pcl::SAC_RANSAC);
  seg.setDistanceThreshold (0.01);


  PointCloudPtr object_candidate (new PointCloud);
  while(cloud_filtered->points.size() > scene_cloud_point_thresh_) {
    seg.setInputCloud (cloud_filtered);
    seg.segment (*inliers, *coefficients);  //Base method for segmentation of a model in a PointCloud
    if (inliers->indices.size () == 0)
       std::cerr << "Could not estimate a planar model for the given dataset." <<std::endl;
      break;
  //................................


  // Extract the planar inliers from the input cloud
   pcl::ExtractIndices<pcl::PointXYZ> extract;
   extract.setInputCloud (cloud_filtered);
   extract.setIndices (inliers);
   extract.setNegative (false);

   // Get the points associated with the planar surface
   extract.filter (*cloud_plane);
   std::cout << "PointCloud representing the planar component: " << cloud_plane->points.size () << " data points." << std::endl;

   // Remove the planar inliers, extract the rest
   extract.setNegative (true);
   extract.filter (*cloud_f);
   *cloud_filtered = *cloud_f;
  }


  // Creating the KdTree object for the search method of the extraction
  pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);
  tree->setInputCloud (cloud_filtered);

  std::vector<pcl::PointIndices> cluster_indices;
  pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
  ec.setClusterTolerance (0.02); // 2cm
  // ec.setClusterTolerance (euclidean_cluster_tolerance_);
  ec.setMinClusterSize (100);
  // ec.setMinClusterSize (object_cloud_point_thresh_);
  ec.setMaxClusterSize (25000);
  // ec.setMaxClusterSize (50000);
  ec.setSearchMethod (tree);
  ec.setInputCloud (cloud_filtered);
  ec.extract (cluster_indices);

  std::cerr << "cluster_inds_vector: " <<  cluster_indices.size()<<std::endl;
  //................................
    PointCloudPtr cloud_cluster (new PointCloud);
  for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin (); it != cluster_indices.end (); ++it)
    {

    for (std::vector<int>::const_iterator pit = it->indices.begin (); pit != it->indices.end (); ++pit)
      cloud_cluster->points.push_back (cloud_filtered->points[*pit]); //*
    cloud_cluster->width = cloud_cluster->points.size ();
    cloud_cluster->height = 1;
    cloud_cluster->is_dense = true;

    std::cout << "PointCloud representing the Cluster: " << cloud_cluster->points.size () << " data points." << std::endl;

      }
            object_cloud = cloud_cluster;

  return;


}

void bboxPublisher(PointCloudPtr object_cloud, std_msgs::Header header) {

  visualization_msgs::MarkerArray line_list;
  visualization_msgs::Marker lines;
  // visualization_msgs::MarkerArray box_points;

  lines.ns = "bounding_box";
  lines.header = header;
  lines.type = visualization_msgs::Marker::LINE_LIST;
  lines.action = visualization_msgs::Marker::ADD;
  lines.id = 1;
  lines.scale.x = 0.01;
  lines.scale.y = 0.01;
  lines.color.a = 1;
  lines.color.r = 1.0f;
  lines.color.g = 1.0f;
  lines.color.b = 1.0f;
  lines.lifetime = ros::Duration(0.01);
  lines.points.clear();

  PCLPoint minp, maxp;
  pcl::getMinMax3D(*object_cloud, minp, maxp);

  geometry_msgs::Point p1,p2,p3,p4,p5,p6,p7,p8;
  p1.x = minp.x; p1.y = minp.y; p1.z = minp.z;
  p2.x = maxp.x; p2.y = minp.y; p2.z = minp.z;
  p3.x = minp.x; p3.y = minp.y; p3.z = maxp.z;
  p4.x = maxp.x; p4.y = minp.y; p4.z = maxp.z;
  p5.x = minp.x; p5.y = maxp.y; p5.z = minp.z;
  p6.x = maxp.x; p6.y = maxp.y; p6.z = minp.z;
  p7.x = minp.x; p7.y = maxp.y; p7.z = maxp.z;
  p8.x = maxp.x; p8.y = maxp.y; p8.z = maxp.z;

  // box_points.points.push_back(p1,p2,p3,p4,p5,p6,p7,p8);
  // boxpoints_pub.publish(box_points);

  lines.points.push_back(p1); lines.points.push_back(p2);
  lines.points.push_back(p1); lines.points.push_back(p3);
  lines.points.push_back(p2); lines.points.push_back(p4);
  lines.points.push_back(p3); lines.points.push_back(p4);
  lines.points.push_back(p5); lines.points.push_back(p6);
  lines.points.push_back(p5); lines.points.push_back(p7);
  lines.points.push_back(p6); lines.points.push_back(p8);
  lines.points.push_back(p7); lines.points.push_back(p8);
  lines.points.push_back(p1); lines.points.push_back(p5);
  lines.points.push_back(p2); lines.points.push_back(p6);
  lines.points.push_back(p3); lines.points.push_back(p7);
  lines.points.push_back(p4); lines.points.push_back(p8);
  line_list.markers.push_back(lines);

  bbox_pub.publish(line_list);

  //ROS_INFO_STREAM("FULLbox POINTS: " << line_list);
  return;
}

void cropboxPublisher(PointCloudPtr object_cloud, std_msgs::Header header) {

  visualization_msgs::MarkerArray line_list;
  visualization_msgs::Marker lines;
  //
  lines.header.frame_id = "/kinect2_rgb_optical_frame";
  lines.header.stamp = ros::Time::now();

  lines.ns = "cropbox_box";
  lines.header = header;
  lines.type = visualization_msgs::Marker::LINE_LIST;
  lines.action = visualization_msgs::Marker::ADD;
  lines.id = 1;
  lines.scale.x = 0.01;
  lines.scale.y = 0.01;
  lines.color.a = 1;
  lines.color.r = 1.0f;
  lines.color.g = 0.0f;
  lines.color.b = 0.0f;

  // lines.pose.position.x =  0.105;
  // lines.pose.position.y = 0.33;
  // lines.pose.position.z = 1.57;

  // lines.pose.orientation.x = 0.0436194;
  // lines.pose.orientation.y = 0;
  // lines.pose.orientation.z = 0;
  // lines.pose.orientation.w = 0.9990482;

  lines.pose.orientation.x = 0;
  lines.pose.orientation.y = 0;
  lines.pose.orientation.z = 0;
  lines.pose.orientation.w = 1;
  lines.lifetime = ros::Duration(0.01);
  lines.points.clear();



  geometry_msgs::Point p1,p2,p3,p4,p5,p6,p7,p8;
  p1.x = roi_min_x_; p1.y = roi_min_y_; p1.z = roi_min_z_;
  p2.x = roi_max_x_; p2.y = roi_min_y_; p2.z = roi_min_z_;
  p3.x = roi_min_x_; p3.y = roi_min_y_; p3.z = roi_max_z_;
  p4.x = roi_max_x_; p4.y = roi_min_y_; p4.z = roi_max_z_;
  p5.x = roi_min_x_; p5.y = roi_max_y_; p5.z = roi_min_z_;
  p6.x = roi_max_x_; p6.y = roi_max_y_; p6.z = roi_min_z_;
  // p7.x = 1.0; p7.y = roi_max_y_; p7.z = roi_max_z_;
  p7.x = roi_min_x_; p7.y = roi_max_y_; p7.z = roi_max_z_;
  p8.x = roi_max_x_; p8.y = roi_max_y_; p8.z = roi_max_z_;

  lines.points.push_back(p1); lines.points.push_back(p2);
  lines.points.push_back(p1); lines.points.push_back(p3);
  lines.points.push_back(p2); lines.points.push_back(p4);
  lines.points.push_back(p3); lines.points.push_back(p4);
  lines.points.push_back(p5); lines.points.push_back(p6);
  lines.points.push_back(p5); lines.points.push_back(p7);
  lines.points.push_back(p6); lines.points.push_back(p8);
  lines.points.push_back(p7); lines.points.push_back(p8);
  lines.points.push_back(p1); lines.points.push_back(p5);
  lines.points.push_back(p2); lines.points.push_back(p6);
  lines.points.push_back(p3); lines.points.push_back(p7);
  lines.points.push_back(p4); lines.points.push_back(p8);
  line_list.markers.push_back(lines);

  cropbox_pub.publish(line_list);

  //ROS_INFO_STREAM("FULLbox POINTS: " << line_list);
  return;
}



void midboxPublisher(PointCloudPtr object_cloud, std_msgs::Header header){
  visualization_msgs::Marker box_points;
  box_points.ns = "mid_box";
  box_points.action=visualization_msgs::Marker::ADD;
  box_points.type=visualization_msgs::Marker::POINTS;
  box_points.lifetime= ros::Duration(0.01);
  box_points.id=2;
  box_points.scale.x = 0.2;
  box_points.scale.y = 0.2;
  PCLPoint minp, maxp;
  pcl::getMinMax3D(*object_cloud, minp, maxp);
  geometry_msgs::Point p1;
  p1.x = (minp.x+maxp.x)/2; p1.y = (minp.y+maxp.y)/2; p1.z = (minp.z+maxp.z)/2;
  box_points.points.push_back(p1);
  midbox_pub.publish(box_points);
  // ROS_INFO_STREAM("mid_box Publish: " << box_points.points[0]);
  return;
}

void minpointPublisher(PointCloudPtr object_cloud, std_msgs::Header header){
  visualization_msgs::Marker min_point;
  min_point.ns = "minpoint_marker";
  min_point.action=visualization_msgs::Marker::ADD;
  min_point.type=visualization_msgs::Marker::POINTS;
  min_point.lifetime= ros::Duration(0.01);
  min_point.id=1;
  min_point.scale.x = 0.2;
  min_point.scale.y = 0.2;
  PCLPoint minp , maxp ;
  pcl::getMinMax3D(*object_cloud, minp, maxp);
  geometry_msgs::Point p1;
  p1.x = minp.x;
  p1.y = minp.y;
  p1.z = minp.z;
  min_point.points.push_back(p1);
  minpoint_pub.publish(min_point);
  // ROS_INFO_STREAM("min_point Publish: " << min_point);
  return;
}

void posePublisher(Eigen::Vector3f p, Eigen::Quaternionf q, std_msgs::Header header){

  geometry_msgs::Point pos;
  tf::pointEigenToMsg(p.cast<double>(), pos);

  geometry_msgs::Quaternion quat;
  tf::quaternionEigenToMsg(q.cast<double>(), quat);

  geometry_msgs::PoseStamped pos_msg;
  pos_msg.pose.position = pos;
  pos_msg.pose.orientation = quat;
  pos_msg.header = header;

  pose_pub.publish(pos_msg);


  return;
}


void halfeastPublisher(PointCloudPtr object_cloud, std_msgs::Header header) {

  visualization_msgs::MarkerArray line_list;
  visualization_msgs::Marker lines;

  lines.ns = "bounding_westbox";
  lines.header = header;
  lines.type = visualization_msgs::Marker::LINE_LIST;
  lines.action = visualization_msgs::Marker::ADD;
  lines.id = 1;
  lines.scale.x = 0.01;
  lines.scale.y = 0.01;
  lines.color.a = 1;
  lines.color.r = 0.0f;
  lines.color.g = 1.0f;
  lines.color.b = 0.0f;
  lines.lifetime = ros::Duration(0.01);
  lines.points.clear();



  PCLPoint minpH, maxpH;
  pcl::getMinMax3D(*object_cloud, minpH, maxpH);
  geometry_msgs::Point p1H;
  p1H.x = (minpH.x+maxpH.x)/2; p1H.y = (minpH.y+maxpH.y)/2; p1H.z = (minpH.z+maxpH.z)/2;

  Eigen::Vector4f minPointH (minpH.x,minpH.y , minpH.z,0);   //region of interest in real world (m)
  Eigen::Vector4f maxPointH (maxpH.x,p1H.y, maxpH.z, 0);   //convert from an Eigen::Vector4f to a point type such pcl::PointXYZ
  //crop box filtering
  PointCloudPtr east_cloud (new PointCloud);
  pcl::CropBox<PCLPoint> cropFilter;  //CropBox is a filter that allows the user to filter all the data inside of a given box.
  cropFilter.setMin(minPointH);
  cropFilter.setMax(maxPointH);
  cropFilter.setInputCloud(object_cloud);
  cropFilter.filter(*east_cloud);

  PCLPoint minp, maxp;
  pcl::getMinMax3D(*east_cloud, minp, maxp);
  geometry_msgs::Point p1e;
  p1e.x = (minp.x+maxp.x)/2; p1e.y = (minp.y+maxp.y)/2; p1e.z = (minp.z+maxp.z)/2;
  // box_points.points.push_back(p1e);

  // pcl::getMinMax3D(*east_cloud, minp, maxp);
  // geometry_msgs::Point p1E;
  // p1E.x = (minp.x+maxp.x)/2; p1E.y = 0.39 - ((minp.y+maxp.y)/2); p1E.z =( 1.72 - ((minp.z+maxp.z)/2));


  geometry_msgs::Point p1,p2,p3,p4,p5,p6,p7,p8;
  p1.x = minp.x; p1.y = minp.y; p1.z = minp.z;
  p2.x = maxp.x; p2.y = minp.y; p2.z = minp.z;
  p3.x = minp.x; p3.y = minp.y; p3.z = maxp.z;
  p4.x = maxp.x; p4.y = minp.y; p4.z = maxp.z;
  p5.x = minp.x; p5.y = maxp.y; p5.z = minp.z;
  p6.x = maxp.x; p6.y = maxp.y; p6.z = minp.z;
  p7.x = minp.x; p7.y = maxp.y; p7.z = maxp.z;
  p8.x = maxp.x; p8.y = maxp.y; p8.z = maxp.z;


  lines.points.push_back(p1); lines.points.push_back(p2);
  lines.points.push_back(p1); lines.points.push_back(p3);
  lines.points.push_back(p2); lines.points.push_back(p4);
  lines.points.push_back(p3); lines.points.push_back(p4);
  lines.points.push_back(p5); lines.points.push_back(p6);
  lines.points.push_back(p5); lines.points.push_back(p7);
  lines.points.push_back(p6); lines.points.push_back(p8);
  lines.points.push_back(p7); lines.points.push_back(p8);
  lines.points.push_back(p1); lines.points.push_back(p5);
  lines.points.push_back(p2); lines.points.push_back(p6);
  lines.points.push_back(p3); lines.points.push_back(p7);
  lines.points.push_back(p4); lines.points.push_back(p8);
  line_list.markers.push_back(lines);




  visualization_msgs::Marker box_points;
  box_points.ns = "midwest_box";
  box_points.header.frame_id  = "/kinect2_rgb_optical_frame";
  box_points.header.stamp = ros::Time::now();
  box_points.action=visualization_msgs::Marker::ADD;
  box_points.type=visualization_msgs::Marker::SPHERE;
  box_points.lifetime= ros::Duration();
  box_points.id=2;
  box_points.scale.x = 0.02;
  box_points.scale.y = 0.02;
  box_points.scale.z = 0.02;
  box_points.color.a = 1;
  box_points.color.r = 0.0f;
  box_points.color.g = 1.0f;
  box_points.color.b = 0.0f;
  box_points.pose.position.x = minp.x ;
   box_points.pose.position.y = p1e.y;
   box_points.pose.position.z = p1e.z;

   box_points.pose.orientation.x = 0.0;
   box_points.pose.orientation.y = 0.0;
   box_points.pose.orientation.z = 0.0;
   box_points.pose.orientation.w = 1.0;

    mideastbox_pub.publish(box_points);

    // mideastboxPublisher(east_cloud);
    eastbox_pub.publish(line_list);

    return;
}

void halfwestPublisher(PointCloudPtr object_cloud, std_msgs::Header header) {



    visualization_msgs::MarkerArray line_list;
    visualization_msgs::Marker lines;

    lines.ns = "bounding_westbox";
    lines.header = header;
    lines.type = visualization_msgs::Marker::LINE_LIST;
    lines.action = visualization_msgs::Marker::ADD;
    lines.id = 1;
    lines.scale.x = 0.01;
    lines.scale.y = 0.01;
    lines.color.a = 1;
    lines.color.r = 0.0f;
    lines.color.g = 0.0f;
    lines.color.b = 1.0f;
    lines.lifetime = ros::Duration(0.01);
    lines.points.clear();



    PCLPoint minpH, maxpH;
    pcl::getMinMax3D(*object_cloud, minpH, maxpH);
    geometry_msgs::Point p1H;
    p1H.x = (minpH.x+maxpH.x)/2; p1H.y = (minpH.y+maxpH.y)/2; p1H.z = (minpH.z+maxpH.z)/2;

    Eigen::Vector4f minPointH (minpH.x, p1H.y, minpH.z,0);   //region of interest in real world (m)
    Eigen::Vector4f maxPointH (maxpH.x,maxpH.y, maxpH.z, 0);   //convert from an Eigen::Vector4f to a point type such pcl::PointXYZ
    //crop box filtering
    PointCloudPtr east_cloud (new PointCloud);
    pcl::CropBox<PCLPoint> cropFilter;  //CropBox is a filter that allows the user to filter all the data inside of a given box.
    cropFilter.setMin(minPointH);
    cropFilter.setMax(maxPointH);
    cropFilter.setInputCloud(object_cloud);
    cropFilter.filter(*east_cloud);

    PCLPoint minp, maxp;
    pcl::getMinMax3D(*east_cloud, minp, maxp);
    geometry_msgs::Point p1e;
    p1e.x = (minp.x+maxp.x)/2; p1e.y = (minp.y+maxp.y)/2; p1e.z = (minp.z+maxp.z)/2;
    // box_points.points.push_back(p1e);

    // pcl::getMinMax3D(*east_cloud, minp, maxp);
    // geometry_msgs::Point p1E;
    // p1E.x = (minp.x+maxp.x)/2; p1E.y = 0.39 - ((minp.y+maxp.y)/2); p1E.z =( 1.72 - ((minp.z+maxp.z)/2));


    geometry_msgs::Point p1,p2,p3,p4,p5,p6,p7,p8;
    p1.x = minp.x; p1.y = minp.y; p1.z = minp.z;
    p2.x = maxp.x; p2.y = minp.y; p2.z = minp.z;
    p3.x = minp.x; p3.y = minp.y; p3.z = maxp.z;
    p4.x = maxp.x; p4.y = minp.y; p4.z = maxp.z;
    p5.x = minp.x; p5.y = maxp.y; p5.z = minp.z;
    p6.x = maxp.x; p6.y = maxp.y; p6.z = minp.z;
    p7.x = minp.x; p7.y = maxp.y; p7.z = maxp.z;
    p8.x = maxp.x; p8.y = maxp.y; p8.z = maxp.z;


    lines.points.push_back(p1); lines.points.push_back(p2);
    lines.points.push_back(p1); lines.points.push_back(p3);
    lines.points.push_back(p2); lines.points.push_back(p4);
    lines.points.push_back(p3); lines.points.push_back(p4);
    lines.points.push_back(p5); lines.points.push_back(p6);
    lines.points.push_back(p5); lines.points.push_back(p7);
    lines.points.push_back(p6); lines.points.push_back(p8);
    lines.points.push_back(p7); lines.points.push_back(p8);
    lines.points.push_back(p1); lines.points.push_back(p5);
    lines.points.push_back(p2); lines.points.push_back(p6);
    lines.points.push_back(p3); lines.points.push_back(p7);
    lines.points.push_back(p4); lines.points.push_back(p8);
    line_list.markers.push_back(lines);




    visualization_msgs::Marker box_points;
    box_points.ns = "midwest_box";
    box_points.header.frame_id  = "/kinect2_rgb_optical_frame";
    box_points.header.stamp = ros::Time::now();
    box_points.action=visualization_msgs::Marker::ADD;
    box_points.type=visualization_msgs::Marker::SPHERE;
    box_points.lifetime= ros::Duration();
    box_points.id=2;
    box_points.scale.x = 0.02;
    box_points.scale.y = 0.02;
    box_points.scale.z = 0.02;
    box_points.color.a = 1;
    box_points.color.r = 0.0f;
    box_points.color.g = 0.0f;
    box_points.color.b = 1.0f;
    box_points.pose.position.x = minp.x  ;
     box_points.pose.position.y = p1e.y;
     box_points.pose.position.z = p1e.z;

     box_points.pose.orientation.x = 0.0;
     box_points.pose.orientation.y = 0.0;
     box_points.pose.orientation.z = 0.0;
     box_points.pose.orientation.w = 1.0;

    midwestbox_pub.publish(box_points);
    // mideastboxPublisher(east_cloud);
    westbox_pub.publish(line_list);

    return;
}


void msgCallback(const sensor_msgs::PointCloud2ConstPtr& pcd){     //sensor messages has various parameters of sensors, PointCloud2 is one of them

    pcl::ScopeTime t("msgCallback");  //Class to measure the time spent in a scope.

    PointCloudPtr read_cloud (new PointCloud);
    pcl::fromROSMsg(*pcd, *read_cloud);  //conversion of point cloud

    Eigen::Vector4f minPoint (roi_min_x_, roi_min_y_, roi_min_z_,0);   //region of interest in real world (m)
    Eigen::Vector4f maxPoint (roi_max_x_, roi_max_y_, roi_max_z_, 0);   //convert from an Eigen::Vector4f to a point type such pcl::PointXYZ
    // Eigen::Vector3f rotationPoint (-2.8797933, 0, 2.8797933);

   std::cerr << "Input PointCloud : " << read_cloud->points.size()<<std::endl;



    //crop box filtering
    PointCloudPtr crop_cloud (new PointCloud);
    pcl::CropBox<PCLPoint> cropFilter;  //CropBox is a filter that allows the user to filter all the data inside of a given box.
    cropFilter.setMin(minPoint);
    cropFilter.setMax(maxPoint);
    // cropFilter.setTranslation(Eigen::Vector3f( -0.0243,0.0014 ,0));
    // cropFilter.setTranslation(Eigen::Vector3f( -0.0597,0.0034 ,0));
    cropFilter.setTranslation(Eigen::Vector3f( 0,0 ,0));

    Eigen::Vector3f rotationPoint ( 0.0719, 0, 0 ); //ADDED Later
    // Eigen::Vector3f rotationPoint ( 0.0854, 0, 0 ); //ADDED Later

    cropFilter.setRotation(rotationPoint);
    cropFilter.setInputCloud(read_cloud);
    cropFilter.filter(*crop_cloud);

    cropboxPublisher(crop_cloud, pcd->header);
    sensor_msgs::PointCloud2 croppub_msg;
    pcl::toROSMsg(*crop_cloud,croppub_msg);
    cropcloud_pub.publish(croppub_msg);


   std::cerr << "PointCloud AFTER Cropping : " << crop_cloud->width * crop_cloud->height<<std::endl;


    //voxel grid filtering :http://pointclouds.org/documentation/tutorials/voxel_grid.php

    pcl::PointCloud<pcl::PointXYZ>::Ptr vox_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::VoxelGrid<pcl::PointXYZ> fo;
    fo.setInputCloud(crop_cloud);
    fo.setLeafSize (0.0025f, 0.0025f, 0.0025f);
    fo.filter (*vox_cloud);


   std::cerr << "PointCloud AFTER Voxel filtering: " << vox_cloud->width * vox_cloud->height <<std::endl;

    //StatisticalOutlierRemoval: http://pointclouds.org/documentation/tutorials/statistical_outlier.php
    PointCloudPtr sor_scene(new PointCloud);
    SORfilter(vox_cloud, sor_scene);

    PointCloudPtr scene(new PointCloud);
    // RORfilter(sor_scene, scene);

      // std::cerr << "PointCloud AFTER Filtering: " << scene->width * scene->height<<std::endl;

    //segment object point cloud
    PointCloudPtr object_cloud (new PointCloud);
    std::cerr << "object_cloud before segmentation: " << sor_scene->points.size()<<std::endl;
        objectSegmentation(sor_scene, object_cloud);
    std::cerr << "object_cloud AFTER segmentation: " << object_cloud->points.size()<<std::endl;

    if(object_cloud->points.empty()){
      ROS_ERROR("Object cloud is empty");
      return;
    }

    //publish object pointcloud
    PointCloudPtr output_cloud = object_cloud;
    output_cloud->header = read_cloud->header;


    sensor_msgs::PointCloud2 pub_msg;
    pcl::toROSMsg(*output_cloud,pub_msg);
    cloud_pub.publish(pub_msg);
    // std::cerr << "object_cloud Published: " << object_cloud->points.size()<<std::endl;

    Eigen::Vector4f centroid;
    Eigen::Matrix3f evecs;
    Eigen::Vector3f evals;
    Eigen::Matrix3f cov_matrix;
    pcl::compute3DCentroid(*object_cloud,centroid);
    pcl::computeCovarianceMatrix(*object_cloud, centroid, cov_matrix);
    pcl::eigen33(cov_matrix, evecs, evals);

    Eigen::Quaternionf rot(evecs);
    Eigen::Vector3f pos = centroid.head<3>();

    //publish pose and bounding box
    posePublisher(pos, rot, pcd->header);
    bboxPublisher(object_cloud, pcd->header);
    midboxPublisher(object_cloud, pcd->header);
    minpointPublisher(object_cloud,pcd->header) ;
    halfeastPublisher(object_cloud, pcd->header);
    halfwestPublisher(object_cloud, pcd->header);


    return;
}


int main (int argc, char** argv){

    ros::init(argc, argv, "find_object");
    ros::NodeHandle priv_nh("~");

    priv_nh.param("object_size_min", object_size_threshold_min_, float(0.02));
    priv_nh.param("object_size_max", object_size_threshold_max_, float(0.2));
    priv_nh.param("plane_normal", plane_normal_threshold_, int(0));
    priv_nh.param("use_plane_normal", use_plane_normal_threshold_, bool(false));
    priv_nh.param("roi_min_x", roi_min_x_, float(-0.5));
    priv_nh.param("roi_max_x", roi_max_x_, float(0.5));
    priv_nh.param("roi_min_y", roi_min_y_, float(-0.5));
    priv_nh.param("roi_max_y", roi_max_y_, float(0.5));
    priv_nh.param("roi_min_z", roi_min_z_, float(0.0));
    priv_nh.param("roi_max_z", roi_max_z_, float(1.5));
    priv_nh.param("cluster_tolerance", euclidean_cluster_tolerance_, float(0.05));

    ros::Subscriber sub = priv_nh.subscribe("input_cloud", 1, msgCallback);

    pose_pub = priv_nh.advertise<geometry_msgs::PoseStamped>("output_pose",1);
    bbox_pub = priv_nh.advertise<visualization_msgs::MarkerArray>("output_box", 1);
    eastbox_pub = priv_nh.advertise<visualization_msgs::MarkerArray>("output_eastbox", 1);
    westbox_pub = priv_nh.advertise<visualization_msgs::MarkerArray>("output_westbox", 1);
    cloud_pub = priv_nh.advertise<sensor_msgs::PointCloud2>("output_cloud", 1);
    midbox_pub = priv_nh.advertise<visualization_msgs::Marker>("midbox_box", 1);
    mideastbox_pub = priv_nh.advertise<visualization_msgs::Marker>("midbox_eastbox", 1);
    midwestbox_pub = priv_nh.advertise<visualization_msgs::Marker>("midbox_westbox", 1);
    minpoint_pub = priv_nh.advertise<visualization_msgs::Marker>("minpoint_box", 1);
    cropbox_pub = priv_nh.advertise<visualization_msgs::MarkerArray>("cropoutput_box", 1);
    cropcloud_pub = priv_nh.advertise<sensor_msgs::PointCloud2>("cropoutput_cloud", 1);
    ros::spin();

    return 0;
}
