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


using namespace std;

visualization_msgs::MarkerArray line_list;
visualization_msgs::Marker lines;

ros::Publisher pose_pub;
ros::Publisher bbox_pub;
ros::Publisher cloud_pub;

float object_size_threshold_min_;
float object_size_threshold_max_;
int plane_normal_threshold_;
bool use_plane_normal_threshold_;
float roi_min_x_, roi_max_x_;
float roi_min_y_, roi_max_y_;
float roi_min_z_, roi_max_z_;

typedef pcl::PointXYZ PCLPoint;
typedef pcl::PointCloud<PCLPoint> PointCloud;
typedef PointCloud::Ptr PointCloudPtr;



void SORfilter(const PointCloudPtr input, PointCloudPtr output) {
  pcl::StatisticalOutlierRemoval<PCLPoint> sor;
  sor.setInputCloud (input);
  sor.setMeanK (50);
  sor.setStddevMulThresh (1);
  sor.filter (*output);

  return;
}


void objectSegmentation(const PointCloudPtr input_cloud, PointCloudPtr& object_cloud) {

  int scene_cloud_point_thresh_ = 2000;
  int object_cloud_point_thresh_ = 500;

  PointCloudPtr scene_cloud (new PointCloud);
  pcl::copyPointCloud(*input_cloud, *scene_cloud);
  
  pcl::ModelCoefficients::Ptr plane_coeffs (new pcl::ModelCoefficients);
  pcl::PointIndices::Ptr plane_inliers (new pcl::PointIndices);
  pcl::SACSegmentation<PCLPoint> seg;
  seg.setOptimizeCoefficients (true);
  seg.setModelType (pcl::SACMODEL_PLANE);
  seg.setMethodType (pcl::SAC_RANSAC);
  seg.setDistanceThreshold (0.01);

  PointCloudPtr object_candidate (new PointCloud);
  while(scene_cloud->points.size() > scene_cloud_point_thresh_) {
    seg.setInputCloud (scene_cloud);
    seg.segment (*plane_inliers, *plane_coeffs);
    if (plane_inliers->indices.size () == 0)
      break;

    pcl::ExtractIndices<PCLPoint> eifilter (true);
    eifilter.setInputCloud(scene_cloud);
    eifilter.setIndices(plane_inliers);
    
    if(use_plane_normal_threshold_){
      PointCloudPtr segmented_plane(new PointCloud);
      eifilter.filter(*segmented_plane);

      Eigen::Vector4f centroid;
      Eigen::Matrix3f evecs;
      Eigen::Vector3f evals;
      Eigen::Matrix3f cov_matrix;
      pcl::compute3DCentroid(*segmented_plane,centroid);
      pcl::computeCovarianceMatrix(*segmented_plane, centroid, cov_matrix);
      pcl::eigen33(cov_matrix, evecs, evals);
      Eigen::Vector3f plane_normal = evecs.col(2);
      plane_normal.normalize();
      float theta = asin(plane_normal[2]) * 180/M_PI;

      eifilter.setNegative(true);
      if(abs(theta) > plane_normal_threshold_){        //if plane normal threshold is satisfied
	eifilter.filter(*object_candidate);
	break;
      }
      else{
	eifilter.filter(*scene_cloud);
      }
    }
    else{
      eifilter.setNegative(true);
      eifilter.filter(*object_candidate);
      break;
    }
  }

  if(object_candidate->points.size() < object_cloud_point_thresh_){
    ROS_ERROR("Object cloud size is less than threshold.");
    return;
  }

  //Euclidean cluster extraction
  pcl::search::KdTree<PCLPoint>::Ptr tree (new pcl::search::KdTree<PCLPoint>);
  tree->setInputCloud (object_candidate);
  std::vector<pcl::PointIndices> cluster_inds_vector;
  pcl::EuclideanClusterExtraction<PCLPoint> ec;
  ec.setClusterTolerance (0.01); // 1cm
  ec.setMinClusterSize (object_cloud_point_thresh_);
  ec.setMaxClusterSize (50000);
  ec.setSearchMethod (tree);
  ec.setInputCloud (object_candidate);
  ec.extract (cluster_inds_vector);

  for (int i=0; i < cluster_inds_vector.size(); i++){
    PointCloudPtr cluster (new PointCloud);
    pcl::ExtractIndices<PCLPoint> extract (true);
    extract.setInputCloud(object_candidate);
    pcl::PointIndices::Ptr inds (new pcl::PointIndices);
    inds->indices = cluster_inds_vector[i].indices;
    extract.setIndices(inds);
    extract.filter(*cluster);

    Eigen::Vector4f obj_centroid;
    pcl::compute3DCentroid(*cluster,obj_centroid);
    Eigen::Vector4f plane_norm (plane_coeffs->values[0], plane_coeffs->values[1], plane_coeffs->values[2], plane_coeffs->values[3]);
    float dist = (plane_norm.dot(obj_centroid))/plane_norm.norm();
    if(dist > object_size_threshold_min_ && dist < object_size_threshold_max_){
      object_cloud = cluster;
      break;
    }
  }

  return;
} 


void bboxPublisher(PointCloudPtr object_cloud, std_msgs::Header header){

  visualization_msgs::MarkerArray line_list;
  visualization_msgs::Marker lines;

  lines.ns = "bounding_box";
  lines.header = header;
  lines.type = visualization_msgs::Marker::LINE_LIST;
  lines.action = visualization_msgs::Marker::ADD;
  lines.id = 1;
  lines.color.a = 1;
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


void msgCallback(const sensor_msgs::PointCloud2ConstPtr& pcd){

    pcl::ScopeTime t("imgCallback");

    PointCloudPtr read_cloud (new PointCloud);
    pcl::fromROSMsg(*pcd, *read_cloud);

    Eigen::Vector4f minPoint (roi_min_x_, roi_min_y_, roi_min_z_, 0);                  //region of interest in real world (m)
    Eigen::Vector4f maxPoint (roi_max_x_, roi_max_y_, roi_max_z_, 0);

    //crop box filtering
    PointCloudPtr crop_cloud (new PointCloud);
    pcl::CropBox<PCLPoint> cropFilter;
    cropFilter.setInputCloud(read_cloud);
    cropFilter.setMin(minPoint);
    cropFilter.setMax(maxPoint);
    cropFilter.filter(*crop_cloud);

    //voxel grid filtering
    /*
    pcl::PointCloud<pcl::PointXYZ>::Ptr vox_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::VoxelGrid<pcl::PointXYZ> fo;
    fo.setInputCloud(crop_cloud);
    fo.setLeafSize (0.005f, 0.005f, 0.005f);
    fo.filter (*vox_cloud);
    */

    //SOR filtering
    PointCloudPtr scene(new PointCloud);
    SORfilter(crop_cloud, scene);

    //segment object point cloud
    PointCloudPtr object_cloud (new PointCloud);
    objectSegmentation(scene, object_cloud);
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
    ros::Subscriber sub = priv_nh.subscribe("input_cloud", 1, msgCallback);

    pose_pub = priv_nh.advertise<geometry_msgs::PoseStamped>("output_pose",1000);
    bbox_pub = priv_nh.advertise<visualization_msgs::MarkerArray>("output_box", 1000);
    cloud_pub = priv_nh.advertise<sensor_msgs::PointCloud2>("output_cloud", 1000);

    ros::spin();

    return 0;
}
