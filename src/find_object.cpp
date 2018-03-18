#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <visualization_msgs/MarkerArray.h>

#include <iostream>
#include <boost/thread/thread.hpp>
#include <Eigen/Dense>

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

float object_size_threshold_;
int plane_normal_threshold_;
bool use_plane_normal_threshold_;

typedef pcl::PointXYZ PCLPoint;
typedef pcl::PointCloud<PCLPoint> PointCloud;
typedef PointCloud::Ptr PointCloudPtr;



Eigen::Vector4f minPoint (0.00,-0.20, -0.50, 0);                  //region of interest in real world (m)
Eigen::Vector4f maxPoint (1.50, 0.20, 0.50, 0);


void SORfilter(const PointCloudPtr input, PointCloudPtr output) {
  pcl::StatisticalOutlierRemoval<PCLPoint> sor;
  sor.setInputCloud (input);
  sor.setMeanK (50);
  sor.setStddevMulThresh (1);
  sor.filter (*output);
}


void objectSegmentation(const PointCloudPtr input_cloud, PointCloudPtr object_cloud) {

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
  seg.setDistanceThreshold (0.02);

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
    }
  }

  if(object_candidate->points.size() < object_cloud_point_thresh_){
    ROS_ERROR("Object not found in point cloud");
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
    if(dist < object_size_threshold_){
      object_cloud = cluster;
      break;
    }
  }
} 


void msgCallback(const sensor_msgs::PointCloud2ConstPtr& pcd){

    pcl::ScopeTime t("imgCallback");

    PointCloudPtr read_cloud (new PointCloud);
    pcl::PCLPointCloud2 pcl_cloud;
    pcl_conversions::toPCL(*pcd,pcl_cloud);
    pcl::fromPCLPointCloud2(pcl_cloud,*read_cloud);

    PointCloudPtr crop_cloud (new PointCloud);
    pcl::CropBox<PCLPoint> cropFilter;
    cropFilter.setInputCloud(read_cloud);
    cropFilter.setMin(minPoint);
    cropFilter.setMax(maxPoint);
    cropFilter.filter(*crop_cloud);

    //SOR filtering
    PointCloudPtr scene(new PointCloud);
    SORfilter(crop_cloud, scene);

    PointCloudPtr object_cloud (new PointCloud);
    objectSegmentation(scene, object_cloud);
    
    sensor_msgs::PointCloud2 output;
    pcl::toROSMsg(*object_cloud, output);
    output.header = pcd->header;
    bbox_pub.publish(output);

    return;


}


int main (int argc, char** argv){

    ros::init(argc, argv, "find_object");
    ros::NodeHandle priv_nh("~");

    priv_nh.param("object_size", object_size_threshold_, float(0.2));
    priv_nh.param("plane_normal", plane_normal_threshold_, int(0));
    priv_nh.param("use_plane_normal", use_plane_normal_threshold_, bool(false));

    ros::Subscriber sub = priv_nh.subscribe("input_cloud", 1, msgCallback);

    pose_pub = priv_nh.advertise<geometry_msgs::PoseStamped>("object_pose",1);
    bbox_pub = priv_nh.advertise<sensor_msgs::PointCloud2>("bounding_box", 1);

    ros::spin();

    return 0;
}
