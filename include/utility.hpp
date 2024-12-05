#ifndef _UTILITY_H_
#define _UTILITY_H_

#include <opencv2/opencv.hpp>
#include <pcl/range_image/range_image.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/png_io.h>
#include <pcl/features/normal_3d.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/point_cloud.h>
#include <pcl/visualization/range_image_visualizer.h>
#include <pcl/console/time.h>
#include <pcl/console/print.h>
#include <pcl/pcl_macros.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/common/common.h>
#include <pcl/registration/icp.h>
#include <vector>
#include <string>
#include <array>
#include <filesystem>

#define PI 3.14159265
constexpr int IN_COLS = 4;
constexpr int OUT_COLS = 4;
constexpr float ang_bottom = 22;

using namespace std;
using namespace pcl::console;
using namespace pcl;
namespace fs = std::filesystem;

typedef pcl::PointXYZI PointType;
typedef pcl::PointCloud<PointType> PointCloudType;

// // Vel 64
// // extern const float ang_bottom = 20; //17.6;
// // extern const int groundScanInd = 78; // 50; //on 64(0.78125) -> 78 on 100
// #define ang_bottom 20
#define groundScanInd 78

static bool inline endsWith(string const &str, string const &suffix) {
    if (str.length() < suffix.length()) 
        return false;
    return str.compare(str.length() - suffix.length(), suffix.length(), suffix) == 0;
}

static bool inline startsWith(string const &str, string const &suffix) {
    if (str.length() < suffix.length()) 
        return false;
    return str.compare(0, suffix.length(), suffix) == 0;
}

static void inline s_mkdir(string dir_path) {
    if (mkdir(dir_path.c_str(), 0777) == -1) {
        if (errno != EEXIST) {
            cerr << "Failed to create save directory " << dir_path << endl;
            cerr << strerror(errno) << endl;
            exit(0);
        }
    }
}

vector<string> listDir(string dirpath);
int mkpath(string s);
void groundRemoval(pcl::RangeImage rangeImage, cv::Mat groundMat, int N_SCAN, int Horizon_SCAN);
void get_filtered_lidar(PointCloudType::Ptr input_cloud, float min_x, float max_x, \
                                        float min_y, float max_y, float min_z, float max_z);
void get_label(string &&label_path, vector<vector<vector<int>>> &labels,
               int width, int height,
               float max_angle_width, float max_angle_height);
int pcd_to_bin(const std::string &file_name, const PointCloudType &cloud, const bool intensity = true);
vector<array<float, IN_COLS>> read_bin(string &bin_path);


#endif