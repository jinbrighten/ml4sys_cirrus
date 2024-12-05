#include "utility.hpp"



vector<string> listDir(string dirpath) {
    DIR *dir_ptr;
    struct dirent *diread;
    vector<string> file_names;
    if ((dir_ptr=opendir(dirpath.c_str())) != nullptr) {
        while ((diread=readdir(dir_ptr)) != nullptr) {
            file_names.push_back(diread->d_name);
        }
        closedir(dir_ptr);
    } else {
        cerr << "Failed to open datset directory." << endl;
    }
    sort(file_names.begin(), file_names.end());
    return file_names;
}



int mkpath(string s) {
    size_t pos=0;
    string dir;
    int mdret;

    if (s[s.size()-1]!= '/') {
        s+= '/';
    }
    while((pos=s.find_first_of('/', pos))!=string::npos) {
        dir = s.substr(0, pos++);
        if(dir.size()==0) continue;
        s_mkdir(dir);
    }
    return 0;
}




void groundRemoval(pcl::RangeImage rangeImage, cv::Mat groundMat, int N_SCAN, int Horizon_SCAN)
{
    int lowerInd, upperInd;
    float diffX, diffY, diffZ, angle;
    for (size_t j = 0; j < Horizon_SCAN; ++j)
    {
        for (size_t i = N_SCAN - 1; N_SCAN - groundScanInd < i; --i)
        {

            lowerInd = j + (i)*Horizon_SCAN;
            upperInd = j + (i + 1) * Horizon_SCAN;

            if (isinf(rangeImage[lowerInd].range) == 1 ||
                isinf(rangeImage[upperInd].range) == 1 ||
                isinf(rangeImage[lowerInd].x) == 1 ||
                isinf(rangeImage[upperInd].x) == 1)
            {
                // no info to check, invalid points
                groundMat.at<int8_t>(i, j) = -1;
                continue;
            }

            diffX = rangeImage[upperInd].x - rangeImage[lowerInd].x;
            diffY = rangeImage[upperInd].y - rangeImage[lowerInd].y;
            diffZ = rangeImage[upperInd].z - rangeImage[lowerInd].z;

            angle = atan2(diffZ, sqrt(diffX * diffX + diffY * diffY)) * 180 / M_PI;

            if (abs(angle) <= 10)
            {
                groundMat.at<int8_t>(i, j) = 1;
                groundMat.at<int8_t>(i + 1, j) = 1;
            }
        }
    }
}

void get_filtered_lidar(PointCloudType::Ptr input_cloud, float min_x, float max_x, \
                                        float min_y, float max_y, float min_z, float max_z)
{
    // Define the region
    pcl::PassThrough<PointType> pass_filter;
    pass_filter.setInputCloud(input_cloud);
    pass_filter.setFilterFieldName("x");
    pass_filter.setFilterLimits(min_x, max_x);
    pass_filter.filter(*input_cloud);

    pass_filter.setInputCloud(input_cloud);
    pass_filter.setFilterFieldName("y");
    pass_filter.setFilterLimits(min_y, max_y);
    pass_filter.filter(*input_cloud);

    pass_filter.setInputCloud(input_cloud);
    pass_filter.setFilterFieldName("z");
    pass_filter.setFilterLimits(min_z, max_z);
    pass_filter.filter(*input_cloud);
}


// 0: type 1: truncated 2: occluded 3: alpha 4: xmin 5: ymin 6: xmax 7: ymax
// 8: dim_h 9: dim_w 10: dim_l 11: loc_x 12: loc_y 13: loc_z 14: rot_y
// (N, 15) to (N, 8, 3)
void get_label(string &&label_path, vector<vector<vector<int>>> &labels,
               int width, int height,
               float max_angle_width, float max_angle_height)
{
    std::ifstream file(label_path);
    std::string line;
    float verticalAngle, horizonAngle, range;
    size_t rowIdn, columnIdn;
    PointType thisPoint;

    float angular_resolution_x = max_angle_width / width;
    float angular_resolution_y = max_angle_height / height;

    while (std::getline(file, line))
    {
        vector<std::string> line_parts;

        std::istringstream iss(line);
        std::string part;

        while (iss >> part)
        {
            line_parts.push_back(part);
        }

        // xmin, ymin, xmax, ymax
        float xmin = std::stof(line_parts[7]), ymin = std::stof(line_parts[8]), xmax = std::stof(line_parts[9]), ymax = std::stof(line_parts[10]);
        // height, width, length (h, w, l)
        float h = std::stof(line_parts[3]), w = std::stof(line_parts[4]), l = std::stof(line_parts[5]);
        // location (x,y,z) in camera coord.
        float x = std::stof(line_parts[0]), y = std::stof(line_parts[1]), z = std::stof(line_parts[2]);
        float ry = std::stof(line_parts[6]); // yaw angle (around Y-axis in camera coordinates) [-pi..pi]

        // cout<< x << " " << y << " " << z << " " << h << " " << w << " " << l << " " << ry << endl;

        vector<vector<float>> trackletBox = {
            {-l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2},
            {w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2},
            {0, 0, 0, 0, h, h, h, h}};

        vector<vector<float>> rotMat = {
            {cos(ry), -sin(ry), 0},
            {sin(ry), cos(ry), 0},
            {0, 0, 1}};

        vector<vector<float>> box3d(8, vector<float>(3, 0.0f));
        vector<vector<int>> box3d_proj(8, vector<int>(2, 0));
        for (int i = 0; i < 8; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                for (int k = 0; k < 3; k++)
                {
                    box3d[i][j] += rotMat[j][k] * trackletBox[k][i];
                }
            }
            box3d[i][0] += x;
            box3d[i][1] += y;
            box3d[i][2] += z;

            thisPoint.x = box3d[i][0];
            thisPoint.y = box3d[i][1];
            thisPoint.z = box3d[i][2];
            // find the row and column index in the iamge for this point
            verticalAngle = atan2(thisPoint.z, sqrt(thisPoint.x * thisPoint.x + thisPoint.y * thisPoint.y)) * 180 / M_PI;
            rowIdn = pcl_lrintf((verticalAngle + ang_bottom) / angular_resolution_y);
            horizonAngle = atan2(thisPoint.x, thisPoint.y) * 180 / M_PI;
            columnIdn = pcl_lrintf(-((horizonAngle - 90.0) / angular_resolution_x) + width / 2);
            if (columnIdn >= width)
                columnIdn -= width;
            box3d_proj[i][0] = columnIdn;
            box3d_proj[i][1] = rowIdn;
            // cout << rowIdn << " " << columnIdn << " ";
        }
        labels.push_back(box3d_proj);
        // cout << endl;
    }
    file.close();
}

vector<array<float, IN_COLS>> read_bin(string &bin_path) {
    ifstream file(bin_path, ios::binary | ios::ate);

    if (!file) {
        cerr << "Failed to open " << bin_path << endl;
        exit(1);
    }

    streamsize size = file.tellg();
    file.seekg(0, ios::beg);

    if (size % (sizeof(float)*IN_COLS) != 0) {
        cerr << "Invalid file size" << endl;
        exit(1);
    }

    int num_points = size / (sizeof(float)*IN_COLS);
    vector<array<float, IN_COLS>> points(num_points);
    for (int i=0; i<num_points; i++) {
        file.read(reinterpret_cast<char*>(points[i].data()), sizeof(float)*IN_COLS);
    }
    file.close();

    return points;
}


int pcd_to_bin(const std::string &file_name, const PointCloudType &cloud, const bool intensity)
{
    FILE *output_file = fopen(file_name.c_str(), "wb");
    if(output_file){
        if(intensity){
            for(const PointType& point : cloud.points){
                fwrite(&point.x, sizeof(float), 1, output_file);
                fwrite(&point.y, sizeof(float), 1, output_file);
                fwrite(&point.z, sizeof(float), 1, output_file);
                fwrite(&point.intensity, sizeof(float), 1, output_file);
            }
        }
        else{
            for(const PointType& point : cloud.points){
                fwrite(&point.x, sizeof(float), 1, output_file);
                fwrite(&point.y, sizeof(float), 1, output_file);
                fwrite(&point.z, sizeof(float), 1, output_file);
            }
        }
        fclose(output_file);
    }
    else
        return -1;

    return 0;
}