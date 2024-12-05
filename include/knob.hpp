#ifndef _KNOB_HPP_
#define _KNOB_HPP_
#define FAR_RANGE_VALUE 1000

#include "utility.hpp"

/*
 *  Sample points by controlling the resolution of the point cloud. Points which has the range value within
 *  [minRange, maxRange] are filtered. Other points will be taken without filtering.
 *  Arguments:
 *      inputPoints: input points in the form of [x, y, z, intensity, elongation]
 *      outputPoints: output points
 *      minRange: minimum range of filtering
 *      maxRange: maximum range of filtering
 *      width: horizontal number of bins
 *      height: vertical number of bins
 *      inclination: vertical angles of the range image
 *  Return: None
 */
void sample_by_resolution(vector<array<float, IN_COLS>> &inputPoints, vector<array<float, OUT_COLS>> &outputPoints,
        float minRange, float maxRange, int width, int height);

void padding_neighbor(int* paddedMat, int width, int nearby_points, int row_index, int col_index, bool direction, int** indexMat);

void get_valid_points(int* lr_mat, int width, int* paddedMat, int** indexMat, int nearby_points, int row_index);

void projectPointCloud (vector<array<float, IN_COLS>> &inPoints, bool interpolation,
                        int seq_num, int width, int height, 
                        float **rangeMat, int **indexMat);

/*  
 *  Make range image from raw point cloud bin file based on each points range value.
 *  For point which has smaller range value than range threshold, apply resolution knob and make range image with reduced width size.
 *  For point which has larger range value than range threshold, apply edge knob and make range image with full width size.
 *  After making range image, sample points from reduced range image for applying resolution knob.
 */
void sample_by_reso_comb(vector<array<float, IN_COLS>> &inputPoints, vector<array<float, OUT_COLS>> &outputPoints, 
        int width, int width_sample, int height, float max_angle_height, float **edge_rangeMat, int **edge_indexMat, float range_threshold);

void sampling_multi_resolution(vector<array<float, IN_COLS>> &inPoints, vector<array<float, OUT_COLS>> &outPoints, 
        array<int, 6> width_sample_list, int height, float maxAngleHeight);

// Extract edge points with padded nearby points
void ROIextraction(vector<array<float, IN_COLS>> &inPoints, float** rangeMat, 
                int** indexMat, int row_size, int nearby_points, int height, int width,
                float curv_threshold, int diff_threshold, vector<array<float, OUT_COLS>> &outPoints);

#endif