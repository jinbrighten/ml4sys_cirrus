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
void padding_neighbor(int* paddedMat, int width, int nearby_points, int row_index, int col_index, bool direction, int** indexMat);

void get_valid_points(int* lr_mat, int width, int* paddedMat, int** indexMat, int nearby_points, int row_index);

void projectPointCloud (vector<array<float, IN_COLS>> &inPoints, bool interpolation,
                        int seq_num, int width, int height, 
                        float **rangeMat, int **indexMat);

void sampling_foreground(vector<array<float, IN_COLS>> &inPoints, float** rangeMat, 
                int** indexMat, int row_size, int nearby_points, int height, int width,
                float curv_threshold, array<int, 6> diff_thr_list, vector<array<float, OUT_COLS>> &outPoints);


void sampling_multi_resolution(vector<array<float, OUT_COLS>> &inPoints, vector<array<float, OUT_COLS>> &outPoints, 
        array<int, 6> width_sample_list, int height, float maxAngleHeight);



#endif