#include "knob.hpp"

void padding_neighbor(int* paddedMat, int width, int nearby_points, int row_index, int col_index, bool direction, int** indexMat)
{   
    int count = 0;
    if (direction){
        for(int a=0; a<200; a++){
            if (a+col_index == width)
                col_index = col_index - width;
            if(indexMat[row_index][col_index+a] != -1){
                count++;
                paddedMat[col_index+a] = 1;
                if (count == nearby_points)
                    break;
            }
        }
    }
    else{
        for (int a=0; a<200; a++){
            if(col_index-a == -1)
                col_index = col_index + width;
            if (indexMat[row_index][col_index-a] != -1){
                count++;
                paddedMat[col_index-a] = 1;
                if (count == nearby_points)
                    break;
            }
        }
    }
}



void get_valid_points(int* lr_mat, int width, int* paddedMat, int** indexMat, int nearby_points, int row_index)
{
    bool direction;

    for (int j = 0; j < width; j++)
    {
        if(lr_mat[j] == 1)
            direction = false;
        else if(lr_mat[j] == 2)
            direction = true;
        else
            continue;
        padding_neighbor(paddedMat, width, nearby_points, row_index, j, direction, indexMat);
    }
}

void sampling_foreground(vector<array<float, IN_COLS>> &inPoints, float** rangeMat, 
                int** indexMat, int row_size, int nearby_points, int height, int width,
                float curv_threshold, array<int, 6> diff_thr_list, vector<array<float, OUT_COLS>> &outPoints)
{      
    float max_angle_height = 28;
    float angular_resolution_y = max_angle_height / height;
    float angular_resolution_x = 360.0 / width;

    int pointNum = inPoints.size();
    int near_points = 0;
    int far_points = 0;
    for(int i=0; i<pointNum; i++){
        array<float, IN_COLS> point = inPoints[i];
        float range, verticalAngle, horizontalAngle, minDiff, row, column;
        int rowIdn, colIdn;

        range = sqrt(point[0]*point[0] + point[1]*point[1] + point[2]*point[2]);

        verticalAngle = asin(point[2]/range) * 180 / M_PI;
        row = (verticalAngle + ang_bottom) / angular_resolution_y;
        rowIdn = lrint(static_cast<double>(row));
        if (rowIdn >= height){
            if (rowIdn == height)
                rowIdn = height - 1;
            continue;
        }
        if (rowIdn < 0)
            continue;

        horizontalAngle = atan2(point[0], point[1]) * 180 / M_PI;
        column = -(horizontalAngle-90.0)/360 * width + width/2;
        colIdn = lrint(static_cast<double>(column));
        if (colIdn >= width){
            colIdn -= width;
            column -= width;
        }

        if (range < rangeMat[rowIdn][colIdn]) {
            rangeMat[rowIdn][colIdn] = range;
            indexMat[rowIdn][colIdn] = i;
        }
    }

    for (int i = 0; i < height; i++){
        int* paddedMat = new int[width];
        int* lrMat = new int[width];
        for(int j=0; j<width; j++) {
            lrMat[j] = 0;
            paddedMat[j] = 0;
        }
        for (int j = 0; j < width; j++){
            if(rangeMat[i][j] == INFINITY){
                continue;
            }
            if(indexMat[i][j] == -1)
                continue;
            float diffRange = 0;
            bool flag = false;
            for(int k = 0; k < 2*row_size + 1; k++){
                if(j-row_size+k < 0){
                    if(rangeMat[i][j-row_size+k+width] != INFINITY)
                        diffRange += rangeMat[i][j-row_size+k+width];
                    else{
                        flag = true;
                        continue;
                    }
                }
                else if(rangeMat[i][(j-row_size+k)%width] != INFINITY)
                    diffRange += rangeMat[i][(j-row_size+k)%width];
                else{
                    flag = true;
                    continue;
                }
            }

            if(flag)
                continue;
            else{
                diffRange -= rangeMat[i][j]*(2*row_size + 1);
                float left_diff = rangeMat[i][j+row_size] - rangeMat[i][j];
                float right_diff = rangeMat[i][j-row_size] - rangeMat[i][j];
                float curvature = std::pow(diffRange, 2);
                float range = rangeMat[i][j];

                int diff_threshold;
                if (range <= 5)
                    diff_threshold = diff_thr_list[0];
                else if (range <= 10)
                    diff_threshold = diff_thr_list[1];
                else if (range <= 15)
                    diff_threshold = diff_thr_list[2];
                else if (range <= 20)
                    diff_threshold = diff_thr_list[3];
                else if (range <= 30)
                    diff_threshold = diff_thr_list[4];
                else
                    diff_threshold = diff_thr_list[5];

                if(curvature >= curv_threshold){
                    if(left_diff > diff_threshold && right_diff > diff_threshold)
                        continue;
                    else if(left_diff > diff_threshold)
                        lrMat[j] = 1;
                    else if(right_diff > diff_threshold)
                        lrMat[j] = 2;
                }
            }
        }
        get_valid_points(lrMat, width, paddedMat, indexMat, nearby_points, i);
        for(int j=0; j<width; j++){
            if (indexMat[i][j] != -1 && paddedMat[j] == 1){
                array<float, IN_COLS> inPoint = inPoints[indexMat[i][j]];
                array<float, OUT_COLS> outPoint;
                copy(inPoint.begin(), inPoint.begin()+OUT_COLS, outPoint.begin());
                outPoints.push_back(outPoint);
            }
        }
        delete [] lrMat;
        delete [] paddedMat;
    }
}


void sampling_multi_resolution(vector<array<float, OUT_COLS>> &inPoints, vector<array<float, OUT_COLS>> &outPoints, 
        array<int, 6> width_sample_list, int height, float maxAngleHeight){
    
    float angular_resolution_y = maxAngleHeight / height;
    float ***reso_rangeMat = new float**[6];
    int ***reso_indexMat = new int**[6];
    
    for (int i=0; i<6; i++){
        reso_rangeMat[i] = new float*[height];
        reso_indexMat[i] = new int*[height];
        for (int j=0; j<height; j++){
            reso_rangeMat[i][j] = new float[width_sample_list[i]];
            reso_indexMat[i][j] = new int[width_sample_list[i]];
            for (int k=0; k<width_sample_list[i]; k++){
                reso_rangeMat[i][j][k] = INFINITY;
                reso_indexMat[i][j][k] = -1;
            }
        }
    }
    
    int pointNum = inPoints.size();
    for (int i = 0; i < pointNum; i++){
        array<float, OUT_COLS> point = inPoints[i];
        float range, verticalAngle, horizontalAngle, row, reso_column;
        int rowIdn, reso_colIdn, range_interval;
        range = sqrt(point[0]*point[0] + point[1]*point[1] + point[2]*point[2]);

        if (range <= 5)
            range_interval = 0;
        else if (range <= 10)
            range_interval = 1;
        else if (range <= 15)
            range_interval = 2;
        else if (range <= 20)
            range_interval = 3;
        else if (range <= 30)
            range_interval = 4;
        else
            range_interval = 5;

        verticalAngle = asin(point[2]/range) * 180 / M_PI;
        row = (verticalAngle + ang_bottom) / angular_resolution_y;
        rowIdn = lrint(static_cast<double>(row));
        if (rowIdn >= height){
            if (rowIdn == height)
                rowIdn = height - 1;
            continue;
        }
        if (rowIdn < 0)
            continue;
        
        horizontalAngle = atan2(point[0], point[1]) * 180 / M_PI;
        reso_column = -(horizontalAngle-90.0)/360 * width_sample_list[range_interval] + width_sample_list[range_interval]/2;
        reso_colIdn = lrint(static_cast<double>(reso_column));
        if (reso_colIdn >= width_sample_list[range_interval])
            reso_colIdn -= width_sample_list[range_interval];

        if (range < reso_rangeMat[range_interval][rowIdn][reso_colIdn]) {
            reso_rangeMat[range_interval][rowIdn][reso_colIdn] = range;
            reso_indexMat[range_interval][rowIdn][reso_colIdn] = i;
        }

    }
    
    for (int i = 0; i < 6; i++){
        for (int j = 0; j < height; j++){
            for (int k = 0; k < width_sample_list[i]; k++){
                if (reso_indexMat[i][j][k] != -1){
                    array<float, OUT_COLS> inPoint = inPoints[reso_indexMat[i][j][k]];
                    array<float, OUT_COLS> outPoint;
                    copy(inPoint.begin(), inPoint.begin()+OUT_COLS, outPoint.begin());
                    outPoints.push_back(outPoint);
                }
            }
            delete[] reso_rangeMat[i][j];
            delete[] reso_indexMat[i][j];
        }
        delete[] reso_rangeMat[i];
        delete[] reso_indexMat[i];
    }
    delete[] reso_rangeMat;
    delete[] reso_indexMat;
}