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

void ROIextraction(vector<array<float, IN_COLS>> 
&inPoints, float** rangeMat, 
                int** indexMat, int row_size, int nearby_points, int height, int width,
                float curv_threshold, int diff_threshold, vector<array<float, OUT_COLS>> &outPoints)
{      
    // float **vis_curv = new float*[height];
    // int **vis_padd = new int*[height];
    // int **vis_lr = new int*[height];
    
    // for (int i = 0; i < height; i++){
    //     vis_curv[i] = new float[width];
    //     vis_padd[i] = new int[width];
    //     vis_lr[i] = new int[width];
    //     for (int j = 0; j < width; j++){
    //         vis_curv[i][j] = 0;
    //         vis_padd[i][j] = 0;
    //         vis_lr[i][j] = 0;
    //     }
    // }
    //  long comment coding. long comment coding. long comment coding. long comment coding. long comment coding. long comment coding. long comment coding. long comment coding. long comment coding. long comment coding. long comment coding. long comment coding.
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
                // vis_curv[i][j] = curvature;

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
        // vis_lr[i] = lrMat;
        // vis_padd[i] = paddedMat;
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
    // save vis_ files as txt file
    // ofstream vis_curv_file("../vis_wointerp_curv.txt");
    // ofstream vis_padd_file("../vis_padd.txt");
    // ofstream vis_lr_file("../vis_lr.txt");
    // for(int i=0; i<height; i++){
    //     for(int j=0; j<width; j++){
    //         vis_curv_file << vis_curv[i][j] << " ";
    //         vis_padd_file << vis_padd[i][j] << " ";
    //         vis_lr_file << vis_lr[i][j] << " ";
    //     }
    //     vis_curv_file << endl;
    //     vis_padd_file << endl;
    //     vis_lr_file << endl;
    // }
}

void RI_interpolation(float column, float row, int i, float range, float range_threshold, int width, int height, 
                    float **edge_rangeMat, int **edge_indexMat, int rowIdn, int edge_colIdn){
    int floor_x = pcl_lrint (std::floor (column)), floor_y = pcl_lrint (std::floor (row)),
    ceil_x  = pcl_lrint (std::ceil (column)),  ceil_y  = pcl_lrint (std::ceil (row));

    // interpolate by 2x4 size
    int neighbor_x[8], neighbor_y[8];
    neighbor_x[0]=floor_x-1; neighbor_y[0]=floor_y;
    neighbor_x[1]=floor_x; neighbor_y[1]=floor_y;
    neighbor_x[2]=ceil_x; neighbor_y[2]=floor_y;
    neighbor_x[3]=ceil_x+1; neighbor_y[3]=floor_y;
    neighbor_x[4]=floor_x-1; neighbor_y[4]=ceil_y;
    neighbor_x[5]=floor_x; neighbor_y[5]=ceil_y;
    neighbor_x[6]=ceil_x; neighbor_y[6]=ceil_y;
    neighbor_x[7]=ceil_x+1; neighbor_y[7]=ceil_y;

    for (int j=0; j<8; ++j)
    {
        int n_x=neighbor_x[j], n_y=neighbor_y[j];
        if (n_x==edge_colIdn && n_y==rowIdn)
            continue;
        if (n_x >= 0 && n_x < width && n_y >= 0 && n_y < height)
        {
            if (edge_indexMat[n_y][n_x] == -1)
            {
                float neighbor_range = edge_rangeMat[n_y][n_x];
                edge_rangeMat[n_y][n_x] = (std::isinf (neighbor_range) ? range : (std::min) (neighbor_range, range));
            }
        }
    }

    float& range_at_image_point = edge_rangeMat[rowIdn][edge_colIdn];
    int& valid = edge_indexMat[rowIdn][edge_colIdn];
    bool addCurrentPoint=false, replace_with_current_point=false;
    
    if (valid==-1)
    {
        replace_with_current_point = true;
    }
    else
    {
        // std::cout << "valid 1, " << rowIdn<<","<<edge_colIdn<<"\n";
        if (range < range_at_image_point)
        {
            replace_with_current_point = true;
            // std::cout << "range: " << range << " range_at_image_point: " << range_at_image_point << "\n";
        }
    }
    
    if (replace_with_current_point)
    {
        range_at_image_point = range;
        if (range >= range_threshold){
            valid = i;
        }
    }
}


void sample_by_reso_comb(vector<array<float, IN_COLS>> &inputPoints, vector<array<float, OUT_COLS>> &outputPoints, 
        int width, int width_sample, int height, float max_angle_height, float **edge_rangeMat, int **edge_indexMat, float range_threshold){

    float angular_resolution_y = max_angle_height / height;
    float angular_resolution_x = 360.0 / width;

    float** reso_rangeMat = new float*[height];
    int** reso_indexMat = new int*[height];
    for(int i=0; i<height; i++) {
        reso_rangeMat[i] = new float[width_sample];
        reso_indexMat[i] = new int[width_sample];
        for(int j=0; j<width_sample; j++) {
            reso_rangeMat[i][j] = INFINITY;
            reso_indexMat[i][j] = -1;
        }
    }

    int pointNum = inputPoints.size();
    int near_points = 0;
    int far_points = 0;
    for(int i=0; i<pointNum; i++){
        array<float, IN_COLS> point = inputPoints[i];
        float range, verticalAngle, horizontalAngle, minDiff, row, column, reso_column;
        int rowIdn, edge_colIdn, reso_colIdn;

        range = sqrt(point[0]*point[0] + point[1]*point[1] + point[2]*point[2]);

        verticalAngle = asin(point[2]/range) * 180 / M_PI;
        row = (verticalAngle + ang_bottom) / angular_resolution_y;
        rowIdn = pcl_lrint(row);
        if (rowIdn >= height){
            if (rowIdn == height)
                rowIdn = height - 1;
            continue;
        }
        if (rowIdn < 0)
            continue;

        // cout << "rowIdn = " << rowIdn << endl;
        horizontalAngle = atan2(point[0], point[1]) * 180 / M_PI;
        column = -(horizontalAngle-90.0)/360 * width + width/2;
        reso_column = -(horizontalAngle-90.0)/360 * width_sample + width_sample/2;
        edge_colIdn = pcl_lrint(column);
        reso_colIdn = pcl_lrint(reso_column);
        if (edge_colIdn >= width){
            edge_colIdn -= width;
            column -= width;
        }

        if (edge_colIdn < 0)
            cout << "edge_colIdn = " << edge_colIdn << endl;

        if (reso_colIdn >= width_sample)
            reso_colIdn -= width_sample;        

        if (range < edge_rangeMat[rowIdn][edge_colIdn]) {
            edge_rangeMat[rowIdn][edge_colIdn] = range;
            if (range >= range_threshold){
                edge_indexMat[rowIdn][edge_colIdn] = i;
            }
        }
        
        if (range < range_threshold){
            if (range < reso_rangeMat[rowIdn][reso_colIdn]) {
                reso_rangeMat[rowIdn][reso_colIdn] = range;
                reso_indexMat[rowIdn][reso_colIdn] = i;
            }
        }

        RI_interpolation(column, row, i, range, range_threshold, width, height, edge_rangeMat, edge_indexMat, rowIdn, edge_colIdn);
    }

    outputPoints.clear();
    for(int i=0; i<height; i++){
        for(int j=0; j<width_sample; j++){
            if (reso_indexMat[i][j] != -1){
                array<float, IN_COLS> inPoint = inputPoints[reso_indexMat[i][j]];
                array<float, OUT_COLS> outPoint;
                copy(inPoint.begin(), inPoint.begin()+OUT_COLS, outPoint.begin());
                outputPoints.push_back(outPoint);
            }
        }
        delete[] reso_rangeMat[i];
        delete[] reso_indexMat[i];
    } 
    delete[] reso_rangeMat;
    delete[] reso_indexMat;
    // cout << "   Raw; near points: " << near_points << " far points: " << far_points << endl;
    // save edge_rangeMat as txt file
    // ofstream edge_file("../edge_rangeMat_RIwointerpolate.txt");
    // for(int i=0; i<height; i++){
    //     for(int j=0; j<width; j++){
    //         edge_file << edge_rangeMat[i][j] << " ";
    //     }
    //     edge_file << endl;
    // }
    // edge_file.close();
}



void sampling_multi_resolution(vector<array<float, IN_COLS>> &inPoints, vector<array<float, OUT_COLS>> &outPoints, 
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
        array<float, IN_COLS> point = inPoints[i];
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
        rowIdn = pcl_lrint(row);
        if (rowIdn >= height){
            if (rowIdn == height)
                rowIdn = height - 1;
            continue;
        }
        if (rowIdn < 0)
            continue;
        
        horizontalAngle = atan2(point[0], point[1]) * 180 / M_PI;
        reso_column = -(horizontalAngle-90.0)/360 * width_sample_list[range_interval] + width_sample_list[range_interval]/2;
        reso_colIdn = pcl_lrint(reso_column);
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
                    array<float, IN_COLS> inPoint = inPoints[reso_indexMat[i][j][k]];
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