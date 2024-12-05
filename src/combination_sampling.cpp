#include "knob.hpp"
#include "utility.hpp"

#include <dirent.h>
#include <ctime>

static int width_sample = 100;
static int diff_threshold = 5;
static float range_threshold = 20;
static string frame_num = "000000";
static bool verbose = false;
static bool frame_mode = false;
static string input_path = "/data/3d/kitti/training/velodyne/";
static string output_path = "/data/3d/kitti_sampled/training/pre_infer/comb_knob/";

static void print_help(string program_name) {
    cout << "Options:" << endl;
    cout << "   -v: verbose print." << endl;
    cout << "   -w: width of the range image (horizontal number of bins)." << endl;
    cout << "   -d: difference threshold." << endl;
    cout << "   -r: range threshold." << endl;
    cout << "   -f: frame number." << endl;
    cout << "   -h: print this page." << endl;
}

static void parse_opt(int argc, char **argv) {
    int c;
    while ((c=getopt(argc, argv, "vhw:d:r:f:")) != -1) {
        switch (c) {
            case 'w':
                width_sample_list = parse_int_list(optarg);
                break;
            case 'd':
                diff_threshold = atoi(optarg);
                break;
            case 'r':
                range_threshold = atof(optarg);
                break;
            case 'f':
                frame_mode = true;
                frame_num = optarg;
                break;
            case 'h':
                print_help(argv[0]);
                exit(0);
        }
    }
    if (verbose) {
        cout << "Combination knob; Sampling points selected by its range value." << endl;
    }
}

int main (int argc, char** argv) {
    parse_opt(argc, argv);
    string save_path;
    save_path = output_path + to_string(width_sample)+"/"+to_string(diff_threshold)+"/"+to_string(int(range_threshold))+"/";
    mkpath(save_path);
    if (verbose) {
        cout << "----------------------------------------" << endl;
        cout << "range_threshold: " << range_threshold << endl;
        cout << "width_sample: " << width_sample << endl;
        cout << "diff_threshold: " << diff_threshold << endl;
        cout << "input_path: " << input_path << endl;
        cout << "save_path: " << save_path << endl;
    }

    int width = 4500;
    int height = 100;
    int row_size = 5;
    float maxAngleWidth = 360.0;
    float maxAngleHeight = 28.0;
    float curv_threshold = 0.5;
    int nearby_points = 50;

    float time_start = 0;
    float time_end = 0;
    float space_saving = 0;
    float latency = 0;
    float assign_latency = 0;
    float reso_latency = 0;
    float edge_latency = 0;
    float reso_start = 0;
    float edge_start = 0;

    vector<string> file_names = listDir(input_path);
    int chunk_size = 500;
    int i = 0;
    for (const auto &file_name: file_names) {
        i++;
        if (file_name.size()<=4 || !endsWith(file_name, ".bin")) 
            continue;
        
        if (frame_mode) {
            if (!startsWith(file_name, frame_num)) 
                continue;
        }

        string file_path = input_path + file_name;
        ifstream file(file_path, ios::binary | ios::ate);
        int near_points = 0;
        int far_points = 0;

        if(!file) {
            cerr << "Failed to open " << file_path << endl;
            return 1;
        }

        streamsize size = file.tellg();
        file.seekg(0, ios::beg);

        if (size % sizeof(float)*IN_COLS != 0) {
            cerr << "Invalid file size" << endl;
            return 1;
        }
        
        int num_points = size / (sizeof(float)*IN_COLS);
        vector<array<float, IN_COLS>> inPoints(num_points);
        vector<array<float, OUT_COLS>> outPoints;
        for (int i=0; i<num_points; i++) {
            file.read(reinterpret_cast<char*>(inPoints[i].data()), sizeof(float)*IN_COLS);
        }
        file.close();

        float** rangeMat = new float*[height];
        int** indexMat = new int*[height];
        for (int i = 0; i < height; ++i) {
            rangeMat[i] = new float[width];
            indexMat[i] = new int[width];
            for (int j = 0; j < width; ++j) {
                rangeMat[i][j] = INFINITY;
                indexMat[i][j] = -1;
            }
        }

        ROIextraction(inPoints, rangeMat, indexMat, row_size, nearby_points, height, width, curv_threshold, diff_threshold, outPoints);
        sampling_multi_resolution(inPoints, outPoints, width_sample_list, height, maxAngleHeight);
        
        ofstream outFile(save_path + file_name, ios::binary);
        if (!outFile) {
            cerr << "Failed to open output file " << save_path + file_name << endl;
            return 1;
        }
        for (const auto &point: outPoints) { 
            outFile.write(reinterpret_cast<const char*>(point.data()), sizeof(float)*OUT_COLS);
        }
        outFile.close();
        inPoints.clear();
        outPoints.clear();
        for(int i=0; i<height; i++) {
            delete[] rangeMat[i];
            delete[] indexMat[i];
            }
        delete[] rangeMat;
        delete[] indexMat;
    }
    if (verbose) {
        cout << save_path << " is saved." << endl;

    }
}



static int width_sample = 2000;
static int diff_threshold = 0;
static float range_threshold = 20.0;
static string frame_num = "000000";
static bool verbose = false;
static bool frame_mode = false;
static string input_path = "/data/3d/kitti/training/velodyne/";
static string output_path = "/data/3d/kitti_sampled/training/pre_infer/comb_knob/";

static void print_help(string program_name) {
    cout << "Options:" << endl;
    cout << "   -v: verbose print." << endl;
    cout << "   -w: width of the range image (horizontal number of bins)." << endl;
    cout << "   -d: difference threshold." << endl;
    cout << "   -r: range threshold." << endl;
    cout << "   -f: frame number." << endl;
    cout << "   -h: print this page." << endl;
}

static void parse_opt(int argc, char **argv) {
    int c;
    while ((c=getopt(argc, argv, "vhw:d:r:f:")) != -1) {
        switch (c) {
            case 'v':
                verbose = true;
                break;
            case 'w':
                width_sample = atoi(optarg);
                break;
            case 'd':
                diff_threshold = atoi(optarg);
                break;
            case 'r':
                range_threshold = atof(optarg);
                break;
            case 'f':
                frame_mode = true;
                frame_num = optarg;
                break;
            case 'h':
                print_help(argv[0]);
                exit(0);
        }
    }
    if (verbose) {
        cout << "Combination knob; Sampling points selected by its range value." << endl;
    }
}





int main (int argc, char** argv) {
    parse_opt(argc, argv);
    string save_path;
    save_path = output_path + to_string(width_sample)+"/"+to_string(diff_threshold)+"/"+to_string(int(range_threshold))+"/";
    mkpath(save_path);
    if (verbose) {
        cout << "----------------------------------------" << endl;
        cout << "range_threshold: " << range_threshold << endl;
        cout << "width_sample: " << width_sample << endl;
        cout << "diff_threshold: " << diff_threshold << endl;
        cout << "input_path: " << input_path << endl;
        cout << "save_path: " << save_path << endl;
    }

    int width = 4500;
    int height = 100;
    int row_size = 5;
    float maxAngleWidth = 360.0;
    float maxAngleHeight = 28.0;
    float curv_threshold = 0.5;
    int nearby_points = 50;

    float time_start = 0;
    float time_end = 0;
    float space_saving = 0;
    float latency = 0;
    float assign_latency = 0;
    float reso_latency = 0;
    float edge_latency = 0;
    float reso_start = 0;
    float edge_start = 0;

    int chunk_size = 300;
    vector<string> file_names = listDir(input_path);
    int i = 0;
    for (const auto &file_name: file_names) {
        i++;
        if (file_name.size()<=4 || !endsWith(file_name, ".bin")) 
            continue;
        if (i>chunk_size) break;
        
        if (frame_mode) {
            if (!startsWith(file_name, frame_num)) 
                continue;
        }

        string file_path = input_path + file_name;
        ifstream file(file_path, ios::binary | ios::ate);
        int near_points = 0;
        int far_points = 0;

        if(!file) {
            cerr << "Failed to open " << file_path << endl;
            return 1;
        }

        streamsize size = file.tellg();
        file.seekg(0, ios::beg);

        if (size % sizeof(float)*IN_COLS != 0) {
            cerr << "Invalid file size" << endl;
            return 1;
        }
        
        int num_points = size / (sizeof(float)*IN_COLS);
        vector<array<float, IN_COLS>> inPoints(num_points);
        vector<array<float, OUT_COLS>> outPoints;
        for (int i=0; i<num_points; i++) {
            file.read(reinterpret_cast<char*>(inPoints[i].data()), sizeof(float)*IN_COLS);
        }
        file.close();

        time_start = clock();

        float** rangeMat = new float*[height];
        int** indexMat = new int*[height];
        for (int i = 0; i < height; ++i) {
            rangeMat[i] = new float[width];
            indexMat[i] = new int[width];
            for (int j = 0; j < width; ++j) {
                rangeMat[i][j] = INFINITY;
                indexMat[i][j] = -1;
            }
        }   

        reso_start = clock();
        sample_by_reso_comb(inPoints, outPoints, width, width_sample, height, maxAngleHeight, rangeMat, indexMat, range_threshold);
        near_points = outPoints.size();

        edge_start = clock();
        ROIextraction(inPoints, rangeMat, indexMat, row_size, nearby_points, height, width, curv_threshold, diff_threshold, outPoints);
        far_points = outPoints.size() - near_points;
        time_end = clock();
        
        cout << "Sample; near points: " << near_points << " far points: " << far_points << endl;
        ofstream outFile(save_path + file_name, ios::binary);
        if (!outFile) {
            cerr << "Failed to open output file " << save_path + file_name << endl;
            return 1;
        }
        for (const auto &point: outPoints) { 
            outFile.write(reinterpret_cast<const char*>(point.data()), sizeof(float)*OUT_COLS);
        }
        outFile.close();
        space_saving += 1.0 - (float)outPoints.size() / (float)inPoints.size();

        inPoints.clear();
        outPoints.clear();
        for(int i=0; i<height; i++) {
            delete[] rangeMat[i];
            delete[] indexMat[i];
        }
        delete[] rangeMat;
        delete[] indexMat;
        latency += 1000 * (time_end - time_start) / CLOCKS_PER_SEC;
        assign_latency += 1000 * (reso_start - time_start) / CLOCKS_PER_SEC;
        reso_latency += 1000 * (edge_start - reso_start) / CLOCKS_PER_SEC;
        edge_latency += 1000 * (time_end - edge_start) / CLOCKS_PER_SEC;
    }
    if (verbose) {
        if (frame_mode) {
            cout << "Frame: " << frame_num << endl;
            cout << "Space saving: " << space_saving << endl;
            cout << "latency: " << latency  << endl;
            cout << "assign latency: " << assign_latency  << endl;
            cout << "reso latency: " << reso_latency  << endl;
            cout << "edge latency: " << edge_latency  << endl;
        }
        else{
            cout << "Average space saving: " << space_saving / chunk_size << endl;
            cout << "Average latency: " << latency / chunk_size << endl;
            cout << "Average assign latency: " << assign_latency / chunk_size << endl;
            cout << "Average reso latency: " << reso_latency / chunk_size << endl;
            cout << "Average edge latency: " << edge_latency / chunk_size << endl;
        }
    }
}