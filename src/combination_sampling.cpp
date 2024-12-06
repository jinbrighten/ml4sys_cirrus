#include "knob.hpp"
#include "utility.hpp"

#include <dirent.h>
#include <ctime>

static string frame_num = "000000";
static array<int, 6> width_sample_list = {100, 200, 300, 400, 500, 1000};
static array<int, 6> diff_thr_list = {10, 5, 0, 0, 0, 0};

static string input_path = "/data/3d/kitti/training/velodyne/";
static string output_path = "/data/3d/kitti_sampled/training/pre_infer/ml4sys/";


static void print_help(string program_name) {
    cout << "Options:" << endl;
    cout << "   -w: size of the width for each distance (horizontal number of bins)." << endl;
    cout << "   -d: difference threshold for each distance." << endl;
    cout << "   -h: print this page." << endl;
}

static std::array<int, 6> parse_int_list(const char* optarg) {
  std::string token;
  std::stringstream ss(optarg);
  std::array<int, 6> values;

  for (int i = 0; i < 6; i++) {
    if (std::getline(ss, token, ',')) {
      values[i] = std::stoi(token);
    }
  }

  return values;
}


static void parse_opt(int argc, char **argv) {
    int c;
    while ((c=getopt(argc, argv, "hw:d:")) != -1) {
        switch (c) {
            case 'w':
                width_sample_list = parse_int_list(optarg);
                break;
            case 'd':
                diff_thr_list = parse_int_list(optarg);
                break;
            case 'h':
                print_help(argv[0]);
                exit(0);
        }
    }
}

int main (int argc, char** argv) {
    parse_opt(argc, argv);
    string save_path;
    string arguments;
    for (int i = 0; i < width_sample_list.size(); i++) {
        arguments += to_string(width_sample_list[i]) + "_";
    }
    for (int i = 0; i < diff_thr_list.size(); i++) {
        arguments += to_string(diff_thr_list[i]) + "_";
    }
    save_path = output_path + arguments + "/";
    mkpath(save_path);

    int chunk_size = 100;
    int width = 1000;
    int height = 100;
    int row_size = 5;
    float maxAngleWidth = 360.0;
    float maxAngleHeight = 28.0;
    float curv_threshold = 0.5;
    int nearby_points = 50;
    float space_saving = 0;


    vector<string> file_names = listDir(input_path);
    int i = 0;
    for (const auto &file_name: file_names) {
        if (i > chunk_size) break;
        if (file_name.size()<=4 || !endsWith(file_name, ".bin")) 
            continue;
        i++;
        cout << "Processing " << file_name << endl;
        
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
        vector<array<float, OUT_COLS>> fore_outPoints;
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

        sampling_foreground(inPoints, rangeMat, indexMat, row_size, nearby_points, height, width, curv_threshold, diff_thr_list, fore_outPoints);
        sampling_multi_resolution(fore_outPoints, outPoints, width_sample_list, height, maxAngleHeight);
        space_saving = (1 - (float)outPoints.size() / inPoints.size()) * 100;
        cout << "Saving " << save_path + file_name << endl;
        cout << "space_saving: " << space_saving << endl;
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
        fore_outPoints.clear();
        outPoints.clear();
        for(int i=0; i<height; i++) {
            delete[] rangeMat[i];
            delete[] indexMat[i];
            }
        delete[] rangeMat;
        delete[] indexMat;
    }
}