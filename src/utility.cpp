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

