<br />
<div align="center">
  <h1 align="center">ML4SYS Project -  Cirrus: Efficient Combinational Poitn Cloud Sampling Method Using ML</h1>
</div>

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Installation
Our project is dependent on several different libraries. We provide a [Dockerfile]() to build an image.

Although we provided the sample point cloud frame(`sample_data/raw.bin`), you can download the entire KITTI 3D point cloud dataset [HERE](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d).
The directory structure for the dataset should be as follows:
```
/data/3d/kitti/training/
|--- calib/
|--- image_2/
|--- label_2/
|--- velodyne/
```

## How to run
1. Build a sampling program.
```bash
mkdir build
cd build
cmake ..
make
cd ..
```

2. Setup the experiment configuartions in `ray_train.py`.
* `NAME`: Search algorithm to use. __Should be `"bayes"`.__
* `RAW_F1`: Average F1 score of raw point cloud.
* `F1_THRESHOLD`: Allowed F1 score drop by sampling out points.
* `WIDTH_COUNT`, `DIFF_COUNT`: Number of range intervals to apply different sampling configuration value. __Should be 6.__
* `WIDTH_MIN`, `WIDTH_MAX`, `DIFF_MIN`, `DIFF_MAX`: Minimum and maximum value of search space.
* `MLFS_ROOT`: Root directory of the experiment. The directory structure should be as follows:
```
/data/3d/mlfs/bayes/
|--- flag/
|   |--- post_eval/
|   |--- post_infer/
|   |--- pre_eval/
|   |--- pre_infer/
|--- post_eval/
|--- post_infer/
|--- pre_infer/
|--- config_idx.json
```

3. Run `ray_train.py`.
```bash
conda activate base
python ray_train.py
```
If it fails to find the sampling program, modify `ray_train.py: line 71` to proper absolute path.

4. Run inference daemon.
```bash
conda activate base
python inference.py --name bayes --device $GPU_NUM
```
If it fails to find the model configuration or checkpoint file, check `inference.py: line 17 & 25`.

5. Run evaluation daemon.
```bash
conda activate pytorch3d
python evaluate.py --name bayes
```

Example video: `docs/implementation.mov`
![](docs/implementation.mov)
<video width="600" controls>
<source src="docs/implementation.mov" type="video/quicktime"> </video>