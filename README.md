## Installation
Our code is implemented in Python 3.8, PyTorch 1.11.0 and CUDA 11.3.
- Install python Dependencies
```bash
conda create -n capudf python=3.8
conda activate capudf
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
pip install tqdm pyhocon==0.3.57 trimesh PyMCubes scipy point_cloud_utils==0.29.7
```
- Compile C++ extensions
```
cd extensions/chamfer_dist
python setup.py install
```

## Dataset
We provide the processed data for ShapeNetCars, 3DScenes and SRB dataset [here](https://drive.google.com/file/d/1BNzmd_OX0s4lxta3FFeRWRYwfGQOALS_/view?usp=sharing). Unzip it to the `./data` folder. The datasets is organised as follows:
```
│data/
├──shapenetCars/
│  ├── input
│  ├── ground_truth
│  ├── query_data
├──3dscene/
│  ├── input
│  ├── ground_truth
│  ├── query_data
├──srb/
│  ├── input
│  ├── ground_truth
│  ├── query_data
```
We provide all data of the 3DScenes and SRB dataset, and two demos of the ShapeNetCars. The full set data of ShapeNetCars will be uploaded soon.

## Train
You can train our CAP-UDF to reconstruct surfaces from a single point cloud as:

- ShapeNetCars
```
python run.py --gpu 0 --conf confs/shapenetCars.conf --dataname 3e5e4ff60c151baee9d84a67fdc5736 --dir 3e5e4ff60c151baee9d84a67fdc5736
```

- 3DScene
```
python run.py --gpu 0 --conf confs/3dscene.conf --dataname lounge_1000 --dir lounge_1000
```

- SRB
```
python run.py --gpu 0 --conf confs/srb.conf --dataname gargoyle --dir gargoyle
```

You can find the generated mesh and the log in `./outs`.

## Test
You can evaluate the reconstructed meshes and dense point clouds as follows:

- ShapeNetCars
```
python evaluation/shapenetCars/eval_mesh.py --conf confs/shapenetCars.conf --dataname 3e5e4ff60c151baee9d84a67fdc5736 --dir 3e5e4ff60c151baee9d84a67fdc5736
```

- 3DScene
```
python evaluation/3dscene/eval_mesh.py --conf confs/3dscene.conf --dataname lounge_1000 --dir lounge_1000
```

- SRB
```
python evaluation/srb/eval_mesh.py --conf confs/srb.conf --dataname gargoyle --dir gargoyle
```

For evaluating the generated dense point clouds, you can run the `eval_pc.py` of each dataset instead of `eval_mesh.py`. 

## Use Your Own Data
We also provide the instructions for training your own data in the following.

### Data
First, you should put your own data to the `./data/owndata/input` folder. The datasets is organised as follows:
```
│data/
├──shapenetCars/
│  ├── input
│      ├── (dataname).ply/xyz/npy
```
We support the point cloud data format of `.ply`, `.xyz` and `.npy`

### Run
To train your own data, simply run:
```
python run.py --gpu 0 --conf confs/base.conf --dataname (dataname) --dir (dataname)
```

### Notice

In different datasets or your own data, because of the variation in point cloud density, this hyperparameter [scale](https://github.com/junshengzhou/CAP-UDF/blob/eb22ffd4b3f34d4fa74e1863ece7ff37d63d03cc/models/dataset.py#L51) has a very strong influence on the final result, which controls the distance between the query points and the point cloud. So if you want to get better results, you should adjust this parameter. We give `0.25 * np.sqrt(POINT_NUM_GT / 20000)` here as a reference value, and this value can be used for most object-level reconstructions. 

## Citation
If you find our code or paper useful, please consider citing

    @inproceedings{Zhou2022CAP-UDF,
        title = {Learning Consistency-Aware Unsigned Distance Functions Progressively from Raw Point Clouds},
        author = {Zhou, Junsheng and Ma, Baorui and Liu, Yu-Shen and Fang, Yi and Han, Zhizhong},
        booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
        year = {2022}
    }
