# BVMatch: Lidar-Based Place Recognition Using Bird's-Eye View Images

BVMatch is a LiDAR-based place recognition method that is capable of estimating 2D relative poses. It projects LiDAR scans to BV images and extracs BVFT descriptors from the images. Place recognition is achieved using bag-of-words approach, and the relative pose is computed through BVFT descriptor matching. This repo contains the source codes of the BV image generation and the BVFT descriptor construction.  

## Dependencies

`OpenCV >= 3.3`

`Eigen`


## Example usage
Go to the "build" directory, run 

`cmake .. && make` 
`./match_two_scan ../data/scan1.bin ../data/scan2.bin`  

You will see the matching result of two LiDAR scans of the Oxford RobotCar dataset,


## Citation
Please cite this paper if you want to use it in your work,
```bibtex
@article{luo2021bvmatch,
  author={Luo, Lun and Cao, Si-Yuan and Han, Bin and Shen, Hui-Liang and Li, Junwei},
  journal={IEEE Robotics and Automation Letters}, 
  title={BVMatch: Lidar-Based Place Recognition Using Bird's-Eye View Images}, 
  year={2021},
  volume={6},
  number={3},
  pages={6076-6083},
  doi={10.1109/LRA.2021.3091386}
}
```

## Contact
Lun Luo

Zhejiang University

luolun@zju.edu.cn