
#include <opencv2/opencv.hpp>
#include <string>
#include <iostream>
#include <Eigen/Eigen>
#include <vector>
#include <chrono>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>
#include "bvftdescriptors.h"
#include <queue>
#include <fstream>
#include <string>
#include <math.h>
#include <stdlib.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <python2.7/Python.h>


using namespace std;
using namespace cv;
using namespace Eigen;


int readPointCloud(pcl::PointCloud<pcl::PointXYZ> &point_cloud, const std::string filename)
{
  point_cloud.clear();
  std::ifstream binfile(filename.c_str(),std::ios::binary);
  if(!binfile)
  {
    throw std::runtime_error("file \"" +filename+ "\" cannot open");
    return -1;
  }
  else
  {
    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
    vector<float> tmp;
    while(1)
    {
      double s;
      pcl::PointXYZ point;
      binfile.read((char*)&s,sizeof(double));
      if(binfile.eof()) break;
      tmp.push_back(s);
      point.x = s;
      binfile.read((char*)&s,sizeof(double));
      tmp.push_back(s);
      point.y = s;
      binfile.read((char*)&s,sizeof(double));
      tmp.push_back(s);
      point.z = s;
      point_cloud.push_back(point);
   }

   std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
  }
  binfile.close();
  return 1;
}

std::vector<std::string> pyListSortDir(std::string path)
{
    std::vector<std::string> ret;    

    Py_Initialize();

    //PyRun_SimpleString("import os");
    PyObject* module_name = PyString_FromString("os");
    PyObject* os_module = PyImport_Import(module_name);
    PyObject* os_list = PyObject_GetAttrString(os_module, "listdir");
    //PyObject* os_list = PyObject_GetAttrString(os_module, "");

    PyObject *ArgList = PyTuple_New(1);
    PyObject* py_path = PyString_FromString(path.c_str());
    PyTuple_SetItem(ArgList, 0, py_path);

    PyObject* files = PyObject_CallObject(os_list, ArgList);
    //PyObject* 
    PyList_Sort(files);
    //PyObject_CallMethod(PyList,"sort", 'O', files);
    //PyObject* files_sort_ = PyObject_CallObject(files_sort,ArgList);
    for(int i=0; i<PyList_Size(files); i++)
    {
        char  *temp;
        PyObject* item = PyList_GetItem(files,i);
        //std::cout << item << std::endl;
        PyArg_Parse(item,"s",&temp);
        //std::cout << std::string(temp) << std::endl;
        ret.push_back(std::string(temp));
    }
    //PyObject* re = PyRun_SimpleString("os.list()");
    Py_Finalize();
    return ret;

}

void generateBVImage(pcl::PointCloud<pcl::PointXYZ> &point_cloud, int &x_max_ind, int &y_max_ind, Mat &mat_local_image)
{
    float resolution = 0.4;
    pcl::VoxelGrid<pcl::PointXYZ> down_size_filter;
    down_size_filter.setLeafSize(resolution, resolution, resolution/2);
    down_size_filter.setInputCloud(point_cloud.makeShared());
    down_size_filter.filter(point_cloud);

    float x_min = 10000, y_min = 10000, x_max=-100000,y_max=-100000;
    for(int i=0; i<point_cloud.size(); i++)
    {
        if(point_cloud.points[i].y< x_min) x_min=point_cloud.points[i].y;
        if(point_cloud.points[i].y> x_max) x_max=point_cloud.points[i].y;
        if(point_cloud.points[i].x< y_min) y_min=point_cloud.points[i].x;
        if(point_cloud.points[i].x> y_max) y_max=point_cloud.points[i].x;
    }
    int x_min_ind = int(x_min/resolution);
    x_max_ind = int(x_max/resolution);
    int y_min_ind = int(y_min/resolution);
    y_max_ind = int(y_max/resolution);

    int x_num = x_max_ind-x_min_ind+1;
    int y_num = y_max_ind-y_min_ind+1;
    mat_local_image=Mat( y_num, x_num,CV_8UC1, cv::Scalar::all(0));

    for(int i=0; i<point_cloud.size(); i++)
    {
        int x_ind = x_max_ind-int((point_cloud.points[i].y)/resolution);
        int y_ind = y_max_ind-int((point_cloud.points[i].x)/resolution);
        if(x_ind>=x_num || y_ind>=y_num ) continue;
        mat_local_image.at<uint8_t>( y_ind,x_ind) += 1;
    }
    uint8_t max_pixel = 0;
    for(int i=0; i<x_num; i++)
        for(int j=0; j<y_num; j++)
        {
            if (mat_local_image.at<uint8_t>(j, i)>max_pixel) max_pixel=mat_local_image.at<uint8_t>(j, i);
        }
    for(int i=0; i<x_num; i++)
        for(int j=0; j<y_num; j++)
        {
            if(mat_local_image.at<uint8_t>(j, i)*10>100) {mat_local_image.at<uint8_t>(j, i)=100;continue;}
            mat_local_image.at<uint8_t>(j, i)=uint8_t(mat_local_image.at<uint8_t>(j, i)*10);
            if(uint8_t(mat_local_image.at<uint8_t>(j, i))==0) {mat_local_image.at<uint8_t>(j, i)=10;continue;}
        }
}


Mat max_moment_local;
Mat MIM_global;
Mat MIM_local;

int max_global_x_ind;
int max_global_y_ind;

int imagePadding(Mat& img, int &cor_x, int & cor_y)
{
    pad_size=200;
    copyMakeBorder(img, img, pad_size/2,pad_size/2,pad_size/2,pad_size/2, BORDER_CONSTANT, Scalar(10));
    
	int m = getOptimalDFTSize(img.rows);
	int n = getOptimalDFTSize(img.cols);

    int row_pad = (m - img.rows)/2;
    int col_pad = (n - img.cols)/2;
    //take this step to make fft faster
    copyMakeBorder(img, img,row_pad , (m - img.rows)%2?row_pad+1:row_pad, 
                col_pad, (n - img.cols)%2?col_pad+1:col_pad , BORDER_CONSTANT, Scalar(10));
    cor_x += col_pad+pad_size/2;
    cor_y += row_pad+pad_size/2;
}


void match2Image(Mat img1, Mat img2, int max_x1, int max_y1, int max_x2 , int max_y2)
{
    float rows = img1.rows, cols = img1.cols;
    BVFT bvfts1 = detectBVFT(img1);
    BVFT bvfts2 = detectBVFT(img2);

    bvfts2.keypoints.insert(bvfts2.keypoints.end(),bvfts2.keypoints.begin(),bvfts2.keypoints.end());


    Mat temp(bvfts2.keypoints.size(),bvfts2.descriptors.cols,CV_32F,Scalar{0} );
    bvfts2.descriptors.copyTo(temp(Rect(0,0,bvfts2.descriptors.cols,bvfts2.keypoints.size()/2)));
    int areas = 6;
    int feautre_size=bvfts2.descriptors.cols/areas/areas;
    for(int i=0; i<areas*areas; i++)  //areas*areas
    {
        bvfts2.descriptors(Rect((areas*areas-i-1)*feautre_size,0 , feautre_size,bvfts2.keypoints.size()/2)).copyTo(temp(Rect(i*feautre_size, bvfts2.keypoints.size()/2, feautre_size , bvfts2.keypoints.size()/2))); //i*norient
    }
    bvfts2.descriptors = temp.clone();

    BFMatcher matcher;//(NORM_L2, true);
	vector<DMatch> matches;
	matcher.match(bvfts1.descriptors, bvfts2.descriptors, matches);

    vector<Point2f> points1;
    vector<Point2f> points2;
    vector<DMatch>::iterator it_end = matches.end();
    for(vector<DMatch>::iterator it= matches.begin(); it!= it_end;it++)
    {
        Point2f point_local_1 = (Point2f(max_y1,max_x1)- 
                                    Point2f(bvfts1.keypoints[it->queryIdx].pt.y,bvfts1.keypoints[it->queryIdx].pt.x))*0.4;
        points1.push_back(point_local_1);

        point_local_1 = (Point2f(max_y2,max_x2)- 
                                    Point2f(bvfts2.keypoints[it->trainIdx].pt.y,bvfts2.keypoints[it->trainIdx].pt.x))*0.4;
        points2.push_back(point_local_1);
    }
    cv::Mat keypoints1(points1);
    keypoints1=keypoints1.reshape(1,keypoints1.rows); //N*2
    cv::Mat keypoints2(points2);
    keypoints2=keypoints2.reshape(1,keypoints2.rows);

    Mat inliers;

    vector<int> inliers_ind;
    Mat rigid = estimateICP(keypoints1, keypoints2, inliers_ind);

    


    if (inliers_ind.size()<4) cout << "few inlier points" << endl;
    cout << "find transform: \n" <<  rigid << endl;
    //return;
    
    vector<DMatch> good_matches;
    
    for(int i=0; i<inliers_ind.size(); i++)
      good_matches.push_back(matches[inliers_ind[i]]);

    
    Mat matchesGoodImage;

    for(int i=0; i<img1.rows; i++)
      for(int j=0; j<img1.cols; j++)
        if(img1.ptr<uint8_t>(i)[j]<=10) img1.ptr<uint8_t>(i)[j]=10;

    for(int i=0; i<img2.rows; i++)
      for(int j=0; j<img2.cols; j++)
        if(img2.ptr<uint8_t>(i)[j]<=10) img2.ptr<uint8_t>(i)[j]=10;

    normalize(img1, img1, 0, 255, CV_MINMAX);
    normalize(img2, img2, 0, 255, CV_MINMAX);

    drawKeypoints(img1, bvfts1.keypoints,img1,Scalar(0,0,255),DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    drawKeypoints(img2, bvfts2.keypoints,img2,Scalar(0,0,255),DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

    drawMatches(img1, bvfts1.keypoints, img2, bvfts2.keypoints, good_matches, matchesGoodImage, Scalar::all(-1), 
		Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    
    imshow("matchesGoodImage", matchesGoodImage);  
    imwrite("match.png",matchesGoodImage);
    waitKey(0);
}



int main(int argc, char** argv)
{

    
    string img1_path = argv[1];
    string img2_path = argv[2];
    
    pcl::PointCloud<pcl::PointXYZ> point_cloud1;
    pcl::PointCloud<pcl::PointXYZ> point_cloud2;

    //read point clouds
    readPointCloud(point_cloud1, img1_path);
    readPointCloud(point_cloud2, img2_path);

    //apply rotation transform to test the rotation invariance
    float rotation_angle=137.0/180*CV_PI;
    for(int i=0; i<point_cloud1.size(); i++)
    {
        pcl::PointXYZ tmp = point_cloud1[i];
        point_cloud1[i].x = cos(rotation_angle)*tmp.x + sin(rotation_angle)*tmp.y;
        point_cloud1[i].y = -sin(rotation_angle)*tmp.x + cos(rotation_angle)*tmp.y;
    }

    Mat img1, img2;
    int max_x1,max_y2,max_x2,max_y1;  //used for localizing the center of the images

    //generate bv images
    generateBVImage(point_cloud1, max_x1, max_y1, img1);
    generateBVImage(point_cloud2, max_x2, max_y2, img2);

    //padding to make fft faster
    imagePadding(img1, max_x1, max_y1);
    imagePadding(img2, max_x2, max_y2);

    //perform matching
    match2Image(img1, img2, max_x1,max_y1,max_x2,max_y2);  
}

