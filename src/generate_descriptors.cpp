#include <opencv2/opencv.hpp>
#include <string>
#include <iostream>
#include <Eigen/Eigen>
#include <vector>
#include <chrono>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>
#include <fstream>
#include "bvftdescriptors.h"

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>

#include <python2.7/Python.h>

using namespace cv;
using namespace std;
int readPointCloud(pcl::PointCloud<pcl::PointXYZ> &point_cloud, const std::string filename)
{
  point_cloud.clear();
  std::ifstream binfile(filename.c_str(),std::ios::binary);
  if(!binfile)
  {
    throw std::runtime_error("file cannot open");
    return -1;
  }
  else
  {
    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
    vector<float> tmp;
    while(1)
    {
      short   s;
      
      pcl::PointXYZ point;
      binfile.read((char*)&s,sizeof(short));
      
      if(binfile.eof()) break;
      point.x = s*0.005;
      binfile.read((char*)&s,sizeof(short));
      point.y = s*0.005;
      binfile.read((char*)&s,sizeof(short));
      point.z = s*0.005;

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
int imagePadding(Mat& img, int &cor_x, int & cor_y)
{
    copyMakeBorder(img, img, pad_size/2,pad_size/2,pad_size/2,pad_size/2, BORDER_CONSTANT, Scalar(10));
    
    //Extending image
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



void generateImage(pcl::PointCloud<pcl::PointXYZ> &point_cloud, int &x_max_ind, int &y_max_ind, Mat &mat_local_image)
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
    //std::cout << x_min << ' ' << x_max << ' ' <<  y_min << ' ' << y_max << std::endl;
    //std::cout << x_min_ind << ' ' << x_max_ind << ' ' <<  y_min_ind << ' ' << y_max_ind << std::endl;

    int x_num = x_max_ind-x_min_ind+1;
    int y_num = y_max_ind-y_min_ind+1;
    mat_local_image=Mat( y_num, x_num,CV_8UC1, cv::Scalar::all(0));

    for(int i=0; i<point_cloud.size(); i++)
    {
        int x_ind = x_max_ind-int((point_cloud.points[i].y)/resolution);
        int y_ind = y_max_ind-int((point_cloud.points[i].x)/resolution);
        if(x_ind>=x_num || y_ind>=y_num ) continue;
        mat_local_image.at<uint8_t>( y_ind,x_ind) += 1;
        //std::cout << int(mat_local_image.at<uint8_t>(x_ind, y_ind)) << std::endl;
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
            if(uint8_t(mat_local_image.at<uint8_t>(j, i)*10)>122) {mat_local_image.at<uint8_t>(j, i)=122;continue;}
            mat_local_image.at<uint8_t>(j, i)=uint8_t(mat_local_image.at<uint8_t>(j, i)*10);//1.0/max_pixel*255);
            if(uint8_t(mat_local_image.at<uint8_t>(j, i))==0) {mat_local_image.at<uint8_t>(j, i)=10;continue;}
        }
}

int main(int argc, char** argv)
{
    string bin_path = argv[1];
    string  seq = argv[2];

    vector<string> bin_file_names = pyListSortDir(bin_path);
    cout << bin_file_names.size() << endl;

    chrono::steady_clock::time_point t_start = chrono::steady_clock::now();
    for(int i=0; i<bin_file_names.size(); i++)
    {
        
        pcl::PointCloud<pcl::PointXYZ> point_cloud;
        int max_local_x_ind,max_local_y_ind;
        Mat mat_local_image;


        readPointCloud(point_cloud, bin_path+bin_file_names[i]);

        //generate BV image, recording the cornet points
        generateImage(point_cloud, max_local_x_ind, max_local_y_ind, mat_local_image);

        //padding to make fft faster
        cout << "processing: " << i << "/" << bin_file_names.size() << "," << bin_path  << endl;
        imagePadding(mat_local_image, max_local_x_ind, max_local_y_ind);

        BVFT bvft = detectBVFT(mat_local_image);

        //save descriptor to mat
        writeMatToBin(bvft.descriptors,(seq+"des_"+to_string(10000+i)+"_"+to_string(max_local_x_ind)+","+to_string(max_local_y_ind)+".bin").c_str());
    }
    chrono::steady_clock::time_point t_end_3 = chrono::steady_clock::now();
    chrono::duration<float> time_used=chrono::duration_cast<
    chrono::duration<float>>(t_end_3-t_start);
    cout <<seq << " size " << bin_file_names.size() << " dectection uses " << time_used.count()  << " seconds"<< endl;
}
