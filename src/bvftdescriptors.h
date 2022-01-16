
#ifndef bvftdescriptors_h
#define bvftdescriptors_h
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
int norient = 12;
int nscale = 4;
int pad_size = 138;
using namespace std;
using namespace cv;
using namespace Eigen;

class BVFT{
    public:
    BVFT(){}
    BVFT(vector<KeyPoint>& keypoints, Mat& descriptors)
    :keypoints(keypoints),descriptors(descriptors){}

    vector<KeyPoint> keypoints; // keypoints coordinates
    Mat descriptors;            // keypoint descriptors
    Mat angle;                  // dominant orientations
};


//lesat square ICP using SVD
Mat get_trans_icp(const Mat& src_, const Mat& dst_)
{
    chrono::steady_clock::time_point t1=chrono::steady_clock::now();

    //3*N
    int used_points = src_.cols;
    int samples = used_points;
    //normalize
    Mat src_x = src_.row(0);
    Mat src_y = src_.row(1);
    Mat dst_x = dst_.row(0);
    Mat dst_y = dst_.row(1);
    float mean_src_x = sum(src_x)[0]/used_points;
    float mean_src_y = sum(src_y)[0]/used_points;
    float mean_dst_x = sum(dst_x)[0]/used_points;
    float mean_dst_y = sum(dst_y)[0]/used_points;

    Mat temp_src=src_.clone();
    Mat temp_dst=dst_.clone();
    temp_src.row(0) = src_x-mean_src_x;
    temp_src.row(1) = src_y-mean_src_y;
    temp_dst.row(0) = dst_x-mean_dst_x;
    temp_dst.row(1) = dst_y-mean_dst_y;

    Mat temp_src_x = temp_src.row(0);
    Mat temp_src_y = temp_src.row(1);
    Mat temp_dst_x = temp_dst.row(0);
    Mat temp_dst_y = temp_dst.row(1);

    Mat temp_sqrt_src, temp_sqrt_dst;
    sqrt(temp_src_x.mul(temp_src_x) + temp_src_y.mul(temp_src_y),temp_sqrt_src);
    sqrt(temp_dst_x.mul(temp_dst_x) + temp_dst_y.mul(temp_dst_y),temp_sqrt_dst);
    float mean_src_dis = sum(temp_sqrt_src)[0]/used_points;
    float mean_dst_dis = sum(temp_sqrt_dst)[0]/used_points;

    float src_sf = sqrt(2)/mean_src_dis;
    float dst_sf = sqrt(2)/mean_dst_dis;

    Mat src_trans  =(Mat_<float>(3,3)<<src_sf,0,-src_sf*mean_src_x,
                                        0,src_sf,-src_sf*mean_src_y,
                                        0,0,1);
    Mat dst_trans  =(Mat_<float>(3,3)<<dst_sf,0,-dst_sf*mean_dst_x,
                                        0,dst_sf,-dst_sf*mean_dst_y,
                                        0,0,1);

    temp_src = src_trans*src_;
    temp_dst = dst_trans*dst_;
    chrono::steady_clock::time_point t2=chrono::steady_clock::now();
    chrono::duration<double> ti = chrono::duration_cast<chrono::duration<double>>(t2-t1);
    t1=chrono::steady_clock::now();

    //SVD 
    Mat A(2*samples, 4, CV_32F, Scalar{0});
    Mat xp(2*samples, 1, CV_32F, Scalar{0});
    const float* temp_src_ptr_x =  temp_src.ptr<float>(0);
    const float* temp_src_ptr_y =  temp_src.ptr<float>(1);
    const float* temp_dst_ptr_x =  temp_dst.ptr<float>(0);
    const float* temp_dst_ptr_y =  temp_dst.ptr<float>(1);
    for(int k=0; k<samples; k++)
    {
        
        Mat temp =(Mat_<float>(1,4) << temp_src_ptr_x[k],-temp_src_ptr_y[k],
                                        1,0);
        temp.copyTo(A.row(2*k)); 
        temp = (Mat_<float>(1,4) << temp_src_ptr_y[k],temp_src_ptr_x[k],
                                        0,1);
        temp.copyTo(A.row(2*k+1));// = ;
        xp.ptr<float>(2*k)[0] = temp_dst_ptr_x[k];
        xp.ptr<float>(2*k+1)[0] = temp_dst_ptr_y[k];
    }

    Mat D_t;
    Mat U(samples*2, samples*2, CV_32F, Scalar{0});
    Mat D(samples*2, 4, CV_32F, Scalar{0});
    Mat V(4, 4, CV_32F, Scalar{0});
    SVDecomp(A, D_t,U,V);

    D = Mat::diag(D_t);

    Mat h = V*(1.0/D)*U.t()*xp;

    Mat H = (Mat_<float>(3,3) << h.ptr<float>(0)[0], -h.ptr<float>(1)[0], h.ptr<float>(2)[0],
                                h.ptr<float>(1)[0], h.ptr<float>(0)[0], h.ptr<float>(3)[0],
                                0,0,1);
    Mat inv_dst_trans;
    invert(dst_trans,inv_dst_trans);

    H = inv_dst_trans*H*src_trans;

    SVDecomp(H(Rect(0,0,2,2)), D,U,V);

    Mat S=Mat::eye(2, 2, CV_32F);
    if (determinant(U)*determinant(V)<0)  S.ptr<float>(1)[1] = -1;
    H(Rect(0,0,2,2))=U*S*V.t();
    t2=chrono::steady_clock::now();
    ti = chrono::duration_cast<chrono::duration<double>>(t2-t1);
    return H;

}

//lesat square ICP using SVD
Mat get_trans_icp_3_points(const Mat& src_, const Mat& dst_)
{
    chrono::steady_clock::time_point t1=chrono::steady_clock::now();

    int samples = 3;

    const float * src_ptr_x = src_.ptr<float>(0);
    const float * src_ptr_y = src_.ptr<float>(1);
    const float * dst_ptr_x = dst_.ptr<float>(0);
    const float * dst_ptr_y = dst_.ptr<float>(1);

    float mean_src_x = (src_ptr_x[0]+src_ptr_x[1]+src_ptr_x[2])/3;
    float mean_src_y = (src_ptr_y[0]+src_ptr_y[1]+src_ptr_y[2])/3;
    float mean_dst_x = (dst_ptr_x[0]+dst_ptr_x[1]+dst_ptr_x[2])/3;
    float mean_dst_y = (dst_ptr_y[0]+dst_ptr_y[1]+dst_ptr_y[2])/3;

    Mat temp_src=src_.clone();
    Mat temp_dst=dst_.clone();

    float * temp_src_ptr_x = temp_src.ptr<float>(0);
    float * temp_src_ptr_y = temp_src.ptr<float>(1);
    float * temp_dst_ptr_x = temp_dst.ptr<float>(0);
    float * temp_dst_ptr_y = temp_dst.ptr<float>(1);

    temp_src_ptr_x[0] = src_ptr_x[0] - mean_src_x;
    temp_src_ptr_x[1] = src_ptr_x[1] - mean_src_x;
    temp_src_ptr_x[2] = src_ptr_x[2] - mean_src_x;
    temp_src_ptr_y[0] = src_ptr_y[0] - mean_src_y;
    temp_src_ptr_y[1] = src_ptr_y[1] - mean_src_y;
    temp_src_ptr_y[2] = src_ptr_y[2] - mean_src_y;

    temp_dst_ptr_x[0] = dst_ptr_x[0] - mean_dst_x;
    temp_dst_ptr_x[1] = dst_ptr_x[1] - mean_dst_x;
    temp_dst_ptr_x[2] = dst_ptr_x[2] - mean_dst_x;
    temp_dst_ptr_y[0] = dst_ptr_y[0] - mean_dst_y;
    temp_dst_ptr_y[1] = dst_ptr_y[1] - mean_dst_y;
    temp_dst_ptr_y[2] = dst_ptr_y[2] - mean_dst_y;
    

    float mean_src_dis = sqrt(temp_src_ptr_x[0]*temp_src_ptr_x[0]+temp_src_ptr_y[0]*temp_src_ptr_y[0])
    +sqrt(temp_src_ptr_x[1]*temp_src_ptr_x[1]+temp_src_ptr_y[1]*temp_src_ptr_y[1])
    +sqrt(temp_src_ptr_x[2]*temp_src_ptr_x[2]+temp_src_ptr_y[2]*temp_src_ptr_y[2]);
    mean_src_dis /= 3;
    float mean_dst_dis = sqrt(temp_dst_ptr_x[0]*temp_dst_ptr_x[0]+temp_dst_ptr_y[0]*temp_dst_ptr_y[0])
    +sqrt(temp_dst_ptr_x[1]*temp_dst_ptr_x[1]+temp_dst_ptr_y[1]*temp_dst_ptr_y[1])
    +sqrt(temp_dst_ptr_x[2]*temp_dst_ptr_x[2]+temp_dst_ptr_y[2]*temp_dst_ptr_y[2]);

    mean_dst_dis /= 3;
    float src_sf = sqrt(2)/mean_src_dis;
    float dst_sf = sqrt(2)/mean_dst_dis;

    Mat src_trans  =(Mat_<float>(3,3)<<src_sf,0,-src_sf*mean_src_x,
                                        0,src_sf,-src_sf*mean_src_y,
                                        0,0,1);
    Mat dst_trans  =(Mat_<float>(3,3)<<dst_sf,0,-dst_sf*mean_dst_x,
                                        0,dst_sf,-dst_sf*mean_dst_y,
                                        0,0,1);
    temp_src_ptr_x[0] = src_sf*src_ptr_x[0]-src_sf*mean_src_x;
    temp_src_ptr_x[1] = src_sf*src_ptr_x[1]-src_sf*mean_src_x;
    temp_src_ptr_x[2] = src_sf*src_ptr_x[2]-src_sf*mean_src_x;
    temp_src_ptr_y[0] = src_sf*src_ptr_y[0]-src_sf*mean_src_y;
    temp_src_ptr_y[1] = src_sf*src_ptr_y[1]-src_sf*mean_src_y;
    temp_src_ptr_y[2] = src_sf*src_ptr_y[2]-src_sf*mean_src_y;

    temp_dst_ptr_x[0] = dst_sf*dst_ptr_x[0]-dst_sf*mean_dst_x;
    temp_dst_ptr_x[1] = dst_sf*dst_ptr_x[1]-dst_sf*mean_dst_x;
    temp_dst_ptr_x[2] = dst_sf*dst_ptr_x[2]-dst_sf*mean_dst_x;
    temp_dst_ptr_y[0] = dst_sf*dst_ptr_y[0]-dst_sf*mean_dst_y;
    temp_dst_ptr_y[1] = dst_sf*dst_ptr_y[1]-dst_sf*mean_dst_y;
    temp_dst_ptr_y[2] = dst_sf*dst_ptr_y[2]-dst_sf*mean_dst_y;

    chrono::steady_clock::time_point t2=chrono::steady_clock::now();
    chrono::duration<double> ti = chrono::duration_cast<chrono::duration<double>>(t2-t1);
    t1=chrono::steady_clock::now();

    Mat A(2*samples, 4, CV_32F, Scalar{0});
    Mat xp(2*samples, 1, CV_32F, Scalar{0});

    for(int k=0; k<samples; k++)
    {
        float * A_x_0 = A.ptr<float>(2*k);
        A_x_0[0] = temp_src_ptr_x[k];
        A_x_0[1] = -temp_src_ptr_y[k];
        A_x_0[2] = 1;
        A_x_0[3] = 0;

        float * A_x_1 = A.ptr<float>(2*k+1);
        A_x_1[0] = temp_src_ptr_y[k];
        A_x_1[1] = temp_src_ptr_x[k];
        A_x_1[2] = 0;
        A_x_1[3] = 1;

        xp.ptr<float>(2*k)[0] = temp_dst_ptr_x[k];
        xp.ptr<float>(2*k+1)[0] = temp_dst_ptr_y[k];
    }
    
    Mat D_t;
    Mat U(samples*2, samples*2, CV_32F, Scalar{0});
    Mat D(samples*2, 4, CV_32F, Scalar{0});
    Mat V(4, 4, CV_32F, Scalar{0});
    
    SVDecomp(A, D_t,U,V);
    
    D = Mat::diag(D_t);
    Mat h = V*(1.0/D)*U.t()*xp;
    Mat H = (Mat_<float>(3,3) << h.ptr<float>(0)[0], -h.ptr<float>(1)[0], h.ptr<float>(2)[0],
                                h.ptr<float>(1)[0], h.ptr<float>(0)[0], h.ptr<float>(3)[0],
                                0,0,1);
    
    Mat inv_dst_trans =(Mat_<float>(3,3)<<1.0/dst_sf,0,mean_dst_x,
                                        0,1.0/dst_sf,mean_dst_y,
                                        0,0,1);
    H = inv_dst_trans*H*src_trans;
    SVDecomp(H(Rect(0,0,2,2)), D,U,V);
    Mat S=Mat::eye(2, 2, CV_32F);

    float U_det = U.ptr<float>(0)[0]*U.ptr<float>(1)[1]-U.ptr<float>(0)[1]*U.ptr<float>(1)[0];
    float V_det = V.ptr<float>(0)[0]*V.ptr<float>(1)[1]-V.ptr<float>(0)[1]*V.ptr<float>(1)[0];
    if (U_det*V_det<0)  S.ptr<float>(1)[1] = -1;

    H(Rect(0,0,2,2))=U*S*V.t();

    return H;
    

}

//RANSAC of rigid transform
Mat estimateICP(const Mat& src, const Mat& dst, vector<int>& inliers_ind, int max_iters_user=10000)
{
    //N*2
    int samples = src.rows;

    cout << "perform RANSAC on " << samples <<  " matches" << endl;

    Mat src_homo(samples, 3, CV_32F, Scalar{0});
    Mat dst_homo(samples, 3, CV_32F, Scalar{0});
    src_homo(Rect(2,0,1,samples)) = 1;
    dst_homo(Rect(2,0,1,samples)) = 1;
    src.copyTo(src_homo(Rect(0,0,2,samples)));
    dst.copyTo(dst_homo(Rect(0,0,2,samples)));
    Mat src_homo_t = src_homo.t();
    Mat dst_homo_t = dst_homo.t();

    double  max_iteration=1.0*samples*(samples-1)/(2)+3;
    if(max_iteration>10000)
        max_iteration=10000;
    max_iteration =     max_iters_user;

    int max_consensus_number=2;
    float min_neighbor_dis = 2;
    float err_t = 2;
    int used_points = 3;
    int ind_vec[used_points];
    Mat consensus_T;
    RNG rng((unsigned)time(NULL));

    for(int i=0; i<max_iteration; i++)
    {
      try{
            chrono::steady_clock::time_point t1=chrono::steady_clock::now();
            for(int k=0; k<used_points; k++)
            {
                ind_vec[k]=int((samples-1)*rng.uniform((float)0, (float)1));
            }
            bool is_neighbor=false;
            for(int k=0; k<used_points; k++)
            {
                for(int l=k+1; l<used_points; l++)
                {
                   if(!is_neighbor)  is_neighbor = ind_vec[k]==ind_vec[l];
                   if(!is_neighbor) is_neighbor = sum(abs(src_homo.row(ind_vec[k])-src_homo.row(ind_vec[l])))[0]<3;
                   if(!is_neighbor)  is_neighbor = sum(abs(dst_homo.row(ind_vec[k])-dst_homo.row(ind_vec[l])))[0]<3;
                }
            }
            if (is_neighbor) continue;
 
            Mat src_(3, used_points,  CV_32F, Scalar{0});
            Mat dst_(3, used_points,  CV_32F, Scalar{0});
            for(int k=0; k<used_points; k++)
            {
                Mat temp_t = src_homo(Rect(0,ind_vec[k],3,1)).t();
                temp_t.copyTo(src_.col(k));// = src.row(ind_vec[k]);
                temp_t = dst_homo(Rect(0,ind_vec[k],3,1)).t();
                temp_t.copyTo(dst_.col(k));// = src.row(ind_vec[k]);
            }
       
            Mat T = get_trans_icp_3_points(src_, dst_);
            
            Mat err;

            Mat err_temp=T*src_homo_t-dst_homo_t;
            err_temp=err_temp.mul(err_temp);
            
            reduce(err_temp, err, 0, CV_REDUCE_SUM); //one row

            int consensus_num =  sum((err<err_t*err_t)/255)[0];
            if (consensus_num > max_consensus_number)
            {
                max_consensus_number = consensus_num;
                consensus_T=T.clone();
            }
          }
          catch(cv::Exception& e)
          {
            continue;
          }
    }

    Mat err;
    reduce((consensus_T*src_homo.t()-dst_homo.t()).mul(consensus_T*src_homo.t()-dst_homo.t()), err, 0, CV_REDUCE_SUM);
    for(int i=0; i<err.cols; i++)
    {
        if(err.ptr<float>(0)[i]<err_t*err_t)
            inliers_ind.push_back(i);
    }
    Mat inliers_src(3,inliers_ind.size(),CV_32F,Scalar{0});
    Mat inliers_dst(3,inliers_ind.size(),CV_32F,Scalar{0});
    for(int i=0; i<inliers_ind.size(); i++)
    {

        Mat temp_t = src_homo(Rect(0,inliers_ind[i],3,1)).t();
            temp_t.copyTo(inliers_src.col(i));// = src.row(ind_vec[k]);
            temp_t = dst_homo(Rect(0,inliers_ind[i],3,1)).t();
            temp_t.copyTo(inliers_dst.col(i));// = src.row(ind_vec[k]);
    }
    consensus_T = get_trans_icp(inliers_src, inliers_dst);

    cout <<"find " << inliers_ind.size()<<" inliers" << endl;


    return consensus_T;
}


void writeMatToFile(cv::Mat& m, const char* filename)
{
	std::ofstream fout(filename);
    int mat_type = m.type();
	if (!fout||mat_type>=8)
	{
		std::cout << "File Not Opened or multi-channel mat" << std::endl;  
		return;
	}
    
	for (int i = 0; i<m.rows; i++)
	{
		for (int j = 0; j<m.cols; j++)
		{
            if(mat_type%8==0)
    			fout << int(m.at<uint8_t>(i, j)) << "\t";
            else if(mat_type%8==1)
    			fout << m.at<int8_t>(i, j) << "\t";
            if(mat_type%8==2)
    			fout << m.at<uint16_t>(i, j) << "\t";
            if(mat_type%8==3)
    			fout << m.at<int16_t>(i, j) << "\t";
            if(mat_type%8==4)
    			fout << m.at<int>(i, j) << "\t";
            if(mat_type%8==5)
    			fout << m.at<float>(i, j) << "\t";
            if(mat_type%8==6)
    			fout << m.at<double>(i, j) << "\t";
		}
		fout << std::endl;
	}
 
	fout.close();
}

int writeMatToBin(const cv::Mat& m, const std::string filename)
{
    std::ofstream fout(filename.c_str(), std::ios::binary);
    int mat_type = m.type();
	if (!fout||mat_type>=8)
	{
		std::cout << "File Not Opened or multi-channel mat" << std::endl;  
		return 1;
	}
    
    float s;
    // s=m.rows;
    // fout.write((char*)&s, sizeof(float));
    // s=m.cols;
    // fout.write((char*)&s, sizeof(float));
	for (int i = 0; i<m.rows; i++)
	{
		for (int j = 0; j<m.cols; j++)
		{
            if(mat_type%8==0)
    			s=int(m.at<uint8_t>(i, j));
            else if(mat_type%8==1)
    			s= m.at<int8_t>(i, j);
            if(mat_type%8==2)
    			s= m.at<uint16_t>(i, j);
            if(mat_type%8==3)
    			s= m.at<int16_t>(i, j);
            if(mat_type%8==4)
    			s=m.at<int>(i, j);
            if(mat_type%8==5)
    			s=m.at<float>(i, j);
            if(mat_type%8==6)
    			s=m.at<double>(i, j);
            fout.write((char*)&s, sizeof(float));
		}
	}
 
	fout.close();
    return 0;
}

//custumed functions, may be faster
#define _DBL_EPSILON 2.2204460492503131e-16f
#define atan2_p1 (0.9997878412794807f*57.29577951308232f)
#define atan2_p3 (-0.3258083974640975f*57.29577951308232f)
#define atan2_p5 (0.1555786518463281f*57.29577951308232f)
#define atan2_p7 (-0.04432655554792128f*57.29577951308232f)

Mat matAbsAtan2(Mat & y, Mat &x)
{
    Mat ax = abs(x);
    Mat ay = abs(y);

    //x>y情况
    Mat re1(y.rows, y.cols, CV_32FC1, Scalar{1}), re2(y.rows, y.cols, CV_32FC1, Scalar{1});
    multiply(re1, (ax>=ay)/255, re1, 1,CV_32F);
    Mat c = ay/(ax+_DBL_EPSILON);
    c = re1.mul(c);
    Mat c2 = c.mul(c);
    c2 = (((atan2_p7*c2 + atan2_p5).mul(c2) + atan2_p3).mul(c2) + atan2_p1).mul(c);
    re1 = c2.mul(re1);

    //x<y情况
    multiply(re2, (ax<ay)/255, re2, 1,CV_32F);
    c = ax/(ay+_DBL_EPSILON);
    c = c.mul(re2);
    c2 = c.mul(c);
    c2 = 90.0 - (((atan2_p7*c2 + atan2_p5).mul(c2) + atan2_p3).mul(c2) + atan2_p1).mul(c);
    re2 = c2.mul(re2);

    //两种情况结合
    Mat ret = re1+re2;

    //区分一二象限
    multiply(ret, (x<0)/255, re1, 1,CV_32F);
    multiply(ret, (x>=0)/255, re2, 1,CV_32F);
    re1 = 180.0 - re1;
    multiply(re1, (x<0)/255, re1, 1,CV_32F);
    ret = re1+re2;
    /*
    multiply(ret, (y<0)/255, re1, 1,CV_32F);
    multiply(ret, (y>=0)/255, re2, 1,CV_32F);
    re1 = 360.0 - re1;
    multiply(re1, (y<0)/255, re1, 1,CV_32F);
    ret = re1+re2;
    */
    return ret/180*CV_PI;

}

Mat matCos(Mat& x)
{
    Mat temp_x = CV_PI/2-x;
    Mat x2 = temp_x.mul(temp_x);
    Mat y = temp_x-temp_x.mul(x2)/6+temp_x.mul(x2).mul(x2)/120
    -temp_x.mul(x2).mul(x2).mul(x2)/5040;
    return y;

}

//utils of fft
Mat ifft2shift(Mat& fft_img);
Mat fft2(const Mat& img);
inline Mat ifft2(const Mat& img);

Mat fft2(const Mat& img)
{
    Mat m_src;
    Mat m_fourier(img.rows, img.cols, CV_32FC2, Scalar(0, 0));
    if(img.type() < 8)
    {
	    Mat m_for_fourier[] = { Mat_<float>(img), Mat::zeros(img.size(), CV_32F) };
        merge(m_for_fourier, 2, m_src);
    }
    else if(img.type() < 16) m_src = img;
    else 
    {
        cout << "FFT input type channel error, input type is " << img.type() << endl;
        return Mat(img.rows, img.cols, CV_32FC2, Scalar{0});
    }
	dft(m_src, m_fourier);
    return m_fourier;
}
inline Mat ifft2(const Mat& img)
{
    Mat m_src;
    Mat m_fourier(img.rows, img.cols, CV_32FC2, Scalar(0, 0));
    if(img.type() < 8)
    {
	    Mat m_for_fourier[] = { Mat_<float>(img), Mat::zeros(img.size(), CV_32F) };
        merge(m_for_fourier, 2, m_src);
    }
    else if(img.type() < 16 ) 
    {
        //cout << "CVC2" << endl;
        m_src = img;
    }
    else
    {
        cout << "FFT input type channel error, input type is " << img.type() << endl;
        return Mat(img.rows, img.cols, CV_32FC2, Scalar{0});
    } 
    //cout << img.size << endl;
	
    //cout << m_src.size << endl;
	dft(m_src, m_fourier, DFT_INVERSE|DFT_SCALE);
    return m_fourier;
}
Mat ifft2shift(Mat& fft_img)
{
    int rows= fft_img.rows;
    int cols= fft_img.cols;

    int cy = rows/2+rows%2;
    int cx = cols/2+cols%2;
    
    Mat ret(rows,cols,fft_img.type(),Scalar{0});//fft_img.clone();

    fft_img(Rect(0,rows-cy,cols,cy)).copyTo(ret(Rect(0,0,cols,cy)));
    fft_img(Rect(0,0,cols,rows-cy)).copyTo(ret(Rect(0,cy,cols,rows-cy)));

    Mat tmp=ret.clone();
    tmp(Rect(cols-cx,0,cx,rows)).copyTo(ret(Rect(0,0,cx,rows)));
    tmp(Rect(0,0,cols-cx,rows)).copyTo(ret(Rect(cx,0,cols-cx,rows)));

    return ret;
}

//BVFT descriptor extraction
BVFT detectBVFT(Mat img1)
{
    chrono::steady_clock::time_point t_start = chrono::steady_clock::now();

    normalize(img1,img1,0,255,CV_MINMAX);

    int rows = img1.rows;
    int cols = img1.cols;

    Mat radius(rows, cols, CV_32FC1, Scalar{0});
    Mat theta(rows, cols, CV_32FC1, Scalar{0});

    //low-pass filter
    Mat lowpass_filter(rows, cols, CV_32FC1, Scalar{0});
    float lowpass_cutoff = 0.45;
    int sharpness = 15;
    for (int x=0; x<cols; x+=1){
        for ( int y=0; y<rows; y+=1){
            float x_range = -1/2.0+x*1.0/cols;
            if(cols%2) x_range = -1/2.0+x*1.0/(cols-1);
            float y_range = -1/2.0+y*1.0/rows;
            if(rows%2) y_range = -1/2.0+y*1.0/(rows-1);
            radius.ptr<float>(y)[x] = sqrt(y_range*y_range + x_range*x_range);
            theta.ptr<float>(y)[x] = atan2(-y_range, x_range);
            lowpass_filter.ptr<float>(y)[x] = 1.0/(1+pow(radius.ptr<float>(y)[x]/lowpass_cutoff,2*sharpness));
        }
    }

    radius = ifft2shift(radius);
    theta = ifft2shift(theta);
    lowpass_filter = ifft2shift(lowpass_filter);
    
    radius.at<float>(0,0) = 1;

    Mat sintheta(rows, cols, CV_32FC1, Scalar{0});
    Mat costheta(rows, cols, CV_32FC1, Scalar{0});
    for (int x=0; x<cols; x+=1){
        for ( int y=0; y<rows; y+=1){
            sintheta.ptr<float>(y)[x] = sin(theta.ptr<float>(y)[x]);
            costheta.ptr<float>(y)[x] = cos(theta.ptr<float>(y)[x]);
        }
    }

    //Log-Gabor filter construction
    float min_wavelength = 3;
    float mult = 1.6;
    float sigma_on_f = 0.75;
    vector<Mat> log_gabor;
    for( int s=0; s<nscale; s++ )
    {
        float wavelength = min_wavelength*pow(mult, s);
        float fo = 1.0/wavelength;
        Mat log_gabor_s;
        log(radius/fo, log_gabor_s);
        log_gabor_s = -log_gabor_s.mul(log_gabor_s)/(2*log(sigma_on_f)*log(sigma_on_f));
        exp(log_gabor_s, log_gabor_s);
        log_gabor_s = log_gabor_s.mul(lowpass_filter);
        log_gabor_s.at<float>(0,0)=0;
        log_gabor.push_back(log_gabor_s);
    }
    
    //fft of the input BV image
    Mat img_fft=fft2(img1);

    
    Mat eo[nscale][norient];
    for(int i=0; i<nscale; i++)
        for(int j=0; j<norient; j++)
            eo[i][j] = Mat(rows, cols, CV_32FC2, Scalar{0});

    for (int o=0; o<norient; o++)
    {
        float angle = o*CV_PI/norient;
        float cos_angle = cos(angle);
        float sin_angle = sin(angle);

        Mat spread(rows, cols, CV_32FC1, Scalar{0});

        Mat ytheta = (sintheta*cos_angle-costheta*sin_angle);
        Mat xtheta = (costheta*cos_angle+sintheta*sin_angle);
        Mat dtheta = matAbsAtan2(ytheta,xtheta);
        dtheta =abs(dtheta)*norient;
        dtheta = min(dtheta,CV_PI);
        spread = (matCos(dtheta)+1)/2;

        for (int s=0; s<nscale; s++)
        {
            //oriented Log-Gabor filter
            Mat filter(rows, cols, CV_32FC1, Scalar{0});
            filter = log_gabor[s].mul(spread);           

            //perform convolution using fft
            Mat img_fft_channels[] = { Mat::zeros(img1.size(), CV_32F), Mat::zeros(img1.size(), CV_32F) };
            split(img_fft, img_fft_channels);
            img_fft_channels[0] = img_fft_channels[0].mul(filter);
            img_fft_channels[1] = img_fft_channels[1].mul(filter);
            
            Mat img_fft_filtered;
            merge(img_fft_channels, 2, img_fft_filtered);
            eo[s][o] = ifft2(img_fft_filtered);
        }
    }

    

    //detect FAST keypoints on the BV image
    Ptr<FastFeatureDetector> fast = FastFeatureDetector::create();
    Mat for_fast=img1.clone();
    normalize(img1,for_fast, 0, 255, CV_MINMAX);//=img1.clone();

    vector<KeyPoint> keypoints_raw;
    vector<KeyPoint> keypoints;
    fast->detect(for_fast, keypoints_raw);
    float max_metric = 0;
    KeyPointsFilter::removeDuplicated(keypoints_raw);

    // cout << "detected " << keypoints_raw.size() << " keypoints" << endl;

    int patch_size=138;
    int patch_size_true = 96;
    Mat keypoints_mask(for_fast.rows, for_fast.cols, CV_8U, Scalar{0});

    //rm the keypoints at borders
    for (int i=0; i<keypoints_raw.size(); i++)
    {
        bool at_border = (keypoints_raw[i].pt.x<patch_size/2 || keypoints_raw[i].pt.y<patch_size/2 || keypoints_raw[i].pt.x>img1.cols-patch_size/2 || keypoints_raw[i].pt.y>img1.rows-patch_size/2);
        if(!at_border)
            keypoints.push_back(keypoints_raw[i]);
    }

    // cout << "keep " << keypoints.size() << " keypoints" << endl;

    //calculate log-gabor responses
    Mat CS[norient];
    for (int o=0; o<norient; o++)
        CS[o] = Mat(rows, cols, CV_32FC1, Scalar{0});

    for(int o=0; o<norient; o++)
    {
        for(int s=0; s<nscale; s++)
        {
            Mat img_channels[] = { Mat::zeros(img1.size(), CV_32F), Mat::zeros(img1.size(), CV_32F) };
            split(eo[s][o], img_channels);
            Mat EO_re = img_channels[0];
            Mat EO_im = img_channels[1];
            Mat An;
            magnitude(EO_re, EO_im, An);
            CS[o]+=abs(An);
        }
    }
    
    //build MIM
    Mat MIM(rows, cols, CV_8U, Scalar{0});
    Mat max_response(rows,cols,CV_32FC1,Scalar{0});
    for(int o=0; o<norient; o++)
    {
        MIM = max(MIM, (CS[o] > max_response)/255*(o+1));
        max_response = max(CS[o], max_response);
    }
    
    //depress the pixels with low responses
    Mat mim_mask;
    mim_mask = ((max_response > 0.1))/255;;    

    multiply(MIM, mim_mask , MIM);


    int descriptor_orients = norient/2;
    int areas = 6;
    Mat decriptors(areas*areas*descriptor_orients, keypoints.size(), CV_32F, Scalar{0});
    vector<int> kps_to_ignore(keypoints.size(), 0);
    Mat kps_angle(1,keypoints.size(), CV_32F, Scalar{0});;

    bool descriptor_permute = 1;
    bool descriptor_rotate = 1;

    //weight kernel for dominant orientation computation
    float patch_main_orient_kernel[patch_size][patch_size];
    float patch_kernel_radius = patch_size/2;
    for(int k=0; k<patch_size; k++)
        for(int j=0; j<patch_size; j++)
        {
            float dis = (k-patch_size/2)*(k-patch_size/2)+(j-patch_size/2)*(j-patch_size/2);
            dis=sqrt(dis)/patch_kernel_radius;
            if(dis>1) dis=1;
            patch_main_orient_kernel[k][j]=1-dis;
            patch_main_orient_kernel[k][j]*=patch_main_orient_kernel[k][j];
        }
    
    //describe every keypoint
    for(int k=0; k<keypoints.size(); k++)
    {

        // find the patch position
        int x = keypoints[k].pt.x;
        int y = keypoints[k].pt.y;

        float x_low = max(0,x-patch_size/2-patch_size%2);
        float x_hig = min(x+patch_size/2-patch_size%2,cols);
        float y_low = max(0,y-int(patch_size/2)-patch_size%2);
        float y_hig = min(y+int(patch_size/2)-patch_size%2,rows);

        Mat patch = MIM(Rect(Point(x_low, y_low),Point(x_hig, y_hig))).clone();
        Mat patch_mask = mim_mask(Rect(Point(x_low, y_low),Point(x_hig, y_hig))).clone();
        Mat patch_max_response = max_response(Rect(Point(x_low, y_low),Point(x_hig, y_hig))).clone();
 
        // compute dominant orientation
        float hist[norient+1] = {0};
        float hist_energy[norient+1] = {0};
        for(int hy = 0; hy<patch.rows; hy++)
        {
            uint8_t*  ptr_y = patch.ptr<uint8_t>(hy);
            float*    ptr_m = patch_max_response.ptr<float>(hy);
            for(int hx = 0; hx<patch.cols; hx++)
            {
                hist[ptr_y[hx]]+=patch_main_orient_kernel[hy][hx]*ptr_m[hx];
                hist_energy[ptr_y[hx]] += patch_main_orient_kernel[hy][hx];
            }
        }
        int max_orient=1;
        for(int hist_i=1;hist_i<norient+1; hist_i++)
            if(hist[hist_i]>hist[max_orient]) max_orient = hist_i;
        if (hist[(max_orient-1)==0?norient:(max_orient-1)]>hist[(max_orient+1)==(norient+1)?1:(max_orient+1)]) 
            max_orient = (max_orient-1)==0?norient:(max_orient-1);
        
        kps_angle.ptr<float>(0)[k] = (max_orient);

        //circle shift the values of the MIM patch
        if(descriptor_permute)
        {
            Mat shang = ((norient)+patch)-(max_orient);//-norient/2-1; //uint8 4drop5in
            patch = shang-(shang>=norient)/255*norient +1;
            patch = patch.mul(patch_mask);
        }
        
        //rotate the patch
        if(descriptor_rotate)
        {
            keypoints[k].angle=-180*(max_orient-1)/norient;
            cv::Size dst_sz(patch.cols, patch.rows);    
            cv::Point2f center(static_cast<float>(patch.cols / 2.), static_cast<float>(patch.rows / 2.));
            float angle=-180*(max_orient-1)/norient;  
            
            Mat patch_te = patch.clone();
            cv::Mat rot_mat = cv::getRotationMatrix2D(center, angle, 1.0);
            cv::warpAffine(patch, patch, rot_mat, dst_sz, INTER_NEAREST, BORDER_REPLICATE);
        }
        

        Mat patch_true = patch(Rect((patch_size-patch_size_true)/2
                                        , (patch_size-patch_size_true)/2,
                                    patch_size_true
                                        ,patch_size_true));

        int ys = patch_true.rows;
        int xs = patch_true.cols;
        //describing the patch
        for(int j=0; j<areas; j++)
            for(int i=0; i<areas; i++)
            {
                Mat clip = patch_true(Rect(j*ys/areas,i*xs/areas,
                                    ys/areas,xs/areas)).clone();

                float hist[norient+1] = {0};
                for(int hi =0 ;hi<norient; hi++) hist[hi]=0;
                for(int hy = 0; hy<clip.rows; hy++)
                {
                     uint8_t*  ptr_y = clip.ptr<uint8_t>(hy);
                    for(int hx = 0; hx<clip.cols; hx++)
                    {
                        float weight = (fabs(j*ys/areas+hx-patch_size_true/2) + fabs(i*xs/areas+hy-patch_size_true/2))/patch_size_true;
                        weight = 1-weight*weight;
                        hist[ptr_y[hx]]+=1;
                    }
                }
                float ker = fabs(i-areas/2.0)+fabs(j-areas/2.0);
                ker/=areas;
                ker=1-ker;
                for(int hist_i=0; hist_i<descriptor_orients ;hist_i++)
                    decriptors.ptr<float>(j*areas*descriptor_orients+i*descriptor_orients+hist_i)[k] = (hist[2*hist_i+1]+hist[2*hist_i+2]);//(hist[2*hist_i+1]+hist[2*hist_i+2]);//*ker;//;//
            }
        
        //normalize the descriptor
        float norm_sum = 0;
        for (int norm_i=0; norm_i< decriptors.rows;norm_i++)
            norm_sum+=decriptors.ptr<float>(norm_i)[k]*decriptors.ptr<float>(norm_i)[k];
        norm_sum=sqrt(norm_sum);
        for (int norm_i=0; norm_i< decriptors.rows;norm_i++)
            decriptors.ptr<float>(norm_i)[k]/=norm_sum;
        float sum_main=0, sum_main_rela=0;
    }

    chrono::steady_clock::time_point t_end = chrono::steady_clock::now();
    chrono::duration<float> time_used = chrono::duration_cast<
    chrono::duration<float>>(t_end-t_start);

    // cout <<"----------------Elapsing time: "<< time_used.count()  << " seconds\n"<< endl;

    decriptors=decriptors.t();
    BVFT bvft(keypoints, decriptors);
    bvft.angle = kps_angle.clone();

    return bvft;//BVFT(keypoints, decriptors);
}

#endif
