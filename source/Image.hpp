#pragma once
#include <opencv2/opencv.hpp>
#include <armadillo>
#include <string>
#include "Probability.hpp"
#include <iostream>
#include <filesystem>
#include <fstream>

extern const int IMG_SIZE=100;
extern const int NUM_CLASES=10;

extern const double Zvalue = 1.95; // for 95% confidence level

arma::mat QueryImagePath(const char *path)
{
    cv::Mat image= cv::imread( path, cv::IMREAD_GRAYSCALE  );
    if (!image.data)
    {
        std::cout<<"File doesn't exists"<<std::endl;
        return arma::mat();
    }
    arma::mat Result(IMG_SIZE*IMG_SIZE,1);
    // Our vector SIZE**2 
    for(int i=0;i<IMG_SIZE;++i)
        for(int j=0;j<IMG_SIZE;++j)
            Result( i*IMG_SIZE +j  , 0) = image.at<uchar>(i,j);
    return Result;
}

arma::Row<uchar> Image2vector(std::string path)
{
    cv::Mat image= cv::imread( path, cv::IMREAD_GRAYSCALE  );
    if (!image.data)
    {
        std::cout<<"File doesn't exists"<<std::endl;
        return arma::Row<uchar>();
    }
    arma::Row<uchar> Result(IMG_SIZE*IMG_SIZE);
    // Our vector SIZE**2 
    for(int i=0;i<IMG_SIZE;++i)
        for(int j=0;j<IMG_SIZE;++j)
            Result( i*IMG_SIZE +j ) = image.at<uchar>(i,j);
    return Result;
}

arma::mat QueryImage(cv::Mat image)
{
    
    cv::resize( image, image, cv::Size( IMG_SIZE , IMG_SIZE ) );
    arma::mat Result(IMG_SIZE*IMG_SIZE,1);
    // Our vector SIZE**2 
    for(int i=0;i<IMG_SIZE;++i)
        for(int j=0;j<IMG_SIZE;++j)
            Result( i*IMG_SIZE +j ) = image.at<uchar>(i,j);
    return Result;
}


std::vector<int> ClassOf(const arma::Mat<size_t> &Neighbors,const arma::Row<size_t> &Label_)
{
    int k = Neighbors.n_rows;
    int n = Neighbors.n_cols;
    
    std::vector<int> Classes(n);
    
    for(int j=0;j<n;++j)
    {
        int max_frec = 0;
        std::vector<int> count(NUM_CLASES,0);
        for(int i=0;i<k;++i)
        {
            auto Current = Label_[Neighbors(i,j)];
            ++count[Current];
            if(count[Current] > max_frec)
                max_frec = count[Current];
        }
        std::vector<int> Result;
        for(int i=0;i<NUM_CLASES;++i)
        {
            if(count[i]==max_frec)
                Result.push_back(i);
        }
        Classes[j] = random_choice<int>(Result);
    }
    
    return Classes;
}

std::ofstream& operator<<(std::ofstream &os, arma::Row<uchar> &Vec)
{
    for(int i=0;i<IMG_SIZE*IMG_SIZE;++i)
    {
        //
        os<<(int)Vec(i);
        if(i<IMG_SIZE*IMG_SIZE-1)
            os<<',';
    }
    return os;
}

void UpdateDB()
{
    std::ofstream Data("Data.csv");
    
    std::ofstream Label("Label.csv");
    int counter= 0;
    for(int i=0;i<10;++i)
    {
        std::string path = "./Dataset/"+std::to_string(i);
        for (auto & p : std::filesystem::directory_iterator(path))
        {
            auto Vec = Image2vector(p.path());
            Data<<Vec;
            Data<<std::endl;
            Label<<i<<",";
            ++counter;
        }
    }
    
    Data.flush();
    Data.close();

    Label.flush();
    Label.close();
    std::cout<<"Done! "<<counter<<" entrys found!"<<std::endl;
}


void Score(int okay, int total)
{
    std::cout<<okay<< " correct cases out of "<<total<<std::endl;
    double  p = (float)okay / (float)total;
    double std = std::sqrt( p*(1-p) /(float)total  );
    std::cout<<"Correct ratio "<<p<<std::endl<<"Standard Deviation "<<std<<std::endl;
    std::cout<<"Condifence Interval:\n( "<<p-std*Zvalue<<" , "<<p+std*Zvalue<<" ) at 0.95 confidence level"<<std::endl<<std::endl;
}

int CameraMode()
{
    cv::VideoCapture cap(0);
    
    if(!cap.isOpened())
    {
        std::cout<<"Change the camera"<<std::endl;
        return -1;
    }
    cv::Mat frame;
    
    while (true)
    {
        cap.read(frame);
        cv::namedWindow("camera" );
        cv::imshow("camera",frame);
        if (cv::waitKey(25)==27)
            return 0;
    }
}
