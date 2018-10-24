#include <mlpack/methods/neighbor_search/neighbor_search.hpp>
#include <mlpack/methods/softmax_regression/softmax_regression.hpp>
#include <mlpack/core.hpp>
#include <mlpack/core/data/split_data.hpp>
#include "Image.hpp"
#include <cmath>
#include <ctime>

int main(int argc, char *argv[])
{
    int k=5;
    
    bool test = false;
    
    bool camera = false;
    
    double ratio = 0.2;
    
    if ( argc == 2 )
    {
        if(std::string(argv[1])=="--update-db")
        {
            UpdateDB();
            return 0;
        }
        if(std::string(argv[1])=="--test")
            test = true;
        if(std::string(argv[1])=="--camera")
            camera = true;
        
    }
    
    if ( argc > 2 )
    {
        for(int i=0;i<argc;++i)
        {
            if(std::string(argv[i])=="-k")
                k = std::stoi(argv[i+1]);
            if(std::string(argv[i])=="-r")
                ratio = std::stof(argv[i+1]);
            if(std::string(argv[i])=="--test")
                test = true;
            if(std::string(argv[i])=="--camera")
            camera = true;
        }    
    }
    
    //Initialize data matrix
    arma::mat Data_;
   
    std::cout<<"Loading Data Matrix..."<<std::endl;
    // Load Data, set fatal to true
    mlpack::data::Load("Data.csv",Data_, true );
    
    std::cout<<Data_.n_cols<< " Entrys found"<<std::endl;

    //Initialize vector of labels
    arma::Row<size_t> Label_;
    std::cout<<"Loading Label vector..."<<std::endl;
    mlpack::data::Load("Label.csv",Label_, true );
    
    std::cout<<Label_.n_cols-1<<" Entrys found"<<std::endl;
    
    // Classifier using knn
    mlpack::neighbor::NeighborSearch <> KnnSearcher;
    
    // Classifier using softmax
    mlpack::regression::SoftmaxRegression Regressor(IMG_SIZE*IMG_SIZE,NUM_CLASES,true);
    
    arma::mat trainData;
    arma::mat testData;
    arma::Row<size_t> trainLabel;
    arma::Row<size_t> testLabel;
    if(test)
    {
        std::cout<<"Splitting data..."<<std::endl;
        mlpack::data::Split(Data_, Label_, trainData , testData, trainLabel, testLabel, ratio);
        
        std::cout<<"Traning knn...."<<std::endl;
        KnnSearcher.Train(trainData);
        std::cout<<"OK!"<<std::endl;
        
        std::cout<<"Traning regression ...."<<std::endl;
        Regressor.Train(trainData,trainLabel,NUM_CLASES);
        std::cout<<"OK!"<<std::endl;
    }
    else
    {
        std::cout<<"Traning...."<<std::endl;
        KnnSearcher.Train(Data_);
    }
    
    
    // The matrices we will store output in.
    arma::Mat<size_t> resultingNeighbors;
    arma::mat resultingDistances;
    
    std::string S = "s";
    while (!test && !camera)
    {
        std::cout<<"Image Path > ";
        std::cout.flush();
        
        std::getline(std::cin,S);
        if(S=="exit" || S=="Exit")
            return 0;
        auto Query = QueryImagePath(S.data());
        if(Query.is_empty())
            continue;
        KnnSearcher.Search(Query,k, resultingNeighbors, resultingDistances);
        int c = ClassOf(resultingNeighbors,Label_)[0];
        std::cout<<"Class of that image is "<<c<<std::endl;
    }
    
    if(test)
    {
        std::cout<<"Testing with split ratio = "<<ratio<< ", finding k = "<<k<< " neighbors"<<std::endl;
        std::cout<<"Starting "<<testData.n_cols<<" tests..."<<std::endl;
        
        int correct_knn=0;
        int correct_reg=0;
        
        std::cout<<"Searching on KNN..."<<std::endl;
        KnnSearcher.Search(testData,k, resultingNeighbors, resultingDistances);
        auto ClassesKNN = ClassOf(resultingNeighbors,trainLabel);
        std::cout<<"Done!"<<std::endl;
        
        arma::Row<size_t> ClassesReg;
        std::cout<<"Classifing on regression..."<<std::endl;
        Regressor.Classify(testData,ClassesReg);
        std::cout<<"Done!"<<std::endl;
        
        for(int i=0;i<testData.n_cols;++i)
        {
            if( ClassesKNN[i] == testLabel[i] )
                ++correct_knn;
            if (ClassesReg[i] == testLabel[i])
                ++correct_reg;
        }
        std::cout<<"Score for knn:"<<std::endl;
        Score(correct_knn,testData.n_cols);
        
        std::cout<<"Score for Regression:"<<std::endl;
        Score(correct_reg,testData.n_cols);
    }
    if(camera)
    {
        std::cout<<"Initializing Camera..."<<std::endl;
        cv::VideoCapture cap(0);
    
        if(!cap.isOpened())
        {
            std::cout<<"Change the camera"<<std::endl;
            return -1;
        }
        cv::Mat frame;
        
        auto timer = time(nullptr);
    
        while (true)
        {
            cap.read(frame);
            cv::namedWindow("camera" );
            auto Q = frame.adjustROI(0,0,-80,-80);
           
            cv::cvtColor(Q,frame, cv::COLOR_RGB2GRAY);
            
            cv::imshow("camera",frame);
            auto curr = time(nullptr);
            if(difftime(curr,timer) >= 1)
            {
                auto Query = QueryImage(frame);
                KnnSearcher.Search(Query,k, resultingNeighbors, resultingDistances);
                int c = ClassOf(resultingNeighbors,Label_)[0];
                std::cout<<"Class of current image is "<<c<<std::endl;
                timer = curr;
            }
        
            if (cv::waitKey(25)==27)
                return 0;
        }
    }
    
    return 0;
}

