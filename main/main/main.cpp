//  Developer: Baturay Özgürün
//  Copyright © 2018 Baturay Özgürün
//  If you have any requests or questions please do not hesitate to contact me via baturay.ozgurun@gmail.com

//  Project Name: Computer Vision Project -- Scene Recognition Algorithm with OpenCV
//  Project Desciption: This project aims to perform an scene recognition technique capitalizing on the Bag of Visiual Words method. A small training dataset (street images) is utilized to create vocabulary and calculate histogram distributions. Scene recognition is performed by finding the correlation between the histogram distributions of training (with the referance image) and test images. This operation allows us to evaluate scene similarities. This code also provides a similarity matrix (evaluation of the scene recognition task) and vocabulary (generated using training dataset) in text files.


//Include Headers
#include <iostream>
#include <opencv/highgui.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/features2d.hpp>

cv::Mat DictionaryBuilding(); //Function for Vocabulary Building

//Include Namespaces
using namespace cv;
using namespace std;

int main(){
    Mat Dictionary = DictionaryBuilding(); //Get the dictionary or vacabularies computed in the costom design file
    
    Ptr<FeatureDetector> detector = xfeatures2d::SiftFeatureDetector::create(); //Define feature extacter
    Ptr<DescriptorExtractor> extractor = cv::xfeatures2d::SiftDescriptorExtractor::create(); //Define descriptor
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("FlannBased"); //Define a matcher
    BOWImgDescriptorExtractor bowide(extractor,matcher); //Define BOW extracter
    bowide.setVocabulary(Dictionary); //Set the vocabularies
    
    vector<KeyPoint> keypoints1, keypoints2; //Define keypoint variable for the SIFT
    Mat BOWHistogram, SceneHistogram; //Create matrix variable for the histogram
    
    Mat Scene = imread("data/scene6.jpg",CV_LOAD_IMAGE_GRAYSCALE); //Get the reference image
    detector->detect(Scene, keypoints1); //Detect keypoints
    bowide.compute(Scene, keypoints1, SceneHistogram);//Get the BOWHistogram
    
    ofstream outputfile;
    outputfile.open("results/Similarity Matrix.txt");
    
    for(int i = 1; i <= 2; i++){
        ostringstream scn; //To read image sequentially, we need to define a string for object recognition
        scn << "data/test" << i << ".jpg"; //Assign a name for the images
        Mat img = imread(scn.str(),CV_LOAD_IMAGE_GRAYSCALE); //Read the image
        detector->detect(img, keypoints2); //Detect keypoints
        bowide.compute(img, keypoints2, BOWHistogram);//Get the BOWHistogram
        
        double correlation = compareHist(SceneHistogram, BOWHistogram, 0); //Measure the correlation value btw histograms
        
        //Write the similary of the exemined object on not only the console but also the text file named Similarity Matrix.txt
        if (abs(correlation) >= 0.3){
            outputfile << "The " << i << "th (st,nd) Test Image is Very Similar as the Images in the Dataset." << endl;
            cout << "The " << i << "th (st,nd) Test Image is Very Similar as the Images in the Dataset." << endl;
        }
        else if (abs(correlation) <= 0.3){
            outputfile << "The " << i << "th (st,nd) Test Image is Slightly OR Substantially Different from the Dataset Images." << endl;
            cout << "The " << i << "th (st,nd) Test Image is Slightly OR Substantially Different from the Dataset Images." << endl;
        }
    }
    return 0;
}

//Function for Vocabulary Building
Mat DictionaryBuilding(){
    Mat input, descriptor, featuresUnclustered; //Define matrix variables
    vector<KeyPoint> keypoints; //Define keypoint variable for the SIFT
    cv::xfeatures2d::SiftDescriptorExtractor detector; //Define detector
    
    for(int i=1;i<=7;i++){
        Ptr<Feature2D> f2d = xfeatures2d::SIFT::create();
        ostringstream name; //To read image sequentially, we need to define a string for a training image
        name << "data/scene" << i << ".jpg"; //Assign a name for the training image
        input = imread(name.str(),CV_LOAD_IMAGE_GRAYSCALE); //Read the image
        f2d->detect(input, keypoints); //Detect features
        f2d->compute(input, keypoints, descriptor); //Compute descriptors for each feature point
        featuresUnclustered.push_back(descriptor); //Compress all feature points into a Mat object
    }
    
    int dictionarySize = 200, retries = 1, flags = KMEANS_PP_CENTERS; //Define variables for the bow trainer
    TermCriteria tc(CV_TERMCRIT_ITER, 100, 0.001); //Define term criterial
    BOWKMeansTrainer bowTrainer(dictionarySize, tc, retries, flags); //Obtain Bow Trainer
    Mat Dictionary = bowTrainer.cluster(featuresUnclustered); //Cluster Features
    FileStorage fs("results/Vocabulary.txt", FileStorage::WRITE); //Write the vacabularies into a text file
    fs << "Vocabulary" << Dictionary;
    fs.release();
    return Dictionary; //Take the obtained dictionary to the main file in order to camputer histogram
}
