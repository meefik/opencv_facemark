#include <opencv2/highgui.hpp>
#include <opencv2/face/facemarkLBF.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;
using namespace cv::face;

struct Conf {
    String model_path;
    double scaleFactor;
    int minNeighbors;
    Conf(String s, double d = 1.05, int n = 2){
        model_path = s;
        scaleFactor = d;
        minNeighbors = n;
        face_cascade.load(model_path);
    }
    CascadeClassifier face_cascade;
};

bool vjFaceDetector(InputArray image, OutputArray faces, void *user_data){
    Conf *conf = static_cast<Conf*>(user_data);
    Mat gray;
    if (image.channels() > 1)
        cvtColor(image, gray, COLOR_BGR2GRAY);
    else
        gray = image.getMat().clone();
    equalizeHist(gray, gray);
    vector<Rect> faces_;
    conf->face_cascade.detectMultiScale(gray, faces_, conf->scaleFactor, conf->minNeighbors, CASCADE_SCALE_IMAGE, Size(100, 100) );
    Mat(faces_).copyTo(faces);
    return true;
}

void detect(Ptr<FacemarkLBF> facemark, string model_filename)
{
    VideoCapture capture;
    capture.open(0);
    capture.set(3,640);
    capture.set(4,480);
    capture.set(5,10);

    Mat image;
    facemark->loadModel(model_filename);
    vector<Rect> faces;
    vector<vector<Point2f>> landmarks;

    while ( capture.read(image) ) {
        if( image.empty() ) {
            fprintf(stderr, "No captured frame -- Break!\n");
            break;
        }
        facemark->getFaces(image, faces);
        facemark->fit(image, faces, landmarks);
        for(int j=0;j<faces.size();j++){
            rectangle( image, faces[j], Scalar( 255, 0, 0 ), 1, 1 );
            drawFacemarks(image, landmarks[j], Scalar(0,0,255));
        }
        imshow("cam",image);

        // quit on ESC button
        if(waitKey(1)==27)break;
    }
}

void train(Ptr<FacemarkLBF> facemark)
{
    // load the dataset list
    String imageFiles = "../dlib_faces_5points/images_train.txt";
    String ptsFiles = "../dlib_faces_5points/points_train.txt";
    vector<String> images_train;
    vector<String> landmarks_train;
    loadDatasetList(imageFiles,ptsFiles,images_train,landmarks_train);

    // add the training samples to the trainer
    Mat image;
    vector<Point2f> facial_points;
    for(size_t i=0;i<images_train.size();i++){
        image = imread(images_train[i].c_str());
        loadFacePoints(landmarks_train[i],facial_points);
        facemark->addTrainingSample(image, facial_points);
    }

    // training process
    facemark->training();
}

int main(int argc, char *argv[])
{
    // declare the facemark instance
    // https://docs.opencv.org/trunk/d4/d12/structcv_1_1face_1_1FacemarkLBF_1_1Params.html
    // https://github.com/yulequan/face-alignment-in-3000fps/blob/master/LBF.cpp
    // Video for tests: https://www.youtube.com/watch?v=h-Gcl58WbGQ
    FacemarkLBF::Params params;
    params.n_landmarks = 5; // number of landmark points
    params.initShape_n = 10; // number of multiplier for make data augmentation
    params.stages_n = 10; // amount of refinement stages
    params.tree_n = 20; // number of tree in the model for each landmark point
    params.tree_depth = 5; // he depth of decision tree
    // params.bagging_overlap = 0.4; // overlap ratio for training the LBF feature
    params.pupils[0] = { 0, 1 };
    params.pupils[1] = { 2, 3 };
    // params.feats_m = { 500, 500, 500, 300, 300, 300, 200, 200, 200, 100 };
    // params.radius_m = { 0.3, 0.2, 0.15, 0.12, 0.10, 0.10, 0.08, 0.06, 0.06, 0.05 };
    params.model_filename = "lbfmodel.yaml"; // filename to save the trained model
    params.cascade_face = "../opencv/data/haarcascades/haarcascade_frontalface_default.xml";
    params.verbose = true;
    Ptr<FacemarkLBF> facemark = FacemarkLBF::create(params);

    Conf config(params.cascade_face);
    facemark->setFaceDetector(vjFaceDetector, &config);

    train(facemark);

    detect(facemark, params.model_filename);

    return 0;
}
