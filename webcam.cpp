
// The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
/*

    This example program shows how to find frontal human faces in an image and
    estimate their pose.  The pose takes the form of 68 landmarks.  These are
    points on the face such as the corners of the mouth, along the eyebrows, on
    the eyes, and so forth.


    This example is essentially just a version of the face_landmark_detection_ex.cpp
    example modified to use OpenCV's VideoCapture object to read from a camera instead
    of files.


    Finally, note that the face detector is fastest when compiled with at least
    SSE2 instructions enabled.  So if you are using a PC with an Intel or AMD
    chip then you should enable at least SSE2 instructions.  If you are using
    cmake to compile this program you can enable them by using one of the
    following commands when you create the build project:
        cmake path_to_dlib_root/examples -DUSE_SSE2_INSTRUCTIONS=ON
        cmake path_to_dlib_root/examples -DUSE_SSE4_INSTRUCTIONS=ON
        cmake path_to_dlib_root/examples -DUSE_AVX_INSTRUCTIONS=ON
    This will set the appropriate compiler options for GCC, clang, Visual
    Studio, or the Intel compiler.  If you are using another compiler then you
    need to consult your compiler's manual to determine how to enable these
    instructions.  Note that AVX is the fastest but requires a CPU from at least
    2011.  SSE4 is the next fastest and is supported by most current machines.
*/

#include <dlib/opencv.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_transforms.h>
#include <dlib/image_io.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace dlib;
using namespace std;
using namespace cv;


string type2str(int type) {
  string r;

  uchar depth = type & CV_MAT_DEPTH_MASK;
  uchar chans = 1 + (type >> CV_CN_SHIFT);

  switch ( depth ) {
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
  }

  r += "C";
  r += (chans+'0');

  return r;
}

void render_exterior_landmarks(cv::Mat &img, const dlib::full_object_detection& d, const int start, const int end)
{
    std::vector<cv::Point> points;
    for (int i = start; i <= 16; ++i)
    {
        points.push_back(cv::Point(d.part(i).x(), d.part(i).y()));
    }
    for (int i = end; i > 16; --i)
    {
        points.push_back(cv::Point(d.part(i).x(), d.part(i).y()));
    }
    cv::polylines(img, points, true, cv::Scalar(255,0,0), 2, 16);
}

void render_turd_on_forehead(cv::Mat &img, cv::Mat &sprite, const dlib::full_object_detection& d){
    cv::Point rightEyebrow(d.part(17).x(),d.part(17).y());
    cv::Point leftEyebrow(d.part(22).x(),d.part(22).y());
/*
    //cv::Mat smTurd;
    cout<<img.type()<<endl;
    cout<<"F: "<<rightEyebrow.x<<","<<rightEyebrow.y<<"/"<<rightFore.x<<","<<rightFore.y<<endl;

*/

    int w= 1.5*abs(leftEyebrow.x-rightEyebrow.x);
    int h=w/1.5;
    //cout<< w<<endl;

    cv::Mat small;

    cv::Rect roi( cv::Point( rightEyebrow.x, rightEyebrow.y-h ), cv::Size( w, h ));
    cv::resize(sprite,small,roi.size());
  /*  cv::Mat destinationROI = img( roi );
    small.copyTo( destinationROI );
*/
    cv::Mat imageROI= img(Rect(rightEyebrow.x,rightEyebrow.y-h,w,h));
    cv::addWeighted(imageROI,1.0,small,0.8,0.,imageROI);

}

/*
void render_exterior_landmarks(cv::Mat &img, const dlib::full_object_detection& d)
{
    draw_polyline(img, d, 0, 16);           // Jaw line
    draw_line(img, d, 0, 17);           // right link
    draw_polyline(img, d, 17, 26);           // Left eyebrow + Browline
    draw_polyline(img, d, 17, 16);          // Left eyebrow
    draw_line(img, d, 16, 26);          //
    //draw_polyline(img, d, 22, 26);          // Right eyebrow
    //draw_polyline(img, d, 21, 0);          //
    //draw_polyline(img, d, 27, 30);          // Nose bridge
    //draw_polyline(img, d, 30, 35, true);    // Lower nose
    //draw_polyline(img, d, 36, 41, true);    // Left eye
    //draw_polyline(img, d, 42, 47, true);    // Right Eye
    //draw_polyline(img, d, 48, 59, true);    // Outer lip
    //draw_polyline(img, d, 60, 67, true);    // Inner lip

}
*/
cv::Rect d2cv_Rect(const dlib::rectangle &dlibRect ) {
    long int x0 = dlibRect.left();
    long int x1 = dlibRect.right();
    long int y0 = dlibRect.top();
    long int y1 = dlibRect.top();
    return cv::Rect( cv::Point(x0,y0), cv::Point(x1,y1));

}


int main()
{
    try
    {
        cv::VideoCapture cap(0);
        //image_window win;
        namedWindow("w",1);

        if(!cap.isOpened()) { // check if we succeeded
            return -1;
        }

        // Load face detection and pose estimation models.
        frontal_face_detector detector = get_frontal_face_detector();
        shape_predictor pose_model;
        deserialize("shape_predictor_68_face_landmarks.dat") >> pose_model;
        cv::Mat turd;
        turd= imread("turd.png");

        // Grab and process frames until the main window is closed by the user.
        //while(!win.is_closed())
        while(true)
        {
            // Grab a frame
            cv::Mat tempBig, temp;
            cap >> tempBig;


            // Turn OpenCV's Mat into something dlib can deal with.  Note that this just
            // wraps the Mat object, it doesn't copy anything.  So cimg is only valid as
            // long as temp is valid.  Also don't do anything to temp that would cause it
            // to reallocate the memory which stores the image as that will make cimg
            // contain dangling pointers.  This basically means you shouldn't modify temp
            // while using cimg.

            cv::resize(tempBig, temp, cv::Size(), 1.0/1.0, 1.0/1.0);
            cv_image<bgr_pixel> cimg(temp);

            // Detect faces
            std::vector<dlib::rectangle> faces = detector(cimg);

            if (faces.size()>0) {
                dlib::rectangle mainBBox = faces[0];
                cv::Rect cvBBox = d2cv_Rect(mainBBox);

                full_object_detection shapes = pose_model(cimg, faces[0]);

                //blending zone = (ROI.width(), 0.5*(topLandmark.y, chinLandmark.y))
                //will be off for a big rotation
                /*std::vector<full_object_detection> shapes;
                for (unsigned long i = 0; i < faces.size(); ++i) {
                    shapes.push_back(pose_model(cimg, faces[i]));
                }*/
                /*dlib::array<array2d<rgb_pixel> > face_chips;
                array2d<rgb_pixel> chip;
                std::vector<chip_details> chip_locations = get_face_chip_details(shapes);
                chip_locations[0].angle=0;
                //extract_image_chips(cimg, chip_locations , face_chips);

                //impl::basic_extract_image_chip(cimg, chip_locations[0], chip);
                //win_faces.set_image(tile_images(face_chips));
                */
                //close ROI

                //fill it

                //use it as mask

                //bonus: apply blend area (alpha channel)


                //RotatedRect  fitEllipse(InputArray points)

                // Display it all on the screen
                //win.clear_overlay();
                //win.set_image(cimg);

                render_exterior_landmarks(temp,shapes,0,26);
                render_turd_on_forehead(temp,turd,shapes);
                imshow("w",temp);
                if(waitKey(30) >= 0) {break;}

                //win.set_image(face_chips[0]);
                //win.add_overlay(render_face_detections(shapes));
                //win.add_overlay(faces, rgb_pixel(255,0,0));
            }
        }
    }
    catch(serialization_error& e)
    {
        cout << "You need dlib's default face landmarking model file to run this example." << endl;
        cout << "You can get it from the following URL: " << endl;
        cout << "   http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2" << endl;
        cout << endl << e.what() << endl;
    }
    catch(exception& e)
    {
        cout << e.what() << endl;
    }
}
