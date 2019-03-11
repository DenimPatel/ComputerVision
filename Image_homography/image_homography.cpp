//
//file : Lab4.cpp
//------------------------------------------------------------------
//1. calculates corner points of 2 input images.
//2. using the corner points, finds correspondences of the 2 images
//3. RANSAC
//4. Estimate homography 

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include <opencv/cv.h>
#include <opencv/cxcore.h>
#include <opencv/highgui.h>

#include "utility.h"
#include "corner.h"
#include "correspEstimation.h"
#include "homographyEst.h"

void Result1(IplImage *inImage1, IplImage *inImage2, IplImage *cornerMap1, IplImage *cornerMap2);
void Result2(IplImage *inImage1, IplImage *inImage2, CorspMap *corspMap, const char* outputName);
void Result3(IplImage *inImage1, IplImage *inImage2, CvMat *H, const char* outputName);
void MarkCornerPoints(IplImage *image, IplImage *cornerMap);
void DrawCorrespLine(IplImage *domainImage, IplImage *rangeImage,
IplImage *outImage, CorspMap *corspMap);

int main(int argc, char **argv) {
  // declaration
  IplImage *inImage1 = 0, *inImage2 = 0;
  IplImage *cornerMap1 = 0, *cornerMap2 = 0;
  int  height, width;
  float threshold;
  CorspMap corspMap, inlierMap;
  char domainImageName[80], rangeImageName[80];
  CvMat *H;
  if(argc == 4){
    strcpy(domainImageName, argv[1]);
    strcpy(rangeImageName, argv[2]);
    threshold = atof(argv[3]);
  }
  else{
    printf("\n");
    printf(" Usage: Lab4 domainImage rangeImage [threshold of corner detect]\n");
    printf("\n");
  }
  // load input images
  inImage1 = cvLoadImage(domainImageName, 1);
  if(!inImage1){
    printf("Could not load image file: %s\n", domainImageName);
    exit(0);
  }
  inImage2 = cvLoadImage(rangeImageName, 1);
  if(!inImage2){
    printf("Could not load image file: %s\n", rangeImageName);
    exit(0);
  }
  ////////////////////////////////////////////////////////////
  // 1. Interest Points
  //  find corners using Harris corner detector

  ////////////////////////////////////////////////////////////
  height = inImage1->height;
  width = inImage1->width;
  cornerMap1 = cvCreateImage(cvSize(width, height), IPL_DEPTH_8U, 1);
  cornerMap2 = cvCreateImage(cvSize(width, height), IPL_DEPTH_8U, 1);
  // apply Harris corner detection algorithm into 2 images
  HarrisCornerDectect(inImage1, cornerMap1, threshold);
  HarrisCornerDectect(inImage2, cornerMap2, threshold);
  // write and plot results
  Result1(inImage1, inImage2, cornerMap1, cornerMap2);
  ////////////////////////////////////////////////////////////
  // 2. Putative Correspondences
  //
  // find correspondences using NCC
  //
  ////////////////////////////////////////////////////////////
  InitializeCorspMap(&corspMap); // initialize correspondence map
  // estimate correspondences
  // (inImage1 : domain image, inImage2 : range image)
  CorrespEstimation(inImage1, inImage2, cornerMap1, cornerMap2, &corspMap);
  // write and plot results
  Result2(inImage1, inImage2, &corspMap, "result_step2.jpg");
  printf("Number of Correspondences = %d\n", corspMap.len);
  ////////////////////////////////////////////////////////////
  //  3. RANSAC
  // find inliers
  //
  ////////////////////////////////////////////////////////////
  H = cvCreateMat(3, 3, CV_32FC1);
  InitializeCorspMap(&inlierMap); // initialize inliers map
  RansacHomograhyEstimation(&corspMap, &inlierMap, H);
  // write and plot
  Result2(inImage1, inImage2, &inlierMap, "result_step3_inliers.jpg");
  Result3(inImage1, inImage2, H, "result_step3.jpg");
  printf("Number of results inliers = %d\n", inlierMap.len);

  ////////////////////////////////////////////////////////////
  //               4. Refinement              //
  //              recalculate the homoraphy              //
  ////////////////////////////////////////////////////////////
  HomograhyEstimation(&inlierMap, H);
  Result3(inImage1, inImage2, H, "result_step4.jpg");
  // release the images and matrix
  cvReleaseImage(&inImage1);
  cvReleaseImage(&inImage2);
  cvReleaseMat(&H);
  return 0;
}
//
// function : input_image->imageData
// usage : Result1(inImage1, inImage2, cornerMap1, cornerMap2);
// ----------------------------------------------------------------
// This function writes and plots the results of step 1
// (interest points).
//
void Result1(IplImage *inImage1, IplImage *inImage2,
        IplImage *cornerMap1, IplImage *cornerMap2){

  IplImage *outImage1 = 0, *outImage2 = 0, *outImage3 = 0;
  int height, width, channels;
  // create the output images
  height = inImage1->height;
  width = inImage1->width;
  channels = inImage1->nChannels;
  outImage1 = cvCreateImage(cvSize(width, height), IPL_DEPTH_8U, channels);
  outImage2 = cvCreateImage(cvSize(width, height), IPL_DEPTH_8U, channels);
  // draw a circle on the corner point
  cvCopy(inImage1, outImage1);
  cvCopy(inImage2, outImage2);
  MarkCornerPoints(outImage1, cornerMap1);
  MarkCornerPoints(outImage2, cornerMap2);
  // create the output images : 2 images in the same output image
  outImage3 = cvCreateImage(cvSize(width * 2, height), IPL_DEPTH_8U, channels);
  CombineTwoImages(outImage1, outImage2, outImage3);
  // display the result image
  cvNamedWindow("output image", CV_WINDOW_AUTOSIZE);
  cvShowImage("output image", outImage3);
  cvWaitKey(0);
  cvDestroyWindow("output image");
  // write output image
  WriteImage(outImage1, "result_corner1.jpg");
  WriteImage(outImage2, "result_corner2.jpg");
  WriteImage(outImage3, "result_step1.jpg");
  CombineTwoImages(inImage1, inImage2, outImage3);
  WriteImage(outImage3, "inputs.jpg");
  cvReleaseImage(&outImage1);
  cvReleaseImage(&outImage2);
  cvReleaseImage(&outImage3);
}
//
// function : Result2
// usage : Result2(inImage1, inImage2, corspMap, name);
// -----------------------------------------------------
// This function writes and plots the results
// (putative correspondences).
//
void Result2(IplImage *inImage1, IplImage *inImage2, CorspMap *corspMap,
 const char* outputName){
  IplImage
  *outImage = 0;
  int
  height, width, channels;
  // create the output images
  height
  = inImage1->height;
  width
  = inImage1->width;
  channels = inImage1->nChannels;
  // create the output images : 2 images in the same output image
  outImage = cvCreateImage(cvSize(width * 2, height), IPL_DEPTH_8U, channels);
  // draw correspondence lines on the resultant images
  DrawCorrespLine(inImage1, inImage2, outImage, corspMap);
  // display the result image
  cvNamedWindow("output image", CV_WINDOW_AUTOSIZE);
  cvShowImage("output image", outImage);
  cvWaitKey(0);
  cvDestroyWindow("output image");
  // write output image
  WriteImage(outImage, outputName);
  cvReleaseImage(&outImage);
}
//
// function : Result3
// usage : Result3(inImage1, inImage2, H, name);
// -----------------------------------------------------
// This function writes and plots the transformed image
// results.
//
void Result3(IplImage *inImage1, IplImage *inImage2, CvMat *H,
  const char* outputName){
  IplImage
  *outImage1 = 0;
  int
  height, width, channels;
  // get the input image data
  height
  = inImage1->height;
  width
  = inImage1->width;
  channels = inImage1->nChannels;
  outImage1 = cvCreateImage(cvSize(width, height), IPL_DEPTH_8U, channels);
  // transform inImage1 using H
  TransformImage(inImage1, outImage1, H);
  // create the output images : 2 images in the same output image
  IplImage * resultImage;
  resultImage = cvCreateImage(cvSize(width * 2, height), IPL_DEPTH_8U, channels);
  CombineTwoImages(outImage1, inImage2, resultImage);
  // display the result image
  cvNamedWindow("output image", CV_WINDOW_AUTOSIZE);
  cvShowImage("output image", resultImage);
  cvWaitKey(0);
  cvDestroyWindow("output image");
  // write output image
  WriteImage(resultImage, outputName);
  char name[80];
  sprintf(name, "transformed_%s", outputName);
  WriteImage(outImage1, name);
  cvReleaseImage(&resultImage);
  cvReleaseImage(&outImage1);
}
//
// function : MarkCornerPoints
// usage : MarkCornerPoints(image, cornerMap);
// -------------------------------------------------
// This function draws marks in input image corresponding
// to the corner map.
//
void MarkCornerPoints(IplImage *image, IplImage *cornerMap) {
  int i, j;
  uchar *cornerMapData = 0;
  int height
  = cornerMap->height;
  int width
  = cornerMap->width;
  int mapStep = cornerMap->widthStep;
  cornerMapData = (uchar *)cornerMap->imageData;
  for(i = 0; i < height; i++){
    for(j = 0; j < width; j++){
      if(cornerMapData[i*mapStep + j] == true){
        cvCircle(image, cvPoint(j, i), 2, cvScalar(0, 255, 0), 2);
      }
    }
  }
}
//
// function : DrawCorrespLine
// usage : DrawCorrespLine(image1, image2, outImage, corspMap);
// -------------------------------------------------------------
// This function draws lines for correspondences of image 1 and 2.
//
void DrawCorrespLine(IplImage *domainImage, IplImage *rangeImage,
          IplImage *outImage, CorspMap *corspMap){
  int a, b, i;
  uchar *outImageData = 0, *domainImageData = 0, *rangeImageData = 0;
  CvPoint rangePos;
  CvPoint domainPos;
  int height = rangeImage->height;
  int width  = rangeImage->width;


  if(height == outImage->height && width * 2 == outImage->width){
    a = 1; b = 0;
  }else if(height * 2 == outImage->height && width == outImage->width){
    a = 0; b = 1;
  }else{
    printf("Error\n");
    exit(0);
  }
  outImageData = (uchar *)outImage->imageData;
  domainImageData = (uchar *)domainImage->imageData;
  rangeImageData = (uchar *)rangeImage->imageData;
  // create output image that contains ref & test images
  CombineTwoImages(domainImage, rangeImage, outImage);
  // draw correspondence lines & corner points
  for(i = 0; i < corspMap->len; i++){
    rangePos = cvPoint(corspMap->rangeImagePositionJ[i] + width * a,
      corspMap->rangeImagePositionI[i] + height * b);
    domainPos = cvPoint(corspMap->domainImagePositionJ[i],
      corspMap->domainImagePositionI[i]);

    cvCircle(outImage, domainPos, 2, cvScalar(0, 255, 0), 2);
    cvCircle(outImage, rangePos, 2, cvScalar(255, 0, 0), 2);
    cvLine(outImage, domainPos, rangePos, cvScalar(0, 0, 255), 1);
  }
}
