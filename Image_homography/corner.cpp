//
// file : corner.cpp
//-----------------------------
// this file contains functions for corner 
// detection
//
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <opencv/cv.h>
#include <opencv/cxcore.h>
#include <opencv/highgui.h>
#include "utility.h"
#include "corner.h"
void HarrisCornerDectect(IplImage *inImage,
  IplImage *cornerMap, float threshold)
  {
    IplImage *tempImage = 0;
    IplImage *fxfxImage = 0, *fxfyImage = 0, *fyfyImage = 0;
    IplImage *eigenvalueImage = 0;
    float *eigenData = 0;
    float *fxfxData = 0, *fxfyData = 0, *fyfyData = 0;
    float fxfx, fxfy, fyfy;
    float slambda, lambda1, lambda2, lambda3;
    int i, j, k;
    int height, width, channels, step;
    int eigStep;
    CvPoint anchor;
    float dx[9] = {-1, 0, 1, -1, 0, 1, -1, 0, 1}; // derivative masks
    float dy[9] = {1, 1, 1, 0, 0, 0, -1, -1, -1}; // derivative masks
    float sMask[9] = {1, 1, 1, 1, 1, 1, 1, 1, 1}; // window sum mask
    CvMat *Dx = cvCreateMat(3, 3, CV_32FC1); // derivative mask
    CvMat *Dy = cvCreateMat(3, 3, CV_32FC1); // derivative mask
    CvMat *window = cvCreateMat(3, 3, CV_32FC1); // window sum mask
    CvMat *G = cvCreateMat(2, 2, CV_32FC1); // Harris Matrix
    CvMat *q = cvCreateMat(2, 2, CV_32FC1); // eigenvector of G
    CvMat *lambda = cvCreateMat(2, 1, CV_32FC1); // eigenvalue of G
    // assign predefined values to matrices
    Array2CvMat(dx, Dx, 3, 3);
    Array2CvMat(dy, Dy, 3, 3);
    Array2CvMat(sMask, window, 3, 3);
    height = inImage->height;
    width = inImage->width;
    channels = inImage->nChannels;
    step = inImage->widthStep;
    eigStep = cornerMap->widthStep;
    // create the processing image
    tempImage = cvCreateImage(cvSize(width, height), IPL_DEPTH_8U, channels);
    fxfxImage = cvCreateImage(cvSize(width, height), IPL_DEPTH_32F, channels);
    fxfyImage = cvCreateImage(cvSize(width, height), IPL_DEPTH_32F, channels);
    fyfyImage = cvCreateImage(cvSize(width, height), IPL_DEPTH_32F, channels);
    eigenvalueImage = cvCreateImage(cvSize(width, height), IPL_DEPTH_32F, 1);
    fxfxData = (float *)fxfxImage->imageData;
    fxfyData = (float *)fxfyImage->imageData;
    fyfyData = (float *)fyfyImage->imageData;
    eigenData = (float *)eigenvalueImage->imageData;
    // LPF filtering to reduce noise
    cvSmooth(inImage, tempImage, CV_GAUSSIAN, 3, 0, 0);
    // get fxfx, fxfy, fyfy images
    anchor = cvPoint(0, 0);
    GetffImage(tempImage, fxfxImage, Dx, Dx, anchor, WINDOW_OPTION, PARAM1, PARAM2);
    GetffImage(tempImage, fyfyImage, Dy, Dy, anchor, WINDOW_OPTION, PARAM1, PARAM2);
    GetffImage(tempImage, fxfyImage, Dx, Dy, anchor, WINDOW_OPTION, PARAM1, PARAM2);
    cvReleaseImage(&tempImage);
    // every fxfx, fxfy, and fyfy image pixel is summed by window
    // to construct the matrix
    // [window sum (fxfx) window sum (fxfy)]
    // [window sum (fxfy) window sum (fyfy)]
    // find small eigenvalues for each pixel
    for(i = 0; i < height; i++){
      for(j = 0; j < width; j++){
        for(k = 0; k < channels; k++){
          fxfx = fxfxData[i*step + j*channels + k];
          fxfy = fxfyData[i*step + j*channels + k];
          fyfy = fyfyData[i*step + j*channels + k];
          // set matrix G = [window sum (fxfx) window sum (fxfy)]
          // [window sum (fxfy) window sum (fyfy)]
          cvmSet(G, 0, 0, fxfx);
          cvmSet(G, 0, 1, fxfy);
          cvmSet(G, 1, 0, fxfy);
          cvmSet(G, 1, 1, fyfy);
          // eigen value decomapStepmposition
          cvEigenVV(G, q, lambda);
          // lambda = eigenvalues of G (descending order)
          // q = corresponding orthogonal eigenvectors (rows)
          if(channels == 3){
            if(k == 0)
            lambda1 = cvmGet(lambda, 1, 0); // lambda for B
            else if(k == 1)
            lambda2 = cvmGet(lambda, 1, 0); // lambda for G
            else
            lambda3 = cvmGet(lambda, 1, 0); // lambda for R
          }else{ // channels == 1
            lambda1 = cvmGet(lambda, 1, 0);
          }
        }
        if(channels == 3){
          // slambda = length of the vector [lambda1, lambda2, lambda3]
          slambda = pow(pow(lambda1,2) + pow(lambda1,2) + pow(lambda1,2), .5);
        }else{
          slambda = lambda1;
        }
        // store the small eigen values that are normalized by threshold
        eigenData[i*eigStep + j] = slambda / threshold;
      }
    }
    // fine local maximum corner points
    FindLocalMaxPoint(eigenvalueImage, cornerMap, RANGE_NEIGHBOR);
    // release images
    cvReleaseImage(&tempImage);
    cvReleaseImage(&fxfxImage);
    cvReleaseImage(&fxfyImage);
    cvReleaseImage(&fyfyImage);
    cvReleaseImage(&eigenvalueImage);
    // release matrices
    cvReleaseMat(&Dx); cvReleaseMat(&Dy);
    cvReleaseMat(&window);
    cvReleaseMat(&G); cvReleaseMat(&q); cvReleaseMat(&lambda);
  }
  void GetffImage(IplImage *inImage, IplImage *outImage,
    CvMat *kernel1st, CvMat *kernel2nd, CvPoint anchor,
    int windowOption, int param1, int param2)
    {
      IplImage *f1Image = 0, *f2Image = 0, *tempImage = 0;
      float *f1Data = 0, *f2Data = 0;
      float *outData;
      int height, width, step, channels;
      int i, j, k;
      // create the output image
      height = inImage->height;
      width = inImage->width;
      channels = inImage->nChannels;
      step = inImage->widthStep;
      f1Image = cvCreateImage(cvSize(width, height), IPL_DEPTH_32F, channels);
      f2Image = cvCreateImage(cvSize(width, height), IPL_DEPTH_32F, channels);
      tempImage = cvCreateImage(cvSize(width, height), IPL_DEPTH_32F, channels);
      f1Data = (float *)f1Image->imageData;
      f2Data = (float *)f2Image->imageData;
      outData = (float *)outImage->imageData;
      // copy input image to float precision image
      CvImageCopyUchar2Float(inImage, tempImage);
      cvFilter2D(tempImage, f1Image, kernel1st, anchor);
      cvFilter2D(tempImage, f2Image, kernel2nd, anchor);
      for(i = 0; i < height; i++){
        for(j = 0; j < width; j++){
          for(k = 0; k < channels; k++){
            outData[i*step + j*channels + k]
            = f1Data[i*step + j*channels + k]
            * f2Data[i*step + j*channels + k];
          }
        }
      }
      // window sum of fxfx, fxfy, or fyfy
      cvCopyImage (outImage, tempImage);
      cvSmooth(tempImage, outImage, windowOption, param1, param2);
      cvReleaseImage(&tempImage);
      cvReleaseImage(&f1Image);
      cvReleaseImage(&f2Image);
    }
    void FindLocalMaxPoint(IplImage *inImage, IplImage *map, int range)
    {
      int r, sum, numOfNeighbor;
      int i, j, ii, jj, posI, posJ;
      float current;
      float *inData = 0;
      uchar *mapData = 0;
      int height = inImage->height;
      int width = inImage->width;
      int step = map->widthStep;
      r = range / 2;
      numOfNeighbor = (2*r + 1) * (2*r + 1);
      inData = (float *)inImage->imageData;
      mapData = (uchar *)map->imageData;
      for(i = 0; i < height; i++){
        for(j = 0; j < width; j++){
          // mark the corner on image
          // write the corner position
          current = inData[i*step + j];
          if(current < 1){ // lambda < threshold
            mapData[i*step + j] = false;
          }else{
            // check neighbors
            sum = 0;
            for(ii = -r; ii <= r; ii++){
              for(jj = -r; jj <= r; jj++){
                posI = min(max((i+ii), 0), height - 1);
                posJ = min(max((j+jj), 0), width - 1);
                sum += (current >= inData[posI*step + posJ]);
              }
            }
            // if current pixel is maximum in its neighbors
            // sum == numOfNeighbor
            if(sum == numOfNeighbor)
            mapData[i*step + j] = true;
            else
            mapData[i*step + j] = false;
          }
        }
      }
    }
