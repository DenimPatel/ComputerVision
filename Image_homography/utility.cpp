//
// file : utility.cpp
//----------------------------------------
// this file contains utility functions to
// deal with general processes.
//
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <opencv/cv.h>
#include <opencv/cxcore.h>
#include <opencv/highgui.h>
#include "utility.h"
void Array2CvMat(float *arr, CvMat *cvArr, int row, int column)
{
  int i, j;
  for(i = 0; i < row; i++){
    for(j = 0; j < column; j++){
      cvmSet(cvArr, i, j, arr[i*column + j]);
    }
  }
}
void CvMat2Array(CvMat *cvArr, float *arr, int row, int column)
{
  int i, j;
  for(i = 0; i < row; i++){
    for(j = 0; j < column; j++){
      arr[i*column + j] = cvmGet(cvArr, i, j);
    }
  }
}
void CvImageCopyFloat2Uchar(IplImage *src, IplImage *dst)
{
  int i, j, k;
  float pixel;
  int height = src->height;
  int width = src->width;
  int channels = src->nChannels;
  int step = dst->widthStep;
  uchar *dstData = (uchar *)dst->imageData;
  float *srcData = (float *)src->imageData;
  // copy float precision image to uchar precision image
  for(i = 0; i < height; i++){
    for(j = 0; j < width; j++){
      for(k = 0; k < channels; k++){
        pixel = srcData[i*step + j*channels + k];
        pixel = (pixel > 255 ? 255 : pixel);
        pixel = (pixel < 0 ? 0 : pixel);
        dstData[i*step + j*channels + k] = (uchar)pixel;
      }
    }
  }
}
void CvImageCopyUchar2Float(IplImage *src, IplImage *dst)
{
  int i, j, k;
  int height = src->height;
  int width = src->width;
  int channels = src->nChannels;
  int step = src->widthStep;
  float *dstData = (float *)dst->imageData;
  uchar *srcData = (uchar *)src->imageData;
  // copy uchar precision image to float precision image
  for(i = 0; i < height; i++){
    for(j = 0; j < width; j++){
      for(k = 0; k < channels; k++){
        dstData[i*step + j*channels + k]
        = (float)srcData[i*step + j*channels + k];
      }
    }
  }
}
void InitializeImage(IplImage *image)
{
  int i, j, k;
  int height = image->height;
  int width = image->width;
  int channels = image->nChannels;
  int step = image->widthStep;
  uchar *imageData = (uchar *)image->imageData;
  for(i = 0; i < height; i++){
    for(j = 0; j < width; j++){
      for(k = 0; k < channels; k++){
        imageData[i*step + j*channels + k]
        = 0;
      }
    }
  }
}
void CombineTwoImages(IplImage *image1, IplImage *image2,
  IplImage *outImage)
  {
    int i, j, k;
    uchar *outImageData = 0, *image1Data = 0, *image2Data = 0;
    int height = image1->height;
    int width = image1->width;
    int step = image1->widthStep;
    int channels = image1->nChannels;
    int outWidth = outImage->width;
    int outHeight = outImage->height;
    int outStep = outImage->widthStep;
    if(outWidth == width * 2 && outHeight == height){
    }else if(outWidth == width && outHeight == height * 2){
    }else{
      printf("image combining error\n");
      exit(0);
    }
    outImageData = (uchar *)outImage->imageData;
    image1Data = (uchar *)image1->imageData;
    image2Data = (uchar *)image2->imageData;
    for(i = 0; i < outHeight; i++){
      for(j = 0; j < outWidth; j++){
        for(k = 0; k < channels; k++){
          if(i < height && j < width){
            outImageData[i*outStep + j*channels + k]
            = image1Data[i*step + j*channels + k];
          }else if((i >= height && j < width)){
            outImageData[i*outStep + j*channels + k]
            = image2Data[(i-height)*step + j*channels + k];
          }else if((i < height && j >= width)){
            outImageData[i*outStep + j*channels + k]
            = image2Data[i*step + (j-width)*channels + k];
          }else{
            printf("there is no i > height & j > width \n");
            exit(0);
          }
        }
      }
    }
  }
  void WriteImage(IplImage *image, const char* imageName) {
    if(!cvSaveImage(imageName, image)){
      printf("Could not save: %s\n", imageName);
    }
  }
  //
  // function : MakeImageBlock
  // usage : MakeImageBlock(block, image, centPosI, centPosJ);
  // ------------------------------------------------------------
  // This function copies a block region of the image into a block
  // for example, if block size is 3 by 3 and the position of the block
  // is i, j on the image. the resultant block will be 3 by 3 and
  // the block will be copied by image(i-1, j-1) ... image(i+1, j+1).
  //
  void MakeImageBlock(IplImage *block, IplImage *image,
    int centPosI, int centPosJ)
    {
      uchar *blockData = 0, *imageData = 0;
      int blockHeight, blockWidth, imageHeight, imageWidth;
      int blockStep, channels, imageStep;
      int i, j, k, posI, posJ;
      blockHeight = block->height;
      blockWidth = block->width;
      imageHeight = image->height;
      imageWidth = image->width;

      channels = block->nChannels;
      blockStep = block->widthStep;
      imageStep = image->widthStep;
      blockData = (uchar *)block->imageData;
      imageData = (uchar *)image->imageData;
      for(i = 0; i < blockHeight; i++){
        for(j = 0; j < blockWidth; j++){
          for(k = 0; k < channels; k++){
            posI = centPosI + i - blockHeight / 2;
            posJ = centPosJ + j - blockWidth / 2;
            posI = min(max(posI, 0), imageHeight - 1);
            posJ = min(max(posJ, 0), imageWidth - 1);
            blockData[i*blockStep + j*channels + k]
            = imageData[posI*imageStep + posJ*channels + k];
          }
        }
      }
    }
    //
    // function : TransformImage
    // usage : TransformImage(inImage, outImage, H); 
    // ---------------------------------------------
    // This function transforms input image using H
    //
    void TransformImage(IplImage *inImage, IplImage *outImage, CvMat *H)
    {
      uchar *inData;
      uchar *outData;
      int height, width, step, channels;
      int i, j, k;
      // get the input image data
      height = inImage->height;
      width = inImage->width;
      step = inImage->widthStep;
      channels = inImage->nChannels;
      inData = (uchar *)inImage->imageData;
      outData = (uchar *)outImage->imageData;
      // apply the transform to get the target image
      // --------------------------------------------
      // out(x’) = in(x) : has 2 implementation forms
      // case 1 : out(Hx) = in(x)
      // case 2 : out(x’) = in(inv(H)x’)
      CvMat *invH = cvCreateMat(3, 3, CV_32FC1);
      cvInvert(H, invH);
      float h[9];
      if(IMPLEMENTATION == 1){ // case 1 : out(Hx) = in(x)
        CvMat2Array(H, h, 3, 3);
      }else{ // case 2 : out(x’) = in(inv(H)x’)
        CvMat2Array(invH, h, 3, 3);
      }
      int ii, jj;
      float x1, x2, x3;
      for(i = 0; i < height-3; i++){ // case 1 : i, j : x, ii, jj : x’, x’ = Hx
        for(j = 0; j < width; j++){ // case 2 : i, j : x’, ii, jj : x, x = invHx’
          for(k = 0; k < channels; k++){ // x : domain, x’ : range
            x1 = h[0] * j + h[1] * i + h[2];
            x2 = h[3] * j + h[4] * i + h[5];
            x3 = h[6] * j + h[7] * i + h[8];
            ii = min(height - 1, max(0, (int)(x2 / x3)));
            jj = min(width - 1, max(0, (int)(x1 / x3)));
            if(IMPLEMENTATION == 1){ // case 1 : out(Hx) = in(x)
              outData[ii*step + jj*channels + k]
              = inData[i*step + j*channels + k];
            }else{ // case 2 : out(x’) = in(inv(H)x’)
              if(ii == 0 || ii == height -1 || jj == 0 || jj == width - 1){
                outData[i*step + j*channels + k] = 0;
              }else{
                outData[i*step + j*channels + k]
                = inData[ii*step + jj*channels + k];
              }
            }
          }
        }
      }
    }
