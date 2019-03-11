//
// file : utility.h
//----------------------------------------
// this file contains utility functions to
// deal with general processes. 
//
#define OFF 0
#define ON 1
#define EPS 0.5
#define IMPLEMENTATION 2
#define min(a, b) ((a <= b) ? a : b)
#define max(a, b) ((a >= b) ? a : b)
void Array2CvMat(float *arr, CvMat *cvArr, int row, int column);
void CvMat2Array(CvMat *cvArr, float *arr, int row, int column);
void CvImageCopyFloat2Uchar(IplImage *src, IplImage *dst);
void CvImageCopyUchar2Float(IplImage *src, IplImage *dst);
void InitializeImage(IplImage *image);
void CombineTwoImages(IplImage *image1, IplImage *image2, IplImage *outImage);
void WriteImage(IplImage *image, const char* name);
void MakeImageBlock(IplImage *block, IplImage *image, int centPosI, int centPosJ);
void TransformImage(IplImage *inImage, IplImage *outImage, CvMat *H);
