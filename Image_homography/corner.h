//

// file : corner.h
//-----------------------------
// this file contains functions for corner
// detection
//
// parameters for window sum (sum fxfx, ...)

#define WINDOW_OPTION CV_BLUR
//#define WINDOW_OPTION CV_GAUSSIAN
#define PARAM1 3
#define PARAM2 3
#define RANGE_NEIGHBOR 6

void HarrisCornerDectect(IplImage *inImage, IplImage *cornerMap,
float threshold);
void GetffImage(IplImage *inImage, IplImage *outImage,
    CvMat *kernel1st, CvMat *kernel2nd, CvPoint anchor,
int windowOption, int param1, int param2);
void FindLocalMaxPoint(IplImage *inImage, IplImage *map, int range);
