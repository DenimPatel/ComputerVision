//
// file : homographyEst.h
//------------------------------------------------
// this file contains functions for homography 
// estimation
//
#define T_SQUARE 36 // t = sqrt(6)* sigma and set sigma = sqrt(6)
void HomograhyEstimation(CorspMap *inlierMap, CvMat *H);
void RansacHomograhyEstimation(CorspMap *corspMap, CorspMap *inlierMap, CvMat *H);
void ComputeHomography(float domainPosiitons[][2], float rangePosiitons[][2],
                      int numOfCorresp, CvMat *H);
void DataNormalization(int numOfPositions, float positions[][2], CvMat *T);
void CalculateDistance(CvMat *H, CorspMap *corspMap, CorspMap *inlierMap);
bool IsGoodSample(float points[][2], int numOfPoints);
bool IsColinear(CvMat *A, CvMat *B, CvMat *C);
