//
// file : correspEstimation.h
//-----------------------------
// this file contains functions for correspondence
// estimation 
//
#define WINDOWSIZE_NCC 9
#define SEARCH_RANGE 256
#define MAX_POINT_SIZE 500
typedef struct{
  int len;
  int rangeImagePositionI[MAX_POINT_SIZE];
  int rangeImagePositionJ[MAX_POINT_SIZE];
  int domainImagePositionI[MAX_POINT_SIZE];
  int domainImagePositionJ[MAX_POINT_SIZE];
}CorspMap;

void CorrespEstimation(IplImage *domainImage, IplImage *rangeImage,
              IplImage *domainCornerMap, IplImage *rangeCornerMap,
              CorspMap *corspMap);
bool FindCorrespodence(IplImage *rangeImage, IplImage *domainBlock,
                      IplImage *rangeCornerMap,
                      int cornerPosI, int cornerPosJ,
                      int *pCorspPosI, int *pCorspPosJ,
                      int searchRange);
float NCC(IplImage *block1, IplImage *block2);
void UpdateCorrespMap(CorspMap *corspMap, int domainPosI, int domainPosJ,
int rangePosI, int rangePosJ);
void InitializeCorspMap(CorspMap *corspMap);
void CopyCorspMap(CorspMap *dst, CorspMap *src);
