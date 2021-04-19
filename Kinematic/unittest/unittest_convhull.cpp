#include "convhull/convhull_volume.h"
#include <Eigen/Dense>
#include <gtest/gtest.h>

TEST(ConvHull, PolygonVol)
{
  Eigen::MatrixXi faces(3, 4);
  faces << 0, 1, 2, 3,
           1, 3, 3, 1,
           2, 2, 0, 0;
           
  for(int i = 0; i < 100; i++)
  {
    Eigen::MatrixXd verts = Eigen::MatrixXd::Random(3, 4);
    double vol = polygonVolume(verts, faces);

    Eigen::Matrix3d tet;
    tet.col(0) = verts.col(0) - verts.col(3);
    tet.col(1) = verts.col(1) - verts.col(3);
    tet.col(2) = verts.col(2) - verts.col(3);
    double vol_ref = tet.determinant() / 6;

    EXPECT_LT(std::abs(vol_ref-vol), 1e-6);

  }
}

TEST(ConvHull, CubeVol)
{
  Eigen::MatrixXi faces(12, 3);
  faces << 0, 1, 4,
           1, 5, 4,
           1, 2, 5,
           2, 6, 5,
           2, 7, 6,
           2, 3, 7,
           3, 4, 7,
           3, 0, 4, 
           4, 5, 7,
           5, 6, 7,
           0, 3, 1,
           1, 3, 2;
  faces.transposeInPlace();

  Eigen::MatrixXd verts(8, 3);
  verts << 0, 0, 0,
           1, 0, 0,
           1, 1, 0, 
           0, 1, 0,
           0, 0, 1,
           1, 0, 1,
           1, 1, 1, 
           0, 1, 1;
  verts.transposeInPlace();

  double vol = polygonVolume(verts, faces);

  EXPECT_EQ(vol, 1.0f);
}

TEST(ConvHull, ConvexHull_uperbound)
{
  for (int i = 0; i < 100; i++)
  {
    Eigen::MatrixXd verts = Eigen::MatrixXd::Random(3, 200);   

    double vol = convexHullVolume(verts);
    EXPECT_LT(vol, 8);
  }
}

TEST(ConvHull, ConvexBall_lowerbound)
{
  for (int i = 0; i < 10; i++)
  {
    Eigen::MatrixXd verts = Eigen::MatrixXd::Random(3, 2000);   

    double vol = convexHullVolume(verts);
    EXPECT_LT(vol, 8);
    EXPECT_GT(vol, 7.58);
  }
}
