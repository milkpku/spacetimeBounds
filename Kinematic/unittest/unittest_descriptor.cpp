#include "utils/Descriptor.h"
#include <gtest/gtest.h>

namespace {
std::string char_file("humanoid3d.txt");
}

using namespace kin;

TEST(Descriptor, momentum)
{
  CharPose character(char_file);
  std::vector<double> p = momentum(&character);
}

TEST(Descriptor, kinematicEnergy_no_inf)
{
  CharPose character(char_file);
  double energy = kinematicEnergy(&character, false);

  EXPECT_LT(abs(energy), 1e-8);
}

TEST(Descriptor, gravitationalEnergy)
{
  CharPose character(char_file);
  double energy = gravitationalEnergy(&character);

  dVec pose = character.getPose();

  pose(1) += 1.0;
  character.setPose(pose);

  double energy_new = gravitationalEnergy(&character);

  double err = energy_new - energy - 9.8 * character.getCharMass();

  EXPECT_LT(abs(err), 1e-8);
}

TEST(Descriptor, jointVelActivation)
{
  CharPose character(char_file);
  std::vector<double> w;
  w.resize(15, 0);
  w[1] = 1;
  w[10] = 1;
  double activation = jointVelActivation(&character, w);
  EXPECT_EQ(activation, 0);
}

TEST(Descriptor, bodyConvexHullVolume)
{
  CharPose character(char_file);
  std::vector<int> b_id = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14};
  float vol = bodyConvexHullVolume(&character, b_id);
  EXPECT_GT(vol, 0);
}

TEST(Descriptor, jointConvexHullVolume)
{
  CharPose character(char_file);
  std::vector<int> j_id = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14};
  float vol = jointConvexHullVolume(&character, j_id);
  EXPECT_GT(vol, 0);
}

TEST(Descriptor, jointConvexHullVolume_special)
{
  CharPose character(char_file);
  dVec pose(43);
  pose << 0.58097345,0.79772097,-0.16030939,0.87853807,0.02499798,-0.47620127,-0.027899,0.99906868,-0.01961466,0.02750356,-0.02684709,0.97187734,-0.06675785,0.04450668,0.22139804,0.95758319,-0.16542706,0.06017464,0.22813883,-0.6183266,0.98026448,0.15066035,0.04973928,0.1179364,0.946055,-0.31407487,0.03549786,-0.07125171,1.89685118,0.94326651,0.13686667,0.15416288,0.26028776,-1.15701187,0.99125797,0.09999075,-0.04501165,0.07337309,0.94231844,0.32443854,0.04605841,0.06822278,1.71324003;
  character.setPose(pose);
  std::vector<int> j_id = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14};
  float vol = jointConvexHullVolume(&character, j_id);
  EXPECT_GT(vol, 0);
}
