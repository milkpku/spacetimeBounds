#include "utils/CharPose.h"
#include <gtest/gtest.h>

#include <fstream>

namespace {
std::string char_file("atlas.txt");
std::string zero_pos("atlas_rest_joint_pos.txt");
std::string link_zero_pos("atlas_rest_link_pos.txt");
std::string one_pos("atlas_one_joint_pos.txt");
std::string link_one_pos("atlas_one_link_pos.txt");
}

using namespace kin;

TEST(Atlas, Init)
{
  CharPose character(char_file);
  std::ofstream out("atlas_zero_pose.txt");
  out << character.getPose();
  out.close();
}

// TEST(Atlas, JointZeroQuat)
// {
//   CharDesc data(char_file);
//   CharPose character(data);

//   for (int i = 0; i < character.getNumJoints(); i++)
//   {
//     dQuat quat = character.getJointLocalQuat(i);
//     EXPECT_EQ(quat.w(), 1) << data.joints[i].Type;
//     EXPECT_EQ(quat.x(), 0);
//     EXPECT_EQ(quat.y(), 0);
//     EXPECT_EQ(quat.z(), 0);
//   }
// }

TEST(Atlas, JointZeroPos)
{
  CharPose character(char_file);

  std::ifstream in(zero_pos);

  for (int i = 0; i < character.getNumJoints(); i++)
  {
    dVec3 j_pos = character.getJointGlobalPos(i);

    double x, y, z;
    in >> x >> y >> z;

    EXPECT_LT(abs(x - j_pos.x()), 1e-7) << "Joint" << i;
    EXPECT_LT(abs(y - j_pos.y()), 1e-7) << "Joint" << i;
    EXPECT_LT(abs(z - j_pos.z()), 1e-7) << "Joint" << i;
  }
}

TEST(Atlas, LinkZeroPos)
{
  CharPose character(char_file);

  std::ifstream in(link_zero_pos);

  for (int i = 0; i < character.getNumJoints(); i++)
  {
    dVec3 j_pos = character.getBodyGlobalPos(i);

    double x, y, z;
    in >> x >> y >> z;

    EXPECT_LT(abs(x - j_pos.x()), 1e-7) << "Joint" << i;
    EXPECT_LT(abs(y - j_pos.y()), 1e-7) << "Joint" << i;
    EXPECT_LT(abs(z - j_pos.z()), 1e-7) << "Joint" << i;
  }
}

TEST(Atlas, JointOnePos)
{
  CharPose character(char_file);

  std::ifstream in(one_pos);

  dVec pose = character.getPose();
  pose.tail(30).array() = 1.0;
  character.setPose(pose);

  for (int i = 0; i < character.getNumJoints(); i++)
  {
    dVec3 j_pos = character.getJointGlobalPos(i);

    double x, y, z;
    in >> x >> y >> z;

    EXPECT_LT(abs(x - j_pos.x()), 2e-7) << "Joint" << i;
    EXPECT_LT(abs(y - j_pos.y()), 2e-7) << "Joint" << i;
    EXPECT_LT(abs(z - j_pos.z()), 2e-7) << "Joint" << i;
  }
}

TEST(Atlas, LinkOnePos)
{
  CharPose character(char_file);

  std::ifstream in(link_one_pos);

  dVec pose = character.getPose();
  pose.tail(30).array() = 1.0;
  character.setPose(pose);

  for (int i = 0; i < character.getNumJoints(); i++)
  {
    dVec3 j_pos = character.getBodyGlobalPos(i);

    double x, y, z;
    in >> x >> y >> z;

    EXPECT_LT(abs(x - j_pos.x()), 2e-7) << "Joint" << i;
    EXPECT_LT(abs(y - j_pos.y()), 2e-7) << "Joint" << i;
    EXPECT_LT(abs(z - j_pos.z()), 2e-7) << "Joint" << i;
  }
}