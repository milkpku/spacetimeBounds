#include "utils/CharPose.h"
#include <gtest/gtest.h>

#include <fstream>

namespace {
std::string char_file("humanoid3d.txt");
std::string zero_pos("humanoid3d_pos_zero.txt");
}

using namespace kin;

TEST(CharPose, Init)
{
  CharPose character(char_file);
  std::ofstream out("humanoid3d_zero_pose.txt");
  out << character.getPose();
  out.close();
}

TEST(CharPose, JointZeroQuat)
{
  CharDesc data(char_file);
  CharPose character(data);

  for (int i = 0; i < character.getNumJoints(); i++)
  {
    dQuat quat = character.getJointLocalQuat(i);
    EXPECT_EQ(quat.w(), 1) << data.joints[i].Type;
    EXPECT_EQ(quat.x(), 0);
    EXPECT_EQ(quat.y(), 0);
    EXPECT_EQ(quat.z(), 0);
  }
  
}

TEST(CharPose, JointZeroPos)
{
  CharPose character(char_file);

  std::ifstream in(zero_pos);

  for (int i = 0; i < character.getNumJoints(); i++)
  {
    dVec3 j_pos = character.getJointGlobalPos(i);

    double x, y, z;
    in >> x >> y >> z;

    EXPECT_LT(abs(x - j_pos.x()), 1e-8) << "Joint" << i;
    EXPECT_LT(abs(y - j_pos.y()), 1e-8) << "Joint" << i;
    EXPECT_LT(abs(z - j_pos.z()), 1e-8) << "Joint" << i;
  }
}

TEST(CharPose, buildState)
{

  CharPose character(char_file);

  std::ifstream state_in("state_20.txt");
  std::ifstream pose_in("pose_20.txt");
  std::ifstream vel_in("pose_20.txt");

  dVec state(196), pose(43), vel(43);

  for (int i = 0; i < state.size(); i++)
    state_in >> state(i);

  for (int i = 0; i < pose.size(); i++)
    pose_in >> pose(i);

  for (int i = 0; i < vel.size(); i++)
    vel_in >> vel(i);


  character.setPose(pose);
  character.setVel(vel);
  
  dVec state_calc = character.buildState();
}

TEST(CharPose, heading)
{

  CharPose character(char_file);
  
  dVec3 head_vec(1, 2, 3);
  character.setHeadingVec(head_vec);

  for (int i = 0; i < 100; i++)
  {
    double theta = rand() / RAND_MAX;
    theta = 2 * (theta - 0.5) * 3.1;
    dVec q(4);
    q(0) = cos(theta/2);
    q(1) = 0;
    q(2) = sin(theta/2);
    q(3) = 0;
  
    double theta_calc = character.getHeadingTheta(q);

    EXPECT_EQ(theta_calc, theta) << theta << " " << theta_calc;
  }
}

TEST(CharPose, printStateDiff)
{
  CharPose character(char_file);
  character.printStateDiff();
}

TEST(CharPose, printBodyInertia)
{
  CharPose character(char_file);
  character.printBodyInertia();
}
