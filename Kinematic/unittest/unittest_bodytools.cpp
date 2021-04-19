#include "utils/BodyTools.h"
#include <gtest/gtest.h>

using namespace kin;

TEST(BodyTools, boxAABB)
{
  for(int i = 0; i < 100; i++)
  {
    auto shape_param = dVec3::Random();
    auto quat = dQuat::UnitRandom();

    auto aabb = AABB(BodyShape::BOX, quat, shape_param);
  }
}

TEST(BodyTools, sphereAABB)
{
  using namespace kin;
  for(int i = 0; i < 100; i++)
  {
    auto shape_param = dVec3::Random();
    auto quat = dQuat::UnitRandom();

    auto aabb = AABB(BodyShape::SPHERE, quat, shape_param);
  }
}

TEST(BodyTools, capsuleAABB)
{
  using namespace kin;
  for(int i = 0; i < 100; i++)
  {
    auto shape_param = dVec3::Random();
    auto quat = dQuat::UnitRandom();

    auto aabb = AABB(BodyShape::CAPSULE, quat, shape_param);
  }
}

TEST(BodyTools, boxInertia)
{
  using namespace kin;
  for(int i = 0; i < 100; i++)
  {
    double mass = 1.0;
    auto shape_param = dVec3::Random();

    auto iner_vec = inertia(BodyShape::BOX, mass, shape_param);
  }
}

TEST(BodyTools, sphereInertia)
{
  using namespace kin;
  for(int i = 0; i < 100; i++)
  {
    double mass = 1.0;
    auto shape_param = dVec3::Random();

    auto iner_vec = inertia(BodyShape::SPHERE, mass, shape_param);
  }
}

TEST(BodyTools, capsuleInertia)
{
  using namespace kin;
  for(int i = 0; i < 100; i++)
  {
    double mass = 1.0;
    auto shape_param = dVec3::Random();

    auto iner_vec = inertia(BodyShape::CAPSULE, mass, shape_param);
  }
}

