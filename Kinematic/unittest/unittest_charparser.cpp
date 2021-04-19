#include "utils/CharParser.h"
#include <vector>
#include <gtest/gtest.h>

namespace {
std::string char_file("humanoid3d.txt");
}

using namespace kin;

TEST(CharParser, LoadJson)
{
  CharDesc character(char_file);

  EXPECT_EQ(character.joints.size(), 15);
  EXPECT_EQ(character.bodys.size(), 15);
}

TEST(CharParser, JointStr)
{
  CharDesc character(char_file);

  std::vector<std::string> names = { "root", "chest", "neck" };
  std::vector<std::string> types = { "none", "spherical", "spherical" };

  for (int i = 0; i < names.size(); i++)
  {
    EXPECT_EQ(character.joints[i].Name, names[i]) << character.joints[i].Name;
    EXPECT_EQ(character.joints[i].Type, types[i]) << character.joints[i].Type;
  }
}

TEST(CharParser, BodyStr)
{
  CharDesc character(char_file);

  std::vector<std::string> names = { "root", "chest", "neck" };
  std::vector<std::string> shapes = { "sphere", "sphere", "sphere" };

  for (int i = 0; i < names.size(); i++)
  {
    EXPECT_EQ(character.bodys[i].Name, names[i]) << character.bodys[i].Name;
    EXPECT_EQ(character.bodys[i].Shape, shapes[i]) << character.bodys[i].Shape;
  }
}

TEST(CharParser, BodyInt)
{
  CharDesc character(char_file);

  std::vector<int> ids = { 0, 1, 2, 3 };
  std::vector<int> colgroups = { 1, 1, 1, 1 };

  for (int i = 0; i < ids.size(); i++)
  {
    EXPECT_EQ(character.bodys[i].ID, ids[i]) << character.bodys[i].ID;
    EXPECT_EQ(character.bodys[i].ColGroup, colgroups[i]) << character.bodys[i].ColGroup;
  }
}
