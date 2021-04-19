#include "KinematicCore.h"
#include "utils/CharPose.h"

cKinematicCore::cKinematicCore(const std::string charfile, const double scale)
{
  _char_pose.Init(charfile, scale);
}

void cKinematicCore::setPose(const std::vector<double>& pose)
{
  Eigen::VectorXd in_pose;
  ConvertVector(pose, in_pose);
  _char_pose.setPose(in_pose);
}

void cKinematicCore::setVel(const std::vector<double>& vel)
{
  Eigen::VectorXd in_vel;
  ConvertVector(vel, in_vel);
  _char_pose.setVel(in_vel);
}

void cKinematicCore::setHeadingVec(const std::vector<double>& head)
{
  Eigen::VectorXd in_head;
  ConvertVector(head, in_head);
  assert(head.size() == 3 && "Heading direction size missmatch");
  _char_pose.setHeadingVec(in_head);
}

double cKinematicCore::getHeadingTheta(const std::vector<double>& ori)
{
  Eigen::VectorXd in_ori;
  ConvertVector(ori, in_ori);
  return _char_pose.getHeadingTheta(in_ori);
}

std::vector<double> cKinematicCore::getPose() const
{
  std::vector<double> out_pose;
  Eigen::VectorXd pose = _char_pose.getPose();
  ConvertVector(pose, out_pose);
  return out_pose;
}

std::vector<double> cKinematicCore::getVel() const
{
  std::vector<double> out_vel;
  Eigen::VectorXd vel = _char_pose.getVel();
  ConvertVector(vel, out_vel);
  return out_vel;
}

std::vector<double> cKinematicCore::buildState() const
{
  Eigen::VectorXd state = _char_pose.buildState();
  std::vector<double> out_state;
  ConvertVector(state, out_state);
  return out_state;
}

std::vector<double> cKinematicCore::buildState(const std::vector<double>& pos, const std::vector<double>& rot, const bool global_root) const
{
  assert(pos.size() == 3 && "origin_pos should be (x, y, z)");
  assert(rot.size() == 4 && "origin_rot should be (w, x, y, z)");
  Eigen::Vector3d in_pos(pos[0], pos[1], pos[2]);
  Eigen::Quaterniond in_rot(rot[0], rot[1], rot[2], rot[3]);
  in_rot.normalize();
  Eigen::VectorXd state = _char_pose.buildState(in_pos, in_rot, global_root);
  std::vector<double> out_state;
  ConvertVector(state, out_state);
  return out_state;
}

std::vector<double> cKinematicCore::getJointPos(int id) const
{
  std::vector<double> out_pos;
  Eigen::VectorXd pos = _char_pose.getJointGlobalPos(id);
  ConvertVector(pos, out_pos);
  return out_pos;
}

std::vector<double> cKinematicCore::getJointLocalOri(int id) const
{
  std::vector<double> out_aa;
  Eigen::Quaterniond quat = _char_pose.getJointLocalQuat(id);
  Eigen::AngleAxisd aa(quat);
  Eigen::VectorXd aa_vec = aa.angle() * aa.axis();
  ConvertVector(aa_vec, out_aa);
  return out_aa;
}

std::vector<double> cKinematicCore::getJointOmg(int id) const
{
  std::vector<double> out_omg;
  Eigen::VectorXd omg = _char_pose.getJointGlobalOmg(id);
  ConvertVector(omg, out_omg);
  return out_omg;
}

std::vector<double> cKinematicCore::getJointLocalOmg(int id) const
{
  std::vector<double> out_omg;
  Eigen::VectorXd omg = _char_pose.getJointLocalOmg(id);
  ConvertVector(omg, out_omg);
  return out_omg;
}

std::vector<double> cKinematicCore::getJointVel(int id) const
{
  std::vector<double> out_vel;
  Eigen::VectorXd vel = _char_pose.getJointGlobalVel(id);
  ConvertVector(vel, out_vel);
  return out_vel;
}

std::vector<double> cKinematicCore::getBodyPos(int id) const
{
  std::vector<double> out_pos;
  Eigen::VectorXd pos = _char_pose.getBodyGlobalPos(id);
  ConvertVector(pos, out_pos);
  return out_pos;
}

std::vector<double> cKinematicCore::getBodyOmg(int id) const
{
  std::vector<double> out_omg;
  Eigen::VectorXd omg = _char_pose.getBodyGlobalOmg(id);
  ConvertVector(omg, out_omg);
  return out_omg;
}

std::vector<double> cKinematicCore::getBodyVel(int id) const
{
  std::vector<double> out_vel;
  Eigen::VectorXd vel = _char_pose.getBodyGlobalVel(id);
  ConvertVector(vel, out_vel);
  return out_vel;
}

std::vector<double> cKinematicCore::getCoMPos() const
{
  Eigen::Vector3d com_pos = _char_pose.getCoMPos();
  std::vector<double> out_com_pos;
  ConvertVector(com_pos, out_com_pos);
  return out_com_pos;
}

std::vector<double> cKinematicCore::getCoMVel() const
{
  Eigen::Vector3d com_vel = _char_pose.getCoMVel();
  std::vector<double> out_com_vel;
  ConvertVector(com_vel, out_com_vel);
  return out_com_vel;
}

std::vector<double> cKinematicCore::getJointGlobalPoses(const std::vector<int>& ids) const
{
  Eigen::VectorXi in_ids;
  ConvertVector(ids, in_ids);
  Eigen::VectorXd poses = _char_pose.getJointGlobalPoses(in_ids);
  std::vector<double> out_poses;
  ConvertVector(poses, out_poses);
  return out_poses;
}

std::vector<double> cKinematicCore::slerp(
    const std::vector<double>& pose0, const std::vector<double>& pose1, const double t) const
{
  std::vector<double> out_pose;
  Eigen::VectorXd in_pose0, in_pose1;
  ConvertVector(pose0, in_pose0);
  ConvertVector(pose1, in_pose1);
  Eigen::VectorXd poset = _char_pose.slerp(in_pose0, in_pose1, t);
  ConvertVector(poset, out_pose);
  return out_pose;
}

std::vector<double> cKinematicCore::actionAsOffset(
    const std::vector<double>& pose, const std::vector<double>& action) const
{
  Eigen::VectorXd in_pose, in_action;
  ConvertVector(pose, in_pose);
  ConvertVector(action, in_action);
  Eigen::VectorXd new_action = _char_pose.actionAsOffset(in_pose, in_action);
  std::vector<double> out_act;
  ConvertVector(new_action, out_act);
  return out_act;
}

std::vector<double> cKinematicCore::expMapToTargetPose(
    const std::vector<double>& exp_map, const bool padding) const
{
  Eigen::VectorXd in_exp;
  ConvertVector(exp_map, in_exp);
  Eigen::VectorXd targ_pose = _char_pose.expMapToTargetPose(in_exp, padding);
  std::vector<double> out_pose;
  ConvertVector(targ_pose, out_pose);
  return out_pose;
}

std::vector<double> cKinematicCore::calcStateDiff(
    const std::vector<double>& pose0, const std::vector<double>& pose1,
    bool rel_root_pos, bool rel_root_ori, bool rel_endeffector)
{
  Eigen::VectorXd in_pose0, in_pose1;
  ConvertVector(pose0, in_pose0);
  ConvertVector(pose1, in_pose1);
  Eigen::VectorXd pose_diff = _char_pose.calcStateDiff(in_pose0, in_pose1, rel_root_pos, rel_root_ori, rel_endeffector);
  std::vector<double> out_diff;
  ConvertVector(pose_diff, out_diff);
  return out_diff;
}

void cKinematicCore::printStateDiff()
{
  _char_pose.printStateDiff();
}

void cKinematicCore::setStateDiffLim(const std::vector<double>& state_diff)
{
  Eigen::VectorXd in_state_diff;
  ConvertVector(state_diff, in_state_diff);
  _char_pose.setStateDiffLim(in_state_diff);
}

bool cKinematicCore::checkStateDiff(
    const std::vector<double>& pose0, const std::vector<double>& pose1,
    bool rel_root_pos, bool rel_root_ori, bool rel_endeffector)
{
  Eigen::VectorXd in_pose0, in_pose1;
  ConvertVector(pose0, in_pose0);
  ConvertVector(pose1, in_pose1);
  Eigen::VectorXd pose_diff = _char_pose.calcStateDiff(in_pose0, in_pose1, rel_root_pos, rel_root_ori, rel_endeffector);
  Eigen::VectorXd pose_lim = _char_pose.getStateDiffLim();
  assert(pose_diff.size() == pose_lim.size() && "state lim size not correct");
  double min = (pose_lim - pose_diff).minCoeff();
  return (min < 0);
}

std::vector<bool> cKinematicCore::checkStateDiffVec(
    const std::vector<double>& pose0, const std::vector<double>& pose1,
    bool rel_root_pos, bool rel_root_ori, bool rel_endeffector)
{
  Eigen::VectorXd in_pose0, in_pose1;
  ConvertVector(pose0, in_pose0);
  ConvertVector(pose1, in_pose1);
  Eigen::VectorXd pose_diff = _char_pose.calcStateDiff(in_pose0, in_pose1, rel_root_pos, rel_root_ori, rel_endeffector);
  Eigen::VectorXd pose_lim = _char_pose.getStateDiffLim();
  assert(pose_diff.size() == pose_lim.size() && "state lim size not correct");
  std::vector<bool> out_bool;
  out_bool.resize(pose_diff.size());
  for (int i = 0; i < pose_diff.size(); i++)
    out_bool[i] = pose_diff(i) > pose_lim(i);
  return out_bool;
}

double cKinematicCore::calcReward(
    const std::vector<double>& pose0, const std::vector<double>& vel0,
    const std::vector<double>& pose1, const std::vector<double>& vel1)
{
  Eigen::VectorXd in_pose0, in_vel0, in_pose1, in_vel1;
  ConvertVector(pose0, in_pose0);
  ConvertVector(vel0, in_vel0);
  ConvertVector(pose1, in_pose1);
  ConvertVector(vel1, in_vel1);
  double rwd = _char_pose.calcReward(in_pose0, in_vel0, in_pose1, in_vel1);

  return rwd;
}

std::vector<double> cKinematicCore::getErrorVec() const
{
  Eigen::VectorXd err = _char_pose.getErrorVec();
  std::vector<double> out_err;
  ConvertVector(err, out_err);
  return out_err;
}

double cKinematicCore::lowestHeight(const std::vector<double>& pose)
{
  Eigen::VectorXd in_pose;
  ConvertVector(pose, in_pose);
  _char_pose.setPose(in_pose);
  return _char_pose.lowestHeight();
}

void cKinematicCore::ConvertVector(const Eigen::VectorXd& in_vec, std::vector<double>& out_vec) const
{
  int size = static_cast<int>(in_vec.size());
  out_vec.resize(size);
  std::memcpy(out_vec.data(), in_vec.data(), size * sizeof(double));
}

void cKinematicCore::ConvertVector(const Eigen::VectorXi& in_vec, std::vector<int>& out_vec) const
{
  int size = static_cast<int>(in_vec.size());
  out_vec.resize(size);
  std::memcpy(out_vec.data(), in_vec.data(), size * sizeof(int));
}

void cKinematicCore::ConvertVector(const std::vector<double>& in_vec, Eigen::VectorXd& out_vec) const
{
  int size = static_cast<int>(in_vec.size());
  out_vec.resize(size);
  std::memcpy(out_vec.data(), in_vec.data(), size * sizeof(double));
}

void cKinematicCore::ConvertVector(const std::vector<int>& in_vec, Eigen::VectorXi& out_vec) const
{
  int size = static_cast<int>(in_vec.size());
  out_vec.resize(size);
  std::memcpy(out_vec.data(), in_vec.data(), size * sizeof(int));
}
