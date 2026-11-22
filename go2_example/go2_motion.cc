// Minimal TOWR trajectory for a Unitree Go2 on flat ground (walk, trot, bound, ...).
// Output:
//   go2_<gait>.csv         samples every --sample-dt (10 ms): CoM pos/vel/acc, orientation (Euler, quaternion,
//                          angular velocity), per foot pos/vel/force/contact/phase timing
//   go2_<gait>_phases.csv  one row per stance/swing phase: start/end time, foot start/end position
//   go2_<gait>_end.csv     state at the last instant; --start <file> begins the next plan there (chaining)

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <map>
#include <numeric>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <ifopt/constraint_set.h>
#include <ifopt/cost_term.h>
#include <ifopt/ipopt_solver.h>
#include <towr/constraints/dynamic_constraint.h>
#include <towr/initialization/gait_generator.h>
#include <towr/initialization/quadruped_gait_generator.h>
#include <towr/models/endeffector_mappings.h>
#include <towr/models/kinematic_model.h>
#include <towr/models/single_rigid_body_dynamics.h>
#include <towr/nlp_formulation.h>
#include <towr/terrain/examples/height_map_examples.h>
#include <towr/variables/euler_converter.h>
#include <towr/variables/nodes_variables_phase_based.h>
#include <towr/variables/variable_names.h>

namespace go2 {

using towr::LF; using towr::RF; using towr::LH; using towr::RH;

// TOWR leg index order. Go2/Unitree names: LF=FL, RF=FR, LH=RL, RH=RR.
constexpr std::array<towr::QuadrupedIDs, 4> kLegs = {LF, RF, LH, RH};

// CoM height above the feet in the URDF standing pose (thigh 0.8 rad, calf -1.5 rad) [m].
constexpr double kStandingHeight = 0.287;

// Box each foot may move in, relative to the CoM (body frame) [m]. TOWR's box is symmetric about the
// nominal stance, so the box is centred in the depth range. Default: standing height +- 0.07, +-0.10
// fore/aft. Jumps need a deeper crouch (smaller min_below) for a longer push / landing stroke; a
// smaller max_below keeps the knee bent at take-off (calf speed) and a smaller reach_x shortens the
// thigh's lever arm under load (thigh torque).
struct FootWorkspace {
  double reach_x = 0.10;                      // +- fore/aft around the nominal foot
  double min_below = kStandingHeight - 0.07;  // most crouched
  double max_below = kStandingHeight + 0.07;  // most extended
};

// Nominal feet relative to whole-body CoM, from go2_description.urdf at the standing pose.
class Go2KinematicModel : public towr::KinematicModel {
public:
  explicit Go2KinematicModel(FootWorkspace box) : KinematicModel(kLegs.size())
  {
    const double z = -(box.min_below + box.max_below) / 2;
    nominal_stance_.at(LF) <<  0.18,  0.142, z;
    nominal_stance_.at(RF) <<  0.18, -0.142, z;
    nominal_stance_.at(LH) << -0.21,  0.142, z;
    nominal_stance_.at(RH) << -0.21, -0.142, z;
    max_dev_from_nominal_ << box.reach_x, 0.08, (box.max_below - box.min_below) / 2;
  }
};

// Whole robot lumped into one rigid body (mass + composite inertia from URDF).
class Go2DynamicModel : public towr::SingleRigidBodyDynamics {
public:
  Go2DynamicModel() : SingleRigidBodyDynamics(15.2, CompositeInertia(), kLegs.size()) {}

private:
  static Eigen::Matrix3d CompositeInertia()
  {
    Eigen::Matrix3d I;
    I <<  0.188, 0.0,  -0.020,
          0.0,   0.499, 0.0,
         -0.020, 0.0,   0.532;
    return I;
  }
};

towr::RobotModel MakeGo2Model(FootWorkspace box)
{
  towr::RobotModel model;
  model.kinematic_model_ = std::make_shared<Go2KinematicModel>(box);
  model.dynamic_model_   = std::make_shared<Go2DynamicModel>();
  return model;
}

// Goal relative to the start pose, in the start heading frame (x forward, y left).
struct PlanarGoal {
  double x;   // [m]
  double y;   // [m]
  double yaw; // [rad]
};

// Base velocity at the end of the plan, in the goal heading frame.
struct PlanarVelocity {
  double forward;   // [m/s]
  double left;      // [m/s]
  double yaw_rate;  // [rad/s]
};

// TOWR defaults with Go2 limits applied.
towr::Parameters Go2Parameters()
{
  towr::Parameters params;
  // Go2 knee is ~45 Nm; 1000 N default is far beyond what the legs can push.
  params.force_limit_in_normal_direction_ = 350.0;
  return params;
}

// GaitGenerator scales the whole schedule to the total time, its opening and closing Stand too.
// The unscaled length lets LegDurations set those two apart from the strides.
class Go2GaitGenerator : public towr::QuadrupedGaitGenerator {
public:
  double UnscaledTotalTime() const { return std::accumulate(times_.begin(), times_.end(), 0.0); }
};

// One leg's phase durations. stand: length of the opening and closing Stand [s], the strides fill the
// rest of total_time. A chain joins two plans in a four-feet stance while the body moves, so that stance
// must be short (the body may only move ~0.2 m over planted feet). Without stand: TOWR's scaling.
// Needs a schedule that starts and ends with Stand (StridesFor builds one).
std::vector<double> LegDurations(const Go2GaitGenerator& gait, towr::QuadrupedIDs leg, double total_time,
                                 std::optional<double> stand)
{
  if (!stand)
    return gait.GetPhaseDurations(total_time, leg);

  constexpr double kPresetStand = 0.3;  // GaitGenerator::GetStrideStand
  const double unscaled = gait.UnscaledTotalTime();
  const double stride_scale = (total_time - 2 * *stand) / (unscaled - 2 * kPresetStand);
  if (stride_scale <= 0.0)
    throw std::invalid_argument("--stand: two stands leave no time for the strides");

  auto durations = gait.GetPhaseDurations(unscaled, leg);
  durations.front() -= kPresetStand;
  durations.back() -= kPresetStand;
  std::transform(durations.begin(), durations.end(), durations.begin(),
                 [&](double d) { return d * stride_scale; });
  durations.front() += *stand;
  durations.back() += *stand;
  return durations;
}

// Contact schedule: which foot touches ground when, and initial phase lengths.
towr::Parameters MakeGaitParameters(const std::vector<towr::GaitGenerator::Gaits>& strides,
                                    double total_time, std::optional<double> stand)
{
  Go2GaitGenerator gait;
  gait.SetGaits(strides);

  towr::Parameters params = Go2Parameters();
  std::transform(kLegs.begin(), kLegs.end(), std::back_inserter(params.ee_phase_durations_),
                 [&](towr::QuadrupedIDs ee) { return LegDurations(gait, ee, total_time, stand); });
  std::transform(kLegs.begin(), kLegs.end(), std::back_inserter(params.ee_in_contact_at_start_),
                 [&](towr::QuadrupedIDs ee) { return gait.IsInContactAtStart(ee); });
  return params;
}

struct JumpSchedule {
  int jumps;
  double flight;      // [s] all feet in the air; CoM apex rises ~ g * flight^2 / 8 above take-off
  double stance;      // [s] ground time between two jumps: land, absorb, push off
  double total_time;  // [s] the rest is split between the first crouch and the final landing
};

// All four feet together: crouch + push, (flight, stance) x (jumps - 1), flight, land + settle.
// The pronk preset fixes stance:flight at 3:2, so long flights force long stances in which the
// body cannot keep its forward speed (it may only move ~0.2 m over planted feet).
towr::Parameters MakeJumpParameters(const JumpSchedule& j)
{
  const double settle = (j.total_time - j.jumps * j.flight - (j.jumps - 1) * j.stance) / 2;
  if (j.jumps < 1 || settle < 0.2)
    throw std::invalid_argument("jump: total time leaves < 0.2 s for the first and last stance");

  std::vector<double> durations{settle};
  for (int k = 0; k < j.jumps; ++k) {
    durations.push_back(j.flight);
    durations.push_back(k + 1 < j.jumps ? j.stance : settle);
  }
  towr::Parameters params = Go2Parameters();
  params.ee_phase_durations_.assign(kLegs.size(), durations);
  params.ee_in_contact_at_start_.assign(kLegs.size(), true);
  return params;
}

// Flat ground with a chosen friction coefficient; TOWR's example terrains all keep the 0.5 default.
// TOWR's ForceConstraint uses a friction pyramid: |f_x| <= mu f_n and |f_y| <= mu f_n separately.
// For sagittal motion (f_y = 0) that is exactly the friction cone.
class FlatGroundWithFriction : public towr::FlatGround {
public:
  FlatGroundWithFriction(double height, double friction_coeff) : FlatGround(height)
  {
    friction_coeff_ = friction_coeff;
  }
};

// Robot state at the first instant of a plan. A chained plan starts where the previous one ended.
struct StartState {
  Eigen::Vector3d com_pos, com_vel;                // world [m], [m/s]
  Eigen::Vector3d rpy, rpy_rate;                   // ZYX Euler angles [rad] and their rates [rad/s]
  std::array<Eigen::Vector3d, kLegs.size()> feet;  // world, on the ground [m]
};

// Standing still at the origin, feet on the ground below the nominal stance.
StartState StandingStart(const towr::KinematicModel& kinematics)
{
  StartState s;
  s.com_pos << 0.0, 0.0, kStandingHeight;
  s.com_vel.setZero();
  s.rpy.setZero();
  s.rpy_rate.setZero();
  const auto stance_B = kinematics.GetNominalStanceInBase();
  std::transform(kLegs.begin(), kLegs.end(), s.feet.begin(), [&](towr::QuadrupedIDs leg) {
    return Eigen::Vector3d(stance_B.at(leg).x(), stance_B.at(leg).y(), 0.0);
  });
  return s;
}

// Start feet must be on the ground, so the terrain must be flat ground at height 0 for now.
// The plan ends at the goal at standing height, roll = pitch = 0, moving with end_velocity.
towr::NlpFormulation MakeFormulation(const StartState& start, const PlanarGoal& goal,
                                     const PlanarVelocity& end_velocity, const towr::Parameters& params,
                                     towr::HeightMap::Ptr terrain, towr::RobotModel model)
{
  towr::NlpFormulation f;
  f.terrain_ = std::move(terrain);
  f.model_   = std::move(model);
  f.params_  = params;

  f.initial_base_.lin.at(towr::kPos) = start.com_pos;
  f.initial_base_.lin.at(towr::kVel) = start.com_vel;
  f.initial_base_.ang.at(towr::kPos) = start.rpy;
  f.initial_base_.ang.at(towr::kVel) = start.rpy_rate;
  f.initial_ee_W_.assign(start.feet.begin(), start.feet.end());

  const double start_yaw = start.rpy.z();
  const double end_yaw = start_yaw + goal.yaw;
  const Eigen::Vector2d end_xy =
      start.com_pos.head<2>() + Eigen::Rotation2Dd(start_yaw) * Eigen::Vector2d(goal.x, goal.y);
  const Eigen::Vector2d end_velocity_xy =
      Eigen::Rotation2Dd(end_yaw) * Eigen::Vector2d(end_velocity.forward, end_velocity.left);

  // Default bounds pin only final x,y (+ all angles); add z so the robot does not finish crouched.
  f.final_base_.lin.at(towr::kPos) << end_xy, kStandingHeight;
  f.final_base_.lin.at(towr::kVel) << end_velocity_xy, 0.0;
  f.final_base_.ang.at(towr::kPos) << 0.0, 0.0, end_yaw;
  f.final_base_.ang.at(towr::kVel) << 0.0, 0.0, end_velocity.yaw_rate;  // = yaw rate at roll = pitch = 0
  f.params_.bounds_final_lin_pos_ = {towr::X, towr::Y, towr::Z};
  return f;
}

struct GaitPreset {
  towr::GaitGenerator::Gaits stride;
  std::optional<towr::GaitGenerator::Gaits> closing;  // "E" stride that ends the gait cleanly
  int default_strides;
};

// Stand -> (strides-1) x stride -> closing stride -> Stand. Without a closing stride: strides x stride.
std::vector<towr::GaitGenerator::Gaits> StridesFor(const std::string& name, std::optional<int> strides)
{
  using G = towr::GaitGenerator;
  static const std::map<std::string, GaitPreset> kGaits = {
    {"walk",    {G::Walk2, G::Walk2E,     4}},
    {"trot",    {G::Run1,  std::nullopt,  4}},
    {"flytrot", {G::Run2,  G::Run2E,      4}},
    {"pace",    {G::Run3,  G::Run3E,      4}},
    {"bound",   {G::Hop1,  G::Hop1E,      4}},
    {"gallop",  {G::Hop3,  G::Hop3E,      4}},
    {"pronk",   {G::Hop2,  std::nullopt,  3}},
  };
  const GaitPreset& preset = kGaits.at(name);
  const int count = strides.value_or(preset.default_strides);

  std::vector<G::Gaits> sequence{G::Stand};
  sequence.insert(sequence.end(), count - (preset.closing ? 1 : 0), preset.stride);
  if (preset.closing)
    sequence.push_back(*preset.closing);
  sequence.push_back(G::Stand);
  return sequence;
}

// Custom constraint (TOWR has none for this): bounds the foot height above
// terrain at the middle node of every swing phase, i.e. the step apex.
// Requires ee_polynomials_per_swing_phase_ == 2 (default) -> one middle node.
class SwingHeightConstraint : public ifopt::ConstraintSet {
public:
  SwingHeightConstraint(towr::HeightMap::Ptr terrain, towr::QuadrupedIDs ee, ifopt::Bounds apex)
      : ConstraintSet(kSpecifyLater, "swing-height-" + std::to_string(ee)),
        terrain_(std::move(terrain)), motion_id_(towr::id::EEMotionNodes(ee)), apex_(apex) {}

  void InitVariableDependedQuantities(const VariablesPtr& x) override
  {
    motion_ = x->GetComponent<towr::NodesVariablesPhaseBased>(motion_id_);
    swing_nodes_ = motion_->GetIndicesOfNonConstantNodes(); // stance nodes are constant
    SetRows(swing_nodes_.size());
  }

  VectorXd GetValues() const override
  {
    const auto nodes = motion_->GetNodes();
    VectorXd g(GetRows());
    std::transform(swing_nodes_.begin(), swing_nodes_.end(), g.data(), [&](int id) {
      const Eigen::Vector3d p = nodes.at(id).p();
      return p.z() - terrain_->GetHeight(p.x(), p.y());
    });
    return g;
  }

  VecBound GetBounds() const override { return VecBound(GetRows(), apex_); }

  void FillJacobianBlock(std::string var_set, Jacobian& jac) const override
  {
    if (var_set != motion_id_) return;
    const auto nodes = motion_->GetNodes();
    for (int row = 0; row < GetRows(); ++row) {
      const int id = swing_nodes_.at(row);
      const Eigen::Vector3d p = nodes.at(id).p();
      jac.coeffRef(row, Index(id, towr::Z)) = 1.0;
      jac.coeffRef(row, Index(id, towr::X)) = -terrain_->GetDerivativeOfHeightWrt(towr::X_, p.x(), p.y());
      jac.coeffRef(row, Index(id, towr::Y)) = -terrain_->GetDerivativeOfHeightWrt(towr::Y_, p.x(), p.y());
    }
  }

private:
  int Index(int node, int dim) const
  {
    return motion_->GetOptIndex(towr::NodesVariables::NodeValueInfo(node, towr::kPos, dim));
  }

  towr::HeightMap::Ptr terrain_;
  std::string motion_id_;
  ifopt::Bounds apex_;
  std::shared_ptr<towr::NodesVariablesPhaseBased> motion_;
  std::vector<int> swing_nodes_;
};

using ConstraintVec = std::vector<ifopt::ConstraintSet::Ptr>;

// Builds extra constraints once the variables exist; some (e.g. dynamics) need the splines.
using ConstraintFactory = std::function<ConstraintVec(const towr::SplineHolder&)>;

ConstraintVec MakeSwingHeightConstraints(const towr::HeightMap::Ptr& terrain, ifopt::Bounds apex)
{
  ConstraintVec c;
  std::transform(kLegs.begin(), kLegs.end(), std::back_inserter(c), [&](towr::QuadrupedIDs ee) {
    return std::make_shared<SwingHeightConstraint>(terrain, ee, apex);
  });
  return c;
}

// TOWR's ForceConstraint checks push-only, friction pyramid and force limit only at the force
// spline nodes. Between nodes the cubic can overshoot (seen in jumps: 1500 N, |fx|/fz > 100).
// This checks the same conditions every dt while the foot is in contact.
// Flat ground only (normal = +z). Contact times are fixed, so phase durations must not be optimized.
class DenseForceConstraint : public ifopt::ConstraintSet {
public:
  DenseForceConstraint(const towr::SplineHolder& s, towr::QuadrupedIDs ee, double dt, double mu, double f_max)
      : ConstraintSet(kSpecifyLater, "dense-force-" + std::to_string(ee)),
        force_(s.ee_force_.at(ee)), force_id_(towr::id::EEForceNodes(ee)), mu_(mu), f_max_(f_max)
  {
    const double T = force_->GetTotalTime();
    std::vector<double> grid(static_cast<size_t>(std::floor(T / dt + 1e-9)) + 1);
    std::iota(grid.begin(), grid.end(), 0.0);
    std::transform(grid.begin(), grid.end(), grid.begin(), [&](double k) { return k * dt; });
    const auto& schedule = s.phase_durations_.at(ee);
    std::copy_if(grid.begin(), grid.end(), std::back_inserter(times_),
                 [&](double t) { return schedule->IsContactPhase(t); });
    SetRows(times_.size() * kRowsPerSample);
  }

  VectorXd GetValues() const override
  {
    VectorXd g(GetRows());
    for (size_t k = 0; k < times_.size(); ++k) {
      const Eigen::Vector3d f = force_->GetPoint(times_[k]).p();
      g.segment<kRowsPerSample>(k * kRowsPerSample) << f.z(), f.x() - mu_ * f.z(), f.x() + mu_ * f.z(),
                                                       f.y() - mu_ * f.z(), f.y() + mu_ * f.z();
    }
    return g;
  }

  VecBound GetBounds() const override
  {
    const VecBound per_sample = {ifopt::Bounds(0.0, f_max_), ifopt::BoundSmallerZero, ifopt::BoundGreaterZero,
                                 ifopt::BoundSmallerZero, ifopt::BoundGreaterZero};
    VecBound bounds;
    for (size_t k = 0; k < times_.size(); ++k)
      bounds.insert(bounds.end(), per_sample.begin(), per_sample.end());
    return bounds;
  }

  void FillJacobianBlock(std::string var_set, Jacobian& jac) const override
  {
    if (var_set != force_id_) return;
    // Each row is a fixed linear combination of (fx, fy, fz): row_weights * df/dnodes.
    const Eigen::Matrix<double, kRowsPerSample, 3> row_weights = (Eigen::Matrix<double, kRowsPerSample, 3>() <<
        0, 0, 1,   1, 0, -mu_,   1, 0, mu_,   0, 1, -mu_,   0, 1, mu_).finished();
    for (size_t k = 0; k < times_.size(); ++k) {
      const Jacobian df = force_->GetJacobianWrtNodes(times_[k], towr::kPos);
      for (int r = 0; r < kRowsPerSample; ++r)
        for (int dim = 0; dim < 3; ++dim)
          if (row_weights(r, dim) != 0.0)
            for (Jacobian::InnerIterator it(df, dim); it; ++it)
              jac.coeffRef(k * kRowsPerSample + r, it.col()) += row_weights(r, dim) * it.value();
    }
  }

private:
  static constexpr int kRowsPerSample = 5;  // normal in [0, f_max], 4 friction pyramid faces
  towr::NodeSpline::Ptr force_;
  std::string force_id_;
  double mu_, f_max_;
  std::vector<double> times_;
};

ConstraintFactory MakeDenseForceConstraints(double dt, double mu, double f_max, std::vector<towr::QuadrupedIDs> legs)
{
  return [=](const towr::SplineHolder& s) {
    ConstraintVec c;
    std::transform(legs.begin(), legs.end(), std::back_inserter(c), [&](towr::QuadrupedIDs ee) {
      return std::make_shared<DenseForceConstraint>(s, ee, dt, mu, f_max);
    });
    return c;
  };
}

// Holds chosen components of a base spline (e.g. y, or roll and yaw) at their start value, with zero
// rate, at every interior node. Start/end nodes are already pinned by TOWR's own bounds. Base splines
// are cubic Hermite, so held node values + zero slopes keep the component constant for ALL t.
//
// tolerance > 0 turns each pin into a tiny band [value - tol, value + tol]. IPOPT refuses problems
// with more equality rows than free variables (status -10, "not enough degrees of freedom"); TOWR's
// dynamics rows for the held directions are redundant with these pins, so exact pins trip that
// check on longer horizons, while bands do not.
class HoldNodeComponents : public ifopt::ConstraintSet {
public:
  HoldNodeComponents(std::string nodes_id, std::vector<int> dims, double tolerance)
      : ConstraintSet(kSpecifyLater, "hold-" + nodes_id), nodes_id_(std::move(nodes_id)),
        dims_(std::move(dims)), tolerance_(tolerance) {}

  void InitVariableDependedQuantities(const VariablesPtr& x) override
  {
    nodes_ = x->GetComponent<towr::NodesVariables>(nodes_id_);
    const auto nodes = nodes_->GetNodes();
    const towr::Node& start = nodes.front();
    for (int id = 1; id + 1 < static_cast<int>(nodes.size()); ++id)
      for (int dim : dims_) {
        pins_.push_back({Index(id, towr::kPos, dim), start.p()(dim)});
        pins_.push_back({Index(id, towr::kVel, dim), 0.0});
      }
    SetRows(pins_.size());
  }

  VectorXd GetValues() const override
  {
    const VectorXd x = nodes_->GetValues();
    VectorXd g(GetRows());
    std::transform(pins_.begin(), pins_.end(), g.data(), [&](const Pin& p) { return x(p.index); });
    return g;
  }

  VecBound GetBounds() const override
  {
    VecBound bounds;
    std::transform(pins_.begin(), pins_.end(), std::back_inserter(bounds), [&](const Pin& p) {
      return ifopt::Bounds(p.value - tolerance_, p.value + tolerance_);
    });
    return bounds;
  }

  void FillJacobianBlock(std::string var_set, Jacobian& jac) const override
  {
    if (var_set != nodes_id_) return;
    for (int row = 0; row < GetRows(); ++row)
      jac.coeffRef(row, pins_.at(row).index) = 1.0;
  }

private:
  struct Pin {
    int index;     // position in this variable set's optimization vector
    double value;  // value it is held at
  };

  int Index(int node, towr::Dx deriv, int dim) const
  {
    return nodes_->GetOptIndex(towr::NodesVariables::NodeValueInfo(node, deriv, dim));
  }

  std::string nodes_id_;
  std::vector<int> dims_;
  double tolerance_;
  std::shared_ptr<towr::NodesVariables> nodes_;
  std::vector<Pin> pins_;
};

// Makes the right leg the mirror image of its left partner across the sagittal (x-z) plane:
// equal x and z, opposite y. Applies to foot-motion or force variables. Both legs must share the
// same contact schedule; their variable layouts are then identical, index for index.
// Last foothold of one foot (x, y, world) within +-tolerance of a target. Chained plans join in a four-feet
// stance while the body moves on; without this a trot ends with one diagonal pair ~7 cm behind its nominal
// spot, and the next gait starts with those feet at the edge of their range of motion (bound: 40-60 Nm thigh).
class FinalFoothold : public ifopt::ConstraintSet {
public:
  FinalFoothold(towr::QuadrupedIDs leg, Eigen::Vector2d target, double tolerance)
      : ConstraintSet(2, "final-foothold-" + std::to_string(leg)), nodes_id_(towr::id::EEMotionNodes(leg)),
        target_(target), tolerance_(tolerance) {}

  void InitVariableDependedQuantities(const VariablesPtr& x) override
  {
    nodes_ = x->GetComponent<towr::NodesVariables>(nodes_id_);
    const int last = static_cast<int>(nodes_->GetNodes().size()) - 1;
    for (int dim : {towr::X, towr::Y})
      index_[dim] = nodes_->GetOptIndex(towr::NodesVariables::NodeValueInfo(last, towr::kPos, dim));
    if (index_[towr::X] < 0 || index_[towr::Y] < 0)
      throw std::logic_error(GetName() + ": last foothold is not an optimization variable");
  }

  VectorXd GetValues() const override
  {
    const VectorXd x = nodes_->GetValues();
    return Eigen::Vector2d(x(index_[towr::X]), x(index_[towr::Y]));
  }

  VecBound GetBounds() const override
  {
    return {ifopt::Bounds(target_.x() - tolerance_, target_.x() + tolerance_),
            ifopt::Bounds(target_.y() - tolerance_, target_.y() + tolerance_)};
  }

  void FillJacobianBlock(std::string var_set, Jacobian& jac) const override
  {
    if (var_set != nodes_id_) return;
    jac.coeffRef(towr::X, index_[towr::X]) = 1.0;
    jac.coeffRef(towr::Y, index_[towr::Y]) = 1.0;
  }

private:
  std::string nodes_id_;
  Eigen::Vector2d target_;
  double tolerance_;
  std::shared_ptr<towr::NodesVariables> nodes_;
  std::array<int, 2> index_{};
};

// Every foot ends within +-tolerance of its nominal stance spot around the goal pose.
ConstraintFactory MakeFinalFootholdConstraints(const towr::NlpFormulation& f, double tolerance)
{
  const Eigen::Vector3d goal = f.final_base_.lin.p();
  const Eigen::Rotation2Dd heading(f.final_base_.ang.p().z());
  const auto nominal_B = f.model_.kinematic_model_->GetNominalStanceInBase();
  std::array<Eigen::Vector2d, kLegs.size()> targets;
  std::transform(kLegs.begin(), kLegs.end(), targets.begin(), [&](towr::QuadrupedIDs leg) -> Eigen::Vector2d {
    return goal.head<2>() + heading * nominal_B.at(leg).head<2>();
  });
  return [targets, tolerance](const towr::SplineHolder&) {
    ConstraintVec constraints;
    for (towr::QuadrupedIDs leg : kLegs)
      constraints.push_back(std::make_shared<FinalFoothold>(leg, targets.at(leg), tolerance));
    return constraints;
  };
}

// tolerance: same band trick as HoldNodeComponents, to keep IPOPT's equality count in check.
class MirrorLegs : public ifopt::ConstraintSet {
public:
  MirrorLegs(std::string left_id, std::string right_id, double tolerance)
      : ConstraintSet(kSpecifyLater, "mirror-" + left_id + "-" + right_id),
        left_id_(std::move(left_id)), right_id_(std::move(right_id)), tolerance_(tolerance) {}

  void InitVariableDependedQuantities(const VariablesPtr& x) override
  {
    left_  = x->GetComponent<towr::NodesVariables>(left_id_);
    right_ = x->GetComponent<towr::NodesVariables>(right_id_);
    if (left_->GetRows() != right_->GetRows())
      throw std::invalid_argument(GetName() + ": legs need identical contact schedules");

    // g_i = left_i + sign_i * right_i = 0, sign +1 for y (opposite), -1 for x, z (equal)
    right_sign_.resize(left_->GetRows());
    for (int i = 0; i < left_->GetRows(); ++i)
      right_sign_(i) = left_->GetNodeValuesInfo(i).front().dim_ == towr::Y ? 1.0 : -1.0;
    SetRows(left_->GetRows());
  }

  VectorXd GetValues() const override
  {
    return left_->GetValues() + right_sign_.cwiseProduct(right_->GetValues());
  }

  VecBound GetBounds() const override { return VecBound(GetRows(), ifopt::Bounds(-tolerance_, tolerance_)); }

  void FillJacobianBlock(std::string var_set, Jacobian& jac) const override
  {
    const bool is_left = var_set == left_id_;
    if (!is_left && var_set != right_id_) return;
    for (int i = 0; i < GetRows(); ++i)
      jac.coeffRef(i, i) = is_left ? 1.0 : right_sign_(i);
  }

private:
  std::string left_id_, right_id_;
  double tolerance_;
  std::shared_ptr<towr::NodesVariables> left_, right_;  // base type: GetNodeValuesInfo is public here
  Eigen::VectorXd right_sign_;
};

// True if each left/right pair (LF/RF, LH/RH) has the same contact schedule (bound, pronk).
bool LeftRightInPhase(const towr::Parameters& p)
{
  auto same = [&](towr::QuadrupedIDs a, towr::QuadrupedIDs b) {
    return p.ee_phase_durations_.at(a) == p.ee_phase_durations_.at(b) &&
           p.ee_in_contact_at_start_.at(a) == p.ee_in_contact_at_start_.at(b);
  };
  return same(LF, RF) && same(LH, RH);
}

// TOWR's dynamics constraint reduced to its sagittal rows: x force, z force, pitch moment.
// Under the sagittal constraints the y / roll / yaw rows hold by symmetry (pitch-only rotation
// couples no inertia into roll/yaw; mirrored forces cancel). Keeping them anyway makes the Jacobian
// rank-deficient (IPOPT slows down or times out) and inflates the equality count (status -10).
class SagittalDynamicConstraint : public ifopt::ConstraintSet {
public:
  explicit SagittalDynamicConstraint(std::shared_ptr<towr::DynamicConstraint> full)
      : ConstraintSet(kSpecifyLater, "dynamic-sagittal"), full_(std::move(full)) {}

  void InitVariableDependedQuantities(const VariablesPtr& x) override
  {
    full_->LinkWithVariables(x);
    std::vector<int> all(full_->GetRows());
    std::iota(all.begin(), all.end(), 0);
    std::copy_if(all.begin(), all.end(), std::back_inserter(rows_), [](int row) {
      const int dim = row % towr::k6D;  // DynamicConstraint row = 6 * sample + Dim6D
      return dim == towr::LX || dim == towr::LZ || dim == towr::AY;
    });
    SetRows(rows_.size());
  }

  VectorXd GetValues() const override { return full_->GetValues()(rows_); }

  VecBound GetBounds() const override
  {
    const VecBound all = full_->GetBounds();
    VecBound picked;
    std::transform(rows_.begin(), rows_.end(), std::back_inserter(picked), [&](int r) { return all.at(r); });
    return picked;
  }

  void FillJacobianBlock(std::string var_set, Jacobian& jac) const override
  {
    Jacobian full_block(full_->GetRows(), jac.cols());
    full_->FillJacobianBlock(var_set, full_block);
    for (int i = 0; i < static_cast<int>(rows_.size()); ++i)
      for (Jacobian::InnerIterator it(full_block, rows_[i]); it; ++it)
        jac.coeffRef(i, it.col()) = it.value();
  }

private:
  std::shared_ptr<towr::DynamicConstraint> full_;
  std::vector<int> rows_;
};

// TOWR's own dynamics constraint is replaced by SagittalDynamicConstraint in sagittal mode.
towr::Parameters WithoutTowrDynamics(towr::Parameters p)
{
  auto& c = p.constraints_;
  c.erase(std::remove(c.begin(), c.end(), towr::Parameters::Dynamic), c.end());
  return p;
}

// Turns the 3D problem into a sagittal-plane (x-z) motion: base y, roll and yaw held at their start
// values, right legs mirror left legs (positions and forces), within kTol (m, rad, m/s, N; 0 = exact),
// and dynamics enforced in x, z, pitch. Use together with WithoutTowrDynamics(params).
ConstraintFactory MakeSagittalConstraints(const towr::NlpFormulation& f, double kTol)
{
  return [model = f.model_.dynamic_model_, T = f.params_.GetTotalTime(),
          dt = f.params_.dt_constraint_dynamic_, kTol](const towr::SplineHolder& splines) -> ConstraintVec {
    using towr::id::EEForceNodes;
    using towr::id::EEMotionNodes;
    return {
      std::make_shared<SagittalDynamicConstraint>(std::make_shared<towr::DynamicConstraint>(model, T, dt, splines)),
      std::make_shared<HoldNodeComponents>(towr::id::base_lin_nodes, std::vector<int>{towr::Y}, kTol),
      std::make_shared<HoldNodeComponents>(towr::id::base_ang_nodes, std::vector<int>{towr::X, towr::Z}, kTol),
      std::make_shared<MirrorLegs>(EEMotionNodes(LF), EEMotionNodes(RF), kTol),
      std::make_shared<MirrorLegs>(EEMotionNodes(LH), EEMotionNodes(RH), kTol),
      std::make_shared<MirrorLegs>(EEForceNodes(LF), EEForceNodes(RF), kTol),
      std::make_shared<MirrorLegs>(EEForceNodes(LH), EEForceNodes(RH), kTol),
    };
  };
}

// weight * sum of squared foot forces at one foot's force nodes (all axes). TOWR has no cost by default, so
// IPOPT returns the first feasible plan, which can swing between hard braking and pushing (a bound entered at
// speed: 25-40 Nm thigh). This prefers low, even forces. TOWR's own NodeCost weighs only f_z, and its sum
// starts uninitialized (towr/src/node_cost.cc:56).
class ForceCost : public ifopt::CostTerm {
public:
  ForceCost(towr::QuadrupedIDs leg, double weight)
      : CostTerm("force-cost-" + std::to_string(leg)), nodes_id_(towr::id::EEForceNodes(leg)), weight_(weight) {}

  void InitVariableDependedQuantities(const VariablesPtr& x) override
  {
    nodes_ = x->GetComponent<towr::NodesVariables>(nodes_id_);
    for (int i = 0; i < nodes_->GetRows(); ++i)
      if (nodes_->GetNodeValuesInfo(i).front().deriv_ == towr::kPos)
        force_values_.push_back(i);  // the rest are force rates
  }

  double GetCost() const override
  {
    const VectorXd x = nodes_->GetValues();
    return weight_ * std::accumulate(force_values_.begin(), force_values_.end(), 0.0,
                                     [&](double sum, int i) { return sum + x(i) * x(i); });
  }

  void FillJacobianBlock(std::string var_set, Jacobian& jac) const override
  {
    if (var_set != nodes_id_) return;
    const VectorXd x = nodes_->GetValues();
    for (int i : force_values_)
      jac.coeffRef(0, i) = 2.0 * weight_ * x(i);
  }

private:
  std::string nodes_id_;
  double weight_;
  std::shared_ptr<towr::NodesVariables> nodes_;  // base type: GetNodeValuesInfo is public here
  std::vector<int> force_values_;
};

std::vector<ifopt::CostTerm::Ptr> MakeForceCosts(double weight)
{
  std::vector<ifopt::CostTerm::Ptr> costs;
  std::transform(kLegs.begin(), kLegs.end(), std::back_inserter(costs),
                 [&](towr::QuadrupedIDs leg) { return std::make_shared<ForceCost>(leg, weight); });
  return costs;
}

struct Solution {
  towr::SplineHolder splines;
  int ipopt_status;
  Eigen::VectorXd x;  // optimization variables, to warm-start a similar problem
};

// extra: your own constraints, added on top of TOWR's set in params_.constraints_; extra_costs likewise.
// warm_start: variables of a previous solve with the same layout (same phase, polynomial and
// node counts); used as the initial guess instead of TOWR's straight-line one.
Solution Solve(towr::NlpFormulation& formulation, const std::vector<ConstraintFactory>& extra,
               const std::vector<ifopt::CostTerm::Ptr>& extra_costs, double max_cpu_time,
               const Eigen::VectorXd* warm_start = nullptr)
{
  ifopt::Problem nlp;
  towr::SplineHolder splines;
  auto vars = formulation.GetVariableSets(splines);
  auto cons = formulation.GetConstraints(splines);
  for (const auto& make : extra) {
    const auto made = make(splines);
    cons.insert(cons.end(), made.begin(), made.end());
  }
  auto costs = formulation.GetCosts();
  costs.insert(costs.end(), extra_costs.begin(), extra_costs.end());
  std::for_each(vars.begin(),  vars.end(),  [&](auto& v) { nlp.AddVariableSet(v); });
  std::for_each(cons.begin(),  cons.end(),  [&](auto& c) { nlp.AddConstraintSet(c); });
  std::for_each(costs.begin(), costs.end(), [&](auto& c) { nlp.AddCostSet(c); });
  if (warm_start && warm_start->size() == nlp.GetNumberOfOptimizationVariables())
    nlp.SetVariables(warm_start->data());

  auto solver = std::make_shared<ifopt::IpoptSolver>();
  solver->SetOption("jacobian_approximation", "exact");
  solver->SetOption("max_cpu_time", max_cpu_time);
  solver->SetOption("print_level", 0);
  solver->Solve(nlp);
  return {splines, solver->GetReturnStatus(), nlp.GetVariableValues()};
}

// ---------------------------------------------------------------------------
// Trajectory export: sampling (pure) is kept separate from CSV writing (I/O).
// ---------------------------------------------------------------------------

constexpr std::array<const char*, 4> kLegNames = {"LF", "RF", "LH", "RH"};

// One stance or swing phase of one leg.
struct ContactPhase {
  int leg;
  int index;                       // 0-based phase number of this leg
  bool contact;                    // true = stance
  double t_start, t_end;           // [s]
  Eigen::Vector3d p_start, p_end;  // foot position at phase start/end, world [m]
};

using LegSchedules = std::array<std::vector<ContactPhase>, kLegs.size()>;

struct FootSample {
  Eigen::Vector3d pos, vel;  // world [m], [m/s]
  Eigen::Vector3d force;     // ground reaction force on the foot, world [N]
  bool contact;
  int phase;
  double phase_progress;     // 0 at phase start -> 1 at phase end
};

struct TrajectorySample {
  double t;
  Eigen::Vector3d com_pos, com_vel, com_acc;  // world
  Eigen::Vector3d euler;                       // roll, pitch, yaw [rad], ZYX convention
  Eigen::Quaterniond quat;                     // base -> world
  Eigen::Vector3d omega;                       // angular velocity, world [rad/s]
  std::array<FootSample, kLegs.size()> feet;
};

std::vector<ContactPhase> LegPhases(const towr::SplineHolder& s, towr::QuadrupedIDs leg)
{
  const auto& schedule = s.phase_durations_.at(leg);
  const auto& foot = s.ee_motion_.at(leg);
  const auto durations = schedule->GetPhaseDurations();

  std::vector<double> t_end(durations.size());
  std::partial_sum(durations.begin(), durations.end(), t_end.begin());
  t_end.back() = foot->GetTotalTime();  // summing can overshoot T by rounding

  std::vector<int> ids(durations.size());
  std::iota(ids.begin(), ids.end(), 0);
  std::vector<ContactPhase> phases;
  std::transform(ids.begin(), ids.end(), std::back_inserter(phases), [&](int k) {
    const double t0 = k == 0 ? 0.0 : t_end[k - 1];
    const double t1 = t_end[k];
    return ContactPhase{leg, k, schedule->IsContactPhase(0.5 * (t0 + t1)), t0, t1,
                        foot->GetPoint(t0).p(), foot->GetPoint(t1).p()};
  });
  return phases;
}

LegSchedules ExtractContactPhases(const towr::SplineHolder& s)
{
  LegSchedules schedules;
  std::transform(kLegs.begin(), kLegs.end(), schedules.begin(),
                 [&](towr::QuadrupedIDs leg) { return LegPhases(s, leg); });
  return schedules;
}

// Phase containing t. Boundaries belong to the later phase, the final instant to the last.
const ContactPhase& PhaseAt(const std::vector<ContactPhase>& phases, double t)
{
  auto it = std::find_if(phases.begin(), phases.end(),
                         [t](const ContactPhase& p) { return t < p.t_end; });
  return it != phases.end() ? *it : phases.back();
}

FootSample SampleFoot(const towr::SplineHolder& s, const std::vector<ContactPhase>& phases,
                      towr::QuadrupedIDs leg, double t)
{
  const auto motion = s.ee_motion_.at(leg)->GetPoint(t);
  const auto& phase = PhaseAt(phases, t);
  const double progress = (t - phase.t_start) / (phase.t_end - phase.t_start);
  return {motion.p(), motion.v(), s.ee_force_.at(leg)->GetPoint(t).p(),
          phase.contact, phase.index, std::clamp(progress, 0.0, 1.0)};
}

TrajectorySample SampleAt(const towr::SplineHolder& s, const towr::EulerConverter& orientation,
                          const LegSchedules& schedules, double t)
{
  const auto com = s.base_linear_->GetPoint(t);
  TrajectorySample out;
  out.t       = t;
  out.com_pos = com.p();
  out.com_vel = com.v();
  out.com_acc = com.a();
  out.euler   = s.base_angular_->GetPoint(t).p();
  out.quat    = orientation.GetQuaternionBaseToWorld(t);
  out.omega   = orientation.GetAngularVelocityInWorld(t);
  std::transform(kLegs.begin(), kLegs.end(), out.feet.begin(), [&](towr::QuadrupedIDs leg) {
    return SampleFoot(s, schedules.at(leg), leg, t);
  });
  return out;
}

// Uniform grid 0, dt, 2dt, ... that always ends exactly at T.
std::vector<double> SampleTimes(double total_time, double dt)
{
  std::vector<double> t(static_cast<size_t>(std::floor(total_time / dt + 1e-9)) + 1);
  std::iota(t.begin(), t.end(), 0.0);
  std::transform(t.begin(), t.end(), t.begin(), [&](double k) { return k * dt; });
  if (t.back() < total_time - 1e-9)
    t.push_back(total_time);
  return t;
}

std::vector<TrajectorySample> SampleTrajectory(const towr::SplineHolder& s,
                                               const LegSchedules& schedules, double dt)
{
  const towr::EulerConverter orientation(s.base_angular_);
  const auto times = SampleTimes(s.base_linear_->GetTotalTime(), dt);
  std::vector<TrajectorySample> samples;
  std::transform(times.begin(), times.end(), std::back_inserter(samples),
                 [&](double t) { return SampleAt(s, orientation, schedules, t); });
  return samples;
}

// Prints a vector as "x,y,z" (Eigen's own operator<< prints space/newline separated).
struct Csv3 {
  const Eigen::Vector3d& v;
};

std::ostream& operator<<(std::ostream& out, const Csv3& c)
{
  return out << c.v.x() << ',' << c.v.y() << ',' << c.v.z();
}

void WriteSamplesCsv(const std::vector<TrajectorySample>& samples, std::ostream& out)
{
  out << "t,com_x,com_y,com_z,com_vx,com_vy,com_vz,com_ax,com_ay,com_az,"
         "roll,pitch,yaw,qw,qx,qy,qz,wx,wy,wz";
  for (const char* l : kLegNames)
    out << ',' << l << "_px," << l << "_py," << l << "_pz,"
        << l << "_vx," << l << "_vy," << l << "_vz,"
        << l << "_fx," << l << "_fy," << l << "_fz,"
        << l << "_contact," << l << "_phase," << l << "_phase_progress";
  out << '\n' << std::setprecision(8);

  for (const auto& s : samples) {
    out << s.t << ',' << Csv3{s.com_pos} << ',' << Csv3{s.com_vel} << ',' << Csv3{s.com_acc}
        << ',' << Csv3{s.euler} << ',' << s.quat.w() << ',' << s.quat.x() << ','
        << s.quat.y() << ',' << s.quat.z() << ',' << Csv3{s.omega};
    for (const auto& f : s.feet)
      out << ',' << Csv3{f.pos} << ',' << Csv3{f.vel} << ',' << Csv3{f.force} << ','
          << f.contact << ',' << f.phase << ',' << f.phase_progress;
    out << '\n';
  }
}

void WritePhasesCsv(const LegSchedules& schedules, std::ostream& out)
{
  out << "leg,phase,contact,t_start,t_end,duration,"
         "start_x,start_y,start_z,end_x,end_y,end_z\n" << std::setprecision(8);
  for (const auto& leg_phases : schedules)
    for (const auto& p : leg_phases)
      out << kLegNames.at(p.leg) << ',' << p.index << ',' << p.contact << ','
          << p.t_start << ',' << p.t_end << ',' << p.t_end - p.t_start << ','
          << Csv3{p.p_start} << ',' << Csv3{p.p_end} << '\n';
}

// State file: one header line and one row, base then feet in kLegNames order.
std::string StateHeader()
{
  std::string header = "com_x,com_y,com_z,com_vx,com_vy,com_vz,roll,pitch,yaw,droll,dpitch,dyaw";
  for (const char* l : kLegNames)
    header += std::string(",") + l + "_px," + l + "_py," + l + "_pz";
  return header;
}

StartState EndState(const towr::SplineHolder& s)
{
  const double T = s.base_linear_->GetTotalTime();
  StartState end;
  end.com_pos  = s.base_linear_->GetPoint(T).p();
  end.com_vel  = s.base_linear_->GetPoint(T).v();
  end.rpy      = s.base_angular_->GetPoint(T).p();
  end.rpy_rate = s.base_angular_->GetPoint(T).v();
  std::transform(kLegs.begin(), kLegs.end(), end.feet.begin(),
                 [&](towr::QuadrupedIDs leg) { return s.ee_motion_.at(leg)->GetPoint(T).p(); });
  return end;
}

void WriteState(const StartState& s, std::ostream& out)
{
  // full precision: the next plan starts exactly here
  out << StateHeader() << '\n' << std::setprecision(17)
      << Csv3{s.com_pos} << ',' << Csv3{s.com_vel} << ',' << Csv3{s.rpy} << ',' << Csv3{s.rpy_rate};
  for (const auto& foot : s.feet)
    out << ',' << Csv3{foot};
  out << '\n';
}

StartState ReadState(std::istream& in)
{
  std::string header, row;
  std::getline(in, header);
  std::getline(in, row);
  if (header != StateHeader())
    throw std::runtime_error("state file: columns differ from " + StateHeader());

  std::vector<double> v;
  std::stringstream cells(row);
  for (std::string cell; std::getline(cells, cell, ',');)
    v.push_back(std::stod(cell));
  if (v.size() != 12 + 3 * kLegs.size())
    throw std::runtime_error("state file: wrong number of values");

  auto vec3 = [&](size_t i) { return Eigen::Vector3d(v[i], v[i + 1], v[i + 2]); };
  StartState s;
  s.com_pos  = vec3(0);
  s.com_vel  = vec3(3);
  s.rpy      = vec3(6);
  s.rpy_rate = vec3(9);
  for (size_t leg = 0; leg < kLegs.size(); ++leg)
    s.feet[leg] = vec3(12 + 3 * leg);
  return s;
}

struct CliOptions {
  std::string gait = "trot";
  PlanarGoal goal{0.6, 0.0, 0.0};
  double total_time = 2.4;
  ifopt::Bounds swing_apex{0.06, 0.10};
  std::optional<int> strides;          // --strides N
  bool sagittal = false;               // --sagittal
  bool dense_force = false;            // --dense-force  (force limits every dt, not only at nodes)
  std::optional<double> dt_dynamic;    // --dt-dyn S     (TOWR default 0.1 s)
  std::optional<double> base_poly;     // --base-poly S  (TOWR default 0.1 s)
  std::optional<int> force_polys;      // --force-polys N (TOWR default 3 per stance phase)
  double flight = 0.45;                // --flight S     (gait "jump" only)
  double stance = 0.3;                 // --stance S     (gait "jump" only)
  double max_cpu_time = 30.0;          // --max-cpu S
  double sample_dt = 0.01;             // --sample-dt S  (CSV row spacing)
  double friction_coeff = 0.5;         // --mu X         (TOWR default 0.5)
  FootWorkspace foot;                  // --foot-z MIN MAX (feet below CoM), --foot-x DX (fore/aft)
  std::optional<int> swing_polys;      // --swing-polys N (TOWR default 2 per swing phase)
  std::optional<double> ramp_from;     // --ramp-from S  (gait "jump": grow flight from S, warm-started)
  std::optional<double> ramp_x_from;   // --ramp-x-from X (with --ramp-from: then grow distance from X)
  std::optional<std::string> start;    // --start FILE   (state file of a previous plan; default: stand at 0)
  PlanarVelocity end_velocity{0, 0, 0};  // --end-vel FWD LEFT YAW_RATE (goal heading frame)
  std::optional<double> stand;         // --stand S      (opening/closing Stand length; walk/trot/... only)
  std::optional<double> end_feet;      // --end-feet TOL (last footholds within TOL of the nominal stance)
  std::optional<double> force_cost;    // --force-cost W (minimize W * sum of squared foot forces)
  std::string out_prefix;              // --out PREFIX, default go2_<gait>
};

// Flags may appear anywhere; the rest are positional.
CliOptions ParseArgs(const std::vector<std::string>& args)
{
  CliOptions o;
  std::vector<std::string> pos;
  for (size_t i = 0; i < args.size(); ++i) {
    const bool has_value = i + 1 < args.size();
    if (args[i] == "--sagittal")                     o.sagittal = true;
    else if (args[i] == "--dense-force")             o.dense_force = true;
    else if (args[i] == "--strides" && has_value)    o.strides = std::stoi(args[++i]);
    else if (args[i] == "--dt-dyn" && has_value)     o.dt_dynamic = std::stod(args[++i]);
    else if (args[i] == "--base-poly" && has_value)  o.base_poly = std::stod(args[++i]);
    else if (args[i] == "--force-polys" && has_value) o.force_polys = std::stoi(args[++i]);
    else if (args[i] == "--flight" && has_value)     o.flight = std::stod(args[++i]);
    else if (args[i] == "--stance" && has_value)     o.stance = std::stod(args[++i]);
    else if (args[i] == "--max-cpu" && has_value)    o.max_cpu_time = std::stod(args[++i]);
    else if (args[i] == "--sample-dt" && has_value)  o.sample_dt = std::stod(args[++i]);
    else if (args[i] == "--mu" && has_value)         o.friction_coeff = std::stod(args[++i]);
    else if (args[i] == "--foot-z" && i + 2 < args.size()) {
      o.foot.min_below = std::stod(args[++i]);
      o.foot.max_below = std::stod(args[++i]);
    }
    else if (args[i] == "--foot-x" && has_value)     o.foot.reach_x = std::stod(args[++i]);
    else if (args[i] == "--swing-polys" && has_value) o.swing_polys = std::stoi(args[++i]);
    else if (args[i] == "--ramp-from" && has_value)  o.ramp_from = std::stod(args[++i]);
    else if (args[i] == "--ramp-x-from" && has_value) o.ramp_x_from = std::stod(args[++i]);
    else if (args[i] == "--start" && has_value)      o.start = args[++i];
    else if (args[i] == "--end-vel" && i + 3 < args.size()) {
      o.end_velocity.forward  = std::stod(args[++i]);
      o.end_velocity.left     = std::stod(args[++i]);
      o.end_velocity.yaw_rate = std::stod(args[++i]);
    }
    else if (args[i] == "--stand" && has_value)      o.stand = std::stod(args[++i]);
    else if (args[i] == "--end-feet" && has_value)   o.end_feet = std::stod(args[++i]);
    else if (args[i] == "--force-cost" && has_value) o.force_cost = std::stod(args[++i]);
    else if (args[i] == "--out" && has_value)        o.out_prefix = args[++i];
    else                                             pos.push_back(args[i]);
  }
  auto num = [&](size_t i, double fallback) { return i < pos.size() ? std::stod(pos[i]) : fallback; };
  if (!pos.empty()) o.gait = pos[0];
  o.goal = {num(1, o.goal.x), num(2, o.goal.y), num(3, o.goal.yaw)};
  o.total_time = num(4, o.total_time);
  o.swing_apex = ifopt::Bounds(num(5, o.swing_apex.lower_), num(6, o.swing_apex.upper_));
  if (o.out_prefix.empty()) o.out_prefix = "go2_" + o.gait;
  return o;
}

// Finer time grids. TOWR checks dynamics only every dt_constraint_dynamic_; with short phases
// (bounding: ~0.13 s stance, ~0.04 s flight) whole phases fall between checks and the motion
// violates physics there. Range of motion is checked at least as often as dynamics.
// Keep dt_dynamic == base_poly (one check per base polynomial): more checks per polynomial
// over-constrain it and IPOPT stalls (tested: 0.025/0.05, 0.02/0.04 time out; 0.025/0.025 solves).
// force_polys: polynomials per stance phase for each contact force. Long stances that must land,
// crouch and push off (big jumps) need more than TOWR's 3 to shape the force profile.
towr::Parameters WithTimeResolution(towr::Parameters p, std::optional<double> dt_dynamic,
                                    std::optional<double> base_poly, std::optional<int> force_polys)
{
  if (dt_dynamic) {
    p.dt_constraint_dynamic_ = *dt_dynamic;
    p.dt_constraint_range_of_motion_ = std::min(p.dt_constraint_range_of_motion_, *dt_dynamic);
  }
  if (base_poly)
    p.duration_base_polynomial_ = *base_poly;
  if (force_polys)
    p.force_polynomials_per_stance_phase_ = *force_polys;
  return p;
}

// More pieces per swing let a foot follow the body through a long flight. TOWR's SwingConstraint
// assumes exactly 2, so it is dropped otherwise (SwingHeightConstraint assumes 2 as well).
towr::Parameters WithSwingPolynomials(towr::Parameters p, int polys)
{
  p.ee_polynomials_per_swing_phase_ = polys;
  if (polys != 2) {
    auto& c = p.constraints_;
    c.erase(std::remove(c.begin(), c.end(), towr::Parameters::Swing), c.end());
  }
  return p;
}

JumpSchedule TargetJump(const CliOptions& o)
{
  return {o.strides.value_or(3), o.flight, o.stance, o.total_time};
}

// Parameters for one solve. grid, when given, sets both dt_dynamic and base_poly (continuation).
towr::Parameters MakeParameters(const CliOptions& o, const JumpSchedule& jump, std::optional<double> grid)
{
  auto p = o.gait == "jump" ? MakeJumpParameters(jump)
                            : MakeGaitParameters(StridesFor(o.gait, o.strides), o.total_time, o.stand);
  p = WithTimeResolution(p, grid ? grid : o.dt_dynamic, grid ? grid : o.base_poly, o.force_polys);
  return o.swing_polys ? WithSwingPolynomials(p, *o.swing_polys) : p;
}

struct SolveSetup {
  towr::NlpFormulation formulation;
  std::vector<ConstraintFactory> extra;
  std::vector<ifopt::CostTerm::Ptr> costs;
};

SolveSetup MakeSolveSetup(const CliOptions& o, const StartState& start, const towr::Parameters& params,
                          const PlanarGoal& goal)
{
  auto f = MakeFormulation(start, goal, o.end_velocity, o.sagittal ? WithoutTowrDynamics(params) : params,
                           std::make_shared<FlatGroundWithFriction>(0.0, o.friction_coeff),
                           MakeGo2Model(o.foot));
  std::vector<ConstraintFactory> extra;
  if (o.swing_apex.upper_ > 0.0)
    extra.push_back([terrain = f.terrain_, apex = o.swing_apex](const towr::SplineHolder&) {
      return MakeSwingHeightConstraints(terrain, apex);
    });
  if (o.sagittal)
    extra.push_back(MakeSagittalConstraints(f, /*tolerance=*/1e-6));  // 0 -> IPOPT -10 on fine grids
  if (o.dense_force) {
    // Sagittal: right legs mirror left legs, so checking the left ones is enough.
    const std::vector<towr::QuadrupedIDs> legs =
        o.sagittal ? std::vector<towr::QuadrupedIDs>{LF, LH} : std::vector<towr::QuadrupedIDs>(kLegs.begin(), kLegs.end());
    extra.push_back(MakeDenseForceConstraints(params.dt_constraint_dynamic_, o.friction_coeff,
                                              params.force_limit_in_normal_direction_, legs));
  }
  if (o.end_feet)
    extra.push_back(MakeFinalFootholdConstraints(f, *o.end_feet));
  auto costs = o.force_cost ? MakeForceCosts(*o.force_cost) : std::vector<ifopt::CostTerm::Ptr>{};
  return {std::move(f), std::move(extra), std::move(costs)};
}

struct SolveStep {
  JumpSchedule jump;
  std::optional<double> grid;
  PlanarGoal goal;
};

// The goal shrunk or stretched along its own line to forward distance x (y and yaw in proportion)
PlanarGoal GoalAtX(const PlanarGoal& goal, double x)
{
  const double scale = x / goal.x;
  return {x, goal.y * scale, goal.yaw * scale};
}

// Continuation for big jumps: TOWR's straight-line initial guess is too far from a high jump for
// IPOPT, so solve a small jump first and grow the flight time (stance in proportion) in 0.03 s
// steps, warm-starting each solve from the last. First/last stance time and the number of base
// polynomials stay fixed, so every step has the same variable layout (the grid stretches with T).
std::vector<SolveStep> JumpRamp(const JumpSchedule& target, double from_flight, double grid, const PlanarGoal& goal)
{
  const double ends = target.total_time - target.jumps * target.flight - (target.jumps - 1) * target.stance;
  const double segments = std::round(target.total_time / grid);
  std::vector<double> flights;
  for (double f = from_flight; f < target.flight - 1e-9; f += 0.03)
    flights.push_back(f);
  flights.push_back(target.flight);

  std::vector<SolveStep> steps;
  std::transform(flights.begin(), flights.end(), std::back_inserter(steps), [&](double f) {
    const double stance = target.stance * f / target.flight;
    const double T = ends + target.jumps * f + (target.jumps - 1) * stance;
    return SolveStep{{target.jumps, f, stance, T}, T / segments, goal};
  });
  return steps;
}

// Long jumps: a short hop cannot cover the full distance, so the flight ramp runs at forward distance
// from_x and this then moves the goal out to the target in 0.1 m steps of x (y and yaw in proportion,
// so a diagonal or turning jump keeps its direction). Only the goal changes, so the warm start
// carries over exactly.
std::vector<SolveStep> DistanceRamp(const SolveStep& at_full_flight, double from_x, const PlanarGoal& target)
{
  std::vector<SolveStep> steps;
  for (double x = from_x + 0.1; x < target.x - 1e-9; x += 0.1)
    steps.push_back({at_full_flight.jump, at_full_flight.grid, GoalAtX(target, x)});
  steps.push_back({at_full_flight.jump, at_full_flight.grid, target});
  return steps;
}

} // namespace go2

// usage: go2_motion [gait] [goal_x] [goal_y] [goal_yaw] [total_time] [apex_min] [apex_max]
//                   [--strides N] [--sagittal] [--dense-force] [--dt-dyn S --base-poly S] [--force-polys N] [--max-cpu S]
//                   [--sample-dt S] [--mu X] [--flight S --stance S] [--ramp-from S [--ramp-x-from X]]
//                   [--foot-z MIN MAX] [--foot-x DX] [--swing-polys N] [--out PREFIX]
//                   [--start FILE] [--end-vel FWD LEFT YAW_RATE] [--stand S] [--end-feet TOL] [--force-cost W]
// goal x, y, yaw: relative to the start pose, in its heading frame (x forward, y left).
// --start: begin at the state another run wrote to <prefix>_end.csv (chaining plans).
// --end-vel: end moving (goal heading frame) instead of at rest. --stand: opening/closing four-feet
//            stance length [s] instead of TOWR's scaled 0.3 s; keep it short when the body moves there.
// --end-feet: last footholds within +-TOL [m] of the nominal stance at the goal (a neat stance to join from).
// --force-cost: minimize W * sum of squared foot forces instead of taking the first feasible plan.
// gait "jump": --strides = number of jumps, --flight / --stance = air / ground time per jump,
//              --ramp-from = solve from a smaller flight time up (needed for big jumps),
//              --ramp-x-from = do that at distance X, then grow the distance (needed for long jumps).
//        go2_motion trot 0.6 0 0 2.4 0.06 0.10
//        go2_motion bound 2.0 0 0 3.5 0.04 0.10 --strides 10 --sagittal \
//                   --dt-dyn 0.01 --base-poly 0.01 --max-cpu 120 --sample-dt 0.002 --out go2_bound_2m
// apex_max <= 0 disables the swing-height constraint.
// --sagittal: x-z plane motion (base y/roll/yaw held, right legs mirror left). Needs a gait
//             whose left/right legs move together: bound, pronk or jump.
// --dt-dyn / --base-poly: use the same value. Smaller = more physically consistent, slower.
int main(int argc, char** argv)
{
  const go2::CliOptions opt = go2::ParseArgs({argv + 1, argv + argc});
  if (opt.swing_apex.upper_ > 0.0 && opt.swing_polys.value_or(2) != 2) {
    std::cerr << "the swing-height constraint needs 2 swing polynomials; pass apex 0 0 with --swing-polys\n";
    return 2;
  }
  if (opt.ramp_from && (opt.gait != "jump" || !opt.base_poly)) {
    std::cerr << "--ramp-from needs gait jump and --base-poly\n";
    return 2;
  }
  if (opt.ramp_x_from && (!opt.ramp_from || opt.goal.x <= 0.0)) {
    std::cerr << "--ramp-x-from needs --ramp-from and a forward goal (goal_x > 0)\n";
    return 2;
  }

  if (opt.stand && opt.gait == "jump") {
    std::cerr << "--stand is for walk/trot/... gaits; gait jump sets its stances with total time\n";
    return 2;
  }
  go2::StartState start = go2::StandingStart(go2::Go2KinematicModel(opt.foot));
  if (opt.start) {
    std::ifstream state_file(*opt.start);
    if (!state_file) {
      std::cerr << "cannot read " << *opt.start << "\n";
      return 2;
    }
    start = go2::ReadState(state_file);
  }

  std::vector<go2::SolveStep> steps{{go2::TargetJump(opt), std::nullopt, opt.goal}};
  if (opt.ramp_from) {
    const go2::PlanarGoal start = opt.ramp_x_from ? go2::GoalAtX(opt.goal, *opt.ramp_x_from) : opt.goal;
    steps = go2::JumpRamp(go2::TargetJump(opt), *opt.ramp_from, *opt.base_poly, start);
    if (opt.ramp_x_from) {
      const auto distance = go2::DistanceRamp(steps.back(), *opt.ramp_x_from, opt.goal);
      steps.insert(steps.end(), distance.begin(), distance.end());
    }
  }
  go2::Solution solution;
  for (size_t k = 0; k < steps.size(); ++k) {
    const auto params = go2::MakeParameters(opt, steps[k].jump, steps[k].grid);
    if (opt.sagittal && !go2::LeftRightInPhase(params)) {
      std::cerr << "--sagittal needs left/right legs in phase (bound, pronk or jump), not " << opt.gait << "\n";
      return 2;
    }
    auto setup = go2::MakeSolveSetup(opt, start, params, steps[k].goal);
    solution = go2::Solve(setup.formulation, setup.extra, setup.costs, opt.max_cpu_time,
                          k > 0 ? &solution.x : nullptr);
    if (steps.size() > 1)
      std::cout << "ramp " << k + 1 << "/" << steps.size() << ": flight " << steps[k].jump.flight
                << " s, x " << steps[k].goal.x << " m -> IPOPT status " << solution.ipopt_status << "\n";
    if (solution.ipopt_status != 0) break;
  }
  std::cout << "IPOPT status: " << solution.ipopt_status << " (0 = solved)\n";

  const auto schedules = go2::ExtractContactPhases(solution.splines);
  const auto samples = go2::SampleTrajectory(solution.splines, schedules, opt.sample_dt);

  const std::string& prefix = opt.out_prefix;
  std::ofstream samples_csv(prefix + ".csv");
  go2::WriteSamplesCsv(samples, samples_csv);
  std::ofstream phases_csv(prefix + "_phases.csv");
  go2::WritePhasesCsv(schedules, phases_csv);
  std::ofstream end_csv(prefix + "_end.csv");
  go2::WriteState(go2::EndState(solution.splines), end_csv);
  std::cout << "wrote " << prefix << ".csv (samples), " << prefix << "_phases.csv (timing) and "
            << prefix << "_end.csv (end state)\n";
  return solution.ipopt_status == 0 ? 0 : 1;
}
