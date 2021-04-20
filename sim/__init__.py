from .sim_pybullet import PyBulletEngine

engine_dict = {
        "pybullet": PyBulletEngine,
        }

model_name = [
        "humanoid3d",
        "atlas",
        ]

def engine_builder(engine, skeleton, self_collision, timestep, model="humanoid3d"):
  if engine not in engine_dict:
    print("select from %s" % str(engine_dict.keys()))
    assert(False and "not implemented engine %s" % engine)

  if model not in model_name:
    print("select from %s" % str(model_name))
    assert(False and "not implemented engine %s" % engine)

  return engine_dict[engine](skeleton, self_collision, timestep, model)
