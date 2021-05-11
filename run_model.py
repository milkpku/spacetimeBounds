import numpy as np
import time

def write_mocap(outfile, start_phase, frames):
  """
    Inputs:
      filename
      start_phase start phase in reference
      frames    np.array
  """
  timestep = np.ones((len(frames), 1)) / 30
  timestep[-1][0] = 0

  frames = np.concatenate([timestep, frames], axis=1)
  frames = frames.tolist()

  with open(outfile, 'w') as wh:
    wh.write("{\n")
    wh.write('"Loop":"none",\n')
    wh.write('"Start_Phase":%f,\n' % start_phase)
    wh.write('"Frames":\n')
    wh.write('[\n')
    contents = ",\n".join(map(str, frames))
    wh.write(contents + "\n")
    wh.write(']\n')
    wh.write("}")

def test_model(env, model, select_set=None, record=False, random=True):
  env.set_mode(1)
  #env.set_task_t(20)

  s_norm = model["s_norm"]
  actor = model["actor"]

  import torch
  T = lambda x: torch.FloatTensor(x)

  frames = []
  kin_frames = []
  contact_forces = []
  outcount = 0

  from utils.timer import Timer
  dnn_timer = Timer("dnn_time")
  trans_timer = Timer("trans_time")
  proce_timer = Timer("proces_timer")
  step_timer = Timer("step_time")
  step_num = 0
  acc_rwd = 0

  # reset
  if select_set is None:
    if random:
      ob = env.reset(None)
    else:
      ob = env.reset(0.0)
  else:
    if random:
      div = np.random.choice(len(select_set))
    else:
      div = 0
    order = np.random.choice(len(select_set[div]))
    state = select_set[div][order]
    ob = env.reset(state=state)

  start_phase = env.curr_phase

  done = False
  start_time = time.time()
  while True:

    # record
    sim_pose, sim_vel = env._engine.get_pose()
    frames.append(sim_pose)
    kin_t = env.curr_phase * env._mocap._cycletime
    cnt, kin_pose, kin_vel = env._mocap.slerp(kin_t)
    kin_frames.append(kin_pose)

    dnn_timer.start()
    with torch.no_grad():
      trans_timer.start()
      obt = T(ob)
      trans_timer.pause()
      proce_timer.start()
      obt_normed = s_norm(obt)
      ac = actor(obt_normed)
      proce_timer.pause()
      trans_timer.start()
      ac = ac.cpu().numpy()
      trans_timer.pause()
    dnn_timer.pause()
    step_timer.start()
    ob, rwd, done, info = env.step(ac)
    step_timer.pause()

    contact_forces.append(env._contact_forces)

    pos = env._skeleton.get_com_pos()
    z_off = abs(pos[2])

    step_num += 1
    acc_rwd += rwd

    print(rwd)

    if done:
      end_time = time.time()
      duration = end_time - start_time
      speed = step_num / duration

      print("total step %d, rwd %f, rwd_per_step %f, sample speed %f fps" % (step_num, acc_rwd, acc_rwd / step_num, speed))
      print(info)

      if step_num >= 601:

        total_time = duration
        dnn_time = dnn_timer.cumtime
        trans_time=trans_timer.cumtime
        proce_time=proce_timer.cumtime

        step_time= step_timer.cumtime

        dnn_ratio = dnn_time / total_time
        step_ratio= step_time/ total_time

        trans_ratio = trans_time / dnn_time
        proce_ratio = proce_time / dnn_time

        print("total time %f, (dnn, step) = (%f, %f) taking (%f, %f)"
              % (total_time, dnn_time, step_time, dnn_ratio, step_ratio))
        print("dnn time %f, (trans, proce) = (%f, %f), taking (%f, %f)"
              % (dnn_time, trans_time, proce_time, trans_ratio, proce_ratio))

      dnn_timer.reset()
      trans_timer.reset()
      proce_timer.reset()
      step_timer.reset()

      start_time = time.time()

      sim_pose, sim_vel = env._engine.get_pose()
      frames.append(sim_pose)
      kin_t = env.curr_phase * env._mocap._cycletime
      cnt, kin_pose, kin_vel = env._mocap.slerp(kin_t)
      kin_frames.append(kin_pose)

      # save
      if record:
        write_mocap("sim_mocap_%d.txt" % outcount, start_phase, frames)
        write_mocap("kin_mocap_%d.txt" % outcount, start_phase, kin_frames)
        torch.save(contact_forces, "contacts_%d.tar" % outcount)

        outcount += 1
      frames = []
      kin_frames = []
      contact_forces = []

      step_num = 0
      acc_rwd = 0

      if select_set is None:
        if random:
          ob = env.reset(None)
        else:
          ob = env.reset(0.0)
      else:
        if random:
          div = np.random.choice(len(select_set))
        else:
          div = 0
        order = np.random.choice(len(select_set[div]))
        state = select_set[div][order]
        ob = env.reset(state=state)

      start_phase = env.curr_phase

if __name__=="__main__":
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument("argfile", type=str, help="arg file that specifies the environment and training parameters")
  parser.add_argument("--ckpt", type=str, default=None, help="checkpoint that stores trained model")
  parser.add_argument("--record", action="store_true")
  parser.add_argument("--random", action="store_true")
  args = parser.parse_args()

  # load env
  import json
  from env import make_env
  file_args = json.load(open(args.argfile))
  env_name = file_args["env_name"]
  env_args = file_args["env_args"]
  env_args["enable_draw"] = True
  test_env = make_env(env_name, env_args)

  # load model
  import torch
  from model import load_model
  ckpt = args.ckpt
  if args.ckpt:
    ckpt = args.ckpt
  elif "ckpt" in file_args:
    ckpt = file_args["ckpt"]
  else:
    print("please specify checkpoint")
    assert(False)

  model = load_model(ckpt)
  data = torch.load(ckpt)
  if "select_set" in data.keys():
    select_set = data["select_set"]
  else:
    select_set = None
  test_model(test_env, model, select_set, args.record, args.random)
