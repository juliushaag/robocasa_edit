from copy import deepcopy
import mujoco as mj
import mujoco.viewer
import h5py
from pathlib import Path
import numpy as np
import time as timer
from xml.etree import ElementTree as ET
from scipy.spatial.transform import Rotation as  R
from tqdm import tqdm

# requires mujoco > 3.2, scikit-learn, h5py, tqdm and numpy therefore ROBOSUITE and ROBOCASA cant be installed in the same environment
# just point the following path to the respective directories where the libraries are installed

# Path to the robosuite and robocasa directories
ROBOSUITE_PATH = Path("/Users/lovelace/Dev/rc/robosuite/robosuite")
ROBOCASA_PATH = Path("/Users/lovelace/Dev/rc/robocasa/robocasa")

# Which dataset to manipulate
SOURCE_PATH = ROBOCASA_PATH / "../datasets/v0.1/single_stage/kitchen_drawer/CloseDrawer/2024-04-30/demo_gentex_im128_randcams.hdf5"

# Where to save the altered dataset
OUTPUT_PATH = "new_dataset.hdf5"

# How many objects to spawn at max, every scene will contain 1 - MAC_DISTRACTION_OBJECTS objects
MAX_DISTRACTION_OBJECTS = 12

# Render every rollout, will be slower 
RENDER = False

# maximum number of trajectories to edit (for testing purposes)
MAX_TRAJ = 2


def set_state_from_flattened(state, recorded_joints):
    idx_qpos = 1
    idx_qvel = idx_qpos + recorded_joints

    time = state[0]
    qpos = state[idx_qpos:idx_qpos + recorded_joints]
    qvel = state[idx_qvel:idx_qvel + recorded_joints]
    
    return time, qpos, qvel


def run_scene(xml_string : str, states : list, observations, distr_objs : list = None, seed = 32, max_geoms = 32, render = False):

  np.random.seed(seed)

  spec = mj.MjSpec.from_string(xml_string)

  parents = { child : parent for parent in spec.bodies for child in parent.bodies }
  obj = next(body for body in spec.bodies if any(jnt.type == mj.mjtJoint.mjJNT_FREE for jnt in body.joints) and body.name.startswith("distr_counter"))


  parent = parents[obj]
  ndistr_objs = np.random.randint(1, max_geoms)

  for i in range(ndistr_objs):
    file = distr_objs[np.random.randint(0, len(distr_objs))]
   
    # this is a mitigation for the misconfiguration that mujoco searches for the assets in the meshes/ subfolder, which is enabled on default
    obj_string = file.read_text()
    root = ET.fromstring(obj_string)
    asset = root.find("asset")
    meshes = asset.findall("mesh")
    textures = asset.findall("texture")


    for elem in meshes + textures:
      old_path = elem.get("file")
      elem.set("file", str(file.parent / old_path))


    obj_string = ET.tostring(root).decode("utf8") 
    obj_spec = mj.MjSpec.from_string(obj_string)


    frame = parent.add_frame()

    body = frame.attach_body(obj_spec.worldbody.bodies[0].bodies[0], f"added_obj{i}", '')

    joint = body.add_freejoint()
    joint.damping = 0.3
    
    for geom in body.geoms:
      geom.friction = np.array([0.7, 0.2, 0.1])
      geom.solimp = np.array([0.9, 0.95, 0.001, 0.02, 1])
      geom.density=1000
      geom.margin=0.002 #  Adding margin for better contact handling

    body.name = "added_obj{}".format(i)
  
  model = spec.compile()
  data = mj.MjData(model)
  xml_string = spec.to_xml()

  objects = [model.body(i) for i in range(model.nbody) if "obj" in model.body(i).name or "distr" in model.body(i).name]

  recorded_joints = model.nq - 7 * ndistr_objs # pos and quat for every freejoint added 
  
  # set initial state for distr objects
  time, qpos, qvel = set_state_from_flattened(states[0], recorded_joints)

  model_distr_joint = model.joint(model.body(obj.name).jntadr.item())
  root_pos = qpos[model_distr_joint.qposadr.item():model_distr_joint.qposadr.item() + 3]


  spawn_height = root_pos[2]
  floor = model.body("floor_room_main")
  spawn_bbos = model.geom(floor.geomadr.item()).size

  forward = np.array([-spawn_bbos[0], 0, 0])
  right = np.array([0, -spawn_bbos[1], 0])

  transform = R.from_quat(floor.quat, scalar_first=True)

  forward = transform.apply(forward) 
  right = transform.apply(right)

  min_pos = floor.pos - forward - right
  max_pos = floor.pos + forward + right

  dia = max_pos - min_pos

  # make the area a bit smaller to avoid objects being spawned in the walls or too close to the edge
  min_pos += 0.1 * dia
  max_pos -= 0.1 * dia

  forward = np.array([max_pos[0] - min_pos[0], 0, 0])
  right = np.array([0, max_pos[1] - min_pos[1], 0])

  # spawn objects at the height of the distr obj already contained and in the area of the floor
  for i in range(ndistr_objs):
    joint = model.jnt(model.body("added_obj{}".format(i)).jntadr.item())
    
    x, y = np.random.rand(2)

    x = min_pos + x * forward
    y = min_pos + y * right

    data.qpos[joint.qposadr.item(): joint.qposadr.item() + 3] = x + y + np.array([0, 0, spawn_height])
    data.qpos[joint.qposadr.item() + 3: joint.qposadr.item() + 7] = R.from_euler("xyz", [*np.random.randn(2), 0]).as_quat()  

  data.qvel[:] = 0
  new_states = np.zeros((len(states), 1 + len(data.qpos) + len(data.qvel)))
  
  if render:
    viewer = mujoco.viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False)
      
    # disable collision geom rendering
    viewer.opt.geomgroup[0] = 0
    viewer.opt.flags[mj.mjtVisFlag.mjVIS_TRANSPARENT] = 0

    viewer.cam.lookat = [2.5, 0, 1]
    viewer.cam.distance = 5
    viewer.cam.azimuth = 90
    viewer.cam.elevation = -10

   
  # let the simulation run for a while to let the objects fall to the ground
  mj.mj_step(model, data, 2000)



  new_obj_obs = np.zeros((len(states), len(objects), 14))

  obj_observations = np.array(observations["object"]).reshape(len(observations["object"]), -1, 14)

  """
  obs_keys:
  'object', 
  'robot0_agentview_left_image', 'robot0_agentview_right_image', 
  'robot0_base_pos', 'robot0_base_quat', 
  'robot0_base_to_eef_pos', 'robot0_base_to_eef_quat', 'robot0_eef_pos', 'robot0_eef_quat', 
  'robot0_eye_in_hand_image', 'robot0_gripper_qpos', 'robot0_gripper_qvel', 
  'robot0_joint_pos', 'robot0_joint_pos_cos', 'robot0_joint_pos_sin', 'robot0_joint_vel'
  """
 
  for i, state in enumerate(states):
    start = timer.time()

    time, qpos, qvel = set_state_from_flattened(state, recorded_joints)

    data.time = time + data.time
    data.qpos[:len(qpos)] = qpos
    data.qvel[:len(qvel)] = qvel

    # save new states
    new_states[i, :] = np.concat([np.array([time]), data.qpos, data.qvel])

    mj.mj_step(model, data)


    eef_pos, eef_quat = observations["robot0_eef_pos"][i], observations["robot0_eef_quat"][i]

    world_in_eef = np.eye(4)
    world_in_eef[:3, :3] = R.from_quat(eef_quat).as_matrix().T
    world_in_eef[:3, 3] = -world_in_eef[:3, :3].dot(eef_pos)

    for k in range(len(objects)):
      
      obj_id = objects[k].id

      obj_in_world = np.eye(4)
      obj_in_world[:3, :3] = data.xmat[obj_id].reshape(3, 3)
      obj_in_world[:3, 3] = data.xpos[obj_id]

      obj_in_eef = world_in_eef.dot(obj_in_world) 

      new_obj_obs[i, k, :] = np.concatenate([
        data.xpos[obj_id], 
        data.xquat[obj_id][[1, 2, 3, 0]], 
        obj_in_eef[:3, 3], 
        R.from_matrix(obj_in_eef[:3, :3]).as_quat(canonical=True)]
      )

    for o, no in zip(obj_observations[i], new_obj_obs[i]): # for assurance that the new observations are correct
      assert np.allclose(o, no, rtol=1e-2)

    if render: 
      viewer.sync()
      elapsed = timer.time() - start
      diff = 1 / 30 - elapsed
      if diff > 0:
        timer.sleep(diff)

  if render:
    viewer.close()


  return xml_string, new_states, new_obj_obs
    
def update_xml(xml_string) -> str:
  root = ET.fromstring(xml_string)

  asset = root.find("asset")
  meshes = asset.findall("mesh")
  textures = asset.findall("texture")
  
  all_assets = meshes + textures

  for elem in all_assets:
      old_path = elem.get("file")
      if old_path is None: continue

      old_path_split = Path(old_path).parts

      if "robosuite" in old_path_split:        
          ind = old_path_split.index("robosuite")
          
          new_path_split =  old_path_split[ind + 1:]
          new_path = "/".join(new_path_split)
          elem.set("file", str(ROBOSUITE_PATH / new_path))

      elif "robocasa" in old_path_split:
          ind = 0
          for i, p in enumerate(old_path_split):
              if p == "robocasa": ind = i # last occurence of robocasa

          new_path_split = old_path_split[ind + 1:]
          new_path = "/".join(new_path_split)
          elem.set("file", str(ROBOCASA_PATH / new_path))

  return ET.tostring(root).decode("utf8")

if __name__ == "__main__":

  # This is the standard dataset path set by robocasa
  dataset_file = h5py.File(str(SOURCE_PATH), "r")

  # sort demos
  demos = sorted(list(dataset_file["data"].keys()), key=lambda x: int(x[5:]))

  # load distraction objects from robocasa
  distr_objs_path =  ROBOCASA_PATH / "models/assets/objects/objaverse"  
  distr_objs = [elem for elem in distr_objs_path.glob("**/**/*.xml")]

  if MAX_TRAJ: demos = demos[:MAX_TRAJ]


  output_file = h5py.File(str(OUTPUT_PATH), "w")

  output_group = output_file.create_group("data")
  for key, value in dict(dataset_file["data"].attrs).items():
    output_group.attrs[key] = value

  # run every demo in the dataset
  for ep in tqdm(demos):
    states = dataset_file["data/{}/states".format(ep)]

    xml_string = update_xml(dataset_file["data/{}".format(ep)].attrs["model_file"])  

    xml_string, states, new_obs = run_scene(xml_string, states, dataset_file["data"][ep]["obs"], distr_objs, max_geoms=MAX_DISTRACTION_OBJECTS, render=RENDER)

    # save the new states
    ep_group = output_group.create_group(ep)
    
    for key, value in  dict(dataset_file["data/{}".format(ep)].attrs).items():
      ep_group.attrs[key] = value

    ep_group.attrs["model_file"] = xml_string

    ep_group.create_dataset("states", data=states)
    ep_group.create_dataset("actions", data=dataset_file["data/{}/actions".format(ep)])
    ep_group.create_dataset("actions_abs", data=dataset_file["data/{}/actions_abs".format(ep)])

    ep_group.create_dataset("rewards", data=dataset_file["data/{}/rewards".format(ep)])
    ep_group.create_dataset("dones", data=dataset_file["data/{}/dones".format(ep)])
    
    for key, value in dict(dataset_file[f"data/{ep}/obs"]).items():
      if key == "object": value=new_obs.reshape(len(new_obs), -1) 
      ep_group.create_dataset("obs/" + key, data=value)


  output_file.close()
  dataset_file.close()