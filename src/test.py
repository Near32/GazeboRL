from GazeboRL import GazeboRL, Swarm1GazeboRL, init_roscore
import time
import rospy

#commands = {'launch': None}
#launchCom = []

#launchCom.append('rosrun gazebo_ros spawn_model -file /home/kevin/rosbuild_ws/sandbox/GazeboRL/object.urdf -urdf -z 1 -model my_object')
#launchCom.append('roslaunch gazebo_ros empty_world.launch paused:=true use_sim_time:=false gui:=true throttled:=false headless:=false debug:=false')
#launchCom.append('rosrun gazebo_ros spawn_model -file /home/kevin/rosbuild_ws/sandbox/OPUSim/models/target_model/model.sdf -sdf -z 1 -x 0 -y 0 -model my_target')

#launchCom.append('roslaunch OPUSim robot1swarm.launch')
#launchCom.append('roslaunch OPUSim robot2swarm.launch')

#commands['launch'] = launchCom

#env = GazeboRL(commands)

#init_roscore()

env = Swarm1GazeboRL()

env.make()


print('\n\nwait for 10 sec...\n\n')
time.sleep(10)

env.setPause(False)


print('\n\nwait for 10 sec...\n\n')
time.sleep(10)


env.reset()


action = [0.0,2.0]
env.step(action)


print('\n\nwait for 10 sec...\n\n')
time.sleep(10)



env.close()
