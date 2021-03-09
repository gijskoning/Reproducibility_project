from warehouse import Warehouse
import numpy as np
warehouse = Warehouse()
warehouse.reset()
#################### Test _place_robots function ######################
for robot in warehouse.robots:
    assert robot._robot_domain[0] <= robot.get_position[0] <= robot._robot_domain[2] and \
      robot._robot_domain[1] <= robot.get_position[1] <= robot._robot_domain[3], \
      'place robots test: failed. Robot {} is not within its designated domain'.format(robot.id)
print('place robots test: passed')
###################### Test _add_items function #######################
item_rows = np.arange(0, warehouse.n_rows, warehouse.distance_between_shelves)
for item in warehouse.items:
    assert  item.get_position[0] in item_rows, \
    'add items test: failed. Item {} is not on a shelf'.format(item.id)
warehouse._add_items()
print('add items test: passed')
###################### Test remove_items function #####################
warehouse = Warehouse()
warehouse.reset()
pos = warehouse.items[0].get_position
warehouse.robots[0]._pos = pos
warehouse._remove_items()
state = warehouse._get_state()
assert state[pos[0],pos[1], 0] == 0, 'remove items test: failed'
print('remove items test: passed')
################### Test compute rewards function #####################
warehouse = Warehouse()
warehouse.reset()
learning_robot_id = warehouse.learning_robot_id
pos = warehouse.items[0].get_position
robot = warehouse.robots[learning_robot_id]
robot._pos = pos
n_items = robot.items_collected
reward = warehouse._compute_reward(robot)
assert reward == 1, 'compute rewards test: failed. Wrong reward'
assert robot.items_collected > n_items, \
'compute rewards: failed'
warehouse = Warehouse()
warehouse.reset()
robot = warehouse.robots[learning_robot_id]
pos = warehouse.items[0].get_position
robot._pos = pos
for i in range(warehouse.max_n_items):
    warehouse._compute_reward(robot)
assert robot.done == True, \
'compute rewards test: failed. Agent is not done after max_n_items were collected'
print('compute rewards test: passed')
####################### Test action fucntion ##########################
warehouse = Warehouse()
warehouse.reset()
# action 0
initial_positions = []
for robot in warehouse.robots:
    initial_positions.append(robot.get_position)
actions = dict(enumerate(np.zeros(len(warehouse.robots), dtype=np.int)))
warehouse.step(actions)
for robot, initial_position in zip(warehouse.robots, initial_positions):
    assert robot.get_position[1] - 1 == initial_position[1], "action test: failed"
# action 1
initial_positions = []
for robot in warehouse.robots:
    initial_positions.append(robot.get_position)
actions = dict(enumerate(np.ones(len(warehouse.robots), dtype=np.int)))
warehouse.step(actions)
for robot, initial_position in zip(warehouse.robots, initial_positions):
    assert robot.get_position[1] + 1 == initial_position[1], "action test: failed"
# action 2
initial_positions = []
for robot in warehouse.robots:
    initial_positions.append(robot.get_position)
actions = dict(enumerate(2*np.ones(len(warehouse.robots), dtype=np.int)))
warehouse.step(actions)
for robot, initial_position in zip(warehouse.robots, initial_positions):
    assert robot.get_position[0] + 1 == initial_position[0], "action test: failed"
# action 3
initial_positions = []
for robot in warehouse.robots:
    initial_positions.append(robot.get_position)
actions = dict(enumerate(3*np.ones(len(warehouse.robots), dtype=np.int)))
warehouse.step(actions)
for robot, initial_position in zip(warehouse.robots, initial_positions):
    assert robot.get_position[0] - 1 == initial_position[0], "action test: failed"
print('action test: passed')
######################## Test action space #############################
warehouse = Warehouse()
warehouse.reset()
print(warehouse.action_space)
############################ Test graph ###############################
robot = warehouse.robots[1]
graph = warehouse._create_graph(robot)
breakpoint()
