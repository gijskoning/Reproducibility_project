import copy
import numpy as np
from gettext import _current_domain
from .LDM import ldm


class State:
    """
    Abstract superclass for the concrete states
    @param ldm the LDM connection with sumo
    @param lights the list of traffic light IDs (strings)
    """

    def __init__(self, ldm, lights:list):
        """
        @param lights list of traffic light ids
        """
        self._ldm = ldm
            # get height - the number of lanes in total
        self._lanes = []
        if(lights):
            self._lights = lights
            for lightid in lights:
                lanes = self._ldm.getControlledLanes(lightid)
                self._lanes += lanes

        self._max_speeds = {}

        for lane in self._lanes:
            self._max_speeds[lane] = self._ldm.getLaneMaxSpeed(lane)

    def getLanes(self):
        """
        the lanes that this state is working with
        """
        return self._lanes
    
    def getMaxSpeed(self, lane):
        """
        @param lane the lane id
        @return maximum speed for that lane
        """
        return self._ldm.getMaxSpeed(lane)

    def get_action_space(self) -> list:
        """
        @return list of strings, each string representing a valid action.
        """
        raise Exception("not implemented")
    
    def update_state(self):
        """
        Updates the state to match the current state in sumo
        @return new state. Note that self._current_state also changes accordingly.
        """
        raise Exception("not implemented")


class LinearFeatureState(State):
    """
    [sum of wait, sum of vehicle delay, number of vehicle, 
    number of halted, average speed, average acceleration, 
    number of emergency stop] combined with action
    as described in elise's master thesis
    only support one-light scenario
    """

    def __init__(self, ldm):
        State.__init__(self, ldm, ["0"])
        self._prev_speed = {}
        self._actions = ['GrGr', 'ryry', 'rGrG', 'yryr']
        self._current_state = np.zeros((len(self._actions) * len(self._ldm.getControlledLanes("0")) * 7, 1, 1))
    
    def update_state(self):
        lane_states, self._prev_speed, stops = self._get_lane_states(self._prev_speed)
        state = np.array(self._get_linear_state(lane_states, self._ldm.getControlledLanes("0")))
        self._current_state = np.reshape(state, (len(self._actions) * len(self._ldm.getControlledLanes("0")) * 7, 1, 1))
        return self._current_state

    # Override
    def get_action_space(self):
        return ['GrGr', 'rGrG']
    
    def _get_lane_states(self, prev_speed):
        '''
        Go through the list of vehicles, and use each vehicle's state
        to determine the state of the lane it's currently on.
        '''
        lane_stats = {}
        found_stop = False
        stops = []
        # For each vehicle in the list of vehicles
        for vehicle in self._ldm.getVehicles() :
            # Get the current lane of the vehicle
            vehicle_lane = self._ldm.getVehicleLane(vehicle)
            # print "vehiclelane", vehicle_lane

            # Count the number of vehicles on the lane:
            try:
                lane_stats[vehicle_lane]['vehicle_count'] += 1.0
            except KeyError:
                lane_stats[vehicle_lane] = {'vehicle_count': 1.0}

            # Store waiting time of current vehicle in the correct lane
            wait = self._ldm.getVehicleWaitingTime(vehicle)
            try:
                lane_stats[vehicle_lane]['wait'].append(wait)
            except KeyError:
                lane_stats[vehicle_lane]['wait'] = [wait]

            # Get the previous speed of the vehicle to determine acceleration
            try:
                previous_speed = prev_speed[vehicle]
            except KeyError:
                previous_speed = 0.0

            # Get current speed of the vehicle
            speed = self._ldm.getSpeed(vehicle)
            try:
                lane_stats[vehicle_lane]['speed'].append(speed)
            except KeyError:
                lane_stats[vehicle_lane]['speed'] = [speed]

            max_speed = self._ldm.getVehicleMaxSpeed(vehicle)

            try:
                lane_stats[vehicle_lane]['vehicle_delay'].append(max_speed - speed)
            except KeyError:
                lane_stats[vehicle_lane]['vehicle_delay'] = [max_speed - speed]
            # Count number of halted vehicles:
            if speed == 0.0:
                try:
                    lane_stats[vehicle_lane]['halted'] += 1.0
                except KeyError:
                    lane_stats[vehicle_lane]['halted'] = 1.0

            # Get vehicle's acceleration (negative value means deceleration)
            accel = speed - previous_speed
            try:
                lane_stats[vehicle_lane]['acceleration'].append(accel)
            except KeyError:
                lane_stats[vehicle_lane]['acceleration'] = [accel]

            # Count accelerations:
            if accel > 0:
                try:
                    lane_stats[vehicle_lane]['acceleration_count'] += 1.0
                except KeyError:
                    lane_stats[vehicle_lane]['acceleration_count'] = 1.0

            # Count decelerations:
            elif accel < 0:
                try:
                    lane_stats[vehicle_lane]['deceleration_count'] += 1.0
                except KeyError:
                    lane_stats[vehicle_lane]['deceleration_count'] = 1.0

            # Store current speed for use in next time step
            prev_speed[vehicle] = speed

            # If the vehicle decelerates too quickly, it is making an emergency stop
            if accel < -4.5:
                print("EMERGENCY STOP")
                stops.append(1.0)
                # Count emergency stops on the current lane
                try:
                    lane_stats[vehicle_lane]['em_sts'] += 1.0
                except KeyError:
                    lane_stats[vehicle_lane]['em_sts'] = 1.0
                found_stop = True
            else:
                stops.append(0.0)

        return lane_stats, prev_speed, stops

    def _get_linear_state(self, lane_states, controlled_lanes, extra="thesis"):
        state = []
        for lane in controlled_lanes:
            try:
                wait = sum(lane_states[lane]['wait'])
            except KeyError:
                wait = 0
            # print "waiting", wait
            try:
                vehicle_delay = sum(lane_states[lane]['vehicle_delay'])
            except KeyError:
                vehicle_delay = 0
            try:
                number_vehicles = lane_states[lane]['vehicle_count']
            except KeyError:
                number_vehicles = 0
            # print "num_veh", number_vehicles
            try:
                halted_vehicles = lane_states[lane]['halted']
            except KeyError:
                halted_vehicles = 0
            # print "halted", halted_vehicles
            try:
                speed = sum(lane_states[lane]['speed']) / len(lane_states[lane]['speed'])
            except KeyError:
                speed = 0
            # print "speed", speed
            try:
                avg_acceleration = sum(lane_states[lane]['acceleration']) / len(lane_states[lane]['acceleration'])
            except KeyError:
                avg_acceleration = 0
            # print "avg accel", avg_acceleration
            try:
                number_accelerations = lane_states[lane]['acceleration_count']
            except KeyError:
                number_accelerations = 0
            # print "num_accel", number_accelerations
            try:
                number_decelerations = lane_states[lane]['deceleration_count']
            except KeyError:
                number_decelerations = 0
            # print "number decel", number_decelerations
            try:
                em_sts = lane_states[lane]['em_sts']
            except KeyError:
                em_sts = 0
            # print "em_sts", em_sts
            # number_vehicles, halted_vehicles, speed, avg_acceleration, number_accelerations, number_decelerations, em_sts
            if extra == "large":
                state += [wait, number_vehicles, halted_vehicles, speed, avg_acceleration, number_accelerations,
                          number_decelerations, em_sts]
            elif extra == "small":
                state += [number_vehicles, halted_vehicles, wait]
            elif extra == "thesis":
                state += [wait, vehicle_delay, number_vehicles, halted_vehicles, speed, avg_acceleration, em_sts]
            else:
                state += [wait, number_vehicles, halted_vehicles, speed, avg_acceleration]

        if not extra == "thesis":
            for tl in self._ldm.getTrafficLights():
                setting = self._ldm.getLightState(tl)
                action_index = self._actions.index(setting)
                actions = [0 for x in range(0, len(self._actions))]
                actions[action_index] = 1
                state += actions
        else:
            combined_state = []
            for tl in self._ldm.getTrafficLights():
                setting = self._ldm.getLightState(tl)
                action_index = self._actions.index(setting)
                for a in range(0, len(self._actions)):
                    if a == action_index:
                        x = 1
                    else:
                        x = 0
                    for item in state:
                        combined_state.append(x * item)
                state = combined_state
        return state


class DenseState(State):
    """
    the dense state representation as described in my dissertation
    for one frame, it is a [lane_num, width+3] binary matrix
    '3' is a one-hot vector for three light status (red, yellow, green)

    """

    def __init__(self, lights, width, frames, ldm):
        State.__init__(self, ldm, lights)

        # get width
        self.width = width

        self.lane_num = len(self._lanes)
        # 0 for vertical, 1 for horizon
        self.vertical_horizon = []

        # define current state
        self._current_state = np.zeros((self.lane_num, self.width + 3, frames))

        # get coordinates of every lane
        self.all_coordinates = []
        for lane in self._lanes:
            coordinate = self._ldm.getLaneShape(lane)
            if coordinate[0][0] == coordinate[1][0]:
                self.vertical_horizon.append(1)
            else:
                self.vertical_horizon.append(0)
            self.all_coordinates.append(coordinate)
        # Get the size of the state in meters
        tl_state_size = self._get_state_size(self.all_coordinates)
        self.scale_factor = self._get_scale_factor(tl_state_size)

    def update_state(self):
        """
        Updates the state to match the current state in sumo
        @return new state
        """
        state_matrix = np.zeros((len(self._lanes), self.width + 3))
        lights = self._ldm.getLightState("0")
        for index, lane in enumerate(self._lanes):
            vehicles = self._ldm.getLaneVehicles(lane)
            vertical = self.vertical_horizon[index]
            # compute vehicle position information
            for vehicle in vehicles:
                location = self._ldm.getVehiclePosition(vehicle)
                if vertical == 1:
                    x = (int)(np.abs(location[1] - self.all_coordinates[index][0][1]) // self.scale_factor)
                else:
                    x = (int)(np.abs(location[0] - self.all_coordinates[index][0][0]) // self.scale_factor)
                state_matrix[index][x] = 1
            # compute one-hot light vector
            light_color = lights[index]
            light_vector = np.zeros((1, 3))
            if light_color == 'G':
                light_vector[0][2] = 1
            elif light_color == 'y':
                light_vector[0][1] = 1
            else:
                light_vector[0][0] = 1
            # append one-hot light vector to the every lane
            state_matrix[index][-3:] = light_vector
        self._add_state_matrix(state_matrix)
        return self._current_state

    def get_action_space(self):
        # two actions: all vertical green or all horizon green
        a1 = ''
        a2 = ''
        for vertical in self.vertical_horizon:
            if vertical == 1:
                a1 += 'G'
                a2 += 'r'
            else:
                a1 += 'r'
                a2 += 'G'
        return [a1, a2]
    
    def get_height(self):
        """
        @return the height of the matrix
        """
        return self.lane_num

    def get_width(self):
        """
        @return the width of the matrix
        """
        return self.width + 3
    
    def _get_scale_factor(self, state_size):
        """
        get scale factor - the length of one cell
        """
        return state_size / float(self.width)

    def _get_state_size(self, all_coordinates):
        """
        assume every lane has the same length
        """
        one_lane = all_coordinates[0]
        return np.abs(one_lane[0][0] - one_lane[1][0]) + np.abs(one_lane[0][1] - one_lane[1][1])
 
    def _add_state_matrix(self, state_matrix):
        """
        update 'current state'
        """
        temp_state = copy.deepcopy(self._current_state[:, :, 0:-1])
        # First element is the latest state
        self._current_state[:, :, 0] = state_matrix
        # The rest are the 2nd, 3rd and 4th latest
        self._current_state[:, :, 1:] = temp_state


class MatrixState():
    """
    WARNING THIS CLASS HAS NOT BEEN FIXED YET (does not extend State)
    This is the super class that describes some basic functions
    of a matrix respresentation of a state
    """

    def __init__(self, lights, width, height, frames, traci):
        """
        This class stores the lanes it represents and calculates everything
        it needs to rescale the information to a matrix
        """

        self._lights = lights
        self._lanes = []
        # The orientation of the lanes, 0 for vertical, 1 for horizontal
        self.vertical_horizon = []
        for light_i in lights:
            lanes = traci.trafficlight.getControlledLanes(light_i)
            self._lanes += lanes
        self.width = width
        self.height = height

        self._max_speeds = {}

        # Get coordinates of each lane
        all_coordinates = []
        for lane in self._lanes:
            self._max_speeds[lane] = traci.lane.getMaxSpeed(lane)
            lane_coordinates = traci.lane.getShape(lane)
            # if the x coordinates of the begin and end point of a lane are the same, the lane is vertical
            if lane_coordinates[0][0] == lane_coordinates[1][0]:
                self.vertical_horizon.append(0)
            else:
                self.vertical_horizon.append(1)
            all_coordinates.append(lane_coordinates)
        bottom_left, upper_right = self.get_corner_points(all_coordinates)

        # Get the size of the state in meters
        tl_state_size = self._get_state_size(upper_right, bottom_left)
        # Compute how much to scale height/width to fit state matrix
        self.scale_factor = self._get_scale_factor(tl_state_size)
        self.bottom_left = bottom_left

    def getLanes(self):
        """
        the lanes that this state is working with
        """
        return self._lanes

    def getMaxSpeed(self, lane):
        """
        @param lane the lane id
        @return maximum speed for that lane
        """
        return self._max_speeds[lane]

    def get_corner_points(self, coordinate_list):
        """
        Using a list of coordinates, compute the corner points:
        the left bottom corner is defined by the smallest x and y
        coordinate, while the upper right corner is defined by the
        biggest x and y coordinate.

        Keyword arguments:
            coordinate_list -- contains points in the state,
                                including the corners
        Returns: list
        """
        x_coordinates = []
        y_coordinates = []
        for coordinates in coordinate_list:
            for coordinate in coordinates:
                x_coordinate = coordinate[0]
                y_coordinate = coordinate[1]
                x_coordinates.append(x_coordinate)
                y_coordinates.append(y_coordinate)
        smallest_x = sorted(x_coordinates)[0]
        biggest_x = sorted(x_coordinates)[-1]
        smallest_y = sorted(y_coordinates)[0]
        biggest_y = sorted(y_coordinates)[-1]
        return [smallest_x, smallest_y], [biggest_x, biggest_y]

    def _get_state_size(self, upper_right, bottom_left):
        """
        Using the bottom left and upper right corner points,
        compute the size of the state.

        Returns: list
        """
        width = upper_right[0] - bottom_left[0]
        height = upper_right[1] - bottom_left[1]
        return [height, width]

    def _get_scale_factor(self, state_size):
        """
        Using the state size and desired width and height,
        compute the scaling factor required to scalarize a
        SUMO state into the desired width and height

        Keyword arguments:
            state_size -- size of state in meters

        Returns: list
        """
        scale_width = state_size[0] / float(self.width)
        scale_height = state_size[1] / float(self.height)
        return [scale_height, scale_width]

    def reshape_location(self, location):
        """
            Reshape a real-valued location into a set of matrix
            coordinates.

            Keyword arguments:
                location -- real location in the simulation state

            Returns: coordinates rescaled to the matrix size
        """
        reshaped_location = [0, 0]
        reshaped_location[0] = int((location[0] - self.bottom_left[0]) // self.scale_factor[0])
        reshaped_location[1] = int((location[1] - self.bottom_left[1]) // self.scale_factor[1])
        # if the location is exactly the maximum horizontal or vertical value it will be out of bounds
        if reshaped_location[0] == self.width:
            reshaped_location[0] -= 1
        if reshaped_location[1] == self.height:
            reshaped_location[1] -= 1

        return reshaped_location

    # return action space
    def get_action_space(self):
        # two actions: all vertical green or all horizon green
        a1 = ''
        a2 = ''
        for vertical in self.vertical_horizon:
            if vertical == 1:
                a1 += 'G'
                a2 += 'r'
            else:
                a1 += 'r'
                a2 += 'G'
        return [a1, a2]


class PositionMatrix(MatrixState):
    """
    WARNING THIS CLASS HAS NOT BEEN FIXED YET (does not extend State)
    TODO document what this is and does
    """

    def __init__(self, lights, width, height, frames, traci):
        """
        This class stores the state as a binary position matrix as used
        by the DQN networks.
        """
        MatrixState.__init__(self, lights, width, height, frames, traci)

        # Tensorflow expects the input of convolution to be
        # of shape [batch, in_height, in_width, in_channels]
        self._current_state = np.zeros((height, width, frames))

    def update_state(self, traci, rotation=0.):
        """
        Store each vehicle's location in the state representation
        matrix.

        Keyword arguments:
            lanes -- the lanes that are included in this state
            (optional) rotation -- the rotation of this state
                                    compared to the learned
                                    Q-value function (only relevant
                                    when sharing Q-value functions)
        @return new state
        """
        state_matrix = np.zeros((self.height, self.width))

        for lane in self._lanes:
            vehicles = traci.lane.getLastStepVehicleIDs(lane)
            for vehicle in vehicles:
                location = traci.vehicle.getPosition(vehicle)
                [x, y] = self.reshape_location(location)
                state_matrix[x][y] = 1

        if rotation > 0:
            # Rotate state rotation * 90 degrees
            state_matrix = np.rot90(state_matrix, rotation)

        self._add_state_matrix(state_matrix)
        return self._current_state

    def _add_state_matrix(self, state_matrix):
        """
        Controls a moving window state:
        replaces the first element with the new state matrix,
        while shifting the other elements one to the right.

        Keyword arguments:
            state_matrix -- the latest state representation

        Returns: None
        """
        # Deepcopy to prevent overwriting due to Python referencing
        temp_state = copy.deepcopy(self._current_state[:, :, 0:-1])
        # First element is the latest state
        self._current_state[:, :, 0] = state_matrix
        # Rest is the 2nd, 3rd and 4th latest
        self._current_state[:, :, 1:] = temp_state


class PositionLightMatrix(MatrixState):
    """
    WARNING THIS CLASS HAS NOT BEEN FIXED YET (does not extend State)
    This class contains the positions of the cars and the current states
    of the traffic lights.
    """

    def __init__(self, lights, width, height, frames, traci):
        """
        This class is an instance of MatrixState
        """
        MatrixState.__init__(self, lights, width, height, frames, traci)

        self._current_state = np.zeros((width, height, frames))

    def update_state(self, traci, rotation=0.):
        """
        Retrieve the new state information from SUMO.
        Keyword arguments:
            traci -- an instance of TraCI to communicate with SUMO
            (optional) rotation -- the rotation of this state compared
                                    to the learned Q-function.
        @return new state
        """
        state_matrix = np.zeros((self.height, self.width))

        for lane in self._lanes:
            vehicles = traci.lane.getLastStepVehicleIDs(lane)
            for vehicle in vehicles:
                location = traci.vehicle.getPosition(vehicle)
                [x, y] = self.reshape_location(location)
                # Vehicle location
                state_matrix[x][y] = 1

        for light_i in self._lights:
            light_color = traci.trafficlight.getRedYellowGreenState(light_i)
            # Set traffic lights to corresponding colors
            # This is very much hard-coded, needs fixing for
            # other scenarios
            state_matrix = self.stop_light_locations(state_matrix, light_color, traci)

        if rotation > 0:
            # Rotate state rotation * 90 degrees
            state_matrix = np.rot90(state_matrix, rotation)

        self._add_state_matrix(state_matrix)
        return self._current_state

    def _add_state_matrix(self, state_matrix):
        """
        Controls a moving window state:
        replaces the first element with the new state matrix,
        while shifting the other elements one to the right.

        Keyword arguments:
            state_matrix -- the latest state representation

        Returns: None
        """
        # Deepcopy to prevent overwriting due to Python referencing
        temp_state = copy.deepcopy(self._current_state[:, :, 0:-1])
        # First element is the latest state
        self._current_state[:, :, 0] = state_matrix
        # Rest is the 2nd, 3rd and 4th latest
        self._current_state[:, :, 1:] = temp_state

    def stop_light_locations(self, state_matrix, light_color, traci):
        """
        Add the right value for the state of the traffic light at the right
        position in the matrix.
        Keyword arguments:
            state_matrix -- the matrix in which the values are stores
            light_color  -- a tuple containing the state of the traffic light
            traci        -- instance of TraCI to communicate with SUMO
        """
        index = 0
        for lane in self._lanes:
            lane_end = traci.lane.getShape(lane)[1]

            if light_color[index] == 'G':
                val = 0.8
            elif light_color[index] == 'y':
                val = 0.5
            elif light_color[index] == 'r':
                val = 0.2
            x, y = self.reshape_location(lane_end)
            state_matrix[x][y] = val

            index += 1
        return state_matrix


class ValueMatrix(MatrixState):
    """
    WARNING THIS CLASS HAS NOT BEEN FIXED YET (does not extend State)
    This class contains the positions of the cars, the speed of the cars,
    the acceleration of the cars and the states of the traffic lights.
    """

    def __init__(self, lights, width, height, frames, traci, y_t=4):
        """
        This class is an instance of MatrixState
        """
        MatrixState.__init__(self, lights, width, height, frames, traci)

        if frames < 4:
            raise ValueError(("The number of frames need to be 3 for \
                              this type of state representation."))

        self._current_state = np.zeros((height, width, frames))

        self.state_speed = {}
        # One matrix is added per second of yellow time, that is,
        # if the yellow time is four seconds, the last four traffic
        # light matrices are added to the state.
        self.last_colors_dict = {}
        for light_i in self._lights:
            if y_t == 0:
                # In the special case that no static yellow time is employed,
                # a single traffic light matrix is still used, since the
                # current traffic light configuration is part of the state.
                self.last_colors_dict[light_i] = [traci.trafficlight.getRedYellowGreenState(light_i)]
            else:
                self.last_colors_dict[light_i] = [traci.trafficlight.getRedYellowGreenState(light_i) for x in range(y_t)]

    def update_state(self, traci, rotation=0.):
        """
        Retrieve the new state information from SUMO.
        Keyword arguments:
            traci -- an instance of TraCI to communicate with SUMO
            (optional) rotation -- the rotation of this state compared
                                    to the learned Q-function.
        @return new state
        """
        state_matrix = np.zeros(self._current_state.shape)

        for lane in self._lanes:
            vehicles = traci.lane.getLastStepVehicleIDs(lane)
            for vehicle in vehicles:
                vehicle_speed = traci.vehicle.getSpeed(vehicle)
                lane = traci.vehicle.getLaneID(vehicle)
                vehicle_max_speed = traci.lane.getMaxSpeed(lane)
                current_speed = vehicle_speed / vehicle_max_speed

                location = traci.vehicle.getPosition(vehicle)
                [x, y] = self.reshape_location(location)
                # Vehicle location
                state_matrix[x][y][0] = 1
                # Vehicle speed
                state_matrix[x][y][1] = current_speed
                # Vehicle deceleration/acceleration
                try:
                    old_speed = self.state_speed[vehicle]
                except KeyError:
                    old_speed = 0.0

                state_matrix[x][y][2] = (current_speed - old_speed)
                # Update speed dictionary
                self.state_speed[vehicle] = current_speed

        for light_i in self._lights:
            light_color = traci.trafficlight.getRedYellowGreenState(light_i)
            self.update_last_colors(light_color, light_i)
            # Set traffic lights to corresponding colors
            # This is very much hard-coded, needs fixing for
            # other scenarios
            for i, light_color in enumerate(self.last_colors_dict[light_i]):
                state_matrix = self.stop_light_locations(state_matrix, i, light_color, traci)

        if rotation > 0:
            # Rotate state rotation * 90 degrees
            state_matrix = np.rot90(state_matrix, rotation)

        self._current_state = state_matrix
        return self._current_state

    def update_last_colors(self, light_color, tl):
        """
        Controls a moving window state:
        replaces the first element with the new state of a traffic light,
        while shifting the other elements one to the right.

        Keyword arguments:
            light_color -- a tuple containing the state of the traffic light
            tl          -- the id of the traffic light
        """
        temp = copy.deepcopy(self.last_colors_dict[tl][0:-1])
        self.last_colors_dict[tl][0] = light_color
        self.last_colors_dict[tl][1:] = temp

    def stop_light_locations(self, state_matrix, i, light_color, traci):
        """
        Add the right value for the state of the traffic light at the right
        position in the matrix.
        Keyword arguments:
            state_matrix -- the matrix in which the values are stores
            i            -- the time index for the color buffer
            light_color  -- a tuple containing the state of the traffic light
            traci        -- instance of TraCI to communicate with SUMO
        """
        index = 0
        for lane in self._lanes:
            lane_end = traci.lane.getShape(lane)[1]

            if light_color[index] == 'G':
                val = 1.0
            elif light_color[index] == 'y':
                val = 0.6
            elif light_color[index] == 'r':
                val = 0.2
            x, y = self.reshape_location(lane_end)
            state_matrix[x][y][3 + i] = val

            index += 1

        return state_matrix


class LdmMatrixState(State):
    """
    TODO document how this state works and achieves
    """

    def __init__(self, ldm, data, type="byCorners"):
        State.__init__(self, ldm, None)

        if type == "byCorners":
            self.bottomLeftCoords = data[0]
            self.topRightCoords = data[1]
        elif type == "byCenter":
            self.bottomLeftCoords = (data[0][0] - data[1] / 2., data[0][1] - data[2] / 2.)
            self.topRightCoords = (data[0][0] + data[1] / 2., data[0][1] + data[2] / 2.)

    def update_reward(self, local_rewards=True):
        return (self._ldm.getReward())

    def update_state(self):
        # self._ldm.update_vehicle_vector(self.bottomLeftCoords, self.topRightCoords)
        return self._ldm.getMapSliceByCorners(self.bottomLeftCoords, self.topRightCoords)
    
    def size(self) -> list: 
        """
        returns the size of the matrix as a list of 2 elements.
        """
        return [self.topRightCoords[0] - self.bottomLeftCoords[0], self.topRightCoords[1] - self.bottomLeftCoords[1] ]
