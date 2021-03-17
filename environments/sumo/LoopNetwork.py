from .SumoGymAdapter import SumoGymAdapter
import copy

class LoopNetwork(SumoGymAdapter):

    __DEFAULT_PARAMETERS = {
        'scene': 'loop_network',
        'box_bottom_corner': [9, 9], #[10, 12],
        'box_top_corner':  [68, 68], #[66, 68],
        'y_t': 6, # Yellow (traffic light) time
        'resolutionInPixelsPerMeterX': 0.25,
        'resolutionInPixelsPerMeterY': 0.25,
        'car_tm': 6,
        'state_type': 'ldm_state', # The type of state to use as input for the network. ('bin' (Position Matrix), 'bin_light' (Position Light Matrix), 'value' (Value Matrix))
        'scaling_factor': 10,
        'fast': False,
        'speed_dev': 0.0, # Can be used to vary the speeds of cars, according to a normal distribution with mean 1 and standard deviation speed_dev (SUMOs default is 0.1)
        'car_pr': 1.0,
        'route_segments': ['L67 L68 L61 L62 L63 L64 L65 L66 L67 L68 L61 L62 L63 L64 L65 L66 L67 L68 L61 L62 L63 L64 L65 L66 L67 L68 L61 L62 L63 L64 L65 L66 L67 L68 L61 L62',
            'L63 L64 L65 L66 L67 L68 L61 L62 L63 L64 L65 L66 L67 L68 L61 L62 L63 L64 L65 L66 L67 L68 L61 L62 L63 L64 L65 L66 L67 L68 L61 L62 L63 L64 L65 L66 L67 L68 L61 L62 L63 L64 L65 L66'],
        'route_starts': [],
        'route_ends': [],
        'route_max_segments': 1,
        'route_min_segments': 1,
        'local_rewards': False,
        'waiting_penalty': False,
        'reward_type': 'waiting_time',
        'lightPositions': {},#{"0": ((37.5,44.16), (39.2,44.16), (32.5,37.5), (32.5,39.16))},
        'traffic_lights': False
    }

    def __init__(self, seed, parameters=None):
        _parameters = copy.deepcopy(self._DEFAULT_PARAMETERS) # load default parameters of SUMOGymAdapter
        _parameters.update(self.__DEFAULT_PARAMETERS) # load default parameters of GridSumoEnv
        if parameters is not None:
            _parameters.update(parameters) # load parameters given by the user
        super().__init__(_parameters, seed)
        # todo temp for gym adaptation
        self.reward_range = 2
        self.metadata = 2
