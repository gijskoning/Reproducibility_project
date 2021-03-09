import random
import logging
import os
from pathlib import Path
import warnings
import xml.etree.ElementTree as ElementTree
import glob


class SumoHelper(object):
    """
    Object that holds helper functions + information for generating routes
    and scenarios for SUMO
    """

    def __init__(self, parameters, seed=None):
        """
        Initializes SUMOHelper object and checks 1) if the proper types are
        being used for the parameters and 2) if the scenario has the proper
        definitions
        @param port: network socket number to connect SUMO with. Default usually 8000.
        """
        self.parameters = parameters
        assert(self.scenario_check(self.parameters['scene']))
        assert(type(self.parameters['car_pr']) == float)
        assert(type(self.parameters['car_tm']) == int)
        self.seed = seed
        self.sumocfg_name = str(self.seed) + '_scenario.sumocfg'
        self._generate_sumocfg_file()

    def scenario_check(self, scenario):
        """
        Checks if the scenario is well-defined and usable by seeing if all
        the needed files exist.
        """
        dirname = os.path.dirname(__file__)
        self.scenario_path = os.path.join(dirname, 'scenarios', scenario)

        if(self.parameters['generate_conf']):
            self._net_file = os.path.basename(glob.glob(self.scenario_path + '/*.net.xml')[0])
            self._needed_files = [os.path.basename(self._net_file)]
        else:
            self.sumocfg_file = glob.glob(self.scenario_path + '/*.sumocfg')[0]
            self._needed_files = [os.path.basename(self.sumocfg_file)]

        scenario_files = os.listdir(self.scenario_path)
        for n_file in self._needed_files:
            if n_file not in scenario_files:
                logging.error(("The scenario is missing file '{}' in {}, please add it and "
                      "try again.".format(n_file, self.scenario_path)))
                return False

        return True

    def write_route(self, route_dict, car_list):
        """
        Writes the route information and generated vehicles to file
        """

        # Define possible routes as read from file earlier
        setup_string = "<routes>\n\n"

        for route in route_dict:
            setup_string += '    <route id="' + route + '" edges="' + \
                            route_dict[route] + '"/>\n'
        setup_string += '\n'

        route_name = str(self.seed) + '_routes.rou.xml'
        route_file = os.path.join(self.scenario_path, route_name)
        # Write the cars to file as generated earlier
        with open(route_file, 'w') as f:
            f.write(setup_string)
            for t, car in enumerate(car_list):
                if car is not None:
                    car_string = '    <vehicle id="' + car + '_' + str(t) + \
                                 '" route="' + car + '" depart="' + str(t) + \
                                 '" />\n'
                    f.write(car_string)
            f.write('\n</routes>')

    def _generate_sumocfg_file(self):
        self.sumocfg_file = os.path.join(self.scenario_path, self.sumocfg_name)
        self.routefile_name = str(self.seed) + '_routes.rou.xml'
        self._route_file = os.path.join(self.scenario_path, self.routefile_name)
        with open(self.sumocfg_file, 'w') as f:
            f.write('<?xml version="1.0" encoding="UTF-8"?>\n'
                    +'<configuration xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/sumoConfiguration.xsd">\n'
                    +'    <input>\n'
                    +'        <net-file value="' + self._net_file + '"/>\n'
                    +'        <route-files value="' + self.routefile_name + '"/>\n'
                    +'    </input>\n'
                    +'    <time>\n'
                    +'        <begin value="0"/>\n'
                    +'    </time>\n'
                    +'    <report>\n'
                    +'        <verbose value="true"/>\n'
                    +'        <no-step-log value="true"/>\n'
                    +'    </report>\n'
                    +'</configuration>')

    def generate_randomized_route(self):
        if len(self.parameters['route_starts']) > 0:
            route = random.choice(self.parameters['route_starts'])
            route += " "
        else:
            route = ""

        number_of_segments = random.choice(range(self.parameters['route_min_segments'], self.parameters['route_max_segments'] + 1))

        for i in range(number_of_segments):
            route += random.choice(self.parameters['route_segments']) + " "

        if len(self.parameters['route_ends']) > 0:
            route += random.choice(self.parameters['route_ends'])
        else:
            route = route[:-1]
        return route

    def _generate_route_file(self, seed):
        """
        Generates vehicles for each possible route in the scenario and writes
        them to file. Returns the location of the sumocfg file.
        """
        logging.debug(('The seed being used for route generation: {}'.format(seed)))
        random.seed(seed)

        car_list = []
        car_sum = 0

        route_dict = {}
        expected_value = self.parameters['car_tm'] * self.parameters['car_pr']

        for t in range(self.parameters['car_tm']):
            random_number = random.randint(0, 100) * 0.01
            if random_number <= self.parameters['car_pr']:
                route = self.generate_randomized_route()
                key = str(len(route_dict) + 1)
                route_dict[key] = route
                car = key
                car_list.append(car)
                car_sum += 1
            else:
                car_list.append(None)

        self.write_route(route_dict, car_list)

        if float(car_sum) / expected_value >= 10:
            warnings.warn("The expected number of cars is {}, but the "
                          "actual number of cars is {}, which may indicate"
                          " a bug.".format(expected_value, car_sum))

    def __del__(self):
        if(self.parameters['generate_conf']):
            if ('sumocfg_file' in locals()):
                os.remove(self.sumocfg_file)
            if ('_route_file' in locals()):
                os.remove(self._route_file)
