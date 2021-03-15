import os
# import tensorflow as tf
import numpy as np


class Controller(object):
    """
    Controller object that interacts with all the agents in the system,
    runs the coordination algorithms (in the multi-agent case), gets
    agent's actions, passes the environment signals to the agents.
    """

    def __init__(self, parameters: dict, action_map: dict, run):
        """
        @param parameters a dictionary with all kinds of options for the run.
        @param action_map  a dict with factor index numbers (in the factorgraph) as keys
        and a list of allowed actions as values
        @param logger
        """
        # todo fix controller to use pytorch
        # Initialize all factor objects here
        self.parameters = parameters
        self.num_actions = {}
        self.step = {}
        tf.reset_default_graph()
        self.step = 0
        summary_path = 'summaries/' + self.parameters['name'] + '_' + \
            self.parameters['scene'] +  '_' + str(self.parameters['flicker'])
        if not os.path.exists(summary_path):
            os.makedirs(summary_path)
        self.summary_writer = tf.summary.FileWriter(summary_path)
        self.run = run

    def get_actions(self, step_output, ip=None):
        """
        Get each factor's action based on its local observation. Append the given
        state to the factor's replay memory.
        """
        evaluate_policy_output = {}
        evaluate_policy_output.update(self.model.evaluate_policy(step_output['obs'],
                                                                 step_output['prev_action']))                                                                 
        return evaluate_policy_output

    def update(self):
        """
        Sample a batch from the replay memory (if it is completely filled) and
        use it to update the models.
        """
        raise NotImplementedError

    def full_memory(self):
        """
        Check if the replay memories are filled.
        """
        return self.buffer.full()

    def save_graph(self):
        """
        Store all the networks and replay memories.
        """
        if self.step % self.parameters['save_frequency'] == 0:
            # Create factor path if it does not exist.
            path = os.path.join('models', self.parameters['name'], 
                                self.parameters['scene'], 
                                str(self.parameters['flicker']))
            if not os.path.exists(path):
                os.makedirs(path)
            self.model.save_graph(self.step)

    # TODO: replay memory
    def store_memory(self, path):
        # Create factor path if it does not exist.
        path = os.path.join(os.environ['APPROXIMATOR_HOME'], path)
        if not os.path.exists(path):
            os.makedirs(path)
        # Store the replay memory
        self.buffer.store(path)

    def increment_step(self):
        self.model.increment_step()
        self.step = self.model.get_current_step()

    def write_summary(self):
        """
        Saves training statistics to Tensorboard.
        """
        if self.step % self.parameters['summary_frequency'] == 0 and \
           self.parameters['tensorboard']:
            summary = tf.Summary()
            for key in self.stats.keys():
                if len(self.stats[key]) > 0:
                    stat_mean = float(np.mean(self.stats[key]))
                    summary.value.add(tag='{}'.format(key), simple_value=stat_mean)
                    self.run.log_scalar('{}'.format(key), stat_mean, self.step)
                    self.stats[key] = []
            self.summary_writer.add_summary(summary, self.step)
            self.summary_writer.flush()
