import numpy as np
import random
import pickle


class Buffer(dict):
    """
    Dictionary to store transitions in.
    """
    # NOTE: The replay memory is modified so that transitions are stored in a
    # dictionary.

    def __init__(self, parameters, act_size, separate=False):
        """
        Initialize the memory with the right size.
        """

        self.memory_size = parameters['memory_size']
        self.batch_size = parameters['batch_size']
        if parameters['influence']:
            self.seq_len = parameters['inf_seq_len']
        elif parameters['recurrent']:
            self.seq_len = parameters['seq_len']
        else:
            self.seq_len = 1

    def __getitem__(self, key):
        if key not in self.keys():
            self[key] = list()
        return super(Buffer, self).__getitem__(key)

    def sample(self, batch_size):
        """
        Sample a batch from the dataset. This can be implemented in
        different ways.
        """
        raise NotImplementedError

    def get_latest_entry(self):
        """
        Retrieve the entry that has been added last. This can differ
        between sequence en non-sequence samplers.
        """
        raise NotImplementedError

    def full(self):
        """
        Check whether the replay memory has been filled.
        """
        # TODO: returns are only calculated either when time horizon is reached
        # or when the episode is over. When that happens all fields in replay_
        # memory
        # are the same size
        # This means replay memory could
        if 'returns' not in self.keys():
            return False
        else:
            return len(self['returns']) >= self.memory_size

    def store(self, path):
        """
        Store the necessary information to recreate the replay memory.
        """
        with open(path + 'buffer.pkl', 'wb') as f:
            pickle.dump(self.buffer, f, pickle.HIGHEST_PROTOCOL)

    def load(self, path):
        """
        Load a stored replay memory.
        """
        with open(path + 'replay_memory.pkl', 'rb') as f:
            self.buffer = pickle.load(f)

    def empty(self):
        for key in self.keys():
            self[key] = []


class SerialSampling(Buffer):
    """
    Batches of experiences are sampled in a series one after the other.
    This ensures all experiences are used the same number of times. Valid for
    sequences or single experiences.
    """

    def __init__(self, parameters, act_size):
        """
        This is an instance of a plain Replay Memory object. It does not
        need more information than its super class.
        """
        super().__init__(parameters, act_size)

    def sample(self, b, n_sequences, keys=None):
        """
        """
        batch = {}
        if keys is None:
            keys = self.keys()
        for key in keys:
            batch[key] = []
            for s in range(n_sequences):
                start = s*self.seq_len + b*self.batch_size
                end = (s+1)*self.seq_len + b*self.batch_size
                batch[key].extend(self[key][start:end])
            # permut dimensions workers-batch to mantain sequence order
            axis = np.arange(np.array(batch[key]).ndim)
            axis[0], axis[1] = axis[1], axis[0]
            batch[key] = np.transpose(batch[key], axis)
        return batch

    def shuffle(self):
        """
        """
        n = len(self['returns'])
        # Only include complete sequences
        # seq_len = 8
        indices = np.arange(0, n - n % self.seq_len, self.seq_len)
        random.shuffle(indices)
        for key in self.keys():
            shuffled_memory = []
            for i in indices:
                shuffled_memory.extend(self[key][i:i+self.seq_len])
            self[key] = shuffled_memory

    def get_last_entries(self, t, keys=None):
        """
        """
        if keys is None:
            keys = self.keys()
        batch = {}
        for key in keys:
            batch[key] = self[key][-t:]
        return batch

    def zero_padding(self, missing, worker):
        for key in self.keys():
            if key not in ['advantages', 'returns']:
                padding = np.zeros_like(self[key][-1])
                for i in range(missing):
                    self[key].append(padding)
