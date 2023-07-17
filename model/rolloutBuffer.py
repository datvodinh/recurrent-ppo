import torch
import numpy as np



class RolloutBuffer:
    """Save and store Agent's data"""
    def __init__(self,config,state_size,action_size) -> None:
        self.hidden_size        = config["LSTM"]["hidden_size"]
        self.seq_length         = config["LSTM"]["seq_length"]
        self.max_eps_length     = config["LSTM"]["max_eps_length"]
        self.num_game_per_batch = config["num_game_per_batch"]
        self.n_mini_batches     = config["n_mini_batch"]

        self.batch = {
            "states"       : torch.zeros((self.num_game_per_batch,self.max_eps_length,state_size)),
            "h_states"     : torch.zeros((self.num_game_per_batch,self.max_eps_length,self.hidden_size)),
            "c_states"     : torch.zeros((self.num_game_per_batch,self.max_eps_length,self.hidden_size)),
            "actions"      : torch.zeros((self.num_game_per_batch,self.max_eps_length)),
            "values"       : torch.zeros((self.num_game_per_batch,self.max_eps_length)),
            "policy"       : torch.zeros((self.num_game_per_batch,self.max_eps_length,action_size)),
            "probs"        : torch.zeros((self.num_game_per_batch,self.max_eps_length)),
            "dones"        : torch.zeros((self.num_game_per_batch,self.max_eps_length)),
            "action_mask"  : torch.zeros((self.num_game_per_batch,self.max_eps_length,action_size)),
            "rewards"      : torch.zeros((self.num_game_per_batch,self.max_eps_length)),
            "advantages"   : torch.zeros((self.num_game_per_batch,self.max_eps_length)),
            "dones_indices": torch.zeros((self.num_game_per_batch))
        }

        
        self.state_size         = state_size
        self.action_size        = action_size

        self.game_count         = 0 #track current game
        self.step_count         = 0 #track current time step in game

    def add_data(self,state,h_state,c_state,action,value,reward,done,valid_action,prob,policy):
        """Add data to rollout buffer"""
        self.batch["states"][self.game_count][self.step_count]      = state
        self.batch["h_states"][self.game_count][self.step_count]    = h_state
        self.batch["c_states"][self.game_count][self.step_count]    = c_state
        self.batch["actions"][self.game_count][self.step_count]     = action
        self.batch["values"][self.game_count][self.step_count]      = value
        self.batch["policy"][self.game_count][self.step_count]      = policy
        self.batch["probs"][self.game_count][self.step_count]       = prob
        self.batch["dones"][self.game_count][self.step_count]       = done
        self.batch["action_mask"][self.game_count][self.step_count] = valid_action
        self.batch["rewards"][self.game_count][self.step_count]     = reward

    def reset_data(self):
        """Clear all data"""
        self.batch = {
            "states"       : torch.zeros((self.num_game_per_batch,self.max_eps_length,self.state_size)),
            "h_states"     : torch.zeros((self.num_game_per_batch,self.max_eps_length,self.hidden_size)),
            "c_states"     : torch.zeros((self.num_game_per_batch,self.max_eps_length,self.hidden_size)),
            "actions"      : torch.zeros((self.num_game_per_batch,self.max_eps_length)),
            "values"       : torch.zeros((self.num_game_per_batch,self.max_eps_length)),
            "policy"       : torch.zeros((self.num_game_per_batch,self.max_eps_length,self.action_size)),
            "probs"        : torch.zeros((self.num_game_per_batch,self.max_eps_length)),
            "dones"        : torch.zeros((self.num_game_per_batch,self.max_eps_length)),
            "action_mask"  : torch.zeros((self.num_game_per_batch,self.max_eps_length,self.action_size)),
            "rewards"      : torch.zeros((self.num_game_per_batch,self.max_eps_length)),
            "advantages"   : torch.zeros((self.num_game_per_batch,self.max_eps_length)),
            "dones_indices": torch.zeros((self.num_game_per_batch))
        }
        self.game_count = 0 #track current game
        self.step_count = 0 #track current time step in game

    def cal_advantages(self,
                       gamma:float,
                       gae_lambda:float):
        """
        Overview:
            Calculate GAE.

        Arguments:
            - gamma: (`float`): gamma discount.
            - gae_lambda: (`float`): gae_lambda discount.
        """

        last_value               = self.batch["values"][:,-1]
        last_advantage           = torch.zeros_like(last_value)

        for t in range(self.max_eps_length-1,-1,-1):
            mask                          = 1.0 - self.batch["dones"][:,t]
            last_value                    = last_value * mask
            last_advantage                = last_advantage * mask
            delta                         = self.batch["rewards"][:,t] + gamma * last_value - self.batch["values"][:,t]
            last_advantage                = delta + gamma * gae_lambda * last_advantage
            self.batch["advantages"][:,t] = last_advantage
            last_value                    = self.batch["values"][:,t]

    def _arange_data_to_sequences(self, data):
        """
        Overview:
            Splits the povided data into episodes and then into sequences.
            The split points are indicated by the envrinoments' done signals.
        
        Arguments:
            data {`torch.tensor`} -- The to be split data arrange into num_worker, max_eps_length
            
        Returns:
            {`list`} -- Data arranged into sequences of variable length as list
        """
        sequences = []
        max_length = 1
        for w in range(data.shape[0]): #number of game per batch
            start_index = 0
            done_index = int(self.batch["dones_indices"][w])
            # Split trajectory into episodes
            episode = data[w, start_index:done_index + 1]
            # Split episodes into sequences
            if self.seq_length > 0:
                for start in range(0, len(episode), int(self.seq_length)):
                    end = start + int(self.seq_length)
                    sequences.append(episode[start:end])
                max_length = self.seq_length
            else:
                # If the sequence length is not set to a proper value, sequences will be based on episodes
                sequences.append(episode)
                max_length = len(episode) if len(episode) > max_length else max_length
            start_index = done_index + 1
        return sequences, max_length
    
    def _pad_sequence(self, sequence:torch.Tensor, target_length:int) -> torch.Tensor:
        """
        Overview:
            Pads a sequence to the target length using zeros.

        Arguments:
            - sequence {`torch.Tensor`} -- The to be padded array (i.e. sequence)
            - target_length {`int`} -- The desired length of the sequence

        Returns:
            {torch.tensor} -- Returns the padded sequence
        """
        # Determine the number of zeros that have to be added to the sequence
        delta_length = target_length - len(sequence)
        # If the sequence is already as long as the target length, don't pad
        if delta_length <= 0:
            return sequence
        # Construct array of zeros
        if len(sequence.shape) > 1:
            # Case: pad multi-dimensional array (e.g. visual observation)
            padding = torch.zeros(((delta_length,) + sequence.shape[1:]), dtype=sequence.dtype)
        else:
            padding = torch.zeros(delta_length, dtype=sequence.dtype)
        # Concatenate the zeros to the sequence
        return torch.cat((sequence, padding), axis=0)
    
    def prepare_batch(self):
        """
        Overview:
            Flattens the training samples and stores them inside a dictionary. Due to using a recurrent policy,
            the data is split into episodes or sequences beforehand.
        """
        # Supply training samples
        samples = {
            "states"      : self.batch["states"],
            "actions"     : self.batch["actions"],
            "policy"      : self.batch["policy"],
            "action_mask" : self.batch["action_mask"],
            "loss_mask"   : torch.ones((self.num_game_per_batch, self.max_eps_length), dtype=torch.bool), # The loss mask is used for masking the padding while computing the loss function.
            "h_states"    : self.batch["h_states"],
            "c_states"    : self.batch["c_states"],
        }
        # Retrieve unpadded sequence indices
        self.flat_sequence_indices = self._arange_data_to_sequences(
                    torch.arange(0, self.num_game_per_batch * self.max_eps_length).reshape(
                        (self.num_game_per_batch, self.max_eps_length)))[0]
        self.flat_sequence_indices = np.array([t.numpy() for t in self.flat_sequence_indices],dtype=object)

        for key, value in samples.items():
            sequences, max_sequence_length = self._arange_data_to_sequences(value)
            for i, sequence in enumerate(sequences):
                sequences[i] = self._pad_sequence(sequence, max_sequence_length)
            samples[key] = torch.stack(sequences, axis=0) # (target shape: (Sequence, Step, Data ...))
            if (key == "h_states" or key == "c_states"):
                samples[key] = samples[key][:, 0] # Select the very first recurrent cell state of a sequence and add it to the samples
        self.num_sequences = len(sequences)
        self.actual_sequence_length = max_sequence_length
        
        # Add remaining data samples
        samples["values"]     = self.batch["values"]
        samples["probs"]      = self.batch["probs"]
        samples["advantages"] = self.batch["advantages"]

        self.samples = {}
        for key, value in samples.items():
            if not key == "h_states" and not key == "c_states":
                value = value.reshape(value.shape[0] * value.shape[1], *value.shape[2:])
            self.samples[key] = value
    def mini_batch_generator(self):
        """
        Overview:
            A recurrent generator that returns a dictionary containing the data of a whole minibatch.
            In comparison to the none-recurrent one, this generator maintains the sequences of the workers' experience trajectories.
        
        Yields:
            {dict} -- Mini batch data for training
        """
        # Determine the number of sequences per mini batch
        num_sequences_per_batch = self.num_sequences // self.n_mini_batches
        num_sequences_per_batch = [num_sequences_per_batch] * self.n_mini_batches # Arrange a list that determines the sequence count for each mini batch
        remainder               = self.num_sequences % self.n_mini_batches
        for i in range(remainder):
            num_sequences_per_batch[i] += 1 # Add the remainder if the sequence count and the number of mini batches do not share a common divider
        # Prepare indices, but only shuffle the sequence indices and not the entire batch to ensure that sequences are maintained as a whole.
        indices          = torch.arange(0, self.num_sequences * self.actual_sequence_length).reshape(self.num_sequences, self.actual_sequence_length)
        sequence_indices = torch.randperm(self.num_sequences)
        # Compose mini batches
        start = 0
        for num_sequences in num_sequences_per_batch:

            end                         = start + num_sequences
            mini_batch_padded_indices   = indices[sequence_indices[start:end]].reshape(-1)
            # Unpadded and flat indices are used to sample unpadded training data
            mini_batch_unpadded_indices = self.flat_sequence_indices[sequence_indices[start:end].tolist()]
            mini_batch_unpadded_indices = [item for sublist in mini_batch_unpadded_indices for item in sublist]
            mini_batch                  = {}


            for key, value in self.samples.items():
                if key == "h_states" or key == "c_states":
                    # Select recurrent cell states of sequence starts
                    mini_batch[key] = value[sequence_indices[start:end]]
                elif key == "probs" or "advantages" in key or key == "values":
                    # Select unpadded data
                    mini_batch[key] = value[mini_batch_unpadded_indices]
                else:
                    # Select padded data
                    mini_batch[key] = value[mini_batch_padded_indices]

            start = end
            yield mini_batch

                

