import numpy as np

class GreedySearchDecoder(object):

    def __init__(self, symbol_set):
        """
        
        Initialize instance variables

        Argument(s)
        -----------

        symbol_set [list[str]]:
            all the symbols (the vocabulary without blank)

        """

        self.symbol_set = symbol_set


    def decode(self, y_probs):
        """

        Perform greedy search decoding

        Input
        -----

        y_probs [np.array, dim=(len(symbols) + 1, seq_length, batch_size)]
            batch size for part 1 will remain 1, but if you plan to use your
            implementation for part 2 you need to incorporate batch_size

        Returns
        -------

        decoded_path [str]:
            compressed symbol sequence i.e. without blanks or repeated symbols

        path_prob [float]:
            forward probability of the greedy path

        """

        decoded_path = []
        blank = 0
        path_prob = 1

        # TODO:
        # 1. Iterate over sequence length - len(y_probs[0])
        # 2. Iterate over symbol probabilities
        # 3. update path probability, by multiplying with the current max probability
        # 4. Select most probable symbol and append to decoded_path
        # 5. Compress sequence (Inside or outside the loop)

        #return decoded_path, path_prob
        #A greedy decode picks the most likely output
        #y_probs [np.array, dim=(len(symbols) + 1, seq_length, batch_size)]
        decoded_path = []
        blank = 0
        path_prob = 1.0

        _, seq_length, batch_size = y_probs.shape
    
        for t in range(seq_length):
            max_prob_idx = np.argmax(y_probs[:, t, 0])
            max_prob = y_probs[max_prob_idx, t, 0]

            # Update the path probability
            path_prob *= max_prob
        
            if max_prob_idx != blank:
                # if not empty or not repeating, add
                if len(decoded_path) == 0 or decoded_path[-1] != max_prob_idx:
                    decoded_path.append(max_prob_idx)

        decoded_path = ''.join([self.symbol_set[idx - 1] for idx in decoded_path if idx != blank])

        return decoded_path, path_prob




class BeamSearchDecoder(object):

    def __init__(self, symbol_set, beam_width):
        """
        Initialize instance variables

        Argument(s)
        -----------
        symbol_set [list[str]]:
            all the symbols (the vocabulary without blank)

        beam_width [int]:
            beam width for selecting top-k hypotheses for expansion
        """
        self.symbol_set = symbol_set
        self.beam_width = beam_width
        self.blank_symbol = '-' 

    def __init__(self, symbol_set, beam_width):
        """
        Initialize instance variables

        Argument(s)
        -----------
        symbol_set [list[str]]:
            all the symbols (the vocabulary without blank)

        beam_width [int]:
            beam width for selecting top-k hypotheses for expansion
        """
        self.symbol_set = symbol_set
        self.beam_width = beam_width
        self.blank_symbol = '-' 

    def decode(self, y_probs):
        """
        Perform beam search decoding

        Input
        -----

        y_probs [np.array, dim=(len(symbols) + 1, seq_length, batch_size)]
            batch size for part 1 will remain 1, but if you plan to use your
            implementation for part 2 you need to incorporate batch_size

        Returns
        -------

        best_path [str]:
            the symbol sequence with the best path score (forward probability)

        merged_path_scores [dict]:
            all the final merged paths with their scores
        """

        V,T,_ = y_probs.shape

        # add front so i don't need to separately deal with it
        self.symbol_set = ["-"] + self.symbol_set
        best_paths = {'-': 1.0} # initialize

        for t in range(T):
            cur_probs = y_probs[:, t]
            temp_best_paths = dict()
            for path, score in best_paths.items():
                for s, s_prob in enumerate(cur_probs):
                    tail = path[-1]
                    if tail == '-':
                        new_path = path[:-1] + self.symbol_set[s]
                    # if not prev, and not blank in the end
                    elif tail != self.symbol_set[s] and not (t==T-1 and self.symbol_set[s] == "-"):
                        new_path = path + self.symbol_set[s]
                    # repeat, just squeeze together
                    else: 
                        new_path = path

                    # merging prob
                    if new_path in temp_best_paths:
                        temp_best_paths[new_path] += s_prob * score
                    else:
                        temp_best_paths[new_path] = s_prob * score
            # prune
            top_k = sorted(temp_best_paths.items(), key=lambda x: x[1], reverse=True)
            best_paths = dict(top_k[:self.beam_width])
            
        best_path = max(best_paths, key=lambda k: best_paths[k])
        
        merged_path_scores = dict()
        for path, score in temp_best_paths.items():
            if path[-1] == "-":
                merged_path_scores[path[:-1]] = score     
            else:
                merged_path_scores[path] = score
        
        return best_path, merged_path_scores