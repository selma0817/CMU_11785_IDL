import numpy as np


class CTC(object):

    def __init__(self, BLANK=0):
        """
        
        Initialize instance variables

        Argument(s)
        -----------
        
        BLANK (int, optional): blank label index. Default 0.

        """

        # No need to modify
        self.BLANK = BLANK


    def extend_target_with_blank(self, target):
        """Extend target sequence with blank.

        Input
        -----
        target: (np.array, dim = (target_len,))
                target output
        ex: [B,IY,IY,F]

        Return
        ------
        extSymbols: (np.array, dim = (2 * target_len + 1,))
                    extended target sequence with blanks
        ex: [-,B,-,IY,-,IY,-,F,-]

        skipConnect: (np.array, dim = (2 * target_len + 1,))
                    skip connections
        ex: [0,0,0,1,0,0,0,1,0]
        """
        extended_symbols = [self.BLANK]
        for symbol in target:
            extended_symbols.append(symbol)
            extended_symbols.append(self.BLANK)
        #N = len(extended_symbols)
        N = len(extended_symbols)
        skip_connect = [0] * N
        for i in range(2, N):
            if extended_symbols[i] is not self.BLANK and extended_symbols[i-2] != extended_symbols[i]:
                skip_connect[i] = 1
        extended_symbols = np.array(extended_symbols).reshape((N,))
        skip_connect = np.array(skip_connect).reshape((N,))
        return extended_symbols, skip_connect

    def get_forward_probs(self, logits, extended_symbols, skip_connect):
        """Compute forward probabilities.

        Input
        -----
        logits: (np.array, dim = (input_len, len(Symbols)))
                predict (log) probabilities

                To get a certain symbol i's logit as a certain time stamp t:
                p(t,s(i)) = logits[t, qextSymbols[i]]

        extSymbols: (np.array, dim = (2 * target_len + 1,))
                    extended label sequence with blanks

        skipConnect: (np.array, dim = (2 * target_len + 1,))
                    skip connections

        Return
        ------
        alpha: (np.array, dim = (input_len, 2 * target_len + 1))
                forward probabilities

        """

        S, T = len(extended_symbols), len(logits)
        alpha = np.zeros(shape=(T, S))

        # -------------------------------------------->
        # TODO: Intialize alpha[0][0]
        # for each timestep we get a prob of certain symbol. after adding blanks between each symbol
        # it becomes a matrix of size input * (2 * target_size + 1)
        alpha[0][0] = logits[0, extended_symbols[0]]  # First blank
        # TODO: Intialize alpha[0][1]
        alpha[0][1] = logits[0, extended_symbols[1]]  # First target symbol
        # at time 0 it could only be blank for the first symbol

        # TODO: Compute all values for alpha[t][sym] where 1 <= t < T and 1 <= sym < S (assuming zero-indexing)
        # IMP: Remember to check for skipConnect when calculating alpha
        # <---------------------------------------------

        for t in range(1, T):
            alpha[t][0] = alpha[t-1][0]*logits[t,extended_symbols[0]] 
            for s in range(1, S):
                # alpha(t,i) = alpha(t-1,i) + alpha(t-1,i-1)
                alpha[t][s] = alpha[t-1][s]*logits[t,extended_symbols[s]] + alpha[t-1][s-1]*logits[t,extended_symbols[s]]
                # if (i > 2 && Sext(i) != Sext(i-2))
                if s >= 1 and skip_connect[s] == 1:
                    # alpha(t,i) += alpha(t-1,i-2)
                    alpha[t][s] += alpha[t-1][s-2]*logits[t,extended_symbols[s]]
        return alpha
        

    def get_backward_probs(self, logits, extended_symbols, skip_connect):
        """Compute backward probabilities.

        Input
        -----
        logits: (np.array, dim = (input_len, len(symbols)))
                predict (log) probabilities

                To get a certain symbol i's logit as a certain time stamp t:
                p(t,s(i)) = logits[t,extSymbols[i]]

        extSymbols: (np.array, dim = (2 * target_len + 1,))
                    extended label sequence with blanks

        skipConnect: (np.array, dim = (2 * target_len + 1,))
                    skip connections

        Return
        ------
        beta: (np.array, dim = (input_len, 2 * target_len + 1))
                backward probabilities
        
        """

        S, T = len(extended_symbols), len(logits)
        # output table
        beta = np.zeros(shape=(T, S))

        # -------------------------------------------->
        # TODO
        # <--------------------------------------------
        # initialize at t = T 
        # First, at t = T betahat(T,N) = y(T,S(N)) betahat(T,1:N-1) = 0
        # either last symbol or blank
        # beta[T-1][S-1] = 1
        # beta[T-1][S-2] = 1


        beta[T-1][S-1] = 1
        beta[T-1][S-2] = 1

        for t in range(T - 2, -1, -1):
        #   last symbol explicitly
            #betahat(t,N) = s(t,N)*betahat(t+1,N)
            beta[t][S-1] = logits[t + 1, extended_symbols[S-1]] * beta[t + 1][S-1]
            for s in range(S - 2, -1, -1):
            # Update beta[t][sym] using the next symbol and the current symbol
            # betahat(t,i) = s(t,i)*(betahat(t+1,i) + betahat(t+1,i+1))
            # betahat(t,i) = y(t,S(i))*(betahat(t+1,i) + betahat(t+1,i+1))
            # beta_hat(t+1, r) and beta_hat(t+1, r+1)
                beta[t][s] = logits[t + 1, extended_symbols[s + 1]] * beta[t + 1][s + 1]  + logits[t + 1, extended_symbols[s]] * beta[t + 1][s]

            # Include the skip connection if applicable, beta_hat(t+1, r+2)
                if s < S - 3 and skip_connect[s + 2]:
                    beta[t][s] += logits[t + 1, extended_symbols[s + 2]] * beta[t + 1][s + 2]
        return beta
        
        

    def get_posterior_probs(self, alpha, beta):
        """Compute posterior probabilities.

        Input
        -----
        alpha: (np.array, dim = (input_len, 2 * target_len + 1))
                forward probability

        beta: (np.array, dim = (input_len, 2 * target_len + 1))
                backward probability

        Return
        ------
        gamma: (np.array, dim = (input_len, 2 * target_len + 1))
                posterior probability

        """

        [T, S] = alpha.shape
        gamma = np.zeros(shape=(T, S))
        sumgamma = np.zeros((T,))

        # -------------------------------------------->
        # TODO
        # <---------------------------------------------
        gamma = alpha * beta 
        sumgamma = np.sum(alpha * beta, axis=1, keepdims=True)
        return gamma/sumgamma
        

class CTCLoss(object):

    def __init__(self, BLANK=0):
        """

        Initialize instance variables

        Argument(s)
        -----------
        BLANK (int, optional): blank label index. Default 0.
        
        """
        # -------------------------------------------->
        # No need to modify
        super(CTCLoss, self).__init__()

        self.BLANK = BLANK
        self.gammas = []
        self.ctc = CTC()
        # <---------------------------------------------

    def __call__(self, logits, target, input_lengths, target_lengths):

        # No need to modify
        return self.forward(logits, target, input_lengths, target_lengths)

    def forward(self, logits, target, input_lengths, target_lengths):
        """CTC loss forward

        Computes the CTC Loss by calculating forward, backward, and
        posterior proabilites, and then calculating the avg. loss between
        targets and predicted log probabilities

        Input
        -----
        logits [np.array, dim=(seq_length, batch_size, len(symbols)]:
            log probabilities (output sequence) from the RNN/GRU

        target [np.array, dim=(batch_size, padded_target_len)]:
            target sequences

        input_lengths [np.array, dim=(batch_size,)]:
            lengths of the inputs

        target_lengths [np.array, dim=(batch_size,)]:
            lengths of the target

        Returns
        -------
        loss [float]:
            avg. divergence between the posterior probability and the target

        """

        # No need to modify
        self.logits = logits
        self.target = target
        self.input_lengths = input_lengths
        self.target_lengths = target_lengths

        #####  IMP:
        #####  Output losses should be the mean loss over the batch

        # No need to modify
        B, _ = target.shape
        total_loss = np.zeros(B)
        self.extended_symbols = []

        for batch_itr in range(B):
            # -------------------------------------------->
            # Computing CTC Loss for single batch
            # Process:
            #     Truncate the target to target length
            #     Truncate the logits to input length
            #     Extend target sequence with blank
            #     Compute forward probabilities
            #     Compute backward probabilities
            #     Compute posteriors using total probability function
            #     Compute expected divergence for each batch and store it in totalLoss
            #     Take an average over all batches and return final result
            # <---------------------------------------------

            # -------------------------------------------->
            # TODO
            # <---------------------------------------------
            # Truncate the target to target length
            # target [np.array, dim=(batch_size, padded_target_len)]:
            # target sequences
            # target_lengths [np.array, dim=(batch_size,)]:
            # lengths of the target

            target = self.target[batch_itr, :target_lengths[batch_itr]]

            # Truncate the logits to input length
            #  logits [np.array, dim=(seq_length, batch_size, len(symbols)]:
            logits = self.logits[:input_lengths[batch_itr], batch_itr, :]

            # Extend target sequence with blank
            extended_symbols, skip_connect = self.ctc.extend_target_with_blank(target)
            # Compute forward probabilities
            alpha = self.ctc.get_forward_probs(logits, extended_symbols, skip_connect)
            # Compute backward probabilities
            beta = self.ctc.get_backward_probs(logits, extended_symbols, skip_connect)
            #Compute posteriors using total probability function
            gamma = self.ctc.get_posterior_probs(alpha, beta)
            # Compute expected divergence for each batch and store it in totalLoss
            self.gammas.append(gamma)
            # average loss, gamma* log logit
            loss =  -np.sum(gamma*np.log(logits[:,extended_symbols]))
            total_loss[batch_itr] = loss

        total_loss = np.sum(total_loss) / B
        
        return total_loss
        #raise NotImplementedError
        

    def backward(self):
        """
        
        CTC loss backard

        Calculate the gradients w.r.t the parameters and return the derivative 
        w.r.t the inputs, xt and ht, to the cell.

        Input
        -----
        logits [np.array, dim=(seqlength, batch_size, len(Symbols)]:
            log probabilities (output sequence) from the RNN/GRU

        target [np.array, dim=(batch_size, padded_target_len)]:
            target sequences

        input_lengths [np.array, dim=(batch_size,)]:
            lengths of the inputs

        target_lengths [np.array, dim=(batch_size,)]:
            lengths of the target

        Returns
        -------
        dY [np.array, dim=(seq_length, batch_size, len(extended_symbols))]:
            derivative of divergence w.r.t the input symbols at each time

        """

        # No need to modify
        T, B, C = self.logits.shape
        dY = np.full_like(self.logits, 0)

        for batch_itr in range(B):
            # -------------------------------------------->
            # Computing CTC Derivative for single batch
            # Process:
            #     Truncate the target to target length
            #     Truncate the logits to input length
            #     Extend target sequence with blank
            #     Compute derivative of divergence and store them in dY
            # <---------------------------------------------

            # -------------------------------------------->
            # TODO
            # <---------------------------------------------
            
            
            # Truncate the target to target length
            target = self.target[batch_itr, :self.target_lengths[batch_itr]]
            # Truncate the logits to input length
            logits = self.logits[:self.input_lengths[batch_itr], batch_itr, :]

            # Extend target sequence with blank
            extended_symbols, _ = self.ctc.extend_target_with_blank(target)
            # Compute derivative of divergence and store them in dY
            # loss is -sum gamma/yt_s
            # dY dim=(seq_length, batch_size, len(extended_symbols))
            for i in range(self.input_lengths[batch_itr]):
                for j in range(len(extended_symbols)):
                    prob = logits[i, extended_symbols[j]] # yt_s
                    #print(f"j: {j}")
                    #print(f"extend: {extended_symbols[j]}")
                    dY[i, batch_itr, extended_symbols[j]] -= self.gammas[batch_itr][i, j]/ prob
        return dY
       
