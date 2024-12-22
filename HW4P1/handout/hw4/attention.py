import torch

class Softmax:

    '''
    DO NOT MODIFY! AN INSTANCE IS ALREADY SET IN THE Attention CLASS' CONSTRUCTOR. USE IT!
    Performs softmax along the last dimension
    '''
    def forward(self, Z):

        z_original_shape = Z.shape

        self.N = Z.shape[0]*Z.shape[1]
        self.C = Z.shape[2]
        Z = Z.reshape(self.N, self.C)

        Ones_C = torch.ones((self.C, 1))
        self.A = torch.exp(Z) / (torch.exp(Z) @ Ones_C)

        return self.A.reshape(z_original_shape)

    def backward(self, dLdA):

        dLdA_original_shape = dLdA.shape

        dLdA = dLdA.reshape(self.N, self.C)

        dLdZ = torch.zeros((self.N, self.C))
        
        for i in range(self.N):

            J = torch.zeros((self.C, self.C))

            for m in range(self.C):
                for n in range(self.C):
                    if n == m:
                        J[m, n] = self.A[i][m] * (1 - self.A[i][m])
                    else:
                        J[m, n] = -self.A[i][m] * self.A[i][n]

            dLdZ[i, :] = dLdA[i, :] @ J

        return dLdZ.reshape(dLdA_original_shape)

class Attention:
        
        def __init__(self, weights_keys, weights_queries, weights_values):

            """
            Initialize instance variables. Refer to writeup for notation.
            input_dim = D, key_dim = query_dim = D_k, value_dim = D_v

            Argument(s)
            -----------
            
            weights_keys (torch.tensor, dim = (D X D_k)): weight matrix for keys 
            weights_queries (torch.tensor, dim = (D X D_q)): weight matrix for queries 
            weights_values (torch.tensor, dim = (D X D_v)): weight matrix for values 
            
            """

            # Store the given weights as parameters of the class.
            # D, D_k = weights_keys.shape
            # _, D_q = weights_queries.shape
            # _, D_v = weights_values.shape

            self.W_k    = weights_keys
            self.W_q    = weights_queries
            self.W_v    = weights_values

            # Use this object to perform softmax related operations.
            # It performs softmax over the last dimension which is what you'll need.
            self.softmax = Softmax()
            
        def forward(self, X):

            """
            Compute outputs of the self-attention layer.
            Stores keys, queries, values, raw and normalized attention weights.
            Refer to writeup for notation.
            batch_size = B, seq_len = T, input_dim = D, value_dim = D_v

            Note that input to this method is a batch not a single sequence, so doing a transpose using .T can yield unexpected results.
            You should permute only the required axes.

            Input
            -----
            X (torch.tensor, dim = (B, T, D)): Input batch

            Return
            ------
            X_new (torch.tensor, dim = (B, T, D_v)): Output batch

            """

            self.X = X
            (B, T, D) = self.X.shape
        
            # Compute the values of Key, Query and Value
            # x is (B, T, D)
            # W_q is (D, D_k), key and query are same dim D_k and value could be
            # of different dim D_v

            self.Q = self.X @ self.W_q # (B, T, D_k)
            self.K = self.X @ self.W_k # (B, T, D_k)
            self.V = self.X @ self.W_v # (B, T, D_v)

            # Calculate unormalized Attention Scores (logits)

            self.A_w    = torch.bmm(self.Q,  self.K.transpose(1,2)) # (B, T, T) = (B, T, D_k) dot (B, D_k, T)
       

            # Create additive causal attention mask and apply mask
            # Hint: Look into torch.tril/torch.triu and account for batch dimension

            attn_mask    = torch.triu(torch.ones(T, T), diagonal=1).to(self.A_w.device)
 
            # # Calculate/normalize Attention Scores
            self.A_w = self.A_w.masked_fill(attn_mask == 1, float('-inf'))
            self.A_w = self.A_w / torch.sqrt(torch.tensor(self.K.shape[-1], dtype=torch.float32))
            self.A_sig = self.softmax.forward(self.A_w)
            # self.A_sig = self.softmax.forward(attn_mask / torch.sqrt(torch.tensor(self.W_k.shape[1], dtype=torch.float32)))

            # Calculate Attention context 
            X_new         = torch.bmm(self.A_sig, self.V) # (B, T, D_v)
            return X_new
            
        def backward(self, dLdXnew):

            """
            Backpropogate derivatives through the self-attention layer.
            Stores derivatives wrt keys, queries, values, and weight matrices.
            Refer to writeup for notation.
            batch_size = B, seq_len = T, input_dim = D, value_dim = D_v

            Note that input to this method is a batch not a single sequence, so doing a transpose using .T can yield unexpected results.
            You should permute only the required axes.

            Input
            -----
            dLdXnew (torch.tensor, dim = (B, T, D_v)): Derivative of the divergence wrt attention layer outputs

            Return
            ------
            dLdX (torch.tensor, dim = (B, T, D)): Derivative of the divergence wrt attention layer inputs

            """

            # Derivatives wrt attention weights (raw and normalized)
            # dLdXnew : (B, T, D_v)
            # (B, T, D_v) dot (B, D_v, T) -> (B, T, T)
            dLdA_sig = torch.bmm(dLdXnew, self.V.transpose(1, 2)) 
            dLdA_w = self.softmax.backward(dLdA_sig) / torch.sqrt(torch.tensor(self.K.shape[-1], dtype=torch.float32))
            # dLdA_w : (B, T, T)

            # Derivatives wrt keys, queries, and value
            # (B, T, T) dot (B, T, D_v) -> (B, T, D_v)
            self.dLdV      = torch.bmm(self.A_sig.transpose(1, 2), dLdXnew)
            self.dLdK      = torch.bmm(dLdA_w.transpose(1, 2), self.Q)
            self.dLdQ      = torch.bmm(dLdA_w, self.K)

            # Dervatives wrt weight matrices
            # Remember that you need to sum the derivatives along the batch dimension.

            # self.dLdWq     = torch.sum(torch.bmm(self.X.transpose(1, 2), self.dLdQ) ,dim=0)
            # self.dLdWv     = torch.sum(self.X.T @ self.dLdV ,dim=0)
            # self.dLdWk     = torch.sum(self.X.T @ self.dLdK ,dim=0)
            self.dLdWq = torch.sum(torch.bmm(self.X.transpose(1, 2), self.dLdQ), dim=0)
            self.dLdWv = torch.sum(torch.bmm(self.X.transpose(1, 2), self.dLdV), dim=0)
            self.dLdWk = torch.sum(torch.bmm(self.X.transpose(1, 2), self.dLdK), dim=0)

            # # Derivative wrt input
            # # (B T D_v) * (D_v D)
            # dLdX = (
            #     torch.bmm(self.dLdQ, self.W_q.unsqueeze(0).transpose(1, 2)) +
            #     torch.bmm(self.dLdK, self.W_k.unsqueeze(0).transpose(1, 2)) +
            #     torch.bmm(self.dLdV, self.W_v.unsqueeze(0).transpose(1, 2))
            # )  # Shape (B, T, D)
            W_q_expanded = self.W_q.unsqueeze(0).expand(self.dLdQ.shape[0], -1, -1)  # Shape (B, D, D_k)
            W_k_expanded = self.W_k.unsqueeze(0).expand(self.dLdK.shape[0], -1, -1)  # Shape (B, D, D_k)
            W_v_expanded = self.W_v.unsqueeze(0).expand(self.dLdV.shape[0], -1, -1)  # Shape (B, D, D_v)

            # Compute the input gradients
            dLdX = (
                torch.bmm(self.dLdQ, W_q_expanded.transpose(1, 2)) +
                torch.bmm(self.dLdK, W_k_expanded.transpose(1, 2)) +
                torch.bmm(self.dLdV, W_v_expanded.transpose(1, 2))
            )
            return dLdX
