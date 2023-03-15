get_ipython().run_line_magic('run', 'MyNoteBook.py')
import Functions
from sklearn.decomposition import PCA
import pandas as pd
import gzip
with gzip.open('cleaned_data.zip','rb') as data:
    data = pd.read_csv(data,index_col=[0,1])
train_uuid = Functions.get_cross_validation('train',0)
train_uuid_list = []
for uuid in train_uuid:
    if uuid in data.groupby('uuid').count().index:
        train_uuid_list.append(uuid)
train_data = data.loc[train_uuid_list][:]

main_label_list = ['LYING_DOWN','SITTING','FIX_walking','FIX_running','BICYCLING','OR_standing']
main_label_dict = {'LYING_DOWN':0,'SITTING':1,'FIX_walking':2,'FIX_running':3,'BICYCLING':4,'OR_standing':5,'Other':6}
main_label_dict['BICYCLING']
new_label_data = pd.DataFrame()
main_label_list = ['LYING_DOWN','SITTING','FIX_walking','FIX_running','BICYCLING','OR_standing']
main_label_dict = {'LYING_DOWN':0,'SITTING':1,'FIX_walking':2,'FIX_running':3,'BICYCLING':4,'OR_standing':5,'Other':6}


for uuid in data.groupby('uuid').count().index:
    X,Y,M,timestamps,feature_names,label_names = Functions.read_user_data(uuid)
    label = pd.DataFrame(data=Y,columns=label_names)
    label = label[main_label_list]
    new_label = []
    for i in label.index:
        if label.loc[i,:].values.any() == False:
            new_label.append(main_label_dict['Other'])
        else:
            for j in main_label_list:
                if label.loc[i,j] == True:
                    new_label.append(main_label_dict[j])
    muti_index = pd.MultiIndex.from_product([[uuid], X.index], names=['uuid','timestamps'])
    new_label = pd.DataFrame(data = new_label, index = muti_index,columns = ['Status'])
    new_label_data = pd.concat([new_label_data,new_label],axis=0,ignore_index=False)

new_label_data
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
with gzip.open('cleaned_data.zip','rb') as file:
    feature_data = pd.read_csv(file,index_col=[0,1])
# split data
X_train, X_test, Y_train, Y_test = train_test_split(feature_data,new_label_data,test_size= 0.2, random_state = 6)

# fit model 
model = XGBClassifier()
model.fit(X_train, Y_train)
print(model)
# make predictions for test data
Y_pred = model.predict(X_test)
predictions = [round(value) for value in Y_pred]
# evaluate predictions
accuracy = accuracy_score(Y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
import numpy as np

class HMM:
    """
    Order 1 Hidden Markov Model
    Attributes
    ----------
    A : numpy.ndarray
        State transition probability matrix
    B: numpy.ndarray
        Output emission probability matrix with shape(N, number of output types)
    pi: numpy.ndarray
        Initial state probablity vector
    Common Variables
    ----------------
    obs_seq : list of int
        list of observations (represented as ints corresponding to output
        indexes in B) in order of appearance
    T : int
        number of observations in an observation sequence
    N : int
        number of states
    """

    def __init__(self, A, B, pi):
        self.A = A
        self.B = B
        self.pi = pi

    def _forward(self, obs_seq):
        N = self.A.shape[0]
        T = len(obs_seq)

        F = np.zeros((N,T))
        F[:,0] = self.pi * self.B[:, obs_seq[0]]

        for t in range(1, T):
            for n in range(N):
                F[n,t] = np.dot(F[:,t-1], (self.A[:,n])) * self.B[n, obs_seq[t]]

        return F

    def _backward(self, obs_seq):
        N = self.A.shape[0]
        T = len(obs_seq)

        X = np.zeros((N,T))
        X[:,-1:] = 1

        for t in reversed(range(T-1)):
            for n in range(N):
                X[n,t] = np.sum(X[:,t+1] * self.A[n,:] * self.B[:, obs_seq[t+1]])

        return X

    def observation_prob(self, obs_seq):
        """ P( entire observation sequence | A, B, pi ) """
        return np.sum(self._forward(obs_seq)[:,-1])

    def state_path(self, obs_seq):
        """
        Returns
        -------
        V[last_state, -1] : float
            Probability of the optimal state path
        path : list(int)
            Optimal state path for the observation sequence
        """
        V, prev = self.viterbi(obs_seq)

        # Build state path with greatest probability
        last_state = np.argmax(V[:,-1])
        path = list(self.build_viterbi_path(prev, last_state))

        return V[last_state,-1], reversed(path)

    def viterbi(self, obs_seq):
        """
        Returns
        -------
        V : numpy.ndarray
            V [s][t] = Maximum probability of an observation sequence ending
                       at time 't' with final state 's'
        prev : numpy.ndarray
            Contains a pointer to the previous state at t-1 that maximizes
            V[state][t]
        """
        N = self.A.shape[0]
        T = len(obs_seq)
        prev = np.zeros((T - 1, N), dtype=int)

        # DP matrix containing max likelihood of state at a given time
        V = np.zeros((N, T))
        V[:,0] = self.pi * self.B[:,obs_seq[0]]

        for t in range(1, T):
            for n in range(N):
                seq_probs = V[:,t-1] * self.A[:,n] * self.B[n, obs_seq[t]]
                prev[t-1,n] = np.argmax(seq_probs)
                V[n,t] = np.max(seq_probs)

        return V, prev

    def build_viterbi_path(self, prev, last_state):
        """Returns a state path ending in last_state in reverse order."""
        T = len(prev)
        yield(last_state)
        for i in range(T-1, -1, -1):
            yield(prev[i, last_state])
            last_state = prev[i, last_state]

    def simulate(self, T):

        def draw_from(probs):
            return np.where(np.random.multinomial(1,probs) == 1)[0][0]

        observations = np.zeros(T, dtype=int)
        states = np.zeros(T, dtype=int)
        states[0] = draw_from(self.pi)
        observations[0] = draw_from(self.B[states[0],:])
        for t in range(1, T):
            states[t] = draw_from(self.A[states[t-1],:])
            observations[t] = draw_from(self.B[states[t],:])
        return observations,states

    def baum_welch_train(self, observations, criterion=0.05):
        n_states = self.A.shape[0]
        n_samples = len(observations)

        done = False
        while not done:
            # alpha_t(i) = P(O_1 O_2 ... O_t, q_t = S_i | hmm)
            # Initialize alpha
            alpha = self._forward(observations)

            # beta_t(i) = P(O_t+1 O_t+2 ... O_T | q_t = S_i , hmm)
            # Initialize beta
            beta = self._backward(observations)

            xi = np.zeros((n_states,n_states,n_samples-1))
            for t in range(n_samples-1):
                denom = np.dot(np.dot(alpha[:,t].T, self.A) * self.B[:,observations[t+1]].T, beta[:,t+1])
                for i in range(n_states):
                    numer = alpha[i,t] * self.A[i,:] * self.B[:,observations[t+1]].T * beta[:,t+1].T
                    xi[i,:,t] = numer / denom

            # gamma_t(i) = P(q_t = S_i | O, hmm)
            gamma = np.squeeze(np.sum(xi,axis=1))
            # Need final gamma element for new B
            prod =  (alpha[:,n_samples-1] * beta[:,n_samples-1]).reshape((-1,1))
            gamma = np.hstack((gamma,  prod / np.sum(prod))) #append one more to gamma!!!

            newpi = gamma[:,0]
            newA = np.sum(xi,2) / np.sum(gamma[:,:-1],axis=1).reshape((-1,1))
            newB = np.copy(self.B)

            num_levels = self.B.shape[1]
            sumgamma = np.sum(gamma,axis=1)
            for lev in range(num_levels):
                mask = observations == lev
                newB[:,lev] = np.sum(gamma[:,mask],axis=1) / sumgamma

            if np.max(abs(self.pi - newpi)) < criterion and \
                            np.max(abs(self.A - newA)) < criterion and \
                            np.max(abs(self.B - newB)) < criterion:
                done = 1

            self.A[:],self.B[:],self.pi[:] = newA,newB,newpi
