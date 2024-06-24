class StateEmissionPFSA:
    def __init__(self, Q, sigma, I, F, T, E):
        self.Q = Q  # Set of states
        self.sigma = sigma  # Alphabet of symbols
        self.I = I  # Starting probabilities
        self.F = F  # Ending probabilities
        self.T = T  # Transition probabilities
        self.E = E  # Emission probabilities

    def is_valid(self):
        # Check if starting probabilities sum up to 1 for each state
        if sum(self.I.values()) != 1:
            return False

        # Check if F(q) + Σ(T(q, q')) = 1 for each state q
        for state in self.Q:
            total_transition_prob = sum(self.T[state].values())
            if self.F[state] + total_transition_prob != 1:
                return False

        # Check if Σ E(q, x) = 1 for each state q
        for state in self.Q:
            total_emission_prob = sum(self.E[state].values())
            if total_emission_prob != 1:
                return False

        return True
    
    def __str__(self):  
        #return "From str method of Test: a is % s, " "b is % s" % (self.a, self.b)
        return "Start %s," " End %s," " Emit %s," " Transit %s" % (self.I, self.F, self.E, self.T)


corpusN = [
    ["the", "stuff", "will", "sleep"], ["the", "stuff", "like", "the", "stuff", "can", "sleep"], 
    ["the", "stuff", "with", "the", "can", "with", "water", "will", "sleep"], ["the", "can", "with", "the", "stuff", "can", "like"], 
    ["the", "stuff", "like", "the", "can", "will", "water"], ["the", "water", "with", "stuff", "will", "sleep"], 
    ["the", "can", "with", "stuff", "will", "sleep"], ["the", "stuff", "can", "water"]]

Q_m3 = {'q1', 'q2', 'q3'}
sigma_m3 = {'can', 'like', 'sleep', 'stuff', 'the', 'water', 'will', 'with'}
I_m3 = {'q1': 1, 'q2': 0, 'q3':0}
F_m3 = {'q1': 0.1, 'q2': 0.3, 'q3': 0.4}
T_m3 = {
    'q1': {'q1': 0.4, 'q2': 0.3, 'q3': 0.2},
    'q2': {'q1': 0.4, 'q2': 0.1, 'q3': 0.2},
    'q3': {'q1': 0.2, 'q2': 0.1, 'q3': 0.3}
}
E_m3 = {
    'q1': {'can': 0.3, 'like': 0.1, 'sleep': 0.1, 'stuff': 0.1, 'the': 0.1, 'water': 0.1, 'will': 0.1, 'with': 0.1},
    'q2': {'can': 0.1, 'like': 0.1, 'sleep': 0.3, 'stuff': 0.1, 'the': 0.1, 'water': 0.1, 'will': 0.1, 'with': 0.1},
    'q3': {'can': 0.2, 'like': 0.1, 'sleep': 0.1, 'stuff': 0.1, 'the': 0.2, 'water': 0.1, 'will': 0.1, 'with': 0.1}
}

m3 = StateEmissionPFSA(Q_m3, sigma_m3, I_m3, F_m3, T_m3, E_m3)

Q_m4 = {'q1', 'q2', 'q3', 'q4'}
sigma_m4 = {'can', 'like', 'sleep', 'stuff', 'the', 'water', 'will', 'with'}
I_m4 = {'q1': 0, 'q2': 1, 'q3':0, 'q4':0}
F_m4 = {'q1': 0.1, 'q2': 0.3, 'q3': 0.4, 'q4': 0.3}
T_m4 = {
    'q1': {'q1': 0.4, 'q2': 0.3, 'q3': 0.1, 'q4':0.1},
    'q2': {'q1': 0.2, 'q2': 0.1, 'q3': 0.2, 'q4':0.2},
    'q3': {'q1': 0.2, 'q2': 0.1, 'q3': 0.1, 'q4':0.2},
    'q4': {'q1': 0.1, 'q2': 0.2, 'q3': 0.2, 'q4':0.2}
}
E_m4 = {
    'q1': {'can': 0.3, 'like': 0.1, 'sleep': 0.1, 'stuff': 0.1, 'the': 0.1, 'water': 0.1, 'will': 0.1, 'with': 0.1},
    'q2': {'can': 0.1, 'like': 0.1, 'sleep': 0.3, 'stuff': 0.1, 'the': 0.1, 'water': 0.1, 'will': 0.1, 'with': 0.1},
    'q3': {'can': 0.2, 'like': 0.1, 'sleep': 0.1, 'stuff': 0.1, 'the': 0.2, 'water': 0.1, 'will': 0.1, 'with': 0.1},
    'q4': {'can': 0.1, 'like': 0.1, 'sleep': 0.1, 'stuff': 0.3, 'the': 0.1, 'water': 0.1, 'will': 0.1, 'with': 0.1}
}

m4 = StateEmissionPFSA(Q_m4, sigma_m4, I_m4, F_m4, T_m4, E_m4)

Q_m5 = {'q1', 'q2', 'q3', 'q4', 'q5'}
sigma_m5 = {'can', 'like', 'sleep', 'stuff', 'the', 'water', 'will', 'with'}
I_m5 = {'q1': 0, 'q2': 0, 'q3':0, 'q4':0, 'q5': 1}
F_m5 = {'q1': 0.1, 'q2': 0.2, 'q3': 0.3, 'q4': 0.3, 'q5': 0.1}
T_m5 = {
    'q1': {'q1': 0.4, 'q2': 0.2, 'q3': 0.1, 'q4':0.1, 'q5': 0.1},
    'q2': {'q1': 0.2, 'q2': 0.1, 'q3': 0.2, 'q4':0.2, 'q5': 0.1},
    'q3': {'q1': 0.2, 'q2': 0.1, 'q3': 0.1, 'q4':0.2, 'q5': 0.1},
    'q4': {'q1': 0.1, 'q2': 0.2, 'q3': 0.2, 'q4':0.1, 'q5': 0.1},
    'q5': {'q1': 0.1, 'q2': 0.2, 'q3': 0.2, 'q4':0.2, 'q5': 0.2}
}
E_m5 = {
    'q1': {'can': 0.3, 'like': 0.1, 'sleep': 0.1, 'stuff': 0.1, 'the': 0.1, 'water': 0.1, 'will': 0.1, 'with': 0.1},
    'q2': {'can': 0.1, 'like': 0.1, 'sleep': 0.3, 'stuff': 0.1, 'the': 0.1, 'water': 0.1, 'will': 0.1, 'with': 0.1},
    'q3': {'can': 0.2, 'like': 0.1, 'sleep': 0.1, 'stuff': 0.1, 'the': 0.2, 'water': 0.1, 'will': 0.1, 'with': 0.1},
    'q4': {'can': 0.1, 'like': 0.1, 'sleep': 0.1, 'stuff': 0.3, 'the': 0.1, 'water': 0.1, 'will': 0.1, 'with': 0.1},
    'q5': {'can': 0.1, 'like': 0.2, 'sleep': 0.1, 'stuff': 0.2, 'the': 0.1, 'water': 0.1, 'will': 0.1, 'with': 0.1}

}

m5 = StateEmissionPFSA(Q_m5, sigma_m5, I_m5, F_m5, T_m5, E_m5)


# backward function take 3 arguments: a PFSA, a string, and a state.
'''
backward(x1)(q) = E(q, x1)*F(q)
backward(x1...xn)(q) = Σq' [E(q, x1)*T(q,q') x bwd(x2...xn)(q')]
'''
# def backward(pfsa, x, q): 
#     if len(x) == 1:
#         return pfsa.E[q][x[0]] * pfsa.F[q]
#     else:
#         sum_prob = sum(pfsa.E[q][x[0]] * pfsa.T[q][next_q] * backward(pfsa, x[1:], next_q) for next_q in pfsa.Q)
#         return sum_prob
    
def backward_memo(pfsa, x, q, memo={}):
    x_tuple = tuple(x)
    if (x_tuple, q) in memo:
        return memo[(x_tuple, q)]
    if len(x) == 1:
        result = pfsa.E[q][x[0]] * pfsa.F[q]
    else:
        result = sum(pfsa.E[q][x[0]] * pfsa.T[q][next_q] * backward_memo(pfsa, x[1:], next_q, memo) for next_q in pfsa.Q)
        memo[(x_tuple, q)] = result
    return result

'''
forward(ε)(q) = I(q)
forward(x1...xn)(q) = Σq' [fwd(x1...x_n-1)(q') * E(q', xn)*T(q',q)]
'''

# def forward(pfsa, x, q): 
#     if len(x) == 0:
#         return pfsa.I[q]
#     else:
#         sum_prob = sum(forward(pfsa, x[:-1], prev_q) * pfsa.E[prev_q][x[-1]] * pfsa.T[prev_q][q] for prev_q in pfsa.Q)
#         return sum_prob

def forward_memo(pfsa, x, q, memo={}):
    x_tuple = tuple(x) # Convert x to a tuple to make it hashable
    if (x_tuple, q) in memo:
        return memo[(x_tuple, q)]

    if len(x) == 0:
        return pfsa.I[q]
    else:
        sum_prob = sum(forward_memo(pfsa, x[:-1], prev_q, memo) * pfsa.E[prev_q][x[-1]] * pfsa.T[prev_q][q] for prev_q in pfsa.Q)
        
        # Store the computed value in memo for reuse
        memo[(x_tuple, q)] = sum_prob
        
        return sum_prob

'''Pr(S=x1...xn) = Σq∈Q [fwd (x1...xn-1)(q) * E(q,xn)*F(q)'''
def biggest_prefix(pfsa, x): #func for 1b
    sum_prob = sum(forward_memo(pfsa, x[:-1], state) * pfsa.E[state][x[-1]] * pfsa.F[state] for state in pfsa.Q)
    return sum_prob


'''Pr(S=x1...xn) = Σq∈Q [I(q) * bwd (x1...xn)(q)'''
def biggest_suffix(pfsa,x): #func for 1d
    sum_prob = sum(pfsa.I[state] * backward_memo(pfsa, x, state)for state in pfsa.Q)
    return sum_prob

 
'''Pr(Qi = q | S = x1...xn) 
= (forward(x1...xi-1)(q) * backward(xi...xn)(q)) / Pr(S=x1...xn)'''

def state_prob_helper_func(pfsa, x, q, i):
    if i == 1:
        numerator = pfsa.I[q] * backward_memo(pfsa, x, q)
        string_prob = biggest_suffix(pfsa,x)
        result= numerator / string_prob
        return result
    elif i == len(x): # i = n
        if len(x) == 1:
            numerator = forward_memo(pfsa,"", q) * backward_memo(pfsa, x, q)
        else:
            j = len(x) - 1
            numerator = forward_memo(pfsa,x[:j], q) * backward_memo(pfsa, x[j:], q)
        string_prob = biggest_suffix(pfsa,x)
        result= numerator / string_prob
        return result
    else:
        numerator = forward_memo(pfsa,x[:i-1], q) * backward_memo(pfsa, x[i-1:], q)
        string_prob = biggest_suffix(pfsa,x)
        result= numerator / string_prob
        return result

def state_prob(pfsa, x): # I
    parameters = {"Start": {}, "End": {}, "Emit": {s: {} for s in pfsa.Q}} 

    for s in pfsa.Q: # Start values
        result = state_prob_helper_func(pfsa, x, s, 1)
        parameters["Start"] [s] = result
    
    for s in pfsa.Q: # End values
        result = state_prob_helper_func(pfsa, x, s, len(x))
        parameters["End"] [s] = result
    
    for symbol in pfsa.sigma: # Emit values
        for s in pfsa.Q:
            visit_states = [i+1 for i in range(len(x)) if x[i] == symbol] # w/o +1, prints [0, 2, 3] when x = fbffb
            result = 0
            for j in visit_states:
                result += state_prob_helper_func(pfsa, x, s, j) 
            parameters["Emit"][s][symbol] = result

    return parameters



def update_emits_helper(pfsa, list_of_strings):
    new_emit = {s: {} for s in pfsa.Q}
    def compute_numerator_and_denominator(state, symbol):
        numerator = 0
        denominator = 0
        for string in list_of_strings:
            parameters = state_prob(pfsa, string)
            numerator += parameters["Emit"][state][symbol]
            for sym in pfsa.sigma:
                denominator += parameters["Emit"][state][sym]
        return numerator, denominator

    for state in pfsa.Q:
        for symbol in pfsa.sigma:
            numerator, denominator = compute_numerator_and_denominator(state, symbol)
            new_probability = numerator / denominator
            new_emit[state][symbol] = new_probability
    return new_emit

               

def update_start_helper(pfsa, list_of_strings):
    new_start = {}
    def compute_numerator_and_denominator(state):
        numerator = 0
        denominator = 0
        for string in list_of_strings:      
            parameters = state_prob(pfsa, string)      
            numerator += parameters["Start"][state]
            for s in pfsa.Q:
                denominator += parameters["Start"][s]
        return numerator, denominator
    for state in pfsa.Q:
        numerator, denominator = compute_numerator_and_denominator(state)
        new_probability = numerator / denominator
        new_start[state] = new_probability
    return new_start



def state_pair_prob_helper_func(pfsa, x, q, q_prime):
    if len(x) == 1 or len(x) == 0:
        return 0

    else:
        prob = []
        for i in range(len(x)-1):
            if i == 0:
                numerator = forward_memo(pfsa,"", q) * pfsa.E[q][x[i]] * pfsa.T[q][q_prime] * backward_memo(pfsa, x[1:], q_prime)
                string_prob = biggest_suffix(pfsa,x)
                result = numerator / string_prob
                prob.append(result)
                
            else:
                numerator = forward_memo(pfsa,x[:i], q) * pfsa.E[q][x[i]] * pfsa.T[q][q_prime] * backward_memo(pfsa, x[i+1:], q_prime)
                string_prob = biggest_suffix(pfsa,x)
                result = numerator / string_prob
                prob.append(result)

    sum = 0
    for j in range(len(prob)):
        sum += prob[j]  
    return sum

def state_pair_prob(pfsa, x): # part I
    parameters = {"Transit": {s: {} for s in pfsa.Q}} 
    for s in pfsa.Q: # Transmit
        for s_prime in pfsa.Q:
            result = state_pair_prob_helper_func(pfsa, x, s, s_prime)
            parameters["Transit"][s][s_prime] = result
    return parameters


def update_end_helper(pfsa, list_of_strings):
    new_end = {}
    def compute_end_probability(state):
        prob_end_numerator = 0
        prob_end_denominator = 0
        transit = 0
        for string in list_of_strings:
            parameters_end = state_prob(pfsa, string)
            parameters_trans = state_pair_prob(pfsa, string)
            prob_end_numerator += parameters_end["End"][state]

            for next_state in pfsa.Q:
                transit += parameters_trans["Transit"][state][next_state]

        prob_end_denominator = prob_end_numerator + transit
        return prob_end_numerator / prob_end_denominator

    for state in pfsa.Q:
        new_prob_end = compute_end_probability(state)
        new_end[state] = new_prob_end
    return new_end


def update_transit_helper(pfsa, list_of_strings):
    new_transit = {s: {} for s in pfsa.Q}
    def compute_transition_probability(from_state, to_state):
        prob_trans_numerator = 0
        prob_trans_denominator = 0
        transit = 0
        end = 0
        for string in list_of_strings:
            parameters_end = state_prob(pfsa, string)
            parameters_trans = state_pair_prob(pfsa, string)
            prob_trans_numerator += parameters_trans["Transit"][from_state][to_state]
            end += parameters_end["End"][from_state]
            for next_state in pfsa.Q:
                transit += parameters_trans["Transit"][from_state][next_state]
        prob_trans_denominator = end + transit
        return prob_trans_numerator / prob_trans_denominator

    for from_state in pfsa.Q:
        for to_state in pfsa.Q:
            new_prob_trans = compute_transition_probability(from_state, to_state)
            new_transit[from_state][to_state] = new_prob_trans
    return new_transit


def update_pfsa(pfsa, list_of_strings):
    new_start = update_start_helper(pfsa, list_of_strings)
    new_end = update_end_helper(pfsa, list_of_strings)
    new_emit = update_emits_helper(pfsa, list_of_strings)
    new_transit = update_transit_helper(pfsa, list_of_strings)
    new_pfsa = StateEmissionPFSA(pfsa.Q, pfsa.sigma, new_start, new_end, new_transit, new_emit)
    return new_pfsa

def update_pfsa_multiple_times(pfsa, list_of_strings, num_iterations):
    for _ in range(num_iterations):
        pfsa = update_pfsa(pfsa, list_of_strings)
    return pfsa
