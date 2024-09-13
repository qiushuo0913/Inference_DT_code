#%%
import numpy as np
import math
import matplotlib.pyplot as plt
import torch
from scipy import stats
import random



# ensure randomness
def reset_randomness(random_seed):
    torch.manual_seed(random_seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)


# model MIMO channel
class MIMOReflectPathChannel:
    def __init__(self, Ns, num_tx, num_rx):
        """
        Initialize the MIMO reflect path channel model.
        
        Args:
        Ns (int): Number of scattering paths
        num_tx (int): Number of transmit antennas
        num_rx (int): Number of receive antennas
        """
        self.Ns = Ns
        self.num_tx = num_tx
        self.num_rx = num_rx
        self.generate_channel()

    def generate_channel(self):
        """Generate the channel matrix H based on the reflect path model."""
        
        self.a = np.random.normal(0, 1/np.sqrt(2), self.Ns) + \
                 1j * np.random.normal(0, 1/np.sqrt(2), self.Ns)
        
        # Generate uniform random angles for Ωr and Ωt
        self.phi_r = np.random.uniform(-np.pi, np.pi, self.Ns)
        self.phi_t = np.random.uniform(-np.pi, np.pi, self.Ns)
        
        # Calculate Ωr and Ωt
        self.omega_r = np.cos(self.phi_r)
        self.omega_t = np.cos(self.phi_t)
        
        # Initialize the channel matrix
        self.H = np.zeros((self.num_rx, self.num_tx), dtype=complex)
        
        # normalized distance between antennas
        self.delta = 0.5
        
        # Compute the channel matrix
        for i in range(self.Ns):
            if i ==0:
                e_r = 1/math.sqrt(N_r)*np.exp(-1j * 2*np.pi * np.arange(self.num_rx) * self.delta * 0)
                e_t = 1/math.sqrt(N_t)*np.exp(-1j * 2*np.pi * np.arange(self.num_tx) * self.delta * 0)
            else:
                e_r = 1/math.sqrt(N_r)*np.exp(-1j * 2*np.pi * np.arange(self.num_rx) * self.delta * self.omega_r[i])
                e_t = 1/math.sqrt(N_t)*np.exp(-1j * 2*np.pi * np.arange(self.num_tx) * self.delta * self.omega_t[i])
            self.H += np.outer(e_r, e_t.conj()) * self.a[i]
        
       

    def get_channel_matrix(self):
        """Return the generated channel matrix."""
        return self.H
    

def generate_symbols(num_symbols, modulation):
    if modulation == 'BPSK':
        return np.random.choice([-1, 1], size=(2, num_symbols))
    elif modulation == 'QPSK':
        return (1/np.sqrt(2)) * (np.random.choice([-1, 1], size=(2, num_symbols)) + 
                                 1j * np.random.choice([-1, 1], size=(2, num_symbols)))

  
def add_noise(signal, h, snr_db):
    signal_power = np.mean(np.abs(signal)**2)
    noise_power = signal_power / (10**(snr_db/10))
    y = np.dot(h, signal)
    
    noise = np.sqrt(noise_power/2) * (np.random.randn(*y.shape) + 1j * np.random.randn(*y.shape))
    
    return y + noise

def zf_equalizer(H):
    return np.linalg.pinv(H)

# For an input s=[s1,s2] process
def alamouti_encoding(symbol):
    s1, s2 = symbol[0], symbol[1]
    return np.array([[s1, s2], [-np.conj(s2), np.conj(s1)]])

# design for 2*2 MIMO
def alamouti_combining(H, Y1, Y2):
    # the meaning of H: h21-- tranmitter 2 to receiver 1 (h_{ji})
    # size of H : 2*2
    h11, h21 = H[0,0], H[0,1]
    h12, h22 = H[1,0], H[1,1]
    # y12-- receiver 1 time slot 2
    y11, y12 = Y1[0], Y1[1]
    y21, y22 = Y2[0], Y2[1]
    
    s1_est = (np.conj(h11)*y11 + h21*np.conj(y12) + np.conj(h12)*y21 + h22*np.conj(y22))/(abs(h11)**2+abs(h21)**2+abs(h12)**2+abs(h22)**2)
    s2_est = np.conj((h11*np.conj(y12)- np.conj(h21)*y11 + h12*np.conj(y22)- np.conj(h22)*y21)/(abs(h11)**2+abs(h21)**2+abs(h12)**2+abs(h22)**2))
    return np.array([s1_est, s2_est]).reshape(-1,1)



def simulate_retransmission_times(T, B, Ns, num_tx, num_rx, snr_db_range, modulation, scheme, seed):
    # input:
    # T - the maximum retransmission times
    # B - The length of transmission symbols
    # Ns - number of scatters
    # num_tx - number of tranmist antenna
    # num_rx - number of receive antenna
    # snr_db_range - SNR
    # modulation - modulation
    # scheme - app type
    # seed - index of random seed
    
    # one SNR, one Ns -- generate T channels, N symbols.
    H_list = []
    Symbol_list = []
    
    # control the randomness of every experiment
    reset_randomness(seed)
    
    # generate channel
    for _ in range (T):
            channel = MIMOReflectPathChannel(Ns, num_tx, num_rx)
            H = channel.get_channel_matrix()
            H_list.append(H)
    
    # generate symbols
    for _ in range (B):
            symbols = generate_symbols(1, modulation)
            Symbol_list.append(symbols)
    
    for t in range (T):
        H = H_list[t]   
        # actual one dB
        for _, snr_db in enumerate(snr_db_range):
            errors = 0
            for symbols in Symbol_list:
                if scheme == 'multiplexing':
                    X = symbols
                    
                    Y_noisy = add_noise(X, H, snr_db)
                    W = zf_equalizer(H)
                    X_est = np.dot(W, Y_noisy)
                elif scheme == 'alamouti':
                    X = alamouti_encoding(symbols.flatten())
                    
                    Y_noisy = add_noise(X, H, snr_db)
                    X_est = alamouti_combining(H, Y_noisy[0,:], Y_noisy[1,:])
                
                if modulation == 'BPSK':
                    X_dec = np.sign(np.real(X_est))

                elif modulation == 'QPSK':
                    X_dec = (np.sign(np.real(X_est)) + 1j * np.sign(np.imag(X_est)))*(1/np.sqrt(2))

                errors += np.sum(X_dec != symbols)
            
        if errors == 0:
            t_final = t+1
            break
        if t == T-1:
            # print('Outrage!')
            t_final = T
    
    return t_final


# Simulation parameters

SNR_min = -5 # db
SNR_max = 15 # db
SNR_peak = 5 # db
# change db to value
SNR_min_value = 10**(SNR_min/10)
SNR_max_value = 10**(SNR_max/10)
SNR_peak_value = 10**(SNR_peak/10)

N_s_list = np.arange(1, 11, 1)

N_t = 2
N_r = 2
num_symbols = 1


def generate_db_samples(num_samples, min_SNR, max_SNR, peak_SNR):
    # Calculate mean and standard deviation
    mean = peak_SNR
    std_dev = (max_SNR - min_SNR) / 6  # Assume the range covers 6 standard deviations of the distribution (almost 100%)

    
    a = (min_SNR - mean) / std_dev
    b = (max_SNR - mean) / std_dev

    
    truncated_norm = stats.truncnorm(a, b, loc=mean, scale=std_dev)

    
    samples = truncated_norm.rvs(num_samples)

    return samples.tolist()

# generate 500 different values of SNR
reset_randomness(random_seed=0)
gamma_list = generate_db_samples(num_samples=500, min_SNR = SNR_min_value, max_SNR = SNR_max_value, peak_SNR=SNR_peak_value)



def over_run_test(T, B, modulation, scheme):
    t_final_list = []
    Gamma = []
    Num_s = []
    for i in range(len(gamma_list)):
        gamma = gamma_list[i]
        for j in range(len(N_s_list)):
            seed = i*len(N_s_list)+j
            # Randomly select one value from 10 values of Ns
            reset_randomness(seed)
            index = random.randint(0, len(N_s_list)-1)
            N_s = N_s_list[index]
            snr_db_range = np.arange(gamma, gamma+1, 2)
            Gamma.append(gamma)
            Num_s.append(N_s)
            t_final = simulate_retransmission_times(T, B, N_s, N_t, N_r, snr_db_range, modulation, scheme, seed)
            t_final_list.append(t_final)
            
            print('Modulation: %s Scheme: %s SNR: %f Ns: %d Re-time: %f' % (modulation, scheme, gamma, N_s, t_final))
            if t_final == T:
                print('Outrage!')
        
        print(f"{(i+1)*100/(len(gamma_list)):3.0f}% Completed!")
            
            
            
    return Gamma, Num_s, t_final_list
            

            


