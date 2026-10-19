import numpy as np
from scipy.special import ndtri,ndtr
import matplotlib.pyplot as plt
from scipy.stats import chi2,norm

#PARAMS
m_range = (100.0,140.0)
signal_mu,signal_sigma = 126.5,2.0
N_s,N_b = 400,80000
alpha = 0.0218
nbins = 20
threshold = 5
params = {"alpha": alpha, "signal": (signal_mu, signal_sigma)}

bins = np.linspace(*m_range,nbins+1)
centers = .5 * (bins[:-1] + bins[1:])


def generate_background(alpha, m_range,size=1):
    """Generate background events."""
    A,B = m_range

    u = np.random.rand(size)

    vals = np.exp(-alpha*A) - u * (np.exp(-alpha*A) - np.exp(-alpha* B))
    
    return -np.log(vals)/alpha

def generate_signal(m_range,signal_params,size):
    """
    Extract the mass range and the signal params. Then standarizes the mass range into (a,b) with sigma units and generate a random uniform value. 
    Once we have this, use built-in functions from scipy to get the CDF and the CDF inverse (numerically) in order to generate a normally distributed 
    variable from the uniformally one.

    Uses the inverse method, not the acceptance one
    """
    mu, sigma = signal_params

    A,B = m_range
    a,b = (A - mu)/sigma, (B-mu)/sigma # escalem en unitats de sigma
    u = np.random.rand(size)

    prob_limit_baix = ndtr(a) # lower
    prob_limit_alt = ndtr(b) # upper

    prob = (prob_limit_baix + u * (prob_limit_alt - prob_limit_baix)) # fem el canvi de variable trucnada

    inversa = ndtri(prob) # retrobem els vals amb la inversa

    x = mu + sigma * inversa # desfem l'escalament fet abans
    return x

def generate_one_year_of_data(m_range,params,n_b,n_s,bins=bins):


    alpha = params['alpha']
    signal_params = params['signal']

    bkg = generate_background(alpha,m_range,size=n_b)

    signal = generate_signal(m_range,signal_params,size=n_s)

    data = np.concatenate([bkg,signal])

    obs_hist,_ = np.histogram(data,bins=bins)

    return obs_hist

def bkg_expectation(n_b, alpha, bins, m_range): # generem la cdf 
    A, B = m_range
    cdf = (np.exp(-alpha*bins) - np.exp(-alpha*A)) / (np.exp(-alpha*B) - np.exp(-alpha*A))
    return n_b * np.diff(cdf)


def sig_expectation(n_s, mu, sigma, bins): # generem la cdf  
    cdf = ndtr((bins - mu)/sigma)
    return n_s * np.diff(cdf)

def build_T(obs,exp):


    return np.sum((obs-exp)**2/exp)

def conduct_p_test(N_s=N_s,N_b=80000,params=params,bins=bins,m_range=m_range,n_vegades=2000,paint=False):

    chi_2_with,p1,Z1_vals = [],[],[]
    chi_2_gen,p2,Z2_vals = [],[],[]
    dof = nbins - 1 # N fixat
    exp_b = bkg_expectation(N_b,params['alpha'],bins,m_range)
    exp_s = sig_expectation(N_s,mu=params['signal'][0],sigma=params['signal'][1],bins=bins)

    for i in range(n_vegades):

        data = generate_one_year_of_data(m_range,params,N_b,N_s,bins)


        T_1 = build_T(data,exp_b)
        chi_2_with.append(T_1)

        pval_1 = chi2.sf(T_1,dof)
        p1.append(pval_1) # survival function:= 1-cdf()

        Z1 = norm.isf(pval_1) if pval_1 > 0 else np.inf
        Z1_vals.append(Z1)

        # Now we will compare
        without = generate_one_year_of_data(m_range,params,N_b,n_s=0,bins=bins)

        T_2 = build_T(without,exp_b)
        chi_2_gen.append(T_2)

        pval_2 = chi2.sf(T_2,dof)
        p2.append(pval_2) # survival function:= 1-cdf()

        Z2 = norm.isf(pval_2) if pval_2 > 0 else np.inf
        Z2_vals.append(Z2)

    p_vals1 = np.array(p1)
    Z_vals1 = np.array(Z1_vals)


    p_vals2 = np.array(p2)
    Z2_vals = np.array(Z2_vals)


    if paint:

        plt.subplot(1, 2, 1)
        plt.bar(centers, data, width=bins[1]-bins[0],
                color='lightgray', edgecolor='k', label='Simulated Data (Signal + Background)')
        plt.plot(centers, exp_b,'r--', lw=2, label='Expected Background')
        plt.plot(centers, exp_s, 'b-.', lw=2, label='Expected Signal')
        plt.plot(centers, data, 'g-', lw=2, label='Expected Total')

        plt.xlabel(r'$m_{\gamma\gamma}$ [GeV]')
        plt.ylabel('Number of Events')
        plt.title('Signal and Background Comparison')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.hist(p_vals1, bins=30, edgecolor='black', color='skyblue',alpha=.6)
        plt.hist(p_vals2, bins=30, edgecolor='black', color='red',alpha=.6)
        plt.xlabel('p-value (under H₀)')
        plt.ylabel('Number of Experiments')
        plt.title('Distribution of p-values')

        plt.tight_layout()
        plt.show()



    return p_vals1,Z_vals1


def how_many_years(max_year=5,threshold=threshold):

    years = np.linspace(0.1,max_year,50)

    for year in years:
       n_b, n_s = int(round(N_b * year)), int(round(N_s * year))
       p_vals, Z_vals = conduct_p_test(N_b=n_b, N_s=n_s)
       p = np.mean(p_vals)
       z = np.mean(Z_vals)
       print(f"Average p-value for year {year}: {p:.3e}")
       print(f"Average significance for year {year}: {z:.3f} σ")

       if z < threshold:
           continue
       else:
           print(f"Average significance for year {year}: {z:.3f} σ")
           print(f" We got there in {year} years!")
           print(f"We have a p-value {p}")
           return year,p,z

    print(f"Sorry, but we never got there. Trial with {max_year} years. Maybe try with a bit more.")
    print(f"We got at {z}")

    return year,p,z
        




if __name__ == "__main__":


    p,z = conduct_p_test(paint=True) 
    print(f"Average p-value with signal data: {np.mean(p):.3e}")
    print(f"Average significance with signal data: {np.mean(z):.3f} σ")
    

    how_many_years()









