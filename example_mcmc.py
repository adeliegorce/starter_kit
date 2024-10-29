import numpy as np 
import sys, glob, os, time
from multiprocessing import Pool, cpu_count
import emcee

ncpu = cpu_count()
outroot = 'test'
init = [1.,2.]
data = np.arange(1,10)
err = 0.1*data

def generate_model(theta):

    model = np.arange(theta[0],theta[1])
    return model

#likelihood
def lnlike(theta):

    #temperature
    model = generate_model(theta)
    chi2s = -0.5*((data-model)**2/err**2)

    return np.sum(chi2s), np.array(chi2s)

def gaussian_prior(param, mu, sigma):

    if (param<=0):
        return -np.inf
    return -0.5 * ((param - mu) ** 2) / (sigma ** 2)

def flat_prior(param, minp, maxp):

    if (param<minp) or (param>maxp):
        return -np.inf
    else:
        return 0.

def lnprior(theta):
    lp = 0.
    for i in range(len(theta)):
        if not np.isfinite(flat_prior(theta[i],priors[i][0],priors[i][1])):
            return -np.inf   
    return lp

def lnprob(theta):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf, 0.
    ll, chi2s = lnlike(theta)
    return lp + ll, chi2s


def main():

    #################################################### MAIN

    ########################################## INI
    ndim = 2
    nwalkers = 2*ndim
    nsteps = 20000

    #backend
    filename = "%s.h5" %outroot
    backend = emcee.backends.HDFBackend(filename)
    if (os.path.isfile(filename)):
        print('Loading backend...')
        pos = None
    else:
        print('Initialising run...')
        backend.reset(nwalkers, ndim)
        #initial position
        pos     = [init + np.random.randn(ndim)*[.01,.01] for i in range(nwalkers)]
        # check if initial positions are compatible with the priors
        test_pos = [lnprior(theta) for theta in pos]
        i=0
        while np.any(np.isinf(test_pos)):
            pos     = [init + np.random.randn(ndim)*[.01,.01] for i in range(nwalkers)]
            test_pos = [lnprior(theta) for theta in pos] 
            i+=1
            if (i>100):
                raise ValueError('Priors and initial walker positions uncompatible.')
        print('Likelihood for input values: %.1f' %lnlike(init)[0])

    #blobs
    dtype = [("chi2_per_bin", float, (data.size,))] 

    #run
    print('Running MCMC....')
    with Pool(ncpu) as pool: 
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob,pool=pool,backend=backend,blobs_dtype=dtype)
        t0 = time.time()
        sampler.run_mcmc(pos, nsteps)#,progress=True)
        t1 = time.time()
        print('Done in %.1fhrs' %((t1-t0)/60./60.))

    # sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob,backend=backend,blobs_dtype=dtype)
    # sampler.run_mcmc(pos, nsteps,progress=True)

    print("Mean acceptance fraction: {0:.3f}".format(np.mean(sampler.acceptance_fraction)))

if __name__ == '__main__':
    # freeze_support()
    main()


