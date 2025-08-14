import io

# for powerlaw warnings
import warnings
from contextlib import redirect_stdout, redirect_stderr

# remove warnings from powerlaw unless testing
import sys
if not sys.warnoptions:
    warnings.simplefilter("ignore")

import powerlaw
import numpy as np
from tqdm import tqdm

from .constants import *
#from constants import *

supported_distributions = {
    'power_law':                powerlaw.Power_Law,
    'lognormal':                powerlaw.Lognormal,
    'exponential':              powerlaw.Exponential,
    'truncated_power_law':      powerlaw.Truncated_Power_Law,
    'stretched_exponential':    powerlaw.Stretched_Exponential,
    'lognormal_positive':       powerlaw.Lognormal_Positive,
}


import logging
logger = logging.getLogger(WW_NAME) 


class WWFit(object):
    # --------- modified ----------
    #def __init__(self, data, xmin=None, xmax=None, distribution=POWER_LAW):
    def __init__(self, data, xmin=None, xmax=None, total_is=1000, distribution=POWER_LAW):
        assert distribution in [POWER_LAW], distribution
        super(WWFit, self).__init__()

        data = np.asarray(np.sort(data), dtype=np.float64)
        def find_limit(data, x, default):
            if x is None: return default
            return np.argmin(np.abs(data - x))
        self.i_min = find_limit(data, xmin, 0)
        self.i_max = find_limit(data, xmax, len(data) - 1)
        self.xmin = data[self.i_min]
        self.xmax = data[self.i_max]

        self.data = data[self.i_min:self.i_max+1]
        self.N = len(self.data)

        # --------- modified ----------
        # option 1
        #self.total_is = 1000  # total increments
        #self.total_is = min(self.N-1, self.total_is)
        # option 2
        #self.total_is = self.N - 1
        
        assert total_is == None or isinstance(total_is, int), "total_is must be None or an integer!"
        if total_is == None:
            self.total_is = self.N - 1
        else:
            self.total_is = min(self.N-1, total_is)

        self.actual_is = len(range(0, self.N-1, round((self.N-1)/self.total_is))) 
        self.xmins = self.data[range(0, self.N-1, round((self.N-1)/self.total_is))]

        #self.xmins = self.data[:-1]
        self.distribution = distribution

        self.dists = {}
        if self.distribution == POWER_LAW:
            self.fit_power_law()
            self.dists[POWER_LAW] = self

        i = np.argmin(self.Ds)
        self.xmin = self.xmins[i]
        self.alpha = self.alphas[i]
        self.sigma = self.sigmas[i]
        self.D = self.Ds[i]

        # powerlaw package does this, so we replicate it here.
        self.data = self.data[self.data >= self.xmin]

    def __str__(self):
        # --------- modified ----------
        return f"WWFit({self.distribution} xmin: {self.xmin:0.04f}, total_is: {self.total_is} \
                alpha: {self.alpha:0.04f}, sigma: {self.sigma:0.04f}, data: {len(self.data)})"

    def fit_power_law(self):
        log_data    = np.log(self.data, dtype=np.float64)

        # --------- modified ----------
        #self.alphas = np.zeros(self.N-1, dtype=np.float64)
        #self.Ds     = np.ones(self.N-1, dtype=np.float64)
        self.alphas = np.zeros(self.actual_is, dtype=np.float64)
        self.Ds     = np.ones(self.actual_is, dtype=np.float64)  # KS distance        

        # --------- delete ----------
        """
        print(f"self.total_is: {self.total_is}")
        print(f"self.actual_is: {self.actual_is}")
        print(f"self.data: {len(self.data)}")

        print(f"max i: {max(range(0, self.N-1, round((self.N-1)/self.total_is)))}")
        print(f"min i: {min(range(0, self.N-1, round((self.N-1)/self.total_is)))}")

        print(f"self.alphas: {len(self.alphas)}")
        print(f"self.Ds: {len(self.Ds)}")
        print(f"len: {len(range(0, self.N-1, round((self.N-1)/self.total_is)))}")
        """
        # ---------------------------
        
        # original iterates through the entire sequence
        #for i, xmin in enumerate(self.data[:-1]):

        # --------- modified ----------
        # for a more computationally efficient method
        #for idx, i in enumerate(range(0, self.N-1, round((self.N-1)/self.total_is))):
        for idx, i in tqdm(enumerate(range(0, self.N-1, round((self.N-1)/self.total_is)))):
            xmin = self.data[i]
            n = float(self.N - i)
            alpha = 1 + n / (np.sum(log_data[i:]) - n * log_data[i])
            self.alphas[idx] = alpha
            if alpha > 1:
                self.Ds[idx] = np.max(np.abs(
                    1 - (self.data[i:] / xmin) ** (-alpha + 1) -    # Theoretical CDF\
                    np.arange(n) / n                                # Actual CDF
                ))

        # --------- modified ----------
        #self.sigmas = (self.alphas - 1) / np.sqrt(self.N - np.arange(self.N-1))
        self.sigmas = (self.alphas - 1) / np.sqrt(self.N - np.array(range(0, self.N-1, round((self.N-1)/self.total_is))))

    def __getattr__(self, item):
        """ Needed for replicating the behavior of the powerlaw.Fit class"""
        if item in self.dists: return self.dists[item]
        raise AttributeError(item)

    def plot_pdf(self, **kwargs):
        """ Needed for replicating the behavior of the powerlaw.Fit class"""
        return powerlaw.plot_pdf(data=self.data, linear_bins=False, **kwargs)

    def plot_power_law_pdf(self, ax, **kwargs):
        """ Needed for replicating the behavior of the powerlaw.Power_Law class"""
        assert ax is not None

        # Formula taken directly from the powerlaw package.
        bins = np.unique(self.data)
        PDF = (bins ** -self.alpha) * (self.alpha-1) * (self.xmin**(self.alpha-1))

        assert np.min(PDF) > 0

        ax.plot(bins, PDF, **kwargs)
        ax.set_xscale("log")
        ax.set_yscale("log")

    def distribution_compare(self, _dist1, _dist2, **kwargs):
        """
        Mimics the interface of a powerlaw.Fit object by passing through to powerlaw's functional API.
        """
        def get_loglikelihoods(_dist):
            if _dist in ["power_law"]:
                return np.log((self.data ** -self.alpha) * (self.alpha - 1) * self.xmin**(self.alpha-1))
            else:
                if _dist in self.dists: dist = self.dists[_dist]
                else:
                    dist = supported_distributions[_dist](
                        data = self.data, xmin=self.xmin, xmax=None, discrete=False, fit_method="Likelihood",
                        parameter_range=None, parent_Fit=None
                    )
                    self.dists[_dist] = dist

                return dist.loglikelihoods(self.data)

        return powerlaw.loglikelihood_ratio(
            get_loglikelihoods(_dist1),
            get_loglikelihoods(_dist2),
            nested=_dist1 in _dist2, **kwargs
        )

Fit = WWFit

# ---------- modified ----------
# when calling powerlaw methods,
# trap warnings, stdout and stderr
def pl_fit(data=None, xmin=None, xmax=None, total_is=1000, verbose=False, distribution=POWER_LAW, pl_package=WW_POWERLAW_PACKAGE):
    
    if xmax==FORCE:  # 'force'
        xmax=np.max(data)
    
    if pl_package==WW_POWERLAW_PACKAGE and distribution==POWER_LAW:
        logger.info("PL FIT running NEW power law method")
        # --------- modified ----------
        return WWFit(data, xmin=xmin, xmax=xmax, total_is=total_is, distribution=distribution)
        
    else:
        
        logger.info(f"PL FIT running OLD power law method with  xmax={xmax}")
        f = io.StringIO()
        with redirect_stdout(f), redirect_stderr(f), warnings.catch_warnings():
            warnings.simplefilter(action='ignore', category=RuntimeWarning)
            return powerlaw.Fit(data, xmin=xmin, xmax=xmax, total_is=total_is, verbose=verbose, distribution=distribution,
                                xmin_distribution=distribution)
            
            
def pl_compare(fit, dist):
    f = io.StringIO()
    with redirect_stdout(f), redirect_stderr(f), warnings.catch_warnings():
        warnings.simplefilter(action='ignore', category=RuntimeWarning)
        return fit.distribution_compare(dist, TRUNCATED_POWER_LAW, normalized_ratio=True)


# -------------------- taken from weightwatcher.py --------------------

# from .constants import *
from .RMT_Util import *
#from RMT_Util import *
def fit_powerlaw(evals, xmin=None, xmax=None, total_is=1000,
                 plot=True, layer_name="", layer_id=0, plot_id=0, \
                 sample=False, sample_size=None,  savedir=DEF_SAVE_DIR, savefig=True, \
                 thresh=EVALS_THRESH,\
                 fix_fingers=False, finger_thresh=DEFAULT_FINGER_THRESH, xmin_max=None, max_fingers=DEFAULT_MAX_FINGERS, \
                 fit_type=POWER_LAW, pl_package=WW_POWERLAW_PACKAGE):
    """Fit eigenvalues to powerlaw or truncated_power_law
    
        if xmin is 
            'auto' or None, , automatically set this with powerlaw method
            'peak' , try to set by finding the peak of the ESD on a log scale
        
        if xmax is 'auto' or None, xmax = np.max(evals)
        
        svd_method = ACCURATE_SVD (to add TRUNCATED_SVD with some cutoff)
        thresh is a threshold on the evals, to be used for very large matrices with lots of zeros
        
                    
        """
        
    status = None
    
    # defaults for failed status
    alpha = -1
    Lambda = -1
    D = -1
    sigma = -1
    xmin = -1  # not set / error
    num_pl_spikes = -1
    best_fit = UNKNOWN
    fit = None
    num_fingers = 0
    
    raw_fit = None # only for fix fingers
    
    fit_entropy = -1
    
    # check    
    num_evals = len(evals)
    logger.debug("fitting {} on {} eigenvalues".format(fit_type, num_evals))

    # if num_evals < MIN_NUM_EVALS:  # 10 , really 50
    #     logger.warning("not enough eigenvalues, stopping")
    #     status = FAILED
    #     return alpha, Lambda, xmin, xmax, D, sigma, num_pl_spikes, best_fit, status
                        
    # if Power law, Lambda=-1 
    distribution = POWER_LAW
    if fit_type==TRUNCATED_POWER_LAW:
        distribution = 'truncated_power_law'
        
    
    # TODO: replace this with a robust sampler / estimator
    # requires a lot of refactoring below
    if sample and  sample_size is None:
        logger.info("setting sample size to default MAX_NUM_EVALS={}".format(MAX_NUM_EVALS))
        sample_size = MAX_NUM_EVALS
        
    if sample and num_evals > sample_size:
        logger.warning("samping not implemented in production yet")
        logger.info("chosing {} eigenvalues from {} ".format(sample_size, len(evals)))
        evals = np.random.choice(evals, size=sample_size)
                
    if xmax == XMAX_FORCE:
        logger.info("forcing xmax, alpha may be over-estimated")
        xmax = np.max(evals)
    elif isinstance(xmax, int):
        xmax = np.abs(xmax)
        if xmax > len(evals)/2:
            logger.warning("xmax is too large, stopping")
            status = FAILED
        else:
            logger.info(f"clipping off {xmax} top eigenvalues")
            evals = evals[:-xmax] 
            xmax = None
    else:
        logger.debug("xmax not set, fast PL method in place")
        xmax = None
                    
    if fix_fingers==XMIN_PEAK:  # 'xmin_peak'
        #print(f"fix_fingers = {fix_fingers} \n")
        logger.info("fix the fingers by setting xmin to the peak of the ESD")
        try:
            nz_evals = evals[evals > thresh]
            num_bins = 100  # np.min([100, len(nz_evals)])
            h = np.histogram(np.log10(nz_evals), bins=num_bins)
            ih = np.argmax(h[0])
            xmin2 = 10 ** h[1][ih]
            if xmin_max is None:
                xmin_max = 1.5 * xmin2 
            elif xmin_max <  0.95 * xmin2:
                logger.fatal("XMIN max is too small, stopping  ")  
                
            if pl_package!=POWERLAW_PACKAGE:
                logger.fatal(f"Only  {POWERLAW_PACKAGE} supports fix_fingers, pl_package mis-specified, {pl_package}")
                
            xmin_range = (np.log10(0.95 * xmin2), xmin_max)
            logger.info(f"using new XMIN RANGE {xmin_range}")
            # ---------- modified ----------
            fit = pl_fit(data=nz_evals, xmin=xmin_range, xmax=xmax, total_is=total_is, verbose=False, distribution=distribution, pl_package=pl_package)  
            status = SUCCESS 
        except ValueError as err:
            logger.warning(str(err))
            status = FAILED
        except Exception as err:
            logger.warning(str(err))
            status = FAILED
            
    elif fix_fingers==CLIP_XMAX:  # 'clip_xmax'
        #print(f"fix_fingers = {fix_fingers} \n")
        logger.info(f"fix the fingers by fitting a clipped power law using pl_package = {pl_package}, xmax={xmax}")
        try:
            nz_evals = evals[evals > thresh]
            if max_fingers is None or max_fingers < 0 or max_fingers < (1/2)*len(evals):
                max_fingers = DEFAULT_MAX_FINGERS
            logger.debug(f"max_fingers = {MAX_FINGERS}")
                
            fit, num_fingers, raw_fit = fit_clipped_powerlaw(nz_evals, xmax=xmax, total_is=total_is, max_fingers=max_fingers, finger_thresh=finger_thresh, \
                                                    logger=logger, plot=plot,  pl_package=pl_package)  
            status = SUCCESS 
        except ValueError as err:
            logger.warning(str(err))
            status = FAILED
        except Exception as err:
            logger.warning(str(err))
            status = FAILED

    # ---------- DEFAULT OPTION ----------
    # fix_fingers=False, xmin=None, xmax=None (default values)     
    elif xmin is None or xmin == -1: 
        #print(f"fix_fingers = {fix_fingers}, xmin = {xmin} \n")
        logger.info(f"Running powerlaw.Fit no xmin, xmax={xmax}. distribution={distribution} pl_package={pl_package}")
        try:
            nz_evals = evals[evals > thresh]
            # ---------- modified ----------
            fit = pl_fit(data=nz_evals, xmax=xmax, total_is=total_is, verbose=False, distribution=distribution, pl_package=pl_package) 
            status = SUCCESS 
        except ValueError as err:
            logger.warning(str(err))
            status = FAILED
        except Exception as err:
            logger.warning(str(err))
            status = FAILED

    else: 
        #print(f"fix_fingers = {fix_fingers} \n")
        #logger.debug("POWERLAW DEFAULT XMIN SET ")
        try:
            # ---------- modified ----------
            fit = pl_fit(data=evals, xmin=xmin, total_is=total_is,  verbose=False, distribution=distribution, pl_package=pl_package)  
            status = SUCCESS 
        except ValueError as err:
            logger.warning(str(err))
            status = FAILED
        except Exception as  err:
            logger.warning(str(err))
            status = FAILED
                
    if fit is None or fit.alpha is None or np.isnan(fit.alpha):
        logger.warning("Power law fit failed")

        status = FAILED
        
    if status == FAILED:
        logger.warning("power law fit failed, will still attempt plots")
        # --------- modified ---------- (added)
        best_fit = None
        Rs_all = None; ps_all = None
    else:
        alpha = fit.alpha 
        D = fit.D
        sigma = fit.sigma
        xmin = fit.xmin
        xmax = fit.xmax
        num_pl_spikes = len(evals[evals>=fit.xmin])
        if fit_type==TRUNCATED_POWER_LAW:
            alpha = fit.truncated_power_law.alpha
            Lambda = fit.truncated_power_law.Lambda
            
        #logger.debug("finding best distribution for fit, TPL or other ?")
        # we stil check againsgt TPL, even if using PL fit
        # all_dists = [TRUNCATED_POWER_LAW, POWER_LAW, LOG_NORMAL]#, EXPONENTIAL]
        # Rs = [0.0]
        # dists = [POWER_LAW]
        # for dist in all_dists[1:]:
        #     R, p = pl_compare(fit, dist)
        #     if R > 0.1 and p > 0.05:
        #         dists.append(dist)
        #         Rs.append(R)
        #         logger.debug("compare dist={} R={:0.3f} p={:0.3f}".format(dist, R, p))
        # best_fit = dists[np.argmax(Rs)]
        
        #fit_entropy = line_entropy(fit.Ds)

        # --------- modified ---------- (added)
        logger.debug("finding best distribution for fit, TPL or other ?")
        #we stil check againsgt TPL, even if using PL fit
        #all_dists = [TRUNCATED_POWER_LAW, POWER_LAW, LOG_NORMAL]#, EXPONENTIAL]
        all_dists = [TRUNCATED_POWER_LAW, POWER_LAW, LOG_NORMAL, EXPONENTIAL]
        #all_dists = [TRUNCATED_POWER_LAW, LOG_NORMAL, EXPONENTIAL]
        # remove the distributed that is already fitted
        #unfitted_dists = [dist for dist in all_dists if dist != fit_type]

        # moved to pretrained_wfit.py func pretrained_ww_plfit()
        """
        Rs = [0.0]
        Rs_all = []
        ps_all = []
        #dists = [POWER_LAW]
        dists = [TRUNCATED_POWER_LAW]
        for dist in all_dists[1:]:
        #for dist in unfitted_dists:
            R, p = pl_compare(fit, dist)  # the dist is always being compare to TRUNCATED_POWER_LAW, check distribution_compare()
            Rs_all.append(R)
            ps_all.append(p)
            if R > 0.1 and p > 0.05:
                dists.append(dist)
                Rs.append(R)
                logger.debug("compare dist={} R={:0.3f} p={:0.3f}".format(dist, R, p))
        best_fit = dists[np.argmax(Rs)]    
        """    

        # --------- delete ----------
        """
        print(f"alphas: {len(fit.alphas)}")
        print(f"Ds: {len(fit.Ds)}")
        print(f"xmins: {len(fit.xmins)}")
        print(f"fit.N: {fit.N}")
        print(f"fit.total_is: {fit.total_is}")
        print(f"fit.actual_is: {fit.actual_is} \n")
        """
        # ---------------------------

    # defaults for failed status
    if status != SUCCESS:
        alpha = -1
        Lambda = -1
        D = -1
        sigma = -1
        xmin = -1  # not set / error
        num_pl_spikes = -1
        best_fit = UNKNOWN    
        fit = None
        num_fingers = 0
        
        raw_fit = None # only for fix fingers
        
        fit_entropy = -1

    if status != SUCCESS:
        #return alpha, Lambda, xmin, xmax, D, sigma, num_pl_spikes, num_fingers, raw_alpha, status, warning, best_fit, Rs_all, ps_all
        #return alpha, Lambda, xmin, None, D, sigma, num_pl_spikes, num_fingers, -1, status, "", best_fit, None, None
        return alpha, Lambda, xmin, None, D, sigma, num_pl_spikes, num_fingers, -1, status, "", fit

    if plot:

        # --------- modified ----------
        #if savefig:
        #    savedir = join(savedir, )
        
        if status==SUCCESS:
            min_evals_to_plot = (xmin/100)
            
            # recover in the future
            """
            fig2 = fit.plot_pdf(color='b', linewidth=0) # invisbile
            fig2 = fit.plot_pdf(color='r', linewidth=2)
            if fit_type==POWER_LAW:
                if pl_package == WW_POWERLAW_PACKAGE:
                    fit.plot_power_law_pdf(color='r', linestyle='--', ax=fig2)
                else:
                    fit.power_law.plot_pdf(color='r', linestyle='--', ax=fig2)
            
            else:
                fit.truncated_power_law.plot_pdf(color='r', linestyle='--', ax=fig2)
            """

        else:
            xmin = -1
            min_evals_to_plot = (0.4*np.max(evals)/100)            

        print(f"Total original evals: {len(evals)}")
        evals_to_plot = evals[evals>min_evals_to_plot]
        print(f"evals_to_plot size: {len(evals_to_plot)}")

        print(fit)
        print(f"xmin: {fit.xmin}")

        # ---------- original version ----------

        # plot_loghist(evals_to_plot, bins=100, xmin=xmin)
        # title = "Log-Log ESD for {}\n".format(layer_name) 
        
        # if status==SUCCESS:
        #     title = title + r"$\alpha=${0:.3f}; ".format(alpha) + \
        #         r'$D_{KS}=$'+"{0:.3f}; ".format(D) + \
        #         r"$\lambda_{min}=$"+"{0:.3f} ".format(xmin) + \
        #         r"$\sigma=$"+"{0:.3f}".format(sigma) + "\n"
        # else:
        #     title = title + " PL FIT FAILED"

        # plt.title(title)
        # plt.legend()
        # if savefig:
        #     # --------- modified ----------
        #     #plt.savefig("ww.layer{}.esd.png".format(layer_id))
        #     #save_fig(plt, "esd", plot_id, savedir)
        #     save_fig(plt, "esd1", plot_id, savedir)
        # plt.show(); plt.clf()
            

        # # plot eigenvalue histogram
        # #num_bins = 100  # np.min([100,len(evals)])
        # num_bins = 1000
        # plt.hist(evals_to_plot, bins=num_bins, density=True)
        # title = "Lin-Lin ESD for {}".format(layer_name) 
        # plt.title(title)
        # plt.axvline(x=fit.xmin, color='red', label=r'$\lambda_{xmin}$')
        # plt.legend()
        # if savefig:
        #     #plt.savefig("ww.layer{}.esd2.png".format(layer_id))
        #     save_fig(plt, "esd2", plot_id, savedir)
        # plt.show(); plt.clf()

        # # plot log eigenvalue histogram
        # nonzero_evals = evals_to_plot[evals_to_plot > 0.0]
        # plt.hist(np.log10(nonzero_evals), bins=100, density=True)
        # title = "Log-Lin ESD for {}".format(layer_name) 
        # plt.title(title)
        # plt.axvline(x=np.log10(fit.xmin), color='red', label=r'$\lambda_{xmin}$')
        # plt.axvline(x=np.log10(fit.xmax), color='orange',  label=r'$\lambda_{xmax}$')
        # plt.legend()
        # if savefig:
        #     #plt.savefig("ww.layer{}.esd3.png".format(layer_id))
        #     save_fig(plt, "esd3", plot_id, savedir)
        # plt.show(); plt.clf()

        # # plot xmins vs D
        
        # plt.plot(fit.xmins, fit.Ds, label=r'$D_{KS}$')
        # plt.axvline(x=fit.xmin, color='red', label=r'$\lambda_{xmin}$')
        # #plt.plot(fit.xmins, fit.sigmas / fit.alphas, label=r'$\sigma /\alpha$', linestyle='--')
        # plt.xlabel(r'$x_{min}$')
        # plt.ylabel(r'$D_{KS}$')
        # title = r'$D_{KS}$'+ ' vs.' + r'$x_{min},\;\lambda_{xmin}=$'
        # plt.title(title+"{:0.3}".format(fit.xmin))
        # plt.legend()

        # #ax = plt.gca().twinx()
        # #ax.plot(fit.xmins, fit.alphas, label=r'$\alpha(xmin)$', color='g')
        # #ax.set_ylabel(r'$\alpha$')
        # #ax.legend()
        
        # if savefig:
        #     save_fig(plt, "esd4", plot_id, savedir)
        #     #plt.savefig("ww.layer{}.esd4.png".format(layer_id))
        # plt.show(); plt.clf() 
        
        
        # plt.plot(fit.xmins, fit.alphas, label=r'$\alpha(xmin)$')
        # plt.axvline(x=fit.xmin, color='red', label=r'$\lambda_{xmin}$')
        # plt.xlabel(r'$x_{min}$')
        # plt.ylabel(r'$\alpha$')
        # title = r'$\alpha$' + ' vs.' + r'$x_{min},\;\lambda_{xmin}=$'
        # plt.title(title+"{:0.3}".format(fit.xmin))
        # plt.legend()
        # if savefig:
        #     save_fig(plt, "esd5", plot_id, savedir)
        #     #plt.savefig("ww.layer{}.esd5.png".format(layer_id))            
        # plt.show(); plt.clf() 

        # ----------------------------------------                                

        # ---------- modified version ----------

        fig, axs = plt.subplots(2, 3, figsize=(17, 8))  

        num_bins = 500
        counts, binEdges = ax_plot_loghist(axs[0,0], evals_to_plot, bins=num_bins, xmin=xmin)
        title = "Log-Log ESD for {}\n".format(layer_name) 
        
        if status==SUCCESS:
            title = title + r"$\alpha=${0:.3f}; ".format(alpha) + \
                r'$D_{KS}=$'+"{0:.3f}; ".format(D) + \
                r"$x_{min}=$"+"{0:.3f} ".format(xmin) + \
                r"$\sigma=$"+"{0:.3f}".format(sigma) + \
                f" fitted: {len(evals[evals>=xmin])}, total: {num_evals}" + "\n"
        else:
            title = title + " PL FIT FAILED"
        # eye guide
        binEdges = binEdges[1:]
        xc = binEdges[binEdges>=xmin].min()
        yc = counts[binEdges>=xmin].max()
        #b = yc + alpha * xc
        b = (np.log(yc) + alpha * np.log(xc))/np.log(10)
        #xs = np.linspace(np.exp(xmin), np.exp(xmax), 500)
        xs = np.linspace(xmin, xmax, 500)
        #ys = -alpha * xs + b
        #axs[0,0].plot(np.exp(xs), np.exp(ys), c='g')
        ys = 10**b * xs**(-alpha)
        axs[0,0].plot(xs, ys, c='g')        

        axs[0,0].set_xscale('log')    
        axs[0,0].set_yscale('log')

        axs[0,0].set_xlim(evals_to_plot.min(), evals_to_plot.max())
        #axs[0,0].set_xlim(0.0001, 1000)
        #axs[0,0].set_ylim(0.0001, 100)

        axs[0,0].set_title(title)
        axs[0,0].legend()        


        # plot eigenvalue histogram
        #num_bins = 100  # np.min([100,len(evals)])
        num_bins = 1000
        axs[0,1].hist(evals_to_plot, bins=num_bins, density=True)
        title = "Lin-Lin tail for {}".format(layer_name) 
        axs[0,1].set_title(title)
        axs[0,1].axvline(x=fit.xmin, color='red', label=r'$x_{min}$')
        axs[0,1].legend()


        # plot log eigenvalue histogram
        nonzero_evals = evals_to_plot[evals_to_plot > 0.0]
        axs[0,2].hist(np.log10(nonzero_evals), bins=100, density=True)
        title = "Log-Lin tail for {}".format(layer_name) 
        axs[0,2].set_title(title)
        axs[0,2].axvline(x=np.log10(fit.xmin), color='red', label=r'$x_{min}$')
        axs[0,2].axvline(x=np.log10(fit.xmax), color='orange',  label=r'$\lambda_{xmax}$')
        axs[0,2].legend()

        # plot xmins vs D
        
        axs[1,0].plot(fit.xmins, fit.Ds, label=r'$D_{KS}$')
        axs[1,0].axvline(x=fit.xmin, color='red', label=r'$x_{min}$')
        #axs[1,0].plot(fit.xmins, fit.sigmas / fit.alphas, label=r'$\sigma /\alpha$', linestyle='--')
        axs[1,0].set_xlabel(r'$x_{min}$')
        axs[1,0].set_ylabel(r'$D_{KS}$')
        title = r'$D_{KS}$'+ ' vs.' + r'$x_{min},\;x_{min}=$'
        axs[1,0].set_title(title+"{:0.3}".format(fit.xmin))
        axs[1,0].legend()

        #ax = plt.gca().twinx()
        #ax.plot(fit.xmins, fit.alphas, label=r'$\alpha(xmin)$', color='g')
        #ax.set_ylabel(r'$\alpha$')
        #ax.legend()    
        
        
        axs[1,1].plot(fit.xmins, fit.alphas, label=r'$\alpha(xmin)$')
        axs[1,1].axvline(x=fit.xmin, color='red', label=r'$x_{xmin}$')
        axs[1,1].set_xlabel(r'$x_{min}$')
        axs[1,1].set_ylabel(r'$\alpha$')
        title = r'$\alpha$' + ' vs.' + r'$x_{min},\;x_{min}=$'
        axs[1,1].set_title(title+"{:0.3}".format(fit.xmin))
        axs[1,1].legend()
        if savefig:
            save_fig(plt, "plfit", plot_id, savedir)
            #plt.savefig("ww.layer{}.esd5.png".format(layer_id))                   
        plt.show(); plt.clf() 

        # ----------------------------------------   


    raw_alpha = -1
    if raw_fit is not None:
        raw_alpha = raw_fit.alpha
        
    warning = ""   
    if alpha < OVER_TRAINED_THRESH:
        warning = OVER_TRAINED
    elif alpha > UNDER_TRAINED_THRESH:
        warning = UNDER_TRAINED
        
    #return alpha, Lambda, xmin, xmax, D, sigma, num_pl_spikes, num_fingers, raw_alpha, status, warning
    #return alpha, Lambda, xmin, xmax, D, sigma, num_pl_spikes, num_fingers, raw_alpha, status, warning, best_fit
    #return alpha, Lambda, xmin, xmax, D, sigma, num_pl_spikes, num_fingers, raw_alpha, status, warning, best_fit, Rs_all, ps_all
    return alpha, Lambda, xmin, xmax, D, sigma, num_pl_spikes, num_fingers, raw_alpha, status, warning, fit
