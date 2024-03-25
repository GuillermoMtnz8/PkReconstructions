#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Packages to be loaded. Probably there are duplicated or missing ones
import cobaya
import camb
import numpy as np
import sympy
import math
import matplotlib.pyplot as plt
from scipy.special import erf
from scipy.interpolate import CubicSpline
from scipy.interpolate import interp1d
import scipy.integrate as integrate
import time


# # Cosmological parameters #

# In[2]:


#Cosmological constants
c = 2.99792458E5;   HJPAS = 1/(c/100);

#Parameteres that won't be sampled. These parameters will be the same as the ones given to Cobaya as input. Planck-compatible
gamma = 0.545; OmegakJPAS = 0; AsJPAS = 2.09052E-9; nsJPAS = 0.9626; tauJPAS = 0.06; mnuJPAS = 0.0; nmuJPAS = 3.046;

#A set of cosmological parameters outside the fiducial cosmology matching Cobaya's cosmology
hJPAS = 0.674
OmegabJPASh2 = 0.02212
OmegaCDMJPASh2 = 0.1206

#Indirect cosmological parameters outside the fiducial
H0JPAS = hJPAS*100
OmegabJPAS = OmegabJPASh2/hJPAS**2; OmegaCDMJPAS = OmegaCDMJPASh2/hJPAS**2;


# In[3]:


#Fiducial cosmology functions and constants (including FoG parameter sigmap)
OmegamFid = 0.31417

#At z=1.7 (first bin)
EzFid = 2.6210003044889154
XiFid = 3263.0797256936944
DAFid = 1208.54804655322
sigmapFid = 2.725068353464309


# In[4]:


#LSS parameters for JPAS-like simulations
DeltazJPAS = 0.00364236313918151
fsky = 0.2575

def bJPAS(z):
    return 0.53+0.289*(1+z)**2


# In[5]:


#Power law primordial power spectrum. Las k se introducen en unidades de h
def PrimordialPowerLaw(As,ns,k):
    return As*(k/(0.05/hJPAS))**(ns-1)


# In[6]:


def PrimordialPowerLawSinh(As,ns,k):
    return As*(k/(0.05))**(ns-1)


# # k and z binning #

# In[7]:


#Arrays limits and steps.

#K arrays in h units. 
kminKArrayComplete = 0.001;   kmaxKArrayComplete = 2.4900;  pasoKArrayComplete = 0.025;

#k binning, complete and in a reduced scaleset
KArrayComplete = np.exp(np.arange(math.log(kminKArrayComplete), math.log(kmaxKArrayComplete), pasoKArrayComplete) )
KArray = KArrayComplete[range(121,246)]

#k binning on lower and upper limits
KArrayUpper = np.zeros(len(KArray)); KArrayLower = np.zeros(len(KArray));

for i in range(0, len(KArray)-1):
    KArrayUpper[i] = KArray[i] + (KArray[i+1]-KArray[i])/2;   KArrayLower[i] = KArray[i] - (KArray[i+1]-KArray[i])/2;

KArrayUpper[-1] = KArrayUpper[-2];  KArrayLower[-1] = KArrayLower[-2];

#z binning
zmin = 1.7;   zmax = 2.9;   pasoz = 0.2;

#Original one
zaAntes = np.arange(zmin-0.1, zmax+pasoz/2, pasoz)

#Including z=0
zaAdicional = np.array([0])

#Binning including all lower and upper z-bins limits
zaConBines = np.arange(zmin-pasoz/2, zmax+0.01+pasoz/2, pasoz/2)

#z binning with 0 and including z-bin limits
za = np.concatenate((zaAdicional,zaConBines))

#Positions of upper and lower limits of the z-bins in the za array
positions_Upper = [3, 5, 7, 9, 11, 13, 15]
positions_Lower = [1, 3, 5, 7, 9, 11, 13]


# # P(k) data and densities reading #

# In[8]:


# Define a class to read the simulated data (Pk, densities) and the seed specifying the path as input
def read_data(path_to_data):
    data = {}

    Simulated_pk_filename = path_to_data+'SimulatedDataHighZLocalOscillatoryChecked.dat'
    Simulated_densities = path_to_data+'Densities_HighZ.dat'
    Vector_Seed = path_to_data+'SeedVector.dat'

    data['pkz'] = np.zeros((len(zaAntes), len(KArray)))
    data['ndz'] = np.zeros(len(zaAntes))
    data['vs'] = np.zeros(len(KArray))
    data['tk'] = np.zeros(len(KArray))
    #data['Nk'] = np.zeros(len(KArray))
  
    with open(Simulated_pk_filename) as file:
        for i in range(len(KArray)):
            line = file.readline().split()
            data['pkz'][0][i] = float(line[7])
            data['tk'][i] = float(line[2])
            #data['Nk'][i] = float(line[3])
            
    with open(Simulated_densities) as file:
        for i in range(len(zaAntes)):
            line = file.readline().split()
            data['ndz'][i] = float(line[1])

    with open(Vector_Seed) as file:
        for i in range(len(KArray)):
            line = file.readline().split()
            data['vs'][i] = float(line[0])
                  
            
    return data

# Read data is converted in the dictionary 'data'

data = read_data('/gpfs/users/martinezg/Simulated_Data/')
#data = read_data('/Users/guillermo/Desktop/Simulated_Data/')
data.keys()


# # Classes to interface with Cobaya #

# In[9]:


#If previous is OK, now the classes to interface with Cobaya are created.

#A cobaya theory NodesInPrimordialPk and a cobaya external likelihood Pklike classes are created

#Needed packages
from cobaya.theory import Theory
from cobaya.likelihood import Likelihood


# In[10]:


#Class of the theory, with the PPS modification including the nodes

class NodesInPrimordialPk(Theory):

    def initialize(self): #Initialize self with the k-array
        self.ks = KArray

    #It seems that in here we allocate the values of the parameters to be sampled and definme their names
    def calculate(self, state, want_derived=True, **params_values_dict):

        #Variables k1, k2... P1, P2... Allocated here
        
        number_nodes = 4
        number_nodes_red = number_nodes-2

        megacubo = np.zeros(number_nodes)
        megacubo[0] = params_values_dict['x1']

        megacubo[1] = megacubo[0] + (1 - megacubo[0]) * (1 - (1 - params_values_dict['x2']) ** (1 /(number_nodes_red+1-(2-1)) ))
        megacubo[2] = megacubo[1] + (1 - megacubo[1]) * (1 - (1 - params_values_dict['x3']) ** (1 /(number_nodes_red+1-(3-1)) ))
        #megacubo[i] = megacubo[i-1] + (1 - megacubo[i-1]) * (1 - (1 - params_values_dict['x2']) ** (1 /(number_nodes_red+1-(i-1)) ))
        megacubo[3] = params_values_dict['x4']

        nodes_logk = [(np.log(KArray[-1])-np.log(KArray[0]) ) * megacubo[0]  + np.log(KArray[0]), 
                      (np.log(KArray[-1])-np.log(KArray[0]) ) * megacubo[1] + np.log(KArray[0]),
                      (np.log(KArray[-1])-np.log(KArray[0]) ) * megacubo[2] + np.log(KArray[0]),
                      (np.log(KArray[-1])-np.log(KArray[0]) ) * megacubo[3] + np.log(KArray[0])] 
        nodes_logPPS = [params_values_dict['y1'], params_values_dict['y2'],params_values_dict['y3'],params_values_dict['y4']]



        #nodes_k and nodes_PPS are interpolated
        NodesInterpFunc_nodes = interp1d(nodes_logk, nodes_logPPS,
        kind='linear', fill_value='extrapolate')

        
     #We construct a modified PPS(k) is evaluated at our nodes, evaluated at our array
        state['primordial_scalar_pk'] = {'kmin': KArray[0]*hJPAS, 'kmax': KArray[-1]*hJPAS,
                                            'Pk': np.exp(NodesInterpFunc_nodes(np.log(KArray))), 'log_regular': True}
        
        
    #To be able to evaluate the PPS?
    def get_primordial_scalar_pk(self):
        return self.current_state['primordial_scalar_pk']
        
    #Function that returns the nodes parameters values
    def get_can_support_params(self):
        return ['x1', 'x2', 'x3', 'x4', 'y1', 'y2','y3', 'y4']



# In[11]:


#Class incorporating the monopole and the likelihood. 


class Pklike(Likelihood): #Class is defined

    #global start, end

    #start = time.time()


    def initialize(self):  

        #Path in wich the data are. We call read_data with this path.
        self.data = read_data('/gpfs/users/martinezg/Simulated_Data/')
        #self.data = read_data('/Users/guillermo/Desktop/Simulated_Data/')

        #Grid of K
        self.ks = KArray
        
        #Grid of z to be employed
        self.z_win = za

    
    def get_requirements(self): #¿Por qué es necesario tener requisitos? ¿No puedo usar las funciones sin esto?
                                #¿Interpolator en extended o Complete?
        
        return {'omegam': None,                
                'Pk_interpolator': {'z': self.z_win, 'k_max': 10, 'nonlinear': False, 'vars_pairs': ([['delta_tot', 'delta_tot']])},
                'comoving_radial_distance': {'z': self.z_win},
                'angular_diameter_distance': {'z': self.z_win},
                'Hubble': {'z': self.z_win, 'units': 'km/s/Mpc'},
                'sigma8_z': {'z': self.z_win}, 'fsigma8': {'z': self.z_win},
                 #'fsigma8': {'z': self.z_win, 'units': None},
                'CAMBdata': None}
 
    #Definition of the monopole. It will return:
        #The monopole evaluated at z=1.7 and at the array of k
        #The covariance evaluated at z=1.7 and at the array of k
    
    def monopole(self, **params_dic):
        
        global start
        
        start = time.time()
        
        #start = time.time()
        
        #Options for print with enough decimal precision
        np.set_printoptions(precision=24, suppress=True)

        #CAMB results
        resultsCobaya = self.provider.get_CAMBdata() 
        
        #This is the primordial power spectrum P(k), evaluatedd at KArray with its minimum and maximum limits.
        primordialCobaya = self.provider.get_primordial_scalar_pk()
        
        #Construction of Pmatter(k) from external transfer function T(k) and Cobaya's primordial P(k)

        pkCobaya = self.provider.get_Pk_interpolator(('delta_tot', 'delta_tot'), nonlinear=False) 
        
        # All functions and variables to compute the Kaiser model. It reads the cosmology from info (below)
        
        #Cosmological parameters from CAMB
        Omegam = self.provider.get_param('omegam')  
        
        #Cosmological functions
        Ez = np.sqrt( Omegam*(1+self.z_win)**3+(1-Omegam) ); 
        H = HJPAS * Ez
        f = (Omegam*(1+self.z_win)**3*1/(Ez**2))**gamma
        Xi = self.provider.get_comoving_radial_distance(self.z_win)*hJPAS; #CAMB is called here
        DA = Xi/(1+self.z_win);

    
        #A and R parameters withouth D(z) (thus calculating only Pm(1.7))
        A = bJPAS(za)
        R = f
        
        # Photometric factor
        sigmar = DeltazJPAS*(1+self.z_win)/H

        # Fingers of God effect at z = 1.7 in the fiducial
        def FFog(mu,k):
            return 1/(1+(f[2]*k*mu*sigmapFid)**2)

        # AP effect
        FactorAP = DAFid**2*EzFid/( DA[2]**2*EzFid )

        def Q(mu):
            return ((Ez[2]**2*Xi[2]**2*mu**2-EzFid**2*XiFid**2*(mu**2-1))**0.5/(EzFid*Xi[2]))
 
        def muObs(mu):
            return mu*Ez[2]/(EzFid*Q(mu))
           
        def kObs(mu,k):
            return Q(mu)*k 


        #Galaxy Power spectrum (mu,k) with AP and FoG 
        def Pg(mu,k):
            #return FactorAP*FFog(muObs(mu),kObs(mu,k))*(A[2]+R[2]*muObs(mu)**2)**2 * (   (    (2 * np.pi**2 * k*hJPAS * TKInterpolationFunction(k)**2) * PrimordialPowerLawSinh(AsJPAS,nsJPAS,k*hJPAS)    )   ) *np.exp(-(k*mu*sigmar[2])**2)
            return FactorAP*FFog(muObs(mu),kObs(mu,k))*(A[2]+R[2]*muObs(mu)**2)**2 * (   (   pkCobaya.P(self.z_win[2],kObs(mu,k)*hJPAS)   )   ) *np.exp(-(k*mu*sigmar[2])**2)
            #return FactorAP*FFog(muObs(mu),kObs(mu,k))*(A[2]+R[2]*muObs(mu)**2)**2 * (   (   (2 * np.pi**2 * k*hJPAS * data['tk']**2) * primordialCobaya['Pk']   )   ) *np.exp(-(k*mu*sigmar[2])**2)

        
        #def MatterFromPrimordial(k):
            #return (2 * np.pi**2 * k*hJPAS * TKInterpolationFunction(k)**2) * PrimordialInterpolationFunction(k)

        
        #Monopole galaxy power spectrum

        
        #Trapezoid rule with 2000 steps for computing the Pmonopole(k)
        def Pgmonopole(k):
            mu = np.arange(-1, 1, 1/1000)
            return 1/2 * integrate.trapz(Pg(mu, k), mu)

        PgmonopoleValores = np.zeros(len(KArray))

        for i in range(len(KArray)):
            PgmonopoleValores[i] = Pgmonopole(KArray[i])

        
        #Covariance

        #Angular distance for z upper and lower bins from CAMB
        XiZaLower = self.provider.get_comoving_radial_distance(self.z_win[positions_Lower])*hJPAS
        XiZaUpper = self.provider.get_comoving_radial_distance(self.z_win[positions_Upper])*hJPAS
        
        #Definition of the volume between redshift bins
        Vol = 4*np.pi*fsky/3*(XiZaUpper**3-XiZaLower**3)
   
        #Number of modes. It depends of ksup and kinf corresponding to kupper y klower
        def Nk(ksup,kinf):
            return Vol[0] * (4*np.pi/3*(ksup**3-kinf**3))/((2*np.pi)**3)

        #Nk evaluated for each of our k-bins. Densities are red from self.data['ndz].
        NkEvaluado = np.zeros(len(self.ks))
        for i in range(0, len(self.ks)):
            NkEvaluado[i] = Nk(KArrayUpper[i],KArrayLower[i])  
        
        #Cov evaluated at our k array
        CovEvaluado = 2 *(PgmonopoleValores + 1/self.data['ndz'][0])**2 / NkEvaluado

        
        #We return the value of the monopole at our k-array and of the Covariance Matrix at the same array
        
        
        
  
        return PgmonopoleValores, CovEvaluado


    #Likelihood calculation
    
    def logp(self, **params_values):  
        
        
        #For allocating the monopole values and cov valued
        PMonopoleBineado = np.zeros((7, len(self.ks)))
        CovBineado = np.zeros((7, len(self.ks)))

        #PMonopoleBineado and CovBineado are equal to the values given by the self.monopole
        PMonopoleBineado[0, :len(self.ks)],CovBineado[0, :len(self.ks)] = self.monopole(**params_values)
        

        #Construction of the likelihood like a Chi^2 with the log of determinant term
        lnlike = 0.0
        for i in range(len(KArray)):
            lnlike = lnlike + ((PMonopoleBineado[0][i] - data['pkz'][0][i])**2 *1/CovBineado[0][i] + np.log(CovBineado[0][i]))


        #print(lnlike, end-start)
        #We return - lnlinke
        return -lnlike/2



    #end = time.time()


# In[12]:


# Input given to Cobaya. These are the cosmological parameters it will interpret. It fixed, like a CAMB with fixed cosmology

# We define the dictionary 'info' including all our information, including the likelihood, theory (with the monopole)
# and the priors

info = {'debug': False,                        #Allow to debug
        'likelihood': {'jpass': Pklike},       #Link likelihood (nombre jpass) with the previously defined class?
        'theory': {'camb': {"external_primordial_pk": True},
                   'my_pk': NodesInPrimordialPk},      #We include the primordial Pk with nodes in the theory class
       'params': {
           
        # Fixed cosmological parameters
        'tau': tauJPAS, 'mnu': mnuJPAS, 'nnu': nmuJPAS,
        'x1': 0.0,
        'x4': 1.0,
        #'ombh2': OmegabJPASh2, 'omch2': OmegaCDMJPASh2, 'H0': H0JPAS,
           
        # Parameters of the nodes, with flat priors
        'y1': {'prior': {'min': -23, 'max': -19}, 'ref': -20, 'latex': 'y_1'},
        'y2': {'prior': {'min': -23, 'max': -19}, 'ref': -20, 'latex': 'y_2'},
        'y3': {'prior': {'min': -23, 'max': -19}, 'ref': -20, 'latex': 'y_3'},
        'y4': {'prior': {'min': -23, 'max': -19}, 'ref': -20, 'latex': 'y_3'},
        'x2': {'prior': {'min': 0.0, 'max': 1.0}, 'ref': 0.5, 'latex': 'x_2'},
        'x3': {'prior': {'min': 0.0, 'max': 1.0}, 'ref': 0.5, 'latex': 'x_2'}, 
        
        # Cosmological parameters to be sampled. Loc y scale are the mean value and the st deviation in a guassian prior
        'ombh2': {'prior': {'dist': 'norm', 'loc': OmegabJPASh2, 'scale': 0.00015}, 'latex': 'Omega_bh^2'},
        'omch2': {'prior': {'dist': 'norm', 'loc': OmegaCDMJPASh2, 'scale': 0.0012}, 'latex': 'Omega_ch^2'},
        'H0': {'prior': {'dist': 'norm', 'loc': H0JPAS, 'scale': 0.54}, 'latex': 'H_0'}},

        "sampler": {"polychord":
                        {"nlive": 175, "precision_criterion": 1e-3}
                    }
           }
        
#Path for output folder and name of the directory. First line: path for Altamira.

info["output"] = "/gpfs/users/martinezg/OutputCobaya/4NodesResults_LocalOscillatory"
#info["output"] = "/Users/guillermo/Desktop/4NodosBuenaPrecision/Results"


# In[13]:


#Cobaya interface

#A model is constructed with the 'info' dictionary
#from cobaya.model import get_model     
#model = get_model(info)          

#x2fixed = (np.log(KArray[54]) - np.log(KArray[0])) / ((np.log(KArray[-1])-np.log(KArray[0]) ))

#x2ordered = (  0 + (1 - 0) * (1 - (1 - x2fixed)
                #** (1 /(2+1-(2-1)) ))  )

#k2ordered = np.exp((np.log(KArray[-1])-np.log(KArray[0]) ) 
                #* x2ordered + np.log(KArray[0]))


#x3fixed = (np.log(KArray[100]) - np.log(KArray[0])) / ((np.log(KArray[-1])-np.log(KArray[0]) ))

#x3ordered =  ((0 + (1 - 0) * (1 - (1 - x2fixed)
                #** (1 /(2+1-(2-1)) ))) + (1 - (0 + (1 - 0) * (1 - (1 - x2fixed) ** (1 /(2+1-(2-1)) ))))
                #* (1 - (1 - x3fixed) ** (1 /(2+1-(3-1)) )))

#k3ordered = np.exp((np.log(KArray[-1])-np.log(KArray[0]) ) 
                #* x3ordered + np.log(KArray[0]))

#Parameters to evaluate the log posterior
#fixed_values = {'tau': tauJPAS, 'mnu': mnuJPAS, 'nnu': nmuJPAS,
                #'x1': 0.0,        
                #'x2': x2fixed,
                #'x3': x3fixed,
                #'x4': 1.0,                                     
#Y axis coordinates
                #'y1': np.log(PrimordialPowerLaw(AsJPAS,nsJPAS,KArray[0])),
                #'y2': np.log(PrimordialPowerLaw(AsJPAS,nsJPAS,k2ordered)),
                #'y3': np.log(PrimordialPowerLaw(AsJPAS,nsJPAS,k3ordered)),
                #'y4': np.log(PrimordialPowerLaw(AsJPAS,nsJPAS,KArray[-1])), 
#Cosmological parameters
    #'H0': H0JPAS, 'ombh2': OmegabJPASh2,
    #'omch2': OmegaCDMJPASh2}

#model.logposterior(fixed_values)         

#camb_results = model.provider.get_CAMBdata();  #Results of CAMB from Cobaya interface

#pk_matter_Cobaya = model.provider.get_Pk_interpolator(('delta_tot', 'delta_tot'), nonlinear=False) #Results of Pm from Cobaya interface


# In[14]:


#Execute in Notebook. 

from cobaya import run
updated_info, sampler = run(info)


# In[ ]:




