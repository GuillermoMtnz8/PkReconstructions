#!/usr/bin/env python
# coding: utf-8

# In[205]:


#Packages to be loaded. Probably there are duplicated or missing ones
import cobaya
import numpy as np
import math
from scipy.special import erf
from scipy.interpolate import CubicSpline
import camb
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


# # Cosmological parameters #

# In[207]:


#Constantes cosmológicas:
c = 2.99792458E5;   HJPAS = 1/(c/100);

#Parámetros que no se van a samplear:
gamma = 0.545; OmegakJPAS = 0; AsJPAS = 2.09052E-9; nsJPAS = 0.9626; 

#Parámetros cosmológicos directos:
hJPASFid = 0.6736; OmegabJPASh2Fid = 0.02237; OmegaCDMJPASh2Fid = 0.1200; 
OmegamJPASFid = 0.3153;

#Parámetros cosmológicos indirectos:
OmegabJPASFid = OmegabJPASh2Fid/hJPASFid**2; OmegaCDMJPASFid = OmegaCDMJPASh2Fid/hJPASFid**2;
OmegaLJPASFid = 1 - OmegamJPASFid;

#Parámetros cosmológicos fuera del fiducial:
hJPAS = hJPASFid + hJPASFid/100;
OmegabJPASh2 = OmegabJPASh2Fid + OmegabJPASh2Fid/100;
OmegaCDMJPASh2 = OmegaCDMJPASh2Fid + OmegaCDMJPASh2Fid/100; 
OmegamJPAS = OmegamJPASFid + OmegamJPASFid/100;

#Parámetros cosmológicos indirectos fuera del fiducial:
OmegabJPAS = OmegabJPASh2/hJPAS**2; OmegaCDMJPAS = OmegaCDMJPASh2/hJPAS**2;
OmegaLJPAS = 1 - OmegamJPAS;


# In[245]:


#Parámetros del Fiducial obtenidos con CAMB:
OmegamFid = 0.31417

#En z = 1.7
EzFid = 2.6210003
XiFid = 3263.07985798
DaFid = 1208.54809555


# # k and z binning #

# In[223]:


#Bineado de k y de z
#Límites y pasos de los arrays. Escalas en unidades de h.
kminKArrayCompleto = 0.001;   kmaxKArrayCompleto = 2.4900;  pasoKArrayCompleto = 0.025;
zmin = 1.7;   zmax = 2.9;   pasoz = 0.2;

#Bines de k, completos y reducidos
KArrayCompleto = np.exp(np.arange(math.log(kminKArrayCompleto), math.log(kmaxKArrayCompleto), pasoKArrayCompleto) )
KArray = KArrayCompleto[range(121,246)]

#Bines de k por arriba y por abajo
KArrayUpper = np.zeros(len(ks)); KArrayLower = np.zeros(len(ks));

for i in range(0, len(ks)-1):
    KArrayUpper[i] = ks[i] + (ks[i+1]-ks[i])/2;   KArrayLower[i] = ks[i] - (ks[i+1]-ks[i])/2;

KArrayUpper[-1] = KArrayUpper[-2];  KArrayLower[-1] = KArrayLower[-2];

#Bines de z:
zaAntes = np.arange(zmin-0.1, zmax+pasoz/2, pasoz)
zaAdicional = np.array([0])
zaConBines = np.arange(zmin-pasoz/2, zmax+0.01+pasoz/2, pasoz/2)
za = np.concatenate((zaAdicional,zaConBines))
positions_Upper = [3, 5, 7, 9, 11, 13, 15]
positions_Lower = [1, 3, 5, 7, 9, 11, 13]



# # P(k) data and densities reading #

# In[210]:


# Define a class to read the simulated data specifying the path as input
def read_data(path_to_data):
    data = {}

    Simulated_pk_filename = path_to_data+'FicticioHighZArrayEnK.dat'
    Simulated_densities = path_to_data+'DensityHighZ.dat'

    data['pkz'] = np.zeros((len(zaAntes), len(KArray)))
    data['ndz'] = np.zeros(len(zaAntes))
  
    with open(Simulated_pk_filename) as file:
        for i in range(len(KArray)):
            line = file.readline().split()
            data['pkz'][0][i] = float(line[2])
            data['pkz'][1][i] = float(line[3])
            data['pkz'][2][i] = float(line[4])
            data['pkz'][3][i] = float(line[5])
            data['pkz'][4][i] = float(line[6])
            data['pkz'][5][i] = float(line[7])
            data['pkz'][6][i] = float(line[8])

    with open(Simulated_densities) as file:
        for i in range(len(zaAntes)):
            line = file.readline().split()
            data['ndz'][i] = float(line[1])

    return data

# Read data se convierte en un diccionario
data = read_data('/Users/guillermo/Desktop/')
data.keys()


# # CAMB settings and results #

# In[211]:


# Let's try to obtain a PPS and Pm with nodes when using CAMB

#Se dan los valores de los nodos
nodes_log_k = [np.log(KArray[0]), np.log(KArray[-1])]
nodes_log_PPS = [3.5, 3.7]

#Se deshace la escala log
nodes_k = np.exp(nodes_log_k)
nodes_PPS = np.exp(nodes_log_PPS)*1e-10

#Aquí se interpolan los nodos (si sonlo son 2, el tipo debe ser lineal)
func = interp1d(nodes_k, nodes_PPS,
                axis=0,  #Este comando interpola entre columnas
                bounds_error=False,
                kind='linear',
                fill_value=(nodes_PPS[0], nodes_PPS[-1]))

#Set de parámetros para CAMB 'pars', con su comología, su primordial modificado y su p de materia modificado con nodos
from camb import model

pars = camb.CAMBparams()
pars.InitPower = camb.initialpower.SplinedInitialPower()
pars.set_cosmology(H0=hJPAS*100, ombh2=OmegabJPASh2, omch2=OmegaCDMJPASh2, mnu=0.0, omk=OmegakJPAS, tau=0.06)
pars.InitPower.set_scalar_log_regular(KArray[0], KArray[-1], func(KArray))
pars.set_matter_power(redshifts=za, kmax=KArrayCompleto[-1])
pars.NonLinear = model.NonLinear_none


# In[212]:


#Resultados de CAMB con los nodos incorporados. Generamos un P de materia y calculamos, por ejemplo sigma8
results = camb.get_results(pars) #Resultados de CAMB

#P de materia
kh, z, pk = results.get_matter_power_spectrum(minkh=KArrayCompleto[0], maxkh=KArrayCompleto[-1], npoints = len(KArrayCompleto))

#Sigma8
s8 = np.array(results.get_sigma8())


# # Classes to interface with Cobaya #

# In[213]:


# I assume this method above is ok, so I will now create the classes to interface with Cobaya
# I will create a cobaya theory NodesInPrimordialPk and a cobaya external likelihood Pklike classes

#Se crean las clases para interactuar con Cobaya: NodesInPrimordialPk (teoría) y Pklike (likelihood)

from cobaya.theory import Theory
from cobaya.likelihood import Likelihood


# In[214]:


#Clase con la teoría, es decir, con la modificación del Primordial para incluir nuestros nodos
class NodesInPrimordialPk(Theory):

    def initialize(self): #Iniciar self con un array de k
        self.ks = KArray

    #Parece que aquí apodamos a las variables de los nodos por sus nombres: k1, k2... pk1, pk2...
    def calculate(self, state, want_derived=True, **params_values_dict):
        
        pivot_scalar = 0.05   #Valor del pivote
        nodes_k = [params_values_dict['k1'], params_values_dict['k2']] #Nombre nodos eje x
        nodes_PPS = [params_values_dict['pk1'], params_values_dict['pk2']] #nombre nodos eje y
        
        #Se interpolan estos nodos.
        Pk_func = interp1d(nodes_k, nodes_PPS,
                axis=0,  # interpolate along columns
                bounds_error=False,
                kind='linear',
                fill_value=(nodes_PPS[0], nodes_PPS[-1]))
        
        #Construimos el PPS(k) en todas las escalas
        state['primordial_scalar_pk'] = {'kmin': self.ks[0], 'kmax': self.ks[-1],
                                         'Pk': Pk_func(self.ks), 'log_regular': True}

    #Metemos en una función el PPS para poder evaluarla 
    def get_primordial_scalar_pk(self):
        return self.current_state['primordial_scalar_pk']
   
    #Función que retorna los parámetros de los nodos
    def get_can_support_params(self):
        return ['k1', 'k2', 'pk1', 'pk2']


# In[243]:


#Clase con el likelihood. Aquí tendremos que introducir nuestro modelo y el cálculo del likelihood

class Pklike(Likelihood): #Se define la clase.
    
    def initialize(self):  

        #Path en el que están los datos. Llamamos a read_data
        self.data = read_data('/Users/guillermo/Desktop/')

        # Se da un grid de zs con extremos en nuestros bines y 150 pasos
        # If you need some quantities at z = 0 you need to have z_win also at zero, please change accordantly
        self.z_win = zaAntes
        self.z_winCompleto = za

   
    #¿Por qué es necesario tener requisitos? ¿No puedo llamar a funciones de CAMB sin antes incluirlas aquí?
    
    def get_requirements(self):
        
        return {'omegam': None,                 
                'Pk_interpolator': {'z': self.z_winCompleto, 'k_max': 10, 'nonlinear': False, 'vars_pairs': ([['delta_tot', 'delta_tot']])},
                'comoving_radial_distance': {'z': self.z_winCompleto},
                'angular_diameter_distance': {'z': self.z_winCompleto},
                'Hubble': {'z': self.z_winCompleto, 'units': 'km/s/Mpc'},
                'sigma8_z': {'z': self.z_winCompleto},
                 #'fsigma8': {'z': self.z_win, 'units': None},
                'CAMBdata': None}

   
    #Aquí defino el monopolo
    
    def monopole(self, **params_dic):

        results = self.provider.get_CAMBdata()   #Parece que esto lee los datos de CAMB
             
        # Aquí se puede acceder al PPS
        ks = KArray
        pps = results.Params.scalar_power(ks)       
        
        #Aquí creamos todas las funciones y variables necesarias para generar el P Kaiser.
        Omegam = self.provider.get_param('omegam');  
        Ez = np.sqrt( Omegam*(1+self.z_winCompleto)**3+(1-Omegam) ); 
        H = HJPAS * Ez
        f = (Omegam*(1+self.z_winCompleto)**3*1/(Ez**2))**gamma
        sigma8z0 = self.provider.get_sigma8_z(self.z_winCompleto[0])
        DeReves = self.provider.get_sigmaR_z(8,self.z_winCompleto)/self.provider.get_sigmaR_z(8,self.z_winCompleto[0])
        De = DeReves[::-1]
        
        def bJPAS(z):
          return 0.53+0.289*(1+z)**2
            
        A = De*bJPAS(za)*sigma8z0
        R = De*f*sigma8z0

        #Fotometría
        DeltazJPAS = 0.00364236313918151
        sigmar = DeltazJPAS*(1+self.z_winCompleto)/H

        # This is the matter power spectrum interpolator:
        pk = self.provider.get_Pk_interpolator(('delta_tot', 'delta_tot'), nonlinear=False)   #Parece que aquí se obtiene el Pmateria
        pk_delta = pk.P(ks, self.z_winCompleto)         # pk_delta is an array de pmateria evaluado en ks and zs

        
        #Fingers of God
        sigmap = (1/(6*np.pi**2)*(De/1)**2*integrate.quad(lambda k:  pk.P(k, self.z_winCompleto[2]) , ks[0], ks[-1])**0.5)

        def FFoG(mu,k):
            return 1/(1+(f[2]*k*mu*sigmap)**2)

        
        #Efecto AP
        Xi = self.provider.get_comoving_radial_distance(self.z_winCompleto)*hJPAS;
        DA = Xi/(1+self.z_winCompleto);
        
        FactorAP = DAFid**2*Ez[0]/( DA[0]**2*EzFid )

        def Q(mu):
            return ((Ez[2]**2*Xi[2]**2*mu**2-EzFid**2*XiFid**2*(mu**2-1))**0.5/(EzFid*Xi[2]))
 
        def muObs(mu):
            return mu*Ez[2]/(EzFid*Q(mu))
           
        def kObs(mu,k):
            return Q(mu)*k 

        #P de galaxias final
        def Pg(mu,k):
            return FactorAP*FFog(muObs(mu),kObs(mu,k))*(A[2]+R[2]*muObs(mu)**2)**2 * PmatInterCAMB(kObs(mu,k))/sigma8z0**2 *np.exp(-(k*mu*sigmar[2])**2)

        #Se usa la regla del trapecio con 2000 pasos
        def Pgmonopole(k):
            mu = np.arange(-1, 1, 1/1000)
            return 1/2 * integrate.trapz(Pg(mu, k), mu)
            
        PgmonopoleValores = np.zeros(len(ks))
        for i in range(0, len(ks)):
             PgmonopoleValores[i] = Pgmonopole(ks[i])

        #Covarianza
        
        #Densidades desde data. #DensityHighZ es equivalente a self.data['ndz]

        #Definición del volumen (requiere distancia angular con unidades y binear en zupper y zlower)
        #Área del cielo
        fsky = 0.2575;

        #Bines de z por arriba y por abajo:

        #Distancia angular para los bines z upper y lower:

        XiZaLower = self.provider.get_comoving_radial_distance(self.z_winCompleto[positions_Lower])*hJPAS
        XiZaUpper = self.provider.get_comoving_radial_distance(self.z_winCompleto[positions_Upper])*hJPAS

        
        #Definición de volumen:
        Vol = 4*np.pi*fsky/3*(XiZaUpper**3-XiZaLower**3)

        #Definición del número de modos

        #Número de modos. Depende de las variables k1 y k2, que debe corresponderse a kupper y klower
        def Nk(k1,k2):
            return Vol[0] * (4*np.pi/3*(k1**3-k2**3))/((2*np.pi)**3)

        #Evaluamos Nk para cada valor de nuestro array de k
        NkEvaluado = np.zeros(len(ks))
        for i in range(0, len(ks)):
            NkEvaluado[i] = Nk(KArrayUpper[i],KArrayLower[i])

        #Definición de la covarianza

        #Va a depender de k1 y k2. No me gusta mucho:
        def Cov(k,k1,k2):
            return 2 * (Pgmonopole(k) + 1/self.data['ndz'][0])**2 / Nk(k1,k2)

        #Evaluamos Cov para nuestros k
        CovEvaluado = 2 *(PgmonopoleValores + 1/self.data['ndz'][0])**2 / NkEvaluado

        
        return PgmonopoleValores


    #Aquí calculo el likelihood
    
    def logp(self, **params_values):       
        
        # Calcular el monopolo:
        
        Pk = self.monopole(**params_values)    #Del anterior código. ¿Debo llamar al monopolo así o con PgmonopoleValores[i]?
        
        # Calcular el loglikelihood
        #Bineamos el P(k) para cuando entre al likelihood:
        PgBineadoz1 = np.zeros(len(ks)); PgBineadoz2 = np.zeros(len(ks));
        PgBineadoz3 = np.zeros(len(ks)); PgBineadoz4 = np.zeros(len(ks));
        PgBineadoz5 = np.zeros(len(ks)); PgBineadoz6 = np.zeros(len(ks));
        PgBineadoz7 = np.zeros(len(ks));

        PgBineado = np.array([PgBineadoz1,PgBineadoz2,PgBineadoz3,PgBineadoz4,PgBineadoz5,PgBineadoz6,PgBineadoz7])

        for i in range(0, len(ks)):
            PgBineado[0][i] = PgmonopoleValores[i]

        #Bineamos la covarianza para cuando entre al likelihood
        CovBineadoz1 = np.zeros(len(ks)); CovBineadoz2 = np.zeros(len(ks));
        CovBineadoz3 = np.zeros(len(ks)); CovBineadoz4 = np.zeros(len(ks));
        CovBineadoz5 = np.zeros(len(ks)); CovBineadoz6 = np.zeros(len(ks));
        CovBineadoz7 = np.zeros(len(ks));

        CovBineado = np.array([CovBineadoz1,CovBineadoz2,CovBineadoz3,CovBineadoz4,CovBineadoz5,CovBineadoz6,CovBineadoz7])

        for i in range(0, len(ks)):
            CovBineado[0][i] = CovEvaluado[i]

        #Construimos el likelihood, siendo j el valor del array de z, i el valor del array de k:

        #Este likelihood es similar a un chi^2. ¿Igual? ¿Factor del log del determinante = log cov?
        def lnlikeSinSumar(j,i):
            return (PgBineado[j][i]-data['pkz'][0][i])**2 * 1/CovBineado[j][i] + np.log(CovBineado[j][i])
    
        #índices en los que queremos evaluar el likelihood y el paso entre ellos. Usamos todos:
        IndicesLikelihood = np.arange(0,len(ks),1)

        #Likelihood sumando a los índices
        lnlike = np.sum(lnlikeSinSumar(0,IndicesLikelihood))
        lnlike
        return lnlike


# In[83]:


# This is how you pass input to Cobaya
# Diccionario que le pasamos a Cobaya, donde linkamos con nuestros códigos de teoría y el likelihood.

info = {'debug': True,                        #Esto permite obtener info de los errores
        'likelihood': {'jpass': Pklike},      #Aquí se engancha el likelihood (nombre jpass) que hemos definido en la clase de arriba
        'theory': {'camb': {"external_primordial_pk": True},
                   'my_pk': NodesInPrimordialPk},      #Aquí le pasamos nuestra clase de teoría, con nombre "my_pk".
       'params': {
        # Parámetros cosmológicos fijados
        'tau': 0.06, 'mnu': 0.00, 'nnu': 3.046,
        # Parámetros nodales, flat priors
        'P1': {'prior': {'min': 1e-10, 'max': 5.6e-9}, 'latex': 'P_1'},
        'P2': {'prior': {'min': 1e-10, 'max': 5.6e-9}, 'latex': 'P_2'},
        # Parámetros cosmológicos a samplear. Loc y scale son, en prior gaussiano, valor medio y desviación estandar.
        'ombh2': {'prior': {'dist': 'norm', 'loc': OmegabJPASh2Fid, 'scale': 0.00015}, 'latex': 'Omega_bh^2'},
        'omch2': {'prior': {'dist': 'norm', 'loc': OmegaCDMJPASFid, 'scale': 0.0012}, 'latex': 'Omega_ch^2'},
        'H0': {'prior': {'dist': 'norm', 'loc': hJPASFid*100, 'scale': 0.54}, 'latex': 'H_0'}}}


# In[ ]:


# Let's reproduce the same matter power spectrum as the one by single camb through the cobaya interface
# Se reproduce el Pmateria construido por CAMB con la interfaz de Cobaya:

from cobaya.model import get_model     
model = get_model(info)          #Se construye un modelo con el diccionario info

model.logposterior({}) 

camb_results = model.provider.get_CAMBdata();

pk = model.provider.get_Pk_interpolator(('delta_tot', 'delta_tot'), nonlinear=False)

