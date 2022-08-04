import numpy as np
from scipy.optimize import minimize

import batman

def init_batman(t, ld_law, nresampling=None, etresampling=None):
    """  
    This function initializes the batman lightcurve generator object.
    Parameters
    ----------
    t: array
      Array containing the times at which the lightcurve will be evaluated. Assumes units of days.
    ld_law: string
      Limb-darkening law used to compute the model. Available ld laws: uniform, linear, quadratic, 
      logarithmic, exponential, squareroot, nonlinear, power2.
    nresampling: int
      Number of resampled points in case resampling is wanted.
    etresampling: float
      Exposure time of the resampling, same units as input time.
    Returns
    -------
    
    params: batman object
      Object containing the parameters of the lightcurve model.
    m: batman object
      Object that enables the lightcurve calculation.
    """

    params = batman.TransitParams()
    params.t0 = 0. 
    params.per = 1. 
    params.rp = 0.1
    params.a = 15.
    params.inc = 87.
    params.ecc = 0. 
    params.w = 90.

    if ld_law == 'linear':
        params.u = [0.5]

    elif ld_law == 'nonlinear':
        params.u = [0.1, 0.1, 0.1, 0.1]

    else:       
        params.u = [0.1,0.3]

    params.limb_dark = ld_law

    if nresampling is None or etresampling is None:
        m = batman.TransitModel(params, t)

    else:
        m = batman.TransitModel(params, t, supersample_factor=nresampling, exp_time=etresampling)

    return params, m

def spam_objective_function(theta, params, m, params_twop, m_twop):
    """
    Objective function that the SPAM algorithm is looking to minimize. 
    """

    u1, u2 = theta
    params_twop.u = [u1, u2]

    return np.sum((m.light_curve(params) - m_twop.light_curve(params_twop))**2)


def transform_coefficients(c1, c2, c3, c4, planet_name=None, planet_data=None, ld_law='quadratic', ndatapoints=1000, method='BFGS', u1_guess=0.5, u2_guess=0.5):
    """
    Given a set of non-linear limb-darkening coefficients (c1, c2, c3 and c4) and either a planet name ('planet_name') or a dictionary with 
    the planet's data ('planet_data'), this function returns the Synthetic-Photometry/Atmosphere-Model (SPAM; https://arxiv.org/abs/1106.4659) 
    limb-darkening coefficients of the star. 
    Reference:
        Howarth, I., 2011, "On stellar limb darkening and exoplanetary transits", MNRAS, 418, 1165
        https://ui.adsabs.harvard.edu/abs/2011MNRAS.418.1165H/abstract
    Parameters
    ----------
    c1: float
      First limb-darkening coefficient of the non-linear law.
    c2: float
      Same as c1, but second.
    c3: float
      Same as c1, but third.
    c4: float
      Same as c1, but fourth.
    planet_name: string
      String with the name of the input planet (e.g., 'WASP-19b'); this will be used to query the planet properties from MAST.
    planet_data: dict
      Dictionary containing the planet properties. In particular, this dictionary should contain the keys 
      'transit_duration', 'orbital_period' (days), 'Rp/Rs', 'a/Rs', 'inclination' (degrees), 'eccentricity' and 'omega' (degrees) 
      for the algorithm to work. Properties in this dictionary take prescedence over the ones retrieved from MAST.
    ld_law: string
      Limb-darkening law for which SPAM coefficients are wanted. Default is 'quadratic', but can also be 'squareroot' or 'logarithmic'. 
    ndatapoints: int
      Number of datapoints that will be used for the lightcurve simulations to extract back the SPAM coefficients.
    method: string
      Minimization method to match lightcurves. Default is 'BFGS', but can be any of the ones available for scipy.optimize.minimize.
      Details: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
    u1_guess: float
      Guess starting value for u1. Default is 0.5
    u2_guess: flat
      Guess starting value for u2. Default is 0.5.
    Returns
    -------
    
    u1: float
      The first limb-darkening coefficient of the selected two-parameter law.
    u2: flat
      Same as u1, but for the second coefficient.
    """

    planet_properties = ['transit_duration', 'orbital_period', 'Rp/Rs', 'a/Rs', 'inclination', 'eccentricity', 'omega']

    # Check if planet name is given
    if planet_name is not None:

        # If planet_name is given, retrieve MAST properties
        mast_planet_data, url = utils.get_target_data(planet_name)

        # Merge the dictionaries, prioritizing the manual input
        mast_planet_data.update(planet_data or {})
        planet_data = mast_planet_data

    if planet_data is None:

        raise Exception("User must input either 'planet_name' and/or 'planet_data' for SPAM to work. See details by doing exoctk.limb_darkening.spam.transform_coefficients?.")

    # Check that all properties exist in the input dictionary
    missing = [planet_property for planet_property in planet_properties if planet_data.get(planet_property) is None]
    if len(missing) > 0:

        # Print current data for user
        print('{} properties'.format(planet_name or 'Planet'))
        for planet_property in planet_properties:
            print('{}: {}'.format(planet_property, planet_data.get(planet_property)))

        raise ValueError("{} missing planet propert{} needed for SPAM to work. Please include this in the 'planet_data' input dictionary".format(len(missing), 'y is' if len(missing) == 1 else 'ies are'))

    # User inputs check done. Now jump into the algorithm. First, define times around transit
    times = np.linspace(-planet_data['transit_duration']/2., planet_data['transit_duration']/2., ndatapoints)

    # Now initialize models
    params, m = init_batman(times, 'nonlinear')
    params_twop, m_twop = init_batman(times, ld_law)

    # Define params according to the planet_data dictionary
    for par in [params, params_twop]:

        par.per = planet_data['orbital_period']
        par.rp = planet_data['Rp/Rs']
        par.a = planet_data['a/Rs']
        par.inc = planet_data['inclination']
        par.ecc = planet_data['eccentricity']
        par.w = planet_data['omega']

    # Set non-linear law coefficients
    params.u = [c1, c2, c3, c4]

    # Allright, now given these models, optimize u1 and u2 such that they match as good as possible the lightcurves of 
    # the non-linear law. We use scipy.optimize for this
    results = minimize(spam_objective_function, [u1_guess, u2_guess], args=(params, m, params_twop, m_twop), method=method.lower())

    return results.x, {prop: val for prop, val in planet_data.items() if prop in planet_properties}


def get_wav_and_coeffs(input_file, ld_law = 'quadratic', model = 'A100'):

    if ld_law == 'quadratic':

        check = 'u1,'

    elif ld_law == 'squareroot':

        check = 's1,'

    wavs, c1, c2, c3, c4, u1, u2 = [], [], [], [], [], [], []

    fin = open(input_file, 'r')

    while True:

        s = fin.readline()

        if s != '':

            spitout = s.split()

            if len(spitout) > 0:

                if len(spitout) > 1:
            
                    if spitout[1] == model:

                        wavs.append( np.double( spitout[0] ) )
               
                if spitout[0] == check:

                    u1.append( np.double(spitout[-2][:-1]) )
                    u2.append( np.double(spitout[-1]) )

                if spitout[0] == 'c1,':

                    c1.append( np.double(spitout[-4][:-1]) )
                    c2.append( np.double(spitout[-3][:-1]) )
                    c3.append( np.double(spitout[-2][:-1]) )
                    c4.append( np.double(spitout[-1]) )

        else:

            break

    return wavs, c1, c2, c3, c4, u1, u2

orders = ['w39_order1', 'w39_order2']

planet_data = {}
planet_data['transit_duration'] = 2.8032 / 24.
planet_data['orbital_period'] = 4.055259
planet_data['Rp/Rs'] = 0.14379920399444782
planet_data['a/Rs'] = 11.328210411169186
planet_data['inclination'] = 87.6798251
planet_data['eccentricity'] = 0.0
planet_data['omega'] = 90.0

ld_law = 'squareroot'
ndatapoints = 537
model = 'A100'

for order in orders:

    fout = open(order+'_'+ld_law+'_spam.txt', 'w')
    input_file = 'results/'+order+'.txt'
    wavs, c1, c2, c3, c4, u1, u2 = get_wav_and_coeffs(input_file, ld_law = ld_law, model = model)
    u1_spam, u2_spam = [], []

    fout.write('# SPAM LD coefficients\n')
    fout.write('# Calculation by Nestor Espinoza (nespinoza@stsci.edu)\n')
    fout.write('# \n')
    fout.write('# Column 1: Wavelength (microns) \n')
    fout.write('# Column 2-5: Non-linear LD coeffs\n')
    fout.write('# Column 6-7: '+ld_law+' LD coeffs\n')
    fout.write('# Column 8-9: SPAM '+ld_law+' LD coeffs\n')
    fout.write('# \n')

    for i in range(len(wavs)):

        out, _ = transform_coefficients(c1[i], c2[i], c3[i], c4[i], 
                                          planet_data = planet_data, 
                                          ld_law=ld_law, 
                                          ndatapoints=ndatapoints, 
                                          u1_guess=u1[i], u2_guess=u2[i])

        u1s, u2s = out
        fout.write('{0:.5f} {1:.10f} {2:.10f} {3:.10f} {4:.10f} {5:.10f} {6:.10f} {7:.10f} {8:.10f}\n'.format(wavs[i] / 10000., c1[i], c2[i], c3[i], c4[i], u1[i], u2[i], u1s, u2s))
