#! /usr/bin/env python
import sys
import os
import numpy as np
import glob
try:     # Python2
  import urllib2 as urllib
except:  # Python3
  import urllib
import argparse
import scipy.interpolate as si
from copy import copy

try:
  import pyfits as fits
except:
  import astropy.io.fits as fits


########## OPTIONS ###################################################

# Order of the interpolation done for sampling intensities:
interpolation_order = 1

# Define if you will apply the corrections or not. First the ATLAS one,
# if True, convert ATLAS intensities using c/lambda**2 (ATLAS intensities
# are given per frequency):
atlas_correction = True
# Now decide if you want to apply photon counting correction, lambda/hc:
photon_correction = True


def parse():
    """
    Parse command-line arguments.

    Returns
    -------
    input_filename: String
       Command-line input set by '-ifile'.
    output_filename: String
       Command-line input set by '-ofile'.  Output file where to store results.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-ifile', default=None)
    parser.add_argument('-ofile', default=None)
    args = parser.parse_args()

    # Set the input file:
    input_filename = 'input_files/all_atlas_lds_kepler.dat'
    if args.ifile is not None:
       input_filename = args.ifile

    # Set the output file:
    output_filename = 'all_atlas_lds_kepler.dat'
    if args.ofile is not None:
       output_filename = args.ofile

    return input_filename, output_filename


def FixSpaces(intervals):
    s = ''
    i = 0
    while True:
        if intervals[i]=='':
            intervals.pop(i)
        else:
            i = i+1
            if len(intervals)==i:
                break
        if len(intervals)==i:
            break
    for i in range(len(intervals)):
        if i!=len(intervals)-1:
            s = s+str(np.double(intervals[i]))+'\t'
        else:
            s = s+str(np.double(intervals[i]))+'\n'
    return s


def getFileLines(fname):
    with open(fname, 'r') as f:
        l = f.readline()
        if l.find('\n') == -1:
            lines = l.split('\r')
        else:
            f.seek(0)
            l = f.read()
            lines = l.split('\n')
    return lines


def getATLASStellarParams(lines):
    for i in range(len(lines)):
        line = lines[i]
        idx = line.find('EFF')
        if idx != -1:
            idx2 = line.find('GRAVITY')
            TEFF    = line[idx +4:idx2-1]
            GRAVITY = line[idx2+8:idx2+8+5]
            VTURB   = line[idx +6:idx2]
            idx = line.find('L/H')
            if idx == -1:
                LH = '1.25'
            else:
                LH = line[idx+4:]
            break
    return str(int(np.double(TEFF))), str(np.round(np.double(GRAVITY),2)), \
           str(np.round(np.double(LH),2))


def getIntensitySteps(lines):
    for j in range(len(lines)):
        line = lines[j]
        idx = line.find('intervals')
        if idx != -1:
            line = lines[j+1]
            intervals = line.split(' ')
            break

    s = FixSpaces(intervals)
    return j+2, s


version = 'v.1.0.'
def get_derivatives(rP, IP):
    """
    This function calculates the derivatives in an intensity profile I(r).
    For a detailed explaination, see Section 2.2 in Espinoza & Jordan (2015).

    INPUTS:
      rP:   Normalized radii, given by r = sqrt(1-mu**2)
      IP:   Intensity at the given radii I(r).

    OUTPUTS:
      rP:      Output radii at which the derivatives are calculated.
      dI/dr:   Measurement of the derivative of the intensity profile.
    """
    ri = rP[1:-1] # Points
    mui = np.sqrt(1-ri**2)
    rib = rP[:-2] # Points inmmediately before
    ria = rP[2:]  # Points inmmediately after
    Ii = IP[1:-1]
    Iib = IP[:-2]
    Iia = IP[2:]

    rbar = (ri+rib+ria)/3.0
    Ibar = (Ii+Iib+Iia)/3.0
    num = (ri-rbar)*(Ii-Ibar) + (rib-rbar)*(Iib-Ibar) + (ria-rbar)*(Iia-Ibar)
    den = (ri-rbar)**2 + (rib-rbar)**2 + (ria-rbar)**2

    return rP[1:-1], num/den


def fix_spaces(the_string):
    """
    This function fixes some spacing issues in the ATLAS model files.
    """
    splitted = the_string.split(' ')
    for s in splitted:
        if(s!=''):
           return s
    return the_string


def fit_exponential(mu, I):
    """
    Calculate the coefficients for the exponential LD law.
    It assumes input intensities are normalized.  For a derivation of the
    least-squares problem solved, see Espinoza & Jordan (2015).

    INPUTS:
      mu:  Angles at which each intensity is calculated (numpy array).
      I:   Normalized intensities (i.e., I(mu)/I(1)) (numpy array).

    OUTPUTS:
      e1:   Coefficient of the linear term of the exponential law.
      e2:   Coefficient of the exponential term of the exponential law.
    """
    # Define A matrix for the linear system:
    A = np.zeros([2,2])
    # Define b vector for the linear system:
    b = np.zeros(2)
    # Obtain alpha_n_k and beta_k that fill the A matrix and b vector.
    # In this case, g_1 = 1-mu, g_2 = 1/(1-exp(mu)):
    A[0,0] = sum((1.0-mu)**2)                    # alpha_{1,1}
    A[0,1] = sum((1.0-mu)*(1./(1.-np.exp(mu))))  # alpha_{1,2}
    A[1,0] = A[0,1]                              # alpha_{2,1} = alpha_{1,2}
    A[1,1] = sum((1./(1.-np.exp(mu)))**2)        # alpha_{2,2}
    b[0] = sum((1.0-mu)*(1.0-I))                 # beta_1
    b[1] = sum((1./(1.-np.exp(mu)))*(1.0-I))     # beta_2
    return np.linalg.solve(A,b)


def fit_logarithmic(mu, I):
    """
    Calculate the coefficients for the logarithmic LD law.
    It assumes input intensities are normalized.  For a derivation of the
    least-squares problem solved, see Espinoza & Jordan (2015).

    INPUTS:
      mu:  Angles at which each intensity is calculated (numpy array).
      I:   Normalized intensities (i.e., I(mu)/I(1)) (numpy array).

    OUTPUTS:
      l1:   Coefficient of the linear term of the logarithmic law.
      l2:   Coefficient of the logarithmic term of the logarithmic law.
    """
    # Define A matrix for the linear system:
    A = np.zeros([2,2])
    # Define b vector for the linear system:
    b = np.zeros(2)
    # Obtain the alpha_n_k and beta_k that fill the A matrix and b vector.
    # In this case, g_1 = 1-mu, g_2 = mu*ln(mu):
    A[0,0] = sum((1.0-mu)**2)               # alpha_{1,1}
    A[0,1] = sum((1.0-mu)*(mu*np.log(mu)))  # alpha_{1,2}
    A[1,0] = A[0,1]                         # alpha_{2,1} = alpha_{1,2}
    A[1,1] = sum((mu*np.log(mu))**2)        # alpha_{2,2}
    b[0] = sum((1.0-mu)*(1.0-I))            # beta_1
    b[1] = sum((mu*np.log(mu))*(1.0-I))     # beta_2
    return np.linalg.solve(A,b)


def fit_square_root(mu, I):
    """
    Calculates the coefficients for the square-root LD law.
    It assumes input intensities are normalized.  For a derivation of the
    least-squares problem solved, see Espinoza & Jordan (2015).

    INPUTS:
      mu:  Angles at which each intensity is calculated (numpy array).
      I:   Normalized intensities (i.e., I(mu)/I(1)) (numpy array).

    OUTPUTS:
      s1:   Coefficient of the linear term of the square-root law.
      s2:   Coefficient of the square-root term of the square-root law.
    """
    # Define A matrix for the linear system:
    A = np.zeros([2,2])
    # Define b vector for the linear system:
    b = np.zeros(2)
    # Obtain the alpha_n_k and beta_k that fill the A matrix and b vector:
    for n in range(1,3,1):
        for k in range(1,3,1):
            A[n-1,k-1] = sum((1.0-mu**(n/2.0))*(1.0-mu**(k/2.0)))
        b[n-1] = sum((1.0-mu**(n/2.0))*(1.0-I))
    x = np.linalg.solve(A,b)
    return x[1],x[0] # x[1] = s1, x[0] = s2


def fit_non_linear(mu, I):
    """
    Calculate the coefficients for the non-linear LD law.
    It assumes input intensities are normalized.  For a derivation of the
    least-squares problem solved, see Espinoza & Jordan (2015).

    INPUTS:
      mu:  Angles at which each intensity is calculated (numpy array).
      I:   Normalized intensities (i.e., I(mu)/I(1)) (numpy array).

    OUTPUTS:
      c1:   Coefficient of the square-root term of the non-linear law.
      c2:   Coefficient of the linear term of the non-linear law.
      c3:   Coefficient of the (1-mu^{3/2}) term of the non-linear law.
      c4:   Coefficient of the quadratic term of the non-linear law.
    """
    # Define A matrix for the linear system:
    A = np.zeros([4,4])
    # Define b vector for the linear system:
    b = np.zeros(4)
    # Obtain the alpha_n_k and beta_k that fill the A matrix and b vector:
    for n in range(1,5,1):
        for k in range(1,5,1):
            A[n-1,k-1] = sum((1.0-mu**(n/2.0))*(1.0-mu**(k/2.0)))
        b[n-1] = sum((1.0-mu**(n/2.0))*(1.0-I))
    return np.linalg.solve(A,b)


def fit_three_parameter(mu, I):
    """
    Calculate the coefficients for the three-parameter LD law.
    It assumes input intensities are normalized.  For a derivation of the
    least-squares problem solved, see Espinoza & Jordan (2015).

    INPUTS:
      mu:  Angles at which each intensity is calculated (numpy array).
      I:   Normalized intensities (i.e., I(mu)/I(1)) (numpy array).

    OUTPUTS:
      b1:   Coefficient of the linear term of the three-parameter law.
      b2:   Coefficient of the (1-mu^{3/2}) part of the three-parameter law.
      b3:   Coefficient of the quadratic term of the three-parameter law.
    """
    # Define A matrix for the linear system:
    A = np.zeros([3,3])
    # Define b vector for the linear system:
    b = np.zeros(3)
    # Obtain the alpha_n_k and beta_k that fill the A matrix and b vector.
    # In this case we skip c1 (i.e., set c1=0):
    for n in range(2,5,1):
        for k in range(2,5,1):
            A[n-2,k-2] = sum((1.0-mu**(n/2.0))*(1.0-mu**(k/2.0)))
        b[n-2] = sum((1.0-mu**(n/2.0))*(1.0-I))
    return np.linalg.solve(A,b)


def fit_quadratic(mu, I):
    """
    Calculate the coefficients for the quadratic LD law.
    It assumes input intensities are normalized.  For a derivation of the
    least-squares problem solved, see Espinoza & Jordan (2015).

    INPUTS:
      mu:  Angles at which each intensity is calculated (numpy array).
      I:   Normalized intensities (i.e., I(mu)/I(1)) (numpy array).

    OUTPUTS:
      u1:   Linear coefficient of the quadratic law.
      u2:   Quadratic coefficient of the quadratic law.
    """
    # Define A matrix for the linear system:
    A = np.zeros([2,2])
    # Define b vector for the linear system:
    b = np.zeros(2)
    # Obtain the alpha_n_k and beta_k that fill the A matrix and b vector:
    for n in range(1,3,1):
        for k in range(1,3,1):
            A[n-1,k-1] = sum(((1.0-mu)**n)*((1.0-mu)**k))
        b[n-1] = sum(((1.0-mu)**n)*(1.0-I))
    return np.linalg.solve(A,b)


def fit_linear(mu, I):
    """
    Calculate the coefficients for the linear LD law.
    It assumes input intensities are normalized.  For a derivation of the
    least-squares problem solved, see Espinoza & Jordan (2015).

    INPUTS:
      mu:  Angles at which each intensity is calculated (numpy array).
      I:   Normalized intensities (i.e., I(mu)/I(1)) (numpy array).

    OUTPUTS:
      a:   Coefficient of the linear law.
    """
    alpha_1_1 = sum((1.0-mu)**2)
    beta_1 = sum((1.0-mu)*(1.0-I))
    a = beta_1/alpha_1_1
    return a


def downloader(url):
    """
    This function downloads a file from the given url using wget.
    """
    file_name = url.split('/')[-1]
    print('\t      + Downloading file {:s} from {:s}.'.format(file_name, url))
    os.system('wget '+url)


def ATLAS_model_search(s_met, s_grav, s_teff, s_vturb):
    """
    Given input metallicities, gravities, effective temperature and
    microturbulent velocity, this function estimates which model is
    the most appropiate (i.e., the closer one in parameter space).
    If the model is not present in the system, it downloads it from
    Robert L. Kurucz's website (kurucz.harvard.edu/grids.html).
    """
    if not os.path.exists('atlas_models'):
       os.mkdir('atlas_models')
       os.mkdir('atlas_models/raw_models')

    # This is the list of all the available metallicities in Kurucz's website:
    possible_mets = np.array([-0.1, -0.2, -0.3, -0.5, -1.0, -1.5, -2.0, -2.5,
                  -3.0, -3.5, -4.0, -4.5, -5.0, 0.0, 0.1, 0.2, 0.3, 0.5, 1.0])
    # And this is the list of all possible vturbs:
    possible_vturb = np.array([0.0, 2.0, 4.0, 8.0])

    # Check if turbulent velocity is given. If not, set to 2 km/s:
    if(s_vturb==-1):
       print('\t > No known turbulent velocity. Setting it to 2 km/s.')
       s_vturb = 2.0
    else:
       if s_vturb not in possible_vturb:
          # Check closest vturb to input:
          vturb_diff = np.inf
          chosen_vturb = np.inf
          for vturb in possible_vturb:
              # Estimate distance between current and input vturb:
              c_vturb_diff = np.abs(vturb - s_vturb)
              if(c_vturb_diff < vturb_diff):
                 chosen_vturb = c_vturb_diff
                 vturb_diff = copy(c_vturb_diff)
          print('\t > For input vturb {} km/s, closest vturb is {} km/s.'
                .format(s_vturb, chosen_vturb))
       else:
          chosen_vturb = s_vturb

    if s_met not in possible_mets:
       # Now check closest metallicity model for input star:
       m_diff = np.inf
       chosen_met = np.inf
       for met in possible_mets:
            # Estimate distance between current and input metallicity:
            c_m_diff = np.abs(met-s_met)
            if(c_m_diff<m_diff):
                chosen_met = met
                m_diff = copy(c_m_diff)

       print('\t > For input metallicity {}, closest metallicity is {}.'
             .format(s_met, chosen_met))
    else:
       chosen_met = s_met

    # Check if the intensity file for the calculated metallicity and vturb is on the atlas_models folder:
    if chosen_met == 0.0:
       met_dir = 'p00'
    elif chosen_met < 0:
       met_string = str(np.abs(chosen_met)).split('.')
       met_dir = 'm'+met_string[0]+met_string[1]
    else:
       met_string = str(np.abs(chosen_met)).split('.')
       met_dir = 'p'+met_string[0]+met_string[1]

    print('\t    + Checking if ATLAS model file is on the system ...')
    # This will make the code below easier to follow:
    amodel = '{:s}k{:.0f}'.format(met_dir, chosen_vturb)
    afile = 'atlas_models/raw_models/i' + amodel

    if os.path.exists(afile + 'new.pck') or \
       os.path.exists(afile + '.pck19')  or \
       os.path.exists(afile + '.pck'):
        print('\t    + Model file found.')
    else:
        # If not in the system, download it from Kurucz's website.
        # First, check all possible files to download:
        print('\t    + Model file not found.')
        response = urllib.urlopen('http://kurucz.harvard.edu/grids/grid' +
                                  met_dir + '/')
        html = response.read()
        ok = True
        filenames = []
        while(ok):
            idx = html.find('>i'+met_dir.lower())
            if(idx==-1):
                ok = False
            else:
                for i in range(30):
                    if(html[idx+i]=='<'):
                        filenames.append(html[idx+1:idx+i])
            html = html[idx+1:]
        hasnew = False
        gotit = False
        # Check that filenames have the desired vturb and prefer *new* models:
        for afname in filenames:
            if 'new' in afname  and  amodel in afname:
                hasnew = True
                gotit = True
                downloader('http://kurucz.harvard.edu/grids/grid'+met_dir+'/'+afname)
                if os.path.exists('atlas_models/raw_models/'):
                    os.rename(afname,'atlas_models/raw_models/'+afname)
                else:
                    os.mkdir('atlas_models/raw_models/')
                    os.rename(afname,'atlas_models/raw_models/'+afname)
        if not hasnew:
            for afname in filenames:
                if '.pck19' in afname and  amodel in afname:
                    gotit = True
                    downloader('http://kurucz.harvard.edu/grids/grid'+met_dir+'/'+afname)
                    if os.path.exists('atlas_models/raw_models/'):
                        os.rename(afname,'atlas_models/raw_models/'+afname)
                    else:
                        os.mkdir('atlas_models/raw_models/')
                        os.rename(afname,'atlas_models/raw_models/'+afname)
            if not gotit:
                for afname in filenames:
                    if amodel+'.pck' in afname:
                        gotit = True
                        downloader('http://kurucz.harvard.edu/grids/grid'+met_dir+'/'+afname)
                        if os.path.exists('atlas_models/raw_models/'):
                            os.rename(afname,'atlas_models/raw_models/'+afname)
                        else:
                            os.mkdir('atlas_models/raw_models/')
                            os.rename(afname,'atlas_models/raw_models/'+afname)
        if not gotit:
            print('\t > No model with closest metallicity of {} and closest '
                  'vturb of {} km/s found.\n\t   Please, modify the input '
                  'values of the target and select other stellar parameters '
                  'for it.'.format(chosen_met, chosen_vturb))
            sys.exit()

    # Check if the models in machine readable form have been generated.
    # If not, generate them:
    if not os.path.exists('atlas_models/' + amodel):
        # Now read the files and generate machine-readable files:
        possible_paths = [afile+'new.pck', afile+'.pck19', afile+'.pck']

        for i in range(len(possible_paths)):
            possible_path = possible_paths[i]
            if os.path.exists(possible_path):
                lines = getFileLines(possible_path)
                # Create folder for current metallicity and turbulent
                # velocity if not created already:
                if not os.path.exists('atlas_models/' + amodel):
                    os.mkdir('atlas_models/' + amodel)
                # Save files in the folder:
                while True:
                    TEFF,GRAVITY,LH = getATLASStellarParams(lines)
                    if not os.path.exists('atlas_models/'+amodel+'/'+TEFF):
                        os.mkdir('atlas_models/'+amodel+'/'+TEFF)
                    idx,mus = getIntensitySteps(lines)
                    save_mr_file = True
                    if os.path.exists('atlas_models/'+amodel+'/'+TEFF+
                                      '/grav_'+GRAVITY+'_lh_'+LH+'.dat'):
                        save_mr_file = False
                    if save_mr_file:
                        f = open('atlas_models/'+amodel+'/'+TEFF+
                                 '/grav_'+GRAVITY+'_lh_'+LH+'.dat','w')
                        f.write('#TEFF:' + TEFF +
                                ' METALLICITY:' + met_dir +
                                ' GRAVITY:' + GRAVITY +
                                ' VTURB:' + str(int(chosen_vturb)) +
                                ' L/H: ' + LH + '\n')
                        f.write('#wav (nm) \t cos(theta):' + mus)
                    for i in range(idx, len(lines)):
                        line = lines[i]
                        idx = line.find('EFF')
                        idx2 = line.find('\x0c')
                        if(idx2!=-1 or line==''):
                            hhhh=1
                        elif(idx!=-1):
                            lines = lines[i:]
                            break
                        else:
                            wav_p_intensities = line.split(' ')
                            s = FixSpaces(wav_p_intensities)
                            if save_mr_file:
                                f.write(s+'\n')
                    if save_mr_file:
                        f.close()
                    if(i==len(lines)-1):
                        break

    # Now, assuming models are written in machine readable form, we can work:
    chosen_met_folder = 'atlas_models/' + amodel

    # Now check closest Teff for input star:
    t_diff = np.inf
    chosen_teff = np.inf
    chosen_teff_folder = ''
    tefffolders = glob.glob(chosen_met_folder+'/*')
    for tefffolder in tefffolders:
        fname = tefffolder.split('/')[-1]
        teff = np.double(fname)
        c_t_diff = abs(teff-s_teff)
        if(c_t_diff<t_diff):
            chosen_teff = teff
            chosen_teff_folder = tefffolder
            t_diff = c_t_diff

    print('\t    + For input effective temperature {:.1f} K, closest value '
          'is {:.0f} K.'.format(s_teff, chosen_teff))
    # Now check closest gravity and turbulent velocity:
    grav_diff = np.inf
    chosen_grav = 0.0
    chosen_fname = ''
    all_files = glob.glob(chosen_teff_folder+'/*')

    for filename in all_files:
        grav = np.double((filename.split('grav')[1]).split('_')[1])
        c_g_diff = abs(grav-s_grav)
        if(c_g_diff<grav_diff):
            chosen_grav = grav
            grav_diff = c_g_diff
            chosen_filename = filename

    # Summary:
    print('\t + For input metallicity {}, effective temperature {} K, and\n'
        '\t   log-gravity {}, and turbulent velocity {} km/s, closest\n'
        '\t   combination is metallicity: {}, effective temperature: {} K,\n'
        '\t   log-gravity {} and turbulent velocity of {} km/s.\n\n'
        '\t + Chosen model file to be used:\n\t\t{:s}.\n'.
          format(s_met, s_teff, s_grav, s_vturb, chosen_met, chosen_teff,
                 chosen_grav, chosen_vturb, chosen_filename))

    return chosen_filename, chosen_teff, chosen_grav, chosen_met,chosen_vturb


def PHOENIX_model_search(s_met, s_grav, s_teff, s_vturb):
    """
    Given input metallicities, gravities, effective temperature and
    microtiurbulent velocity, this function estimates which model is
    the most appropiate (i.e., the closer one in parameter space).
    If the model is not present in the system, it downloads it from
    the PHOENIX public library (phoenix.astro.physik.uni-goettingen.de).
    """

    if not os.path.exists('phoenix_models'):
       os.mkdir('phoenix_models')
       os.mkdir('phoenix_models/raw_models')

    model_path = 'phoenix_models/raw_models/'  # Path to the PHOENIX models

    # In PHOENIX models, all of them are computed with vturb = 2 km/2
    if(s_vturb==-1):
       print('\t    + No known turbulent velocity. Setting it to 2 km/s.')
       s_vturb = 2.0

    possible_mets = np.array([0.0, -0.5, -1.0, 1.0, -1.5, -2.0, -3.0, -4.0])

    if s_met not in possible_mets:
        # Now check closest metallicity model for input star:
        m_diff = np.inf
        chosen_met = np.inf
        for met in possible_mets:
            # Estimate distance between current and input metallicity:
            c_m_diff = np.abs(met-s_met)
            if(c_m_diff<m_diff):
                chosen_met = met
                m_diff = copy(c_m_diff)

        print('\t    + For input metallicity {}, closest value is {}.'.
              format(s_met, chosen_met))
    else:
        chosen_met = s_met

    # Generate the folder name:
    if chosen_met == 0.0:
        met_folder = 'm00'
        model = 'Z-0.0'
    else:
        abs_met = str(np.abs(chosen_met)).split('.')
        if chosen_met<0:
            met_folder = 'm'+abs_met[0]+abs_met[1]
            model = 'Z-'+abs_met[0]+abs_met[1]
        else:
            met_folder = 'p'+abs_met[0]+abs_met[1]
            model = 'Z+'+abs_met[0]+abs_met[1]

    chosen_met_folder = model_path + met_folder

    # Check if folder exists. If it does not, create it and download the
    # PHOENIX models that are closer in temperature and gravity to the
    # user input values:
    if not os.path.exists(chosen_met_folder):
       os.mkdir(chosen_met_folder)
    cwd = os.getcwd()
    os.chdir(chosen_met_folder)

    # See if in a past call the file list for the given metallicity was
    # saved; if not, retrieve it from the PHOENIX website:
    if os.path.exists('file_list.dat'):
       with open('file_list.dat') as f:
           all_files = f.readlines()
       for i in np.arange(len(all_files)):
           all_files[i] = all_files[i].strip()
    else:
       response = urllib.urlopen('ftp://phoenix.astro.physik.uni-goettingen.de/SpecIntFITS/PHOENIX-ACES-AGSS-COND-SPECINT-2011/'+model+'/')
       html = response.read()
       all_files = []
       while True:
            idx = html.find('lte')
            if(idx==-1):
                break
            else:
                idx2 = html.find('.fits')
                all_files.append(html[idx:idx2+5])
            html = html[idx2+5:]
       f = open('file_list.dat','w')
       for file in all_files:
            f.write(file+'\n')
       f.close()
    # Now check closest Teff for input star:
    t_diff = np.inf
    chosen_teff = np.inf
    for file in all_files:
        teff = np.double(file[3:8])
        c_t_diff = abs(teff-s_teff)
        if(c_t_diff<t_diff):
            chosen_teff = teff
            t_diff = c_t_diff

    print('\t    + For input effective temperature {:.1f} K, closest '
          'value is {:.0f} K.'.format(s_teff, chosen_teff))

    teff_files = []
    teff_string = "{:05.0f}".format(chosen_teff)
    for file in all_files:
        if teff_string in file:
            teff_files.append(file)

    # Now check closest gravity:
    grav_diff = np.inf
    chosen_grav = np.inf
    chosen_fname = ''
    for file in teff_files:
        grav = np.double(file[9:13])
        c_g_diff = abs(grav-s_grav)
        if(c_g_diff<grav_diff):
            chosen_grav = grav
            grav_diff = c_g_diff
            chosen_fname = file

    print('\t    + Checking if PHOENIX model file is on the system...')
    # Check if file is already downloaded. If not, download it from the PHOENIX website:
    if not os.path.exists(chosen_fname):
        print('\t    + Model file not found.')
        downloader('ftp://phoenix.astro.physik.uni-goettingen.de/SpecIntFITS/PHOENIX-ACES-AGSS-COND-SPECINT-2011/'+model+'/'+chosen_fname)
    else:
        print('\t    + Model file found.')

    os.chdir(cwd)
    chosen_path = chosen_met_folder + '/'+chosen_fname

    # Summary:
    print('\t + For input metallicity {}, effective temperature {} K, and\n'
          '\t   log-gravity {}, closest combination is metallicity: {},\n'
          '\t   effective temperature: {} K, and log-gravity {}\n\n'
          '\t + Chosen model file to be used:\n\t\t{:s}\n'.format(s_met, s_teff,
                 s_grav, chosen_met, chosen_teff, chosen_grav, chosen_fname))

    return chosen_path, chosen_teff, chosen_grav, chosen_met, s_vturb


def get_response(min_w, max_w, response_function):
    root = "response_functions/standard/"
    if(response_function.lower() == 'kphires'):
       response_file = root + "kepler_response_hires1.txt"
    elif(response_function.lower() == 'kplowres'):
       response_file = root + "kepler_response_lowres1.txt"
    elif(response_function.lower() == 'irac1'):
       response_file = root + "IRAC1_subarray_response_function.txt"
    elif(response_function.lower() == 'irac2'):
       response_file = root + "IRAC2_subarray_response_function.txt"
    elif(response_function.lower() == 'wfc3'):
       response_file = root + "WFC3_response_function.txt"
    else:
       if(os.path.exists("response_functions/"+response_function)):
          response_file = "response_functions/"+response_function
       else:
          print("Error: '{:s}' is not valid.".format(response_function))
          sys.exit()
    # Open the response file, which we assume has as first column wavelength and second column the response:
    w,r = np.loadtxt(response_file, unpack=True)
    if('kepler' in response_file):
         w = 10*w
         if min_w is None:
            min_w = min(w)
         if max_w is None:
            max_w = max(w)
         print('\t > Kepler response file detected.  Switch from '
               'nanometers to Angstroms.')
         print('\t > Minimum wavelength: {} A.\n'
               '\t > Maximum wavelength: {} A.'.format(min(w), max(w)))
    elif('IRAC' in response_file):
         w = 1e4*w
         if min_w is None:
            min_w = min(w)
         if max_w is None:
            max_w = max(w)
         print('\t > IRAC response file detected.  Switch from microns to '
               'Angstroms.')
         print('\t > Minimum wavelength: {} A.\n'
               '\t > Maximum wavelength: {} A.'.format(min(w), max(w)))
    else:
         if min_w is None:
            min_w = min(w)
         if max_w is None:
            max_w = max(w)

    # Fit a univariate spline with s=0 (i.e., a node in each data-point) and k = 1 (linear spline).
    S = si.UnivariateSpline(w,r,s=0,k=1)
    if type(min_w) is list:
       S_wav = []
       S_res = []
       for i in range(len(min_w)):
           c_idx = np.where((w>min_w[i])&(w<max_w[i]))[0]
           c_S_wav = np.append(np.append(min_w[i],w[c_idx]),max_w[i])
           c_S_res = np.append(np.append(S(min_w[i]),r[c_idx]),S(max_w[i]))
           S_wav.append(np.copy(c_S_wav))
           S_res.append(np.copy(c_S_res))
    else:
       idx = np.where((w>min_w)&(w<max_w))[0]
       S_wav = np.append(np.append(min_w,w[idx]),max_w)
       S_res = np.append(np.append(S(min_w),r[idx]),S(max_w))

    return min_w, max_w, S_wav, S_res


def read_ATLAS(chosen_filename, model):
    # Define the ATLAS grid in mu = cos(theta):
    mu = np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.25,
                   0.2, 0.15, 0.125, 0.1, 0.075, 0.05, 0.025, 0.01])
    if(model != 'A100'):
       mu100 = np.array([])
    else:
       mu100 = np.arange(0.01, 1.01, 0.01)

    # Now prepare files and read data from the ATLAS models:
    with open(chosen_filename, 'r') as f:
      lines = f.readlines()
    # Remove comments and blank lines:
    for i in np.flipud(np.arange(len(lines))):
        if lines[i].strip() == "" or lines[i].strip().startswith("#"):
            lines.pop(i)

    nwave = len(lines)
    wavelengths = np.zeros(nwave)
    intensities = np.zeros((nwave, len(mu)))
    I100        = np.zeros((nwave, len(mu100)))
    for i in np.arange(nwave):
        # If no jump of line or comment, save the intensities:
        splitted = lines[i].split()
        if(len(splitted)==18):
            wavelengths[i] = np.double(splitted[0])*10  # nano to angstrom
            intensities[i] = np.array(splitted[1:], np.double)                     # Get the intensities.
            ndigits = len(str(int(intensities[i,1])))
            # Only if I(1) is different from zero, fit the LDs:
            if(intensities[i,0]!=0.0):
                # Kurucz doesn't put points on his files (e.g.: 0.8013 is 8013).
                intensities[i,1:] = intensities[i,1:]/1e5
                # Normalzie intensities wrt the first one:
                intensities[i,1:] = intensities[i,1:]*intensities[i,0]
                # If requested, extract the 100 mu-points, with cubic spline
                # interpolation (k=3) through all points (s=0) as CB11:
                if(model == 'A100'):
                    II = si.UnivariateSpline(mu[::-1], intensities[i,::-1],
                                             s=0, k=3)
                    I100[i] = II(mu100)

    # Select only those with non-zero intensity:
    flag = intensities[:,0] != 0.0
    return wavelengths[flag], intensities[flag], I100[flag], mu, mu100


def read_PHOENIX(chosen_path):
    mu = fits.getdata(chosen_path, 'MU')
    data = fits.getdata(chosen_path)
    CDELT1 = fits.getval(chosen_path, 'CDELT1')
    CRVAL1 = fits.getval(chosen_path, 'CRVAL1')
    wavelengths = np.arange(data.shape[1]) * CDELT1 + CRVAL1
    I = data.transpose()
    return wavelengths, I, mu


def integrate_response_ATLAS(wavelengths, I, I100, mu, mu100, S_res, S_wav,
                             atlas_correction, photon_correction,
                             interpolation_order, model):
    # Define the number of mu angles at which we will perform the integrations:
    if model == "A100":
      nmus = len(mu100)
    else:
      nmus = len(mu)

    # Integrate intensity through each angle:
    I_l = np.array([])
    for i in range(nmus):
        # Interpolate the intensities:
        if(model == "A100"):
            Ifunc = si.UnivariateSpline(wavelengths, I100[:,i], s=0, k=interpolation_order)
        else:
            Ifunc = si.UnivariateSpline(wavelengths, I[:,i], s=0, k=interpolation_order)
        # If several wavelength ranges where given, integrate through each chunk one at a time.
        # If not, integrate the given chunk:
        if type(S_res) is list:
            integration_results = 0.0
            for j in range(len(S_res)):
                if atlas_correction and photon_correction:
                    integrand = (S_res[j]*Ifunc(S_wav[j])) / S_wav[j]
                elif atlas_correction and not photon_correction:
                    integrand = (S_res[j]*Ifunc(S_wav[j])) / (S_wav[j]**2)
                elif not atlas_correction and photon_correction:
                    integrand = (S_res[j]*Ifunc(S_wav[j])) * (S_wav[j])
                else:
                    integrand = S_res[j]*Ifunc(S_wav[j])*S_wav[j]
                integration_results = integration_results + np.trapz(integrand, x=S_wav[j])
        else:
            if atlas_correction and photon_correction:
                integrand = (S_res*Ifunc(S_wav)) / S_wav
            elif atlas_correction and not photon_correction:
                integrand = (S_res*Ifunc(S_wav)) / (S_wav**2)
            elif not atlas_correction and photon_correction:
                integrand = S_res*Ifunc(S_wav) * S_wav
            else:
                integrand = S_res*Ifunc(S_wav)
            integration_results = np.trapz(integrand, x=S_wav)
        I_l = np.append(I_l,integration_results)

    # Normalize profile with respect to I(mu = 1):
    if(model == "A100"):
      I0 = I_l/(I_l[-1])
    else:
      I0 = I_l/(I_l[0])

    return I0


def integrate_response_PHOENIX(wavelengths, I, mu, S_res, S_wav, correction,
                               interpolation_order):
    I_l = np.array([])
    for i in range(len(mu)):
        Ifunc = si.UnivariateSpline(wavelengths, I[:,i], s=0,
                                    k=interpolation_order)
        if type(S_res) is list:
            integration_results = 0.0
            for j in range(len(S_res)):
                if correction:
                    integrand = S_res[j]*Ifunc(S_wav[j])*S_wav[j]
                else:
                    integrand = S_res[j]*Ifunc(S_wav[j])
                integration_results = integration_results + \
                                      np.trapz(integrand, x=S_wav[j])

        else:
            integrand = S_res * Ifunc(S_wav)  #lambda x,I,S: I(x)*S(x)
            if correction:
                integrand *= S_wav            #lambda x,I,S: (I(x)*S(x))*x
            # Integral of Intensity_nu*(Response Function*lambda)*c/lambda**2
            integration_results = np.trapz(integrand, x=S_wav)
        I_l = np.append(I_l,integration_results)

    return I_l/(I_l[-1])


def get_rmax(mu, I0):
    # Apply correction due to spherical extension. First, estimate the r:
    r = np.sqrt(1.0-(mu**2))
    # Estimate the derivatives at each point:
    rPi, m = get_derivatives(r,I0)
    # Estimate point of maximum (absolute) derivative:
    idx_max = np.argmax(np.abs(m))
    r_max = rPi[idx_max]
    # To refine this value, take 20 points to the left and 20 to the right
    # of this value, generate spline and search for roots:
    ndata = 20
    r_maxes = rPi[idx_max-ndata:idx_max+ndata]
    m_maxes = m[idx_max-ndata:idx_max+ndata]
    spl = si.UnivariateSpline(r_maxes[::-1],m_maxes[::-1],s=0,k=4)
    fine_r_max = spl.derivative().roots()
    if(len(fine_r_max)>1):
       abs_diff = np.abs(fine_r_max-r_max)
       iidx_min = np.where(abs_diff == np.min(abs_diff))[0]
       fine_r_max = fine_r_max[iidx_min]
    return r,fine_r_max


def get100_PHOENIX(wavelengths, I, new_mu, idx_new):
    mu100 = np.arange(0.01,1.01,0.01)
    I100 = np.zeros((len(wavelengths),len(mu100)))
    for i in range(len(wavelengths)):
        # Cubic splines (k=3), interpolation through all points (s=0) ala CB11.
        II = si.UnivariateSpline(new_mu,I[i,idx_new],s=0,k=3)
        I100[i] = II(mu100)
    return mu100, I100


def calc_lds(name, response_function, model, atlas_correction,
             photon_correction, s_met, s_grav, s_teff, s_vturb,
             min_w=None, max_w=None, fout=None):
    """
    Generate the limb-darkening coefficients.  Note that response_function
    can be a string with the filename of a response function not in the
    list. The file has to be in the response_functions folder.

    INPUTS:
     name: String
        Name of the object we are working on.
     response_function:
        Number of a standard response function or filename of a response
        function under the response_functions folder.
     model:
        Model atmosphere to be used.
     atlas_correction:
        True if corrections in the integrand of the ATLAS models should
        be applied (i.e., transformation of ATLAS intensities given in
        frequency to per wavelength)
     photon_correction:
        If True, correction for photon-counting devices is used.
     s_met:
        Metallicity of the star.
     s_grav:
        log_g of the star (cgs).
     s_teff:
        Effective temperature of the star (K).
     s_vturb:
        Turbulent velocity in the star (km/s)
     min_w: Float
        Minimum wavelength to integrate (if None, use the minimum wavelength
        of the response function).
     max_w: Float
        Maximum wavelength to integrate (if None, use the maximum wavelength
        of the response function).
     fout: FILE
        If not None, file where to save the LDCs.

    OUTPUTS
       LDC: 1D float tuple
          The linear (a), quadratic (u1, u2), three-parameter (b1, b2, b3),
          non-linear (c1, c2, c3, c4), logarithmic (l1, l2),
          exponential (e1, e2), and square-root laws (s1, s2).
    """

    print('\n\t Reading response functions\n\t --------------------------')

    # Get the response file minimum and maximum wavelengths and all the wavelengths and values:
    min_w, max_w, S_wav, S_res = get_response(min_w, max_w, response_function)

    ######################################################################
    # IF USING ATLAS MODELS....
    ######################################################################
    if 'A' in model:
        # Search for best-match ATLAS9 model for the input stellar parameters:
        print('\n\t ATLAS modelling\n\t ---------------\n'
              '\t > Searching for best-match Kurucz model ...')
        chosen_filename, chosen_teff, chosen_grav, chosen_met, \
           chosen_vturb = ATLAS_model_search(s_met, s_grav, s_teff, s_vturb)

        # Read wavelengths and intensities (I) from ATLAS models.
        # If model is "A100", it also returns the interpolated
        # intensities (I100) and the associated mu values (mu100).
        # If not, those arrays are empty:
        wavelengths, I, I100, mu, mu100 = read_ATLAS(chosen_filename, model)

        # Now use these intensities to obtain the (normalized) integrated
        # intensities with the response function:
        I0 = integrate_response_ATLAS(wavelengths, I, I100, mu, mu100, S_res,
                                    S_wav, atlas_correction, photon_correction,
                                    interpolation_order, model)

        # Finally, obtain the limb-darkening coefficients:
        if(model == "AS"):
            # Save indexes which apply to Sing's (2010) criterion:
            idx_sing = np.where(mu>=0.05)[0]
            c1,c2,c3,c4 = fit_non_linear(mu,I0)
            a = fit_linear(mu[idx_sing],I0[idx_sing])
            u1,u2 = fit_quadratic(mu[idx_sing],I0[idx_sing])
            b1,b2,b3 = fit_three_parameter(mu[idx_sing],I0[idx_sing])
            l1,l2 = fit_logarithmic(mu[idx_sing],I0[idx_sing])
            e1,e2 = fit_exponential(mu[idx_sing],I0[idx_sing])
            s1,s2 = fit_square_root(mu[idx_sing],I0[idx_sing])
        elif(model == "A100"):
            c1,c2,c3,c4 = fit_non_linear(mu100,I0)
            a = fit_linear(mu100,I0)
            u1,u2 = fit_quadratic(mu100,I0)
            b1,b2,b3 = fit_three_parameter(mu100,I0)
            l1,l2 = fit_logarithmic(mu100,I0)
            e1,e2 = fit_exponential(mu100,I0)
            s1,s2 = fit_square_root(mu100,I0)
        else:
            c1,c2,c3,c4 = fit_non_linear(mu,I0)
            a = fit_linear(mu,I0)
            u1,u2 = fit_quadratic(mu,I0)
            b1,b2,b3 = fit_three_parameter(mu,I0)
            l1,l2 = fit_logarithmic(mu,I0)
            e1,e2 = fit_exponential(mu,I0)
            s1,s2 = fit_square_root(mu,I0)

    ######################################################################
    # IF USING PHOENIX MODELS....
    ######################################################################

    elif 'P' in model:
        # Search for best-match PHOENIX model for the input stellar parameters:
        print('\n\t PHOENIX modelling\n\t -----------------\n'
              '\t > Searching for best-match PHOENIX model ...')
        chosen_path, chosen_teff, chosen_grav, chosen_met, \
           chosen_vturb = PHOENIX_model_search(s_met, s_grav, s_teff, s_vturb)

        # Read PHOENIX model wavelenghts, intensities and mus:
        wavelengths, I, mu = read_PHOENIX(chosen_path)

        # Now use these intensities to obtain the (normalized) integrated
        # intensities with the response function:
        I0 = integrate_response_PHOENIX(wavelengths, I, mu, S_res, S_wav,
                                        photon_correction, interpolation_order)

        # Obtain correction due to spherical extension. First, get r_max:
        r, fine_r_max = get_rmax(mu, I0)

        # Now get r for each intensity point and leave out those that have r>1:
        new_r = r/fine_r_max
        idx_new = np.where(new_r<=1.0)[0]
        new_r = new_r[idx_new]
        new_mu = np.sqrt(1.0-(new_r**2))
        new_I0 = I0[idx_new]

        # Now, if the model requires it, obtain 100-mu points interpolated
        # in this final range of "usable" intensities:
        if(model == 'P100'):
            mu100, I100 = get100_PHOENIX(wavelengths, I, new_mu, idx_new)
            I0_100 = integrate_response_PHOENIX(wavelengths, I100, mu100,
                        S_res, S_wav, photon_correction, interpolation_order)

       # Now define each possible model and fit LDs:
        if(model == 'PQS'): # Quasi-spherical model, as defined by Claret et al. (2012), mu>=0.1
            idx = np.where(new_mu>=0.1)[0]
            c1,c2,c3,c4 = fit_non_linear(new_mu[idx],new_I0[idx])
            a = fit_linear(new_mu[idx],new_I0[idx])
            u1,u2 = fit_quadratic(new_mu[idx],new_I0[idx])
            b1,b2,b3 = fit_three_parameter(new_mu[idx],new_I0[idx])
            l1,l2 = fit_logarithmic(new_mu[idx],new_I0[idx])
            e1,e2 = fit_exponential(new_mu[idx],new_I0[idx])
            s1,s2 = fit_square_root(new_mu[idx],new_I0[idx])
        elif(model == 'PS'): # Sing method:
            idx = np.where(new_mu>=0.05)[0]
            c1,c2,c3,c4 = fit_non_linear(new_mu,new_I0)
            a = fit_linear(new_mu[idx],new_I0[idx])
            u1,u2 = fit_quadratic(new_mu[idx],new_I0[idx])
            b1,b2,b3 = fit_three_parameter(new_mu[idx],new_I0[idx])
            l1,l2 = fit_logarithmic(new_mu[idx],new_I0[idx])
            e1,e2 = fit_exponential(new_mu[idx],new_I0[idx])
            s1,s2 = fit_square_root(new_mu[idx],new_I0[idx])
        elif(model == 'P100'):
            c1,c2,c3,c4 = fit_non_linear(mu100,I0_100)
            a = fit_linear(mu100,I0_100)
            u1,u2 = fit_quadratic(mu100,I0_100)
            b1,b2,b3 = fit_three_parameter(mu100,I0_100)
            l1,l2 = fit_logarithmic(mu100,I0_100)
            e1,e2 = fit_exponential(mu100,I0_100)
            s1,s2 = fit_square_root(mu100,I0_100)
        else:
            c1,c2,c3,c4 = fit_non_linear(new_mu,new_I0)
            a = fit_linear(new_mu,new_I0)
            u1,u2 = fit_quadratic(new_mu,new_I0)
            b1,b2,b3 = fit_three_parameter(new_mu,new_I0)
            l1,l2 = fit_logarithmic(new_mu,new_I0)
            e1,e2 = fit_exponential(new_mu,new_I0)
            s1,s2 = fit_square_root(new_mu,new_I0)

    # Stack all LD coefficients into one single tuple:
    LDC = a, u1, u2, b1, b2, b3, c1, c2, c3, c4, l1, l2, e1, e2, s1, s2

    # Save to the file:
    if fout is not None:
        fout.write(70*"#" + "\n")
        fout.write("{:s}  {:s}  {:s}\nTeff={:.1f}K  log(g)={:.1f}  "
                   "[M/H]={:.1f}  vturb={:.1f}\n\n".format(name, model,
                                  response_function, chosen_teff, chosen_grav,
                                  chosen_met, chosen_vturb))
        fout.write("a = {:12.8f}\n"
                   "u1, u2 = {:11.8f}, {:11.8f}\n"
                   "b1, b2, b3 = {:11.8f}, {:11.8f}, {:11.8f}\n"
                   "c1, c2, c3, c4 = {:11.8f}, {:11.8f}, {:11.8f}, {:11.8f}\n"
                   "l1, l2 = {:11.8f}, {:11.8f}\n"
                   "e1, e2 = {:11.8f}, {:11.8f}\n"
                   "s1, s2 = {:11.8f}, {:11.8f}\n\n".format(*LDC))

    print('\t > Done! \n\t {:s}\n'.format(70*'#'))
    return LDC


def lds(ifile=None, ofile=None):
    """
    Compute limb-darkening coefficients.

    Parameters
    ----------
    ifile: String
       Filename with the user inputs.
    ofile: String
       If not None, filename where to write the LCDs.

    Returns
    -------
    LDC: 1D float tuple
       The linear (a), quadratic (u1, u2), three-parameter (b1, b2, b3),
       non-linear (c1, c2, c3, c4), logarithmic (l1, l2),
       exponential (e1, e2), and square-root laws (s1, s2).
    """
    print('\n\t ##########################################################\n'
          '\n\t             Limb Darkening Calculations {:s}\n'
          '\n\t      Author: Nestor Espinoza (nespino@astro.puc.cl)\n'
          '\n\t DISCLAIMER: If you make use of this code for your research,\n'
          '\t please consider citing Espinoza & Jordan (2015)\n'
          '\n\t ##########################################################'.
           format(version))

    if ofile is None:
      fout = None
    else:
        fout = open('results/'+ofile, 'w')
        fout.write(70*"#" + "\n"
         "#\n# Limb Darkening Calculations {}\n"
         "#\n# Limb-darkening coefficients for linear (a), quadratic (u1,u2),\n"
            "# three parameter (b1,b2,b3), non-linear (c1,c2,c3,c4),\n"
            "# logarithmic (l1,l2), exponential (e1,e2), "
               "and square-root laws (s1,s2).\n"
         "#\n# Author:       Nestor Espinoza (nespino@astro.puc.cl) \n"
         "#\n# Contributors: Benjamin Rackham (brackham@email.arizona.com) \n"
            "#               Andres Jordan    (ajordan@astro.puc.cl) \n"
            "#               Ashley Villar    (vvillar@cfa.harvard.edu) \n"
         "#\n# DISCLAIMER: If you make use of this code for your research,\n"
            "#          please consider citing Espinoza & Jordan (2015).\n\n".
             format(version))

    f = open(ifile, 'r')
    while True:
      line = f.readline()
      if line == '':
         break
      elif line[0] != '#':
         splitted = line.split('\t')
         if len(splitted) == 1:
            # If split is not done with tabs, but spaces:
            splitted = line.split()
         name = fix_spaces(splitted[0])
         teff = splitted[1]
         grav = splitted[2]
         met = splitted[3]
         vturb = splitted[4]
         RF = fix_spaces(splitted[5])
         FT = fix_spaces(splitted[6])
         min_w = splitted[7]
         min_w = np.double(min_w)
         max_w = splitted[8]
         max_w = np.double(max_w.split('\n')[0])
         s_teff = np.double(teff)
         s_grav = np.double(grav)
         s_vturb = np.double(vturb)
         s_met = np.double(met)
         if(min_w == -1 or max_w == -1):
            min_w = None
            max_w = None
         response_function = RF
         models = FT.split(',')
         for model in models:
             LDC = calc_lds(name, response_function, model,
                            atlas_correction, photon_correction, s_met, s_grav,
                            s_teff, s_vturb, min_w, max_w, fout)
    if ofile is not None:
        fout.close()
        print('\t > Program finished without problems.\n'
              '\t   The results were saved in the "results" folder.\n')
    return LDC

if __name__ == "__main__":
    ifile, ofile = parse()
    lds(ifile=ifile, ofile=ofile)
