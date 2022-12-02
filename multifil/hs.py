#!/usr/bin/env python
# encoding: utf-8
"""
hs.py - A half-sarcomere model with multiple thick and thin filaments

Created by Dave Williams on 2009-12-31.
"""
import matplotlib.pyplot as plt
import multiprocessing as mp
import sys
import time

import scipy.sparse as sparse

from scipy.sparse.linalg import spsolve 
import scipy
# import torch 

import numpy.linalg
from numba import njit
import numpy as np

import math as m
from . import af
from . import mf
from . import ti

import pdb

def expm_(a, q=2):
    
    '''
    Matrix exponential, based on squaring/scaling pade approximation
    
    '''
    # q = 6
    a2 = a.copy()
    a_norm = np.max(np.sum(np.abs(a2), axis=1)) #np.linalg.norm ( a2, ord = np.inf )
    ee = ( int ) ( np.log2 ( a_norm ) ) + 1
    # exp(A) = (exp(A/s))**s
    s = max ( 0, ee + 1 )
    a2 = a2 / ( 2.0 ** s )
    x = a2.copy()
    I = np.broadcast_to(np.eye(a.shape[1]), a.shape)
    c = .5
    e = I + c*a2
    d = I - c*a2
    p=True

    for k in range ( 2, q + 1 ):
        c = c * float ( q - k + 1 ) / float ( k * ( 2 * q - k + 1 ) )
        # x = np.einsum('mij,mjk->mik',a2,x)
        x = numba_dot(a2,x)
        e = e + c * x
        
        if ( p ):
            d = d + c * x
        else:
            d = d - c * x
        p = not p
    
    # we want e/d - Pade spporximataiton is ratio of two polynomials, e and d
    e = np.linalg.solve(d,e)
    
    # square TO GEt back answer
    e = np.linalg.matrix_power(e, 2**s)

    # for k in range ( 0, s ):
    #     e = numba_dot( e, e )
        
    return e


@njit
def numba_dot(a,b):
    x = np.empty(shape=a.shape)
    for i in range(0,a.shape[0]):
        x[i,:,:] = np.dot(a[i],b[i])
    return x
    
@njit
def cart2pol(x, y):
    '''
    cartesian to polar coordinates
    '''
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)
        

class hs:
    """The half-sarcomere and ways to manage it"""
    VALID_PARAMS = {}
    for component in [mf.ThickFilament, mf.mh.Crossbridge,
                      af.ThinFilament, af.tm.TmSite,
                      ti.Titin]:
        VALID_PARAMS.update(component.VALID_PARAMS)

    def __init__(self, lattice_spacing=None, z_line=None, poisson=None,
                 pCa=None, timestep_len=1, time_dependence=None, starts=None, temp=26.15, **kwargs):
        """ Create the data structure that is the half-sarcomere model

        Parameters:
            lattice_spacing: the surface-to-surface distance (14.0)
            z_line: the length of the half-sarcomere (1250)
            poisson: poisson ratio obeyed when z-line changes. Significant
                values are:
                    * 0.5 - constant volume
                    * 0.0 - constant lattice spacing, default value
                    * any negative value - auxetic lattice spacing
            pCa: pCa, controlling tropomyosin movement and thus binding site
                availability, positive by convention (4.0)
            timestep_len: how many ms per timestep (1)
            time_dependence: a dictionary to override the initial lattice
                spacing, sarcomere length, and pCa at each
                timestep. Each key may contain a list of the values, to be
                iterated over as timesteps proceed. The first entry in these
                lists will override passed initial values. The valid keys
                time_dependence can control are:
                    * "lattice_spacing"
                    * "z_line"
                    * "pCa"
            starts: starting polymer/orientation for thin/thick filaments in
                form ((rand(0,25), ...), (rand(0,3), ...))
        Returns:
            None

        This is the organizational basis that the rest of the model, and
        classes representing the other structures therein will use when
        running. It contains the following properties:

        ## Half-sarcomere properties: these are properties that can be
        interpreted as belonging to the overall model, not to any thick or
        thin filament.

        lattice_spacing:
            the face to face lattice spacing for the whole model
        m_line:
            x axis location of the m line
        h_line:
            x axis location of the h line
        hiding_line:
            x axis location below which actin sites are hidden by actin
            overlap (crossing through the m-line from adjacent half sarc)

        ## Thick Filament Properties: each is a tuple of thick filaments
        (filament_0, filament_1, filament_2, filament_3) where each
        filament_x is giving the actual properties of that particular
        filament.

        thick_location:
            each tuple location is a list of x axis locations
        thick_crowns:
            each tuple location is a tuple of links to crown instances
        thick_link:
            each tuple location is a list consisting of three (one for each
            myosin head in the crown) of either None or a link to a thin_site
        thick_adjacent:
            each tuple location is a tuple of links to adjacent thin filaments
        thick_face:
            each tuple location is a tuple of length six, each location of
            which contains a tuple of links to myosin heads that are facing
            each surrounding thin filament
        thick_bare_zone:
            a single value, the length of each filament before the first crown
        thick_crown_spacing:
            a single value, the distance between two crowns on a single filament
        thick_k:
            a single value, the spring constant of the thick filament between
            any given pair of crowns

        ## Thin Filament Properties: arranged in the same manner as the
        thick filament properties, but for the eight thin filaments

        thin_location:
            each tuple location is a list of x axis locations
        thin_link:
            each tuple location is a list consisting of entries (one for each
            thin_site on the thin_filament) of either a None or a link to a
            thick_crown
        thin_adjacent:
            each tuple location is a tuple of links to adjacent thick filaments
        thin_face:
            each tuple location is a tuple of length three, each location of
            which contains a tuple of links to thin filament sites that are
            facing each surrounding thick filament
        thin_site_spacing:
            the axial distance from one thin filament binding site to another
        thin_k:
            a single value, the spring constant of the thin filament between
            any given pair of thin binding sites

        """
        # Versioning, to be updated when backwards incompatible changes to the
        # data structure are made, not on release of new features
        self.version = 1.4  # Includes support for tropomyosin AND titin

        """ ## Handle Kwargs ## """  # =================================================================================
        # TODO add specifications
        """
        # Syntax for profiles:
        #

        # Syntax for multiple profiles
        # module _ iso (isomer)
        """
        
        # Titin constants
        # Isomer-available
        valid_ti_params = ti.Titin.VALID_PARAMS
        if 'ti_iso' in kwargs.keys():
            for param in valid_ti_params:
                assert param not in kwargs.keys(), "ti_iso cannot be set at the same time as ti parameters"
            ti_params = {"ti_iso": kwargs.pop('ti_iso')}

            # check profiles
            profiles = ti_params['ti_iso']
            total_p = 0
            for profile in profiles:
                total_p += profile['iso_p']
            assert 1.0 - total_p < 0.001,\
                "Please enter TITIN isomer probabilities ROUNDED to the nearest -tenth- of a -percent-."
        else:
            ti_params = {}
            for param in valid_ti_params:
                if param in kwargs.keys():
                    ti_params[param] = kwargs.pop(param)

        # Tropomyosin/Troponin constants
        # Isomer-available
        valid_tm_params = af.tm.TmSite.VALID_PARAMS
        if 'tm_iso' in kwargs.keys():
            for param in valid_tm_params:
                assert param not in kwargs.keys(), "tm_iso cannot be set at the same time as tm parameters"
            af_params = {"tm_iso": kwargs.pop('tm_iso')}

            # check profiles
            profiles = af_params['tm_iso']
            total_p = 0
            for profile in profiles:
                total_p += profile['iso_p']
            assert 1.0 - total_p < 0.001, \
                "Please enter TROPOMYOSIN isomer probabilities ROUNDED to the nearest -tenth- of a -percent-."
        else:
            tm_params = {}
            for param in valid_tm_params:
                if param in kwargs.keys():
                    tm_params[param] = kwargs.pop(param)
            af_params = {'tm_params': tm_params}

        # Actin thin filament constants
        # TODO add isomer support - tricky because of the way that the filament is constructed
        # af_params = {} # constructed in tropomyosin parameter logic
        valid_af_params = af.ThinFilament.VALID_PARAMS
        for param in valid_af_params:
            if param in kwargs.keys():
                af_params[param] = kwargs.pop(param)

        # Crossbridge constants
        valid_mh_params = mf.mh.Crossbridge.VALID_PARAMS
        if 'mh_iso' in kwargs.keys():
            for param in valid_mh_params:
                assert param not in kwargs.keys(), "mh_iso cannot be set at the same time as mh parameters"
            mf_params = {"mh_iso": kwargs.pop('mh_iso')}

            # check profiles
            profiles = mf_params['mh_iso']
            total_p = 0
            for profile in profiles:
                total_p += profile['iso_p']
            assert 1.0 - total_p < 0.001, \
                "Please enter MYOSIN_HEAD isomer probabilities ROUNDED to the nearest -tenth- of a -percent-."
        else:
            mh_params = {}
            for param in valid_mh_params:
                if param in kwargs.keys():
                    mh_params[param] = kwargs.pop(param)
            mf_params = {"mh_params": mh_params}

        # ## Myosin Thick filament constants
        # TODO add isomer support - tricky because of the way that the filament is constructed
        # mf_params = {} # constructed in crossbridge parameter logic
        valid_mf_params = mf.ThickFilament.VALID_PARAMS
        for param in valid_mf_params:
            if param in kwargs.keys():
                mf_params[param] = kwargs.pop(param)

        # print undigested kwargs
        for key in kwargs.keys():
            print("Unknown Kwarg:", key)
        """ ## Finished Handling Kwargs ## """  # ======================================================================

        # Parse initial LS and Z-line
        if time_dependence is not None:
            if 'lattice_spacing' in time_dependence:
                lattice_spacing = time_dependence['lattice_spacing'][0]
            if 'z_line' in time_dependence:
                z_line = time_dependence['z_line'][0]
            if 'pCa' in time_dependence:
                pCa = time_dependence['pCa'][0]
        self.time_dependence = time_dependence
        # The next few lines use detection of None rather than a sensible
        # default value as a passed None is an explicit selection of default
        if lattice_spacing is None:
            lattice_spacing = 14.0
        if z_line is None:
            z_line = 1250   # nm
        if pCa is None:
            pCa = 4.0
        if poisson is None:
            poisson = 0.0
        # Record initial values for use with poisson driven ls
        self._initial_z_line = z_line
        self._initial_lattice_spacing = lattice_spacing
        self.poisson_ratio = poisson
        # Store these values for posterity
        self.lattice_spacing = lattice_spacing
        self.z_line = z_line
        self.pCa = pCa
        self.ca = 10**(-pCa)
        # Create the thin filaments, unlinked but oriented on creation.
        thin_orientations = ([4, 0, 2], [3, 5, 1], [4, 0, 2], [3, 5, 1],
                             [3, 5, 1], [4, 0, 2], [3, 5, 1], [4, 0, 2])
        np.random.seed(None)
        # This gives 25 random starts in orientation to the thin
        if starts is None:
            thin_starts = [np.random.randint(25) for _ in thin_orientations]
        else:
            thin_starts = starts[0]
        self._thin_starts = thin_starts
        thin_ids = range(len(thin_orientations))
        new_thin = lambda thin_id: af.ThinFilament(self, thin_id, thin_orientations[thin_id],
                                                   thin_starts[thin_id], **af_params)
        self.thin = tuple([new_thin(thin_id) for thin_id in thin_ids])
        # Determine the hiding line
        self.hiding_line = None
        self.update_hiding_line()

        '''Begin connecting things'''
        # Create the thick filaments, remembering they are arranged thus:
        # ----------------------------
        # |   Actin around myosin    |
        # |--------------------------|
        # |      a1      a3          |
        # |  a0      a2      a0      |
        # |      M0      M1          |
        # |  a4      a6      a4      |
        # |      a5      a7      a5  |
        # |          M2      M3      |
        # |      a1      a3      a1  |
        # |          a2      a0      |
        # ----------------------------
        # and that when choosing which actin face to link to which thick
        # filament face, use these face orders:
        # ----------------------------------------------------
        # | Myosin face order  |       Actin face order      |
        # |--------------------|-----------------------------|
        # |         a1         |                             |
        # |     a0      a2     |  m0      m1         m0      |
        # |         mf         |      af      OR             |
        # |     a5      a3     |                     af      |
        # |         a4         |      m2         m2      m1  |
        # ----------------------------------------------------
        if starts is None:
            thick_starts = [np.random.randint(1, 4) for _ in range(4)]
        else:
            thick_starts = starts[1]
        self._thick_starts = thick_starts
        self.thick = (
            mf.ThickFilament(self, 0, (
                self.thin[0].thin_faces[1], self.thin[1].thin_faces[2],
                self.thin[2].thin_faces[2], self.thin[6].thin_faces[0],
                self.thin[5].thin_faces[0], self.thin[4].thin_faces[1]),
                             thick_starts[0], **mf_params),
            mf.ThickFilament(self, 1, (
                self.thin[2].thin_faces[1], self.thin[3].thin_faces[2],
                self.thin[0].thin_faces[2], self.thin[4].thin_faces[0],
                self.thin[7].thin_faces[0], self.thin[6].thin_faces[1]),
                             thick_starts[1], **mf_params),
            mf.ThickFilament(self, 2, (
                self.thin[5].thin_faces[1], self.thin[6].thin_faces[2],
                self.thin[7].thin_faces[2], self.thin[3].thin_faces[0],
                self.thin[2].thin_faces[0], self.thin[1].thin_faces[1]),
                             thick_starts[2], **mf_params),
            mf.ThickFilament(self, 3, (
                self.thin[7].thin_faces[1], self.thin[4].thin_faces[2],
                self.thin[5].thin_faces[2], self.thin[1].thin_faces[0],
                self.thin[0].thin_faces[0], self.thin[3].thin_faces[1]),
                             thick_starts[3], **mf_params)
        )
        # Now the thin filaments need to be linked to thick filaments, use
        # the face orders from above and the following arrangement:
        # ----------------------------
        # |   Myosin around actin    |
        # |--------------------------|
        # |      m3      m2      m3  |
        # |          A1      A3      |
        # |      A0      A2          |
        # |  m1      m0      m1      |
        # |      A4      A6          |
        # |          A5      A7      |
        # |      m3      m2      m3  |
        # ----------------------------
        # The following may be hard to read, but it has been checked and
        # may be moderately trusted. CDW-20100406
        self.thin[0].set_thick_faces((self.thick[3].thick_faces[4],
                                      self.thick[0].thick_faces[0], self.thick[1].thick_faces[2]))
        self.thin[1].set_thick_faces((self.thick[3].thick_faces[3],
                                      self.thick[2].thick_faces[5], self.thick[0].thick_faces[1]))
        self.thin[2].set_thick_faces((self.thick[2].thick_faces[4],
                                      self.thick[1].thick_faces[0], self.thick[0].thick_faces[2]))
        self.thin[3].set_thick_faces((self.thick[2].thick_faces[3],
                                      self.thick[3].thick_faces[5], self.thick[1].thick_faces[1]))
        self.thin[4].set_thick_faces((self.thick[1].thick_faces[3],
                                      self.thick[0].thick_faces[5], self.thick[3].thick_faces[1]))
        self.thin[5].set_thick_faces((self.thick[0].thick_faces[4],
                                      self.thick[2].thick_faces[0], self.thick[3].thick_faces[2]))
        self.thin[6].set_thick_faces((self.thick[0].thick_faces[3],
                                      self.thick[1].thick_faces[5], self.thick[2].thick_faces[1]))
        self.thin[7].set_thick_faces((self.thick[1].thick_faces[4],
                                      self.thick[3].thick_faces[0], self.thick[2].thick_faces[2]))
        # Create the titin filaments and link them from thick
        # faces to thin faces
        # |--------------------------------------------------|
        # |            Actin & titin around myosin           |
        # |--------------------------------------------------|
        # |           a1               a3                    |
        # |                                                  |
        # |  a0       t1      a2       t4       a0           |
        # |       t0     t2        t3      t5                |
        # |           M0               M1                    |
        # |       t6     t8        t9      t11               |
        # |  a4       t7      a6       t10      a4           |
        # |                                                  |
        # |           a5     t13       a7       t16      a5  |
        # |               t12    t14        t15    t17       |
        # |                   M2                M3           |
        # |               t18    t20        t21    t23   a1  |
        # |           a1      t19      a3       t22          |
        # |                                                  |
        # |                   a2                a0           |
        # |--------------------------------------------------|
        # ## CHECK_JDP ## Link Thick filament to titin
        # ## Checked - AMA 13JUN19
        """ titin connection loop format:
        link_list = []
        num = 0
        an_list = [0, 1, 2, 2, 3, 0]
        for half in range(0, 2):
            mf_list = [0, 1, 2]
            af_list = [1, 2, 2]

            for quart in range(0, 2):
                for eighth in range(0, 2):
                    for triple in range(0, 3):
                        m_n = eighth + half * 2
                        m_f = mf_list[triple]
                        a_n = an_list[triple + eighth * 3]
                        a_f = af_list[triple]
                        link_list.append((num, m_n, m_f, a_n, a_f))
                        num += 1
                mf_list = [5, 4, 3]
                af_list = [1, 0, 0]
                for i in range(0, len(an_list)):
                    an_list[i] = an_list[i] - half * 8 + 4
            an_list = [5, 6, 7, 7, 4, 5]

        for item in link_list:
            print("ti.Titin(self, " + str(item[0]) + ", ti_thick(" +
                  str(item[1]) + ", " + str(item[2]) + "), ti_thin(" +
                  str(item[3]) + ", " + str(item[4]) + "), **ti_params),")
        # """

        ti_thick = lambda thick_i, j: self.thick[thick_i].thick_faces[j]
        ti_thin = lambda thin_i, j: self.thin[thin_i].thin_faces[j]
        self.titin = (
            ti.Titin(self, 0, ti_thick(0, 0), ti_thin(0, 1), **ti_params),
            ti.Titin(self, 1, ti_thick(0, 1), ti_thin(1, 2), **ti_params),
            ti.Titin(self, 2, ti_thick(0, 2), ti_thin(2, 2), **ti_params),
            ti.Titin(self, 3, ti_thick(1, 0), ti_thin(2, 1), **ti_params),
            ti.Titin(self, 4, ti_thick(1, 1), ti_thin(3, 2), **ti_params),
            ti.Titin(self, 5, ti_thick(1, 2), ti_thin(0, 2), **ti_params),

            ti.Titin(self, 6, ti_thick(0, 5), ti_thin(4, 1), **ti_params),
            ti.Titin(self, 7, ti_thick(0, 4), ti_thin(5, 0), **ti_params),
            ti.Titin(self, 8, ti_thick(0, 3), ti_thin(6, 0), **ti_params),
            ti.Titin(self, 9, ti_thick(1, 5), ti_thin(6, 1), **ti_params),
            ti.Titin(self, 10, ti_thick(1, 4), ti_thin(7, 0), **ti_params),
            ti.Titin(self, 11, ti_thick(1, 3), ti_thin(4, 0), **ti_params),

            ti.Titin(self, 12, ti_thick(2, 0), ti_thin(5, 1), **ti_params),
            ti.Titin(self, 13, ti_thick(2, 1), ti_thin(6, 2), **ti_params),
            ti.Titin(self, 14, ti_thick(2, 2), ti_thin(7, 2), **ti_params),
            ti.Titin(self, 15, ti_thick(3, 0), ti_thin(7, 1), **ti_params),
            ti.Titin(self, 16, ti_thick(3, 1), ti_thin(4, 2), **ti_params),
            ti.Titin(self, 17, ti_thick(3, 2), ti_thin(5, 2), **ti_params),

            ti.Titin(self, 18, ti_thick(2, 5), ti_thin(1, 1), **ti_params),
            ti.Titin(self, 19, ti_thick(2, 4), ti_thin(2, 0), **ti_params),
            ti.Titin(self, 20, ti_thick(2, 3), ti_thin(3, 0), **ti_params),
            ti.Titin(self, 21, ti_thick(3, 5), ti_thin(3, 1), **ti_params),
            ti.Titin(self, 22, ti_thick(3, 4), ti_thin(0, 0), **ti_params),
            ti.Titin(self, 23, ti_thick(3, 3), ti_thin(1, 0), **ti_params),
        )
        '''Initialize the last few variables'''
        # Set the timestep for all our new cross-bridges
        self.timestep_len = timestep_len
        # Track how long we've been running
        self.current_timestep = 0

        # ## Tropomyosin species concentrations
        self.c_tn = None
        self.c_tnca = None
        # TODO determine if we need to self.update_concentrations()

        # ## variables previously initialized in methods (hiding line included above)
        self.last_transitions = None
        self.tm_transitions = None
        self._volume = None
        self.update_volume()

        # Now that we are making constants more accessible, we need to track them
        self.constants = {'ti': {titin.index: titin.constants for titin in self.titin},
                          'af': {actin.index: actin.af_constants for actin in self.thin},
                          'mf': {myosin.index: myosin.mf_constants for myosin in self.thick},
                          'tm': {},
                          'mh': {}}

        for myosin in self.thick:
            self.constants['mh'].update(myosin.mh_constants)
        for actin in self.thin:
            self.constants['tm'].update(actin.tm_constants)
            
        self.temp = temp
            
        # K_matrix is the stiffnesst matrix of all LINEAR ELEMENTS ONLY, does not include titin
        self.K_matrix = self.hs_passive_stiffness_matrix()
        self.V = self.spring_boundary_conditions()
        
        # initaizle the half sarc geometry 
        # bassically self.timestep, but with dt = 100-100ms ms
        # Use Newton method to solve for the new spring configuration
        # do this before any binding to deal with titin's configuration
        # self.settle()
        self.Newton()
        # initialize to (the approximate) steady state 
        # basically assuming the past under constant everything with condtions defined by t=0
        # 'steady state' but it doesnt account for cooperativity, so it's only approximate
        for _ in range(10):
            pass
            # set which tm sites are subject to cooperative effects
            self.set_subject_to_cooperativity()
            # assign each xb its nearest neighbor binding site
            self.set_xb_nearest_binding_site()  
            # Update thin filament tm states
            self.thin_transitions(dt = 100)
            # update thick filament xb states and binding status
            self.thick_transitions(dt = 100)
            # Use Newton method to solve for the new spring configuration 
            self.Newton()
        
        
        
        
        

    def to_dict(self):
        """Create a JSON compatible representation of the thick filament

        Example usage: json.dumps(sarc.to_dict(), indent=1)

        Current output includes:
            version: version of the sarcomere model
            timestep_len: the length of the timestep in ms
            current_timestep: time to get a watch
            lattice_spacing: the thick to thin distance
            z_line: the z_line location
            pCa: the calcium level
            hiding_line: where binding sites become unavailable due to overlap
            time_dependence: dictionary of how "lattice_spacing", "z_line", 
                and "pCa" can change
            last_transitions: keeps track of the last state change by thick
                filament and by crown
            thick: the structures for the thick filaments
            thin: the structures for the thin filaments
        """
        sd = self.__dict__.copy()  # sarc dict
        sd['current_timestep'] = self.current_timestep
        # set act_perm as mean since prop access returns values at every point
        sd['actin_permissiveness'] = np.mean(self.actin_permissiveness)
        sd['thick'] = [t.to_dict() for t in sd['thick']]
        sd['titin'] = [t.to_dict() for t in sd['titin']]
        sd['thin'] = [t.to_dict() for t in sd['thin']]
        return sd

    def from_dict(self, sd):
        """ Load values from a sarcomere dict. Values read in correspond to
        the current output documented in to_dict.
        """
        # Warn of possible version mismatches
        read, current = sd['version'], self.version
        if read != current:
            import warnings
            warnings.warn("Versioning mismatch, reading %0.1f into %0.1f."
                          % (read, current))
        # Get filaments in right orientations
        self.__init__(
            lattice_spacing=sd['_initial_lattice_spacing'],
            z_line=sd['_initial_z_line'],
            poisson=sd['poisson_ratio'],
            pCa=sd['pCa'],
            timestep_len=sd['timestep_len'],
            time_dependence=sd['time_dependence'],
            starts=(sd['_thin_starts'], sd['_thick_starts'])
        )
        # Local keys
        self.current_timestep = sd['current_timestep']
        self._z_line = sd['_z_line']
        self._lattice_spacing = sd['_lattice_spacing']
        self.hiding_line = sd['hiding_line']
        if 'last_transitions' in sd.keys():
            self.last_transitions = sd['last_transitions']
        # Sub-structure keys
        for data, thick in zip(sd['thick'], self.thick):
            thick.from_dict(data)
        for data, thin in zip(sd['thin'], self.thin):
            thin.from_dict(data)

    def print_constants(self, print_address=False):
        """prints all settings in an organized fashion"""
        for f_type, filaments in self.constants.items():
            print(f_type)
            for address, filament in filaments.items():
                address = "\t" + str(address)
                if not print_address:
                    address = ""
                print(address, "\t", end="")

                for constant, value in filament.items():
                    print(constant, "=", value, end=" ")
                if len(filaments.keys()) < 50:
                    print()
                else:
                    print(", ", end="\t")

    def run(self, time_steps=100, callback=None, bar=True, every=5):
        """Run the model for the specified number of time_steps

        Parameters:
            time_steps: number of time steps to run the model for (100)
            callback: function to be executed after each time step to
                collect data. The callback function takes the sarcomere
                in its current state as its only argument. (Defaults to
                the axial force at the M-line if not specified.)
            bar: progress bar control,False means don't display, True
                means give us the basic progress reports, if a function
                is passed, it will be called as f(completed_steps,
                total_steps, sec_left, sec_passed, process_name).
                (Defaults to True)
            every: how many time_steps to update after
        Returns:
            output: the results of the callback after each timestep
            exit_code: how the simulation was terminated
                0 - exited successfully
                1 - general error
                130 - CTRL-C ~ User Interrupt
        """
        # Callback defaults to the axial force at the M-line
        if callback is None:
            callback = self.axial_force
        # ## logic to handle bar is type(True || False || Function)
        use_bar = False
        update_bar = self.print_bar
        if isinstance(bar, bool):
            use_bar = bar
        elif isinstance(bar, type(lambda x: x)):
            use_bar = True
            update_bar = bar
        # Create a place to store callback information and note the time
        output = []
        tic = time.time()
        # Run through each timestep
        for i in range(time_steps):
            try:
                self.timestep(i)
                output.append(callback())
                # Update us on how it went
                toc = int((time.time() - tic) / (i + 1) * (time_steps - i - 1))
                proc_name = mp.current_process().name

                if use_bar and i % every == 0:
                    update_bar(i=i, time_steps=time_steps,
                               toc=toc, tic=time.time() - tic,
                               proc_name=proc_name, output=output)
            except KeyboardInterrupt:
                return output, 130
            except Exception as e:
                import traceback
                print("/n")
                print(e)
                traceback.print_exc()
                return output, 1
        return output, 0

    @staticmethod
    def print_bar(i, time_steps, toc, proc_name, **bar_kwargs):
        if 'tic' in bar_kwargs.keys() and bar_kwargs['tic'] < -1:
            print('Causality has failed')
        sys.stdout.write("\n" + proc_name +
                         " finished timestep %i of %i, %ih%im%is left"
                         % (i + 1, time_steps, toc / 60 / 60, toc / 60 % 60, toc % 60))
        sys.stdout.flush()
    
    def Force_on_each_node(self):
        '''
        
        Force on each node is:
            
            K*x + Fxb + V
            
            K = hs_passive_stiffness_matrix
            F = force from xbs
            F_titin
            
            V = boundary conditions 

        Returns
        -------
        None.

        '''
        
        K = self.K_matrix
        V = self.V
        
        for th in self.thick:
            V[60*th.index + 59] += 6 * th.thick_faces[0].titin_fil.axial_force() 
        for th in self.thin:
            V[240 + 90*th.index + 89] += th.k*self.z_line
            
        
        axial_locations = np.concatenate((np.concatenate([i.axial for i in self.thick]), np.concatenate([i.axial for i in self.thin])))
        
        
        xbs = [xb for th in self.thick for cr in th.crowns for xb in cr.crossbridges if xb.bound_to is not None ]
        
        x = []
        y = []
        data = []
        for xb in xbs:
            
            # # indices corresponding to this crossbridge 
            mf_index = xb.address[1]*60+xb.address[3] 
            af_index = xb.bound_to.address[1]*90 + xb.bound_to.address[2] + 240 # 90 binding sites per thin filament, + 60*4=240 xbs
            
            
            # # axial value of the xb base and binding site, w.r.t the m_line (m_line == 0)
            xb_x = th.axial[xb.address[3]]
            bs_x = self.thin[xb.bound_to.address[1]].axial[xb.bound_to.address[2]]
            
            Fxb = xb.axial_force(bs_x - xb_x, self.lattice_spacing)
            
            x.extend([mf_index,af_index])
            # y.extend([mf_index, af_index])
            data.extend([Fxb,-Fxb])
        
        y1 = [0 for i in x]
        
        Force =  (K @ axial_locations).reshape((960,1)) + sparse.csr_matrix((data,(x,y1)), shape=(960,1)) + V.reshape(960,1)
        
        return Force
        
    def spring_boundary_conditions(self):
        
        '''
        
        actin nodes have different spring offsets, we write k * (rest_i - rest_j) here for all thick and thin filaments. 
        
        This does not include contributions at the last nodes connecting to z-disk, since it depends on z, 
        and we want to update at each step accordingly
        
        F = K_matrix * x + V + F_titin + F_xbs_bs + F_z_actin
        
        
        
        
        '''
        
        v_thick_0 = - self.thick[0].k * np.diff(np.concatenate((self.thick[0].rests,[0])))
        v_thick_1 = - self.thick[1].k * np.diff(np.concatenate((self.thick[1].rests,[0])))
        v_thick_2 = - self.thick[2].k * np.diff(np.concatenate((self.thick[2].rests,[0])))
        v_thick_3 = - self.thick[3].k * np.diff(np.concatenate((self.thick[3].rests,[0])))
        
        v_thin_0 = - self.thin[0].k * np.diff(np.concatenate(([0],self.thin[0].rests)))
        v_thin_1 = - self.thin[1].k * np.diff(np.concatenate(([0],self.thin[1].rests)))
        v_thin_2 = - self.thin[2].k * np.diff(np.concatenate(([0],self.thin[2].rests)))
        v_thin_3 = - self.thin[3].k * np.diff(np.concatenate(([0],self.thin[3].rests)))
        v_thin_4 = - self.thin[4].k * np.diff(np.concatenate(([0],self.thin[4].rests)))
        v_thin_5 = - self.thin[5].k * np.diff(np.concatenate(([0],self.thin[5].rests)))
        v_thin_6 = - self.thin[6].k * np.diff(np.concatenate(([0],self.thin[6].rests)))
        v_thin_7 = - self.thin[7].k * np.diff(np.concatenate(([0],self.thin[7].rests)))

        V = np.concatenate((
            v_thick_0,
            v_thick_1,
            v_thick_2,
            v_thick_3,  
            v_thin_0,
            v_thin_1,
            v_thin_2,
            v_thin_3,
            v_thin_4,
            v_thin_5,
            v_thin_6,
            v_thin_7
            ))
        
        return V
    
    def hs_passive_stiffness_matrix(self):
        '''
        
        gets the spring stiffness matrix K for passive conditions (no bound crossbridges), not counting titin 
        
        for a system of linear springs:
        Force = K * x 
        K is the stiffness matrix
        
        
        Form is:
            Myosin:
                A[0,0] = -2k # myosin is connected to m-line
                A[i,i] = -2k
                A[i,i-1] = A[i-1,i] = +k # aka the two adjacent diagonals, super- and sub-diagonalals
                A[end,end] = -k # see note below
                
            Actin:
                A[0,0] = -k # actin unconnected to m-line
                A[i,i] = -2k 
                A[i,i-1] = A[i-1,i] = +k # aka the two adjacent off diagonals, super- and sub-diagonalals
                A[end,end] = -2k # actin is connected to z disk
                
                
                
                *********
      !!          A[end,end] = -k #
      !!          This should acually be -k + titin_stiffness, but here we set the stiffness matrix for myosin as if it's 
      !!          unconected to z-line, and account for titin elsewhere, when we set the Jacobian, since its stiffness is dependent 
      !!          on the z_line, so it needs to be updated at each time step 
                
                *********

        Then concatenate into big 960*960 matrix, that is >99 % sparse, so we can use sparse matrix methods, 
        ie scipy.sparse
        
        
        This matrix is a constant and is used in finding the Jacobian, we just need to add the stiffness due to any bound xbs, and titin, 
        which both change at each timestep, 
        
        
                
        ''' 
        
        xb_num = len(self.thick[0].axial)
        thick_k = self.thick[0].k
        
        bs_num = len(self.thin[0].axial)
        thin_k = self.thin[0].k
        
        kth0 = np.eye(xb_num, k=0)*-2*thick_k + \
            np.eye(xb_num, k=1)*thick_k + \
                np.eye(xb_num, k=-1)*thick_k; kth0[59,59] = -thick_k
        kth1 = np.eye(xb_num, k=0)*-2*thick_k + \
            np.eye(xb_num, k=1)*thick_k + \
                np.eye(xb_num, k=-1)*thick_k; kth1[59,59] = -thick_k
        kth2 = np.eye(xb_num, k=0)*-2*thick_k + \
            np.eye(xb_num, k=1)*thick_k + \
                np.eye(xb_num, k=-1)*thick_k; kth2[59,59] = -thick_k
        kth3 = np.eye(xb_num, k=0)*-2*thick_k + \
            np.eye(xb_num, k=1)*thick_k + \
                np.eye(xb_num, k=-1)*thick_k; kth3[59,59] = -thick_k
        
        Ka0 = np.eye(bs_num, k=0)*-2*thin_k + \
            np.eye(bs_num, k=1)*thin_k + \
                np.eye(bs_num, k=-1)*thin_k; Ka0[0,0] = -thin_k
                
        Ka1 = np.eye(bs_num, k=0)*-2*thin_k + \
            np.eye(bs_num, k=1)*thin_k + \
                np.eye(bs_num, k=-1)*thin_k; Ka1[0,0] = -thin_k
                
        Ka2 = np.eye(bs_num, k=0)*-2*thin_k + \
            np.eye(bs_num, k=1)*thin_k + \
                np.eye(bs_num, k=-1)*thin_k; Ka2[0,0] = -thin_k
                
        Ka3 = np.eye(bs_num, k=0)*-2*thin_k + \
            np.eye(bs_num, k=1)*thin_k + \
                np.eye(bs_num, k=-1)*thin_k; Ka3[0,0] = -thin_k
                
        Ka4 = np.eye(bs_num, k=0)*-2*thin_k + \
            np.eye(bs_num, k=1)*thin_k + \
                np.eye(bs_num, k=-1)*thin_k; Ka4[0,0] = -thin_k
                
        Ka5 = np.eye(bs_num, k=0)*-2*thin_k + \
            np.eye(bs_num, k=1)*thin_k + \
                np.eye(bs_num, k=-1)*thin_k; Ka5[0,0] = -thin_k
                
        Ka6 = np.eye(bs_num, k=0)*-2*thin_k + \
            np.eye(bs_num, k=1)*thin_k + \
                np.eye(bs_num, k=-1)*thin_k; Ka6[0,0] = -thin_k
                
        Ka7 = np.eye(bs_num, k=0)*-2*thin_k + \
            np.eye(bs_num, k=1)*thin_k + \
                np.eye(bs_num, k=-1)*thin_k; Ka7[0,0] = -thin_k
             
        thick_fill = np.zeros((xb_num,xb_num))
        thin_fill = np.zeros((bs_num,bs_num))
        upper_fill = np.zeros((xb_num,bs_num))
        lower_fill = np.zeros((bs_num,xb_num))
        
        hs_node_matrix = np.bmat([
            [kth0,        thick_fill,  thick_fill,  thick_fill,  upper_fill, upper_fill,  upper_fill, upper_fill, upper_fill, upper_fill, upper_fill, upper_fill],
            [thick_fill,  kth1,        thick_fill,  thick_fill,  upper_fill, upper_fill,  upper_fill, upper_fill, upper_fill, upper_fill, upper_fill, upper_fill],
            [thick_fill,  thick_fill,  kth2,        thick_fill,  upper_fill, upper_fill,  upper_fill, upper_fill, upper_fill, upper_fill, upper_fill, upper_fill],
            [thick_fill,  thick_fill,  thick_fill,  kth3,        upper_fill, upper_fill,  upper_fill, upper_fill, upper_fill, upper_fill, upper_fill, upper_fill],
            [lower_fill,  lower_fill,  lower_fill,  lower_fill,  Ka0,        thin_fill,   thin_fill,  thin_fill,  thin_fill,  thin_fill,  thin_fill,  thin_fill],
            [lower_fill,  lower_fill,  lower_fill,  lower_fill,  thin_fill,  Ka1,         thin_fill,  thin_fill,  thin_fill,  thin_fill,  thin_fill,  thin_fill],
            [lower_fill,  lower_fill,  lower_fill,  lower_fill,  thin_fill,  thin_fill,   Ka2,        thin_fill,  thin_fill,  thin_fill,  thin_fill,  thin_fill],
            [lower_fill,  lower_fill,  lower_fill,  lower_fill,  thin_fill,  thin_fill,   thin_fill,  Ka3,        thin_fill,  thin_fill,  thin_fill,  thin_fill],
            [lower_fill,  lower_fill,  lower_fill,  lower_fill,  thin_fill,  thin_fill,   thin_fill,  thin_fill,  Ka4,        thin_fill,  thin_fill,  thin_fill],
            [lower_fill,  lower_fill,  lower_fill,  lower_fill,  thin_fill,  thin_fill,   thin_fill,  thin_fill,  thin_fill,  Ka5,        thin_fill,  thin_fill],
            [lower_fill,  lower_fill,  lower_fill,  lower_fill,  thin_fill,  thin_fill,   thin_fill,  thin_fill,  thin_fill,  thin_fill,  Ka6,        thin_fill],
            [lower_fill,  lower_fill,  lower_fill,  lower_fill,  thin_fill,  thin_fill,   thin_fill,  thin_fill,  thin_fill,  thin_fill,  thin_fill,  Ka7]
            ])
        
        return sparse.csr_matrix(hs_node_matrix)
    
    def Jacobian_Matrix(self):
        '''
        
        Returns the Jacobian matrix for the current configuration of the Half sarcomere
        J is stored as a sparse scipy matrix
        
        Each entry Jij is the partial derivative of i with respect to j
        
        So in the half sarcomere model, it's the derivative of Force on node i with resepect to the axial position of node j
        
        The force on node i can only be influenced directly by the nodes adjacent to it, or when a crossbridge is bound between node i and j,
        so J ends up being 99% sparse
        
        K is the (sparse) stiffness matrix of the linear springs in the thick and thin filaments, so we just need to add the contributions
        of bound xbs and titin.
        
        J is a square matrix that's of length total_crown_num + total_bs_num = 240 + 720 = 960      
        
        '''
        
        # passive stiffness of each linear spring making up the filaments, not counting titin
        
        K = self.K_matrix

        i = []
        j = []
        Jac = []
        for thick in self.thick: 
            # each thick filament has 6 titin filaments
            # 60*thick_index + 59 is the last node for that thick filament, which is the one attached to the z-disk
            i.extend([60*thick.index + 59])
            j.extend([60*thick.index + 59])
            # multiply by 6 because each thick filament is attached to the z disk by 6 titin filaments, they have identical stiffness
            # Jac.extend([- 6 * thick.thick_faces[0].titin_fil.stiffness() ])
            Jac.extend([- 6 * thick.thick_faces[0].titin_fil.dForce_dMnode(thick.axial[-1])])
            
            for cr in thick.crowns:
                for xb in cr.crossbridges:
                    if xb.state in {'loose', 'tight_1', 'tight_2'}:
                        
                        # get i,j for this bound xb-bs pair to put in the 960*960 array
                        
                        # there are two indexes we need here, the index of the xb and bs w.r.t the filament they are on
                        # and their index in the final 960*960 matrix
                        
                        # for their index w.r.t thier parent filament see xb.address and bs.address
                        # we use those to get the index of the parent filament and the xb/bs number witin that particular parent filament
                        # then  the index wrt the 960*960 array, corresponding to this crossbridge xb:
                            # 60 * thick_fil_index + xb_index
                        mf_index = xb.address[1]*60+xb.address[3] 
                       
                        # and the index of the bound actin bindings site bs:
                            # 90 * thin_fil_index + xb_index + 240 (240 accounts for the 4*60 xbs)
                        af_index = xb.bound_to.address[1]*90 + xb.bound_to.address[2] + 240

                        # # axial value of the xb base and binding site, w.r.t the m_line (m_line == 0)
                        # get the axial values the bs-xb pair
                        xb_x = thick.axial[xb.address[3]]
                        bs_x = self.thin[xb.bound_to.address[1]].axial[xb.bound_to.address[2]]
                        
                        # calculate the partial derivative of the force on the xb with repsect to the binding site axial 
                        Fxb = xb.D_force_D_x((bs_x, xb_x, self.lattice_spacing))
                        
                        # each bound xb-bs pair contributes 4 values to the Jacobian, 
                        # at (i,i) for d_Fxb/d_xb_x
                        # at (j,i) for d_Fbs/d_xb_x
                        # at (i,j) for d_Fxb/d_bs_x
                        # at (j,j) for d_Fbs/d_bs_x
                        i.extend([mf_index,af_index,mf_index,af_index])
                        j.extend([mf_index,mf_index,af_index,af_index])
                        # the value to add is just different by a sign
                        Jac.extend([Fxb,-Fxb,-Fxb,Fxb])
                        
        # i,j are now the list of indicies correspoiding to values in J, 
        # which are the contributions to the Jacobian from titin and xbs                       
        # now add to K for full Jacobian
        Jacobian = K + sparse.csr_matrix((Jac,(i,j)), shape=(960,960))
        
        return Jacobian
        
    def timestep(self, current=None):
        """Move the model one step forward in time, allowing the
        myosin heads a chance to bind and then balancing forces
        """
        # Record our passage through time
        if current is not None:
            self.current_timestep = current
        else:
            self.current_timestep += 1
        
        # set which tm sites are subject to cooperative effects
        self.set_subject_to_cooperativity()
        
        # assign each xb its nearest neighbor binding site
        self.set_xb_nearest_binding_site()  
                
        # Update thin filament tm states
        self.thin_transitions()
        # update thick filament xb states and binding status
        self.thick_transitions()
       
        # Use Newton method to solve for the new spring configuration 
        self.Newton()
        
                
        xbs = [xb for th in self.thick for cr in th.crowns for xb in cr.crossbridges if xb.bound_to is not None ]
        bs = np.array([xb.nearest.axial_location - xb.axial_location for xb in xbs])

        # if len(xbs)!=0:
        #     plt.hist(bs)
        #     plt.show()
            
            # pdb.set_trace()

        
        return
        
    
    def thin_transitions(self, dt=None):
        '''

        Cycle through the tm sites and get the rate matrix Q for each site, then cacluate P = matrix_exp(Q * dt),
        where dt = timestep_len. 
        
        P is then the matrix with elements Pij, which gives the probability a state which begins in state i will end up
        in state j after time dt. 
        We then get a random number and compare with row i of P, where i is corresponds to the current state 
        of the site. 
        
        Because Q only depends on Ca concentration (same for all Tm sites), cooperative status [T/F], and myosin 
        binding status [T/F], we only need to calculate Q and P 2*2=4 times per timestep, per species of Tropomyosin 
        present in the half sarcomere. We then store those matrix and recall them. That way we don't calucate 
        once per tm site (720) times per timestep. 
        

        '''
        
        if dt is None:
            dt = self.timestep_len
        
        # get a list of every actin binding site
        bs_s = [bs for th in self.thin for bs in th.binding_sites ]
        
        # get a list of the current state of each tm site
        old_states = [bs.tm_site.state for th in self.thin for bs in th.binding_sites ]
        
        # get all the constants we need to make the rate matrix Q for every binding site
        tm_site_constants = np.array([[
            bs.tm_site._K1,
            bs.tm_site._K2,
            bs.tm_site._K3,
            bs.tm_site._K4 if bs.tm_site._K4 is not None else 0,
            
            bs.tm_site._k_12,
            bs.tm_site._k_23,
            bs.tm_site._k_34,
            bs.tm_site._k_41,
            
            bs.tm_site._coop,
            1 if bs.tm_site.subject_to_cooperativity else 0,
            
            0 if bs.tm_site.state==3 and bs.bound_to is not None else 1]
            for bs in bs_s])
        
        # we also need caclium concentration in molar
        ca = self.ca 
        
        # pdb.set_trace()
        # get the [720 x 4 x 4] array of Q matricies 
        Q = self.binding_site_rate_matix(tm_site_constants, ca)
        
        # each row in each Q should sum to 0
        # don't manipulate Q outsied of self.binding_site_rate_matrix()
        assert(np.max(np.abs(np.sum(Q[:,:,:], axis = 2)))<10**-11)

        # average Q and store for output to data file at each timestep, see self.tm_rates
        self.tm_rates_ = np.mean(Q,axis=0)
        
        # find the unique Qs and thier index, there should be at most 4 unqiue Qs per species of tm site
        # we only find expm(Q) for those to avoid unecessary caclulations, 
        Q_reduced, Q_ids = np.unique(Q, axis=0, return_inverse=True)
        
        # each row in each Q should sum to 0
        # don't manipulate Q outsied of binding_site_rate_matix function
        assert(np.max(np.abs(np.sum(Q_reduced[:,:,:], axis = 2)))<10**-11)
        
        # get the full [720 x 4 x 4] array of P matricies from the reduced list
        P_reduced = expm_(Q_reduced * dt)
        P = P_reduced[Q_ids]
        
        # get a list of random numbers from uniform dist
        p = np.random.rand(720,1)
        
        # for each of the [4 x 4] probability matrices, we want the row which corresponds to that bs's current state, 
        # for example since P is a matrix with values Pij, we want row i since we want transition probabilities from state i to j, where i is the current state
        Prob = np.empty((720,4))
        for i in range(0,720):
            # bs numeric_state goes from 0-3, (rather than 1-4) so we DONT need -1 here, 
            Prob[i,:] = P[i,old_states[i],:]
        
        # cumsum each item in Prob for next step
        Prob_array = np.cumsum(Prob, axis=1)
        
        # for each bs, the new state is the first entry in each probability vector Prob_array[i] smaller than p[i]
        new_states = np.argmax(p < Prob_array, axis=1) 
        
        # deal out the new states to each tm_site
        for i, bs_ in enumerate(bs_s):
            bs_.tm_site.state = new_states[i]
        
        return                    
                
    @staticmethod
    @njit
    def binding_site_rate_matix(tm_site_constants, ca):
        
        
        '''
        
        Constructs the [720 x 4 x 4] array of rate matrices for each of the tm sites. 
        
        Each [4 x 4] slice is the rate matrix of one tm_site iwth elememnts Qij meaning the rate of transitioning from state i
        to state j
        
        ca is calcium concentation (10**pca)
        
        tm_site_constants[:,0:4] are the equilirim rate constants
        
        tm_site_constants[:,4:8] are the forward rate constants
        
        tm_site_constants[:,8] is the magniutde of cooperativity 
        
        tm_site_constants[:,9] is 0/1 for True/False of whether a bs is subject to cooperateive effets
        
        tm_site_constants[:,10] is whether the site is bound to an xb and therefore cant transition away from state 3
        
        reverse rates are defined as forward_rates/Equilibrium rates
        
        '''
        
        # get the equilibrium reaction rates 
        _K1 = tm_site_constants[:,0]
        _K2 = tm_site_constants[:,1]
        _K3 = tm_site_constants[:,2]
        _K4 = tm_site_constants[:,3]
        
        # coop is magnitude of cooperative effect
        coop = tm_site_constants[:,8]
        # wheterh the tm site is subject to cooperative effeects: 1 for true, 0 for false
        sub_to_coop = tm_site_constants[:,9]
        
        # where (coop * sub_to_coop) = 0 => set coop to 1, ie no cooperative effect (coop*rate = 1*rate = rate)
        # else coop is unchanged, the magnitude of the cooperative effect
        coop[np.where(coop * sub_to_coop==0)] = 1
        
        # forward rate constants
        _k_12 = tm_site_constants[:,4] 
        _k_23 = tm_site_constants[:,5] 
        _k_34 = tm_site_constants[:,6] 
        _k_41 = tm_site_constants[:,7]
        
        # we don't allow 43 or 41 transitions if the binding site is in state 3 AND bound to an xb
        # so we multiply k41 and k43 by s, which is 0 if bs is bound and in state 3, otherwise 1
        s = tm_site_constants[:,10]
        
        ########################################################################################################################
        # 
        #       Tropomyosin rate defintions below
        #
        #
        # 
        ########################################################################################################################
        
        # forward
        k_12 = _k_12 * ca * coop
        k_23 = _k_23 * coop
        k_34 = _k_34 * coop
        k_41 = _k_41 * s 
        
        # backward
        k_21 = _k_12 / _K1
        k_32 = _k_23 / _K2
        k_43 = _k_34 / _K3 * s
        k_14 = 0
        
        # diagonal rates should be set to that rows in Q sum to 0
        k_11 = - (k_12 + k_14)
        k_22 = - (k_23 + k_21)
        k_33 = - (k_32 + k_34)
        k_44 = - (k_41 + k_43)
        
        ########################################################################################################################
        # 
        #       Tropomyosin rate defintions above
        #
        #
        # 
        ########################################################################################################################
        
        # construct Q 
        Q = np.zeros((720,4,4))
        
        Q[:,0,0] = k_11
        Q[:,0,1] = k_12 
        Q[:,0,3] = k_14 
        
        Q[:,1,0] = k_21
        Q[:,1,1] = k_22
        Q[:,1,2] = k_23
        
        Q[:,2,1] = k_32
        Q[:,2,2] = k_33
        Q[:,2,3] = k_34
        
        Q[:,3,2] = k_43
        Q[:,3,3] = k_44
        Q[:,3,0] = k_41
        
        return Q
        
       
    def set_subject_to_cooperativity(self):
        '''
        
        decide which tm sites are subject to cooperativity. 

        The span (state 2 coercion of adjacent sites to state 1 from 
        state 0) is based on the current tension at the binding site 
        co-located under this tropomyosin site. 
        Notes
        -----
        The functional form of the span is determined by:
            
            $$span = 0.5 * base (1 + tanh(steep*(force50 + f)))$$
            
        Where $span$ is the actual span, $base$ is the resting (no force) 
        span distance, $steep$ is how steep the decrease in span is, 
        $force50$ is the force at which the span has decreased by half, and 
        f is the current effective axial force of the thin filament, an 
        estimate of the tension along the thin filament. 
        These properties are stored at the tropomyosin chain level as they 
        are material properties of the entire chain.
        
        
        0.5 * base * (1 + m.tanh(steep * (f50 + f)))
        
        ''' 

        for thin in self.thin:
            for tm in thin.tm:
                
                
                base = tm.span_base
                F_50 = tm.span_force50
                steepness = tm.span_steep
                
                # F is the tension on each node due to crossbridges, summed up to the m line: see thin.tension_at_site in af.py
                F = (np.triu(np.ones(thin.number_of_nodes)) @ thin._axial_thin_filament_forces())[tm.sites[0].binding_site.address[2]::2]
                
                # calculate span 
                span = .5 * base * (1 + np.tanh(steepness * (F_50 + F ) )) 
                
                # truth matrix of which nodes are within a node's span
                d = (np.abs(tm.axial_locations[:,np.newaxis] - tm.axial_locations) - span[:,None])<0
                
                # Truth vector of which nodes are in state 2
                state = [True if s.state==2 else False for s in tm.sites]
                
                # truth matrix * truth vector, then np.any to see if at least one node is within span and in state 2
                coop = np.any(d*state, axis=1)

                # deal out 'True' or 'False' for coop to each tm node
                for index, site in enumerate(tm.sites):
                    site.subject_to_cooperativity = coop[index]
                    
        return   
    
    
      
    @staticmethod
    # @njit
    def xb_rate_matrix(bs, V, rate_factors, ap, ca_c, temp):
        
        '''
        
        Finds the (n x n x L) array where each n*n slice corresponds to one xbs transtions rate matrix, 
        with elements rij, which correspond to transitons rates from state i to state j
        
        each 6x6 slice looks like the following:
        
            Q = np.array([
                      [r11,  r12,   0,    0,    r15,  r16], 
                      [r21,  r22,   r23,  0,    0,    0],
                      [0,    r32,   r33,  r34,  0,    0],
                      [0,    0,     r43,  r44,  r45,  0],
                      [r51,  0,     0,    r54,  r55,  0],
                      [r61,  0,     0,    0,    0,    r66]
                      ])
        
        
        L is the nuimber of crossbridges, so we end up with a (6 x 6 x 720) array 
        
        
        Arguments are:
            bs: (x,y) distance between xb and bs
            V: spring constants of each of the two spings, for each xb, in each state (2*2*2) -> (8 x 720) array
            rate_factors: multiply certain xbs with rate factors, see hs_params in hs.py and mh_params in mh.py, 
                and see rate_factors in hs.thick_transitions() for which are which:
                    rate_factors = np.array([list((                     
                        xb.constants['mh_br'],                          r12  0 
                        xb.constants['mh_r34'],                         r34  1 
                        xb.constants['mh_dr'],                          r45  2 
                        xb.constants['mh_srx'])) for xb in xbs])        r16  3 
                
            ap: binding site availability
            ca_c: calcium concentration in uM
            temp: temp in degree c
        
        
        '''
    
        # get boltzman constant in units of pn*nm
        # self.temp is in units of c, we need to convert to k
        k_t =  1.3810 * 10. ** -23. * (temp + 273.15) * 10. ** 21.  # 10**21
        
        # cartesian to polar transformation
        (r,theta) = cart2pol(bs[:,0], bs[:,1])
        
        # E1, WEAK potential energy when BOUND at bs
        E_weak = (1/2 * V[:,0] * (r -  V[:,1])**2  +  1/2 * V[:,2] * (theta - V[:,3])**2) / k_t
        # E2, STRONG potential Energy when BOUND at bs
        E_strong = (1/2 * V[:,4] * (r -  V[:,5])**2  +  1/2 * V[:,6] * (theta - V[:,7])**2) / k_t
        
        # FREE energy U = scalar + potential energy, 
        # scalar values come from Pate and Cooke 1989 - "A model of crossbridge action: the effects of ATP, and Pi" page 186
        U_free = 0 
        U_SRX = 0       
        U_DRX = -2.3
        U_loose = -4.3 + E_weak
        U_tight_1 = -4.3 + -14.3 + E_strong
        U_tight_2 = -4.3 + -14.3 + -2.12 + E_strong
        
        f_3_4 = V[:,4] * (r -  V[:,5])  +  1/r * V[:,6] * (theta - V[:,7])
        f_3_4_x = V[:,4] * (r -  V[:,5]) * np.cos(theta)  + 1/r * V[:,6] * (theta - V[:,7]) * np.sin(theta)
        
        f_2 = V[:,0] * (r -  V[:,1])  +  1/r * V[:,2] * (theta - V[:,3])
        f_2_x = V[:,0] * (r -  V[:,1]) * np.cos(theta)  + 1/r * V[:,2] * (theta - V[:,3]) * np.sin(theta)
        
        
        # f_x = (g_k * (g_len - g_s) * m.cos(c_ang) +
               # 1 / g_len * c_k * (c_ang - c_s) * m.sin(c_ang))
        
        ############################################################################################################################################
        #
        #
        #
        #   Myosin head
        #   Rate function definitions below
        # 
        #
        #
        ############################################################################################################################################
        
        # for constant rates
        ones = np.ones(len(bs))
        # upper bound, exp(Q) will fail if norm(Q) is to large, also gets rid of inf from 1/exp in backwards rate defs
        upper = 10000
        
        # DRX (1) <-> free (2)
        tau = .72
        r12 = rate_factors[:,0] * tau * np.exp( -E_weak ); r12[np.isnan(r12)] = 0
        r21 = (r12 + .005) / np.exp(U_DRX - U_loose); r21[r21>upper] = upper; r21[np.isnan(r21)] = upper
        
        # free (2) <-> tight_1 (3)
        r23 =  rate_factors[:,4] *( (0.6 *  # reduce overall rate
                (1+# shift rate up to avoid negative rate
                  np.tanh(5 +  # move center of transition to right
                        0.4 * (E_weak - E_strong)) #- 
                  # np.tanh(-5 +  # move center of transition to right
                  #       0.4 * (E_weak - E_strong))
                   )) + .05); r23[np.isnan(r23)] = 0
        r32 = r23 / np.exp(U_loose - U_tight_1); r32[r32>upper] = upper
        
        # tight_1 (3) <-> tight_2 (4)
        A = 1
        r34 = rate_factors[:,1] * (A * .5 * (1 + np.tanh(E_weak - E_strong)) + .03) + np.exp(-f_3_4); r34[r34>upper] = upper
        r43 = r34 / np.exp(U_loose - U_tight_2); r43[r43>upper] = upper
        
        # tight_1 (4) <-> free_2 (5)
        r45 = rate_factors[:,2] * .5 * np.sqrt(U_tight_2 + 23) + np.exp(-f_3_4); r45[r45>upper] = upper
        r54 = 0 * ones
        
        # free_2 (5) <-> DRX (1)
        r51 = .1 * ones 
        r15 = .01 * ones
        
        # # SRX (6) <-> DRX (1)
        # # rate equation from https://doi.org/10.1085/jgp.202012604 pages 6 and 8
        k_0 = .1 # 5/s
        k_max = .4 # 400/s
        b = 1.
        Ca_50 = 10**-6.0 # 10**-pca at which k_0==kmax/2
        # to drx
        r61 = (k_0 + ((k_max-k_0)*ca_c**b)/(Ca_50**b + ca_c**b)) * ones 
        # to srx
        r16 = rate_factors[:,3] * .1 * ones   
        
        # diagonal rates should be set so that each row in the rate matrix sums to 0
        r11 = - (r12 * ap + r15 + r16)        
        r22 = - (r21 + r23)
        r33 = - (r32 + r34)
        r44 = - (r43 + r45)
        r55 = - (r54 + r51)
        r66 = - r61
        
        # ############################################################################################################################################
        #
        #
        #
        #
        #   Myosin head
        #   Rate function definitions above
        # 
        #
        #
        #
        ############################################################################################################################################
        
        
        Q = np.zeros((720,6,6))
        
        Q[:,0,0] = r11
        Q[:,0,1] = r12 * ap
        Q[:,0,4] = r15
        Q[:,0,5] = r16
        
        Q[:,1,0] = r21
        Q[:,1,1] = r22
        Q[:,1,2] = r23
        
        Q[:,2,1] = r32
        Q[:,2,2] = r33
        Q[:,2,3] = r34
        
        Q[:,3,2] = r43
        Q[:,3,3] = r44
        Q[:,3,4] = r45
        
        Q[:,4,3] = r54
        Q[:,4,4] = r55
        Q[:,4,0] = r51
        
        Q[:,5,0] = r61
        Q[:,5,5] = r66


        
        # the final array looks like
        # Q = np.array([
        #               [r11,  r12,   0,    0,    r15,  r16],            [0,0]  [0,1]  [0,2]  [0,3]  [0,4]  [0,5]
        #               [r21,  r22,   r23,  0,    0,    0],              [1,0]  [1,1]  [1,2]  [1,3]  [1,4]  [1,5]
        #               [0,    r32,   r33,  r34,  0,    0],              [2,0]  [2,1]  [2,2]  [2,3]  [2,4]  [2,5]
        #               [0,    0,     r43,  r44,  r45,  0],              [3,0]  [3,1]  [3,2]  [3,3]  [3,4]  [3,5]
        #               [r51,  0,     0,    r54,  r55,  0],              [4,0]  [4,1]  [4,2]  [4,3]  [4,4]  [4,5]
        #               [r61,  0,     0,    0,    0,    r66]             [5,0]  [5,1]  [5,2]  [5,3]  [5,4]  [5,5]
        #               ])
        
        return Q
    
    @staticmethod
    @njit
    def compare_states(a,b):
        
        '''
        
        given two lists of old and new states, returns the ids of xbs which need to bind, 
        unbind, or do nothting (remaing bound or unbound)
        
        a = old state list
        b = new state lits
        
        {1: "DRX", 2: "loose", 3: "tight_1", 4:"tight_2", 5:"free_2", 6:"SRX"} 
        
        
        # binidng transitions are from {1,5,6} to {2,3,4}
        
        # unbinding transitions are from {2,3,4} to {1,5,6}
    
        # bound to bound transitions are from {3,4,5} to {3,4,5}, 
        
        # unbound to unbound transitions are from {1,5,6} to {1,5,6}, 
        
        '''
        
        bind_ids = []
        unbind_ids = []
        do_nothing_ids = []
        
        for i in range(0,len(a)):
            
            if (a[i] == 1 or a[i]==5 or a[i]==6) and (b[i] == 2 or b[i]==3 or b[i]==4):
                bind_ids.append(i)
                
            elif (a[i] == 2 or a[i]==3 or a[i]==4) and (b[i] == 1 or b[i]==5 or b[i]==6):
                unbind_ids.append(i)
                
            elif ((a[i] == 2 or a[i]==3 or a[i]==4) and (b[i] == 2 or b[i]==3 or b[i]==4)) or ((a[i] == 1 or a[i]==5 or a[i]==6) and (b[i] == 1 or b[i]==5 or b[i]==6)):
                do_nothing_ids.append(i)
                
        return bind_ids, unbind_ids, do_nothing_ids

        
    def thick_transitions(self, dt = None):
        
        '''
        
        Transitions every crossbridge into a new state based on it's current configuration.
        
        
                               6
                              SRX 
                               ^
                               |
                               v
        Tight_2 => Free_2 <=>  DRX  <=> loose <=> tight_1 <=> tight_2
                            
           4         5         1         2         3          4
        
        
        
        To get probablity of a transition, we first find the rate matrix Q:
            
            Q =      [r11,  r12*ap,  0,    0,    r15,  r16], 
                     [r21,  r22,     r23,  0,    0,    0],
                     [0,    r32,     r33,  r34,  0,    0],
                     [0,    0,       r43,  r44,  r45,  0],
                     [r51,  0,       0,    r54,  r55,  0],
                     [r61,  0,       0,    0,    0,    r66] 
                     
        Eeach element rij has units of 1/ms, and r12 is multiplied by the actin permissiveness (0 or 1, signifying binding is possible or not). 
        The rows of Q should sum to 0
                     
        Then the probability of a transition is:
            
            P = expm(Q*dt) 
        
        where expm() is the matrix exponential (note that np.exp() only gives element-wise exponential). 
        Rows of P should sum to 1, and will if rate matrix rows sum to 0.
        Also, if the norm of Q is too large, it will be impossible to find expm(Q). So we set rates at a max of 10^6 in the definitions of r21 and r32. 
        
        elements of P are       
                     #     p11  p12  p13  p14  p15  p16                 [0,0]  [0,1]  [0,2]  [0,3]  [0,4]  [0,5]
                     #     p21  p22  p23  p24  p25  p26                 [1,0]  [1,1]  [1,2]  [1,3]  [1,4]  [1,5]
                     #     p31  p32  p33  p34  p35  p36                 [2,0]  [2,1]  [2,2]  [2,3]  [2,4]  [2,5]
                     #     p41  p42  p43  p44  p45  p46                 [3,0]  [3,1]  [3,2]  [3,3]  [3,4]  [3,5]
                     #     p51  p52  p53  p54  p55  p56                 [4,0]  [4,1]  [4,2]  [4,3]  [4,4]  [4,5]
                     #     p61  p62  p63  P64  p56  p66                 [5,0]  [5,1]  [5,2]  [5,3]  [5,4]  [5,5]
        
        The element Pij gives the probability that a myosin head which starts in state i will be in state j after a time dt. 
        
        
        First, get every crossbridge's configuration and rate constants as vectors:
            bs - (x,y) distance between the crossbridge and it's nearest actin binding site neighbor 
            V  - vector of spring stiffness (k) and rest points (r_0) for both torsional and linear springs in both weak and strong states: 
                2*2*2=8 values needed to calculate energy 
            ap = vector of zeros and ones of whether the nearest neighbor actin binding site is possible to bind to (meaning the binding 
                                                                                                                     site the xb is 
                                                                                                                     neaest must be in state 3)
            ca_c - calcium concentration in Molar, used for srx->drx xb transitions
            
        pass all this to self.xb_rate_matrix, to return the [720 x 6 x 6] array of rate matricies, which we need to convert to probabilities
        each 6 x 6 slice is the rate matrix Q of a crossbridge with element Qij being the rate of i changing to j
        its 6 x 6 because there are 6 states xbs can exist in
        
        P = expm(Q * dt) is the matrix of probabilites, with Pij being prob ending up in state j after time dt if you started in state i
        ** expm is matrix exponential, different from element wise exponential, np.exp will only give element wise **
        
        for each [6 x 6] slice in P, get the row i corresponding to the current state
        
        roll 720 random numbers to comare with the 720 [1 x 6] row vectors of probability
        
        set new states, and binding status for each xb
        
        
        _____________
        
        We can set dt = 1000ms to get the stochastic approximate quasi-static (long term) behavior
        This doesn't account for cooperative effects (explicit or implicit), so its more useful for initiliznig teh half sarc.
        for example, in doing force-pca curves at low calcium, 
        
        
        '''
        # set dt, the timestep in P = expm(Q * dt)
        if dt is None:
            dt = self.timestep_len
        
        # current Ca2+ concentration in M
        ca_c = self.ca 
        
        # current lattice spacing
        ls = self.lattice_spacing
      
        # (x,y) distance between each xb and its nearest binding site
        bs = np.array([(xb.nearest.axial_location - xb.axial_location, ls) for th in self.thick for cr in th.crowns for xb in cr.crossbridges])

        # get a list of the old (i.e. current) states
        old_states = np.array([xb.numeric_state for th in self.thick for cr in th.crowns for xb in cr.crossbridges])
    
        # globular = r, converter = theta, 
        # w = weak aka loose, s = strong aka tight
        # V is all the k and r values for the springs which make up the globular and converter domain for weak and strong states of the xbs
        # we get all the values we need here so we can vecotroize rate calculations with numpy instead of looping through each xb
        V = np.array([[
            xb.g.k_w, 
            xb.g.r_w, 
            xb.c.k_w, 
            xb.c.r_w, 
          
            xb.g.k_s, 
            xb.g.r_s, 
            xb.c.k_s, 
            xb.c.r_s, 
             ] for th in self.thick for cr in th.crowns for xb in cr.crossbridges]) # in cr.crossbridges for cr in th.crowns for th in self.thick]

        # list of all xbs
        xbs = [xb for th in self.thick for cr in th.crowns for xb in cr.crossbridges]

        # get constants corresponding to each xb's species type
        # we will multiplyy certain rates by the factors in this array:
            # rate_factors[:,0] -> mh_srx (r16)
            # rate_factors[:,1] -> mh_br (r12)
            # rate_factors[:,2] -> mf_dr (r45)
        # pdb.set_trace()
        rate_factors = np.array([list((
            xb.constants['mh_br'],
            xb.constants['mh_r34'],
            xb.constants['mh_dr'],
            xb.constants['mh_srx'],
            xb.constants['mh_r23'],
            )) for xb in xbs])
        
        # rate_factors = np.array([list(xb.constants.values()) for xb in xbs])

        # list of actin bs availability status
        ap = np.array([xb.nearest.permissiveness for th in self.thick for cr in th.crowns for xb in cr.crossbridges])

        # Q gets exported to a function compiled with numba njit
        # Q is the (720 x 6 x 6) array of rate matricies, each 6 by 6 slice is the transition rate matrix of one of the 720 xbs
        Q = self.xb_rate_matrix(bs, V, rate_factors, ap, ca_c, temp = self.temp)

        # each row in each Q should sum to 0
        # if it doesn't you fucked up
        # don't manipulate Q outsied of xb_rate_matrix function
        # assert(np.max(np.abs(np.sum(Q[:,:,:], axis = 2)))<10**-11)
        if ~(np.max(np.abs(np.sum(Q[:,:,:], axis = 2)))<10**-11):
            pdb.set_trace()

        # From rate matricies Q, we get the Prob matrices P
        # P is the (720 x 6 x 6) array of Prob matricies, each 6 by 6 slice is the transition PROBABILITY matrix of one of the 720 xbs
        P = expm_(Q * dt, q=3) #self.xbs_probs_from_rates(Q, bs, ap, dt, old_states)
        
        # pdb.set_trace()
        # print(np.max(np.abs(np.array([scipy.linalg.expm(q*dt) for q in Q]) - expm_(Q*dt, q=0))))
        
        # for each of the 6,6 probability matrices, we want the row which corresponds to that xbs current state, 
        # for example since P is a matrix with values Pij, we want row i since we want transition probabilities from state i to j
        Prob = np.empty((720,6))
        for i in range(0,720):
            # xb numeric_state goes from 1-6, (rather than 0-5) so we need -1 here
            Prob[i,:] = P[i,old_states[i]-1,:]
        
        # cumsum of each of the prob vectors, to compare against random number
        Prob_array = np.cumsum(Prob, axis=1)
        
        # 720 random numbers [0,1] for stochastic state transisitons for each xb
        p = np.random.rand(720,1)
        
        # for each xb i, the new state is the first entry in each probability vector Prob_array[i] smaller than p[i]
        # we need to then add back 1 since xb numeric_state goes from 1-6, (rather than 0-5)
        new_states = np.argmax(p < Prob_array, axis=1) + 1
        
        # compare old and new states do determine which undergo binding, unbinding, or no change in binding transisitinso. 
        # self.compare_states is compiled with numba as a separate function for speed
        bind_ids, unbind_ids, do_nothing_ids = self.compare_states(old_states, new_states)
        
        # convert state id number to string, since that's how we'll store state names
        num_2_string = {1: "DRX", 2: "loose", 3: "tight_1", 4: "tight_2", 5: "free_2", 6: "SRX"}
        new_states_ = [num_2_string[i] for i in new_states]
        
        # binding transitions are {1,5,6} => {2,3,4,}
        for ind in bind_ids:
            
            # xbs can only bind to a bs site that is not already bound
            # if xb.nearest is unoccupied, then bind
            if xbs[ind].nearest.bound_to is None: 
                # it should only be possible to bind if tm site state is on
                if xbs[ind].nearest.tm_site.state == 3:
                    # set xb.bound_to as the nearest bs
                    xbs[ind].bound_to = xbs[ind].nearest
                    # set that bs as bound to current xb
                    xbs[ind].bound_to.bound_to = xbs[ind]
                    # update xb state
                    xbs[ind].state = new_states_[ind]
                else:
                    # this should never happen..... if it does something is wrong
                    print(r'shouldnt bind, tm state = ' + str(xbs[ind].nearest.tm_site.state)) #
                    xbs[ind].state = "DRX"
                    xbs[ind].bound_to = None
                    # pdb.set_trace()
            
            # if xb.nearest is already bound, just set set state to DRX and don't bind
            elif xbs[ind].nearest.bound_to is not None:
                xbs[ind].state = "DRX"
                xbs[ind].bound_to = None
                
        # unbinding transitions are {2,3,4} => {1,5,6}
        for ind in unbind_ids:
            
            # first unlink the xb from the bs
            xbs[ind].bound_to.bound_to = None
            # then the bs from the xb
            xbs[ind].bound_to = None
            # update state
            xbs[ind].state = new_states_[ind]
            
        # do_nothing_ids conatin bound->bound or unbond->unbound transistions, so we only need
        # to update the state, not the binding status
        for ind in do_nothing_ids:
            # 'do_nothing' is a misnomer, we still need to change the state, just don't change binding status
            xbs[ind].state = new_states_[ind]

        return 
       
    def set_xb_nearest_binding_site(self):
        '''
        
        Sets each crossbrige's nearest actin binding site neighbor. 
        
        
        Each thick and thin filament is sub-divided into 'faces' which are matched to faces on the opposite filament types.
        So a thick face and thin face have a particular orientation and can only interact with each other
        
        so we cycle through each thick filament face, and get every xb on that face and bs on the matching thin filament face
        then find the nearest bs for each xb
        
        
        '''
        
        for th in self.thick:
            for th_face in th.thick_faces:
                
                # get the ids of the xbs on this thick face
                xb_id = [i.index for i in th_face.xb]
                # get the ids of the bs sites on this thick face's matching thin face
                bs_id = [i.index for i in th_face.thin_face.binding_sites]
                
                # the plus 13 is b/c we want the binding site closest to the head of the xb, 
                # not the one closest to the base of the xb and 19.9*cos(47 degrees) = 13
                # the 'nearest' bs is really the one for which r12=exp(-E/kt) is greatest, so 
                # this should really depend on lattice spacing, but bs sites on a face are separated by 36 nm
                # so + 13 should be good enough
                xb_axial = th.axial[xb_id] + 13 
                bs_axial = self.thin[th_face.thin_face.parent_thin.index].axial[bs_id]
                
                # dist matrix, every xb's distance on this thick face to every bs on the matching thin face, 
                # in other words, element i,j is the ith xb's distance to the jth bs site, for this thick face - thin face pair
                dist = np.abs(xb_axial[:, np.newaxis] - bs_axial)
                # get the index of the single closest bs to each xb
                Closest_bs = dist.argmin(axis=1)
                
                # iterate through the xb's on this thick face and assign nearest
                for i, xb in enumerate(th_face.xb):
                    xb.nearest = xb.thin_face.binding_sites[Closest_bs[i]]
                    
                    
        return
                
    def Update_axial_locations(self, new_axial_locations):
        '''
        
        Use the output of X = self.Newton() and update the axial location of all spring nodes.
        X should be of size 1 by (total_xb_num + total_bs_num)=960
        
        There are 60 nodes in each of the 4 thick filaments
        
        There are 90 nodes in each of the 8 thin filaments
        
        
        '''   
        
        # 60 nodes in each of the 4 thick filament
        self.thick[0].axial = new_axial_locations[60*0:60*0+60]
        self.thick[1].axial = new_axial_locations[60*1:60*1+60]
        self.thick[2].axial = new_axial_locations[60*2:60*2+60]
        self.thick[3].axial = new_axial_locations[60*3:60*3+60]
        
        # 90 nodes in each of the 8 thin filaments, + 240 offset to account for 60*4 xb nodes
        self.thin[0].axial = new_axial_locations[240 + 0*90: 240 + 0*90 +90]
        self.thin[1].axial = new_axial_locations[240 + 1*90: 240 + 1*90 +90]
        self.thin[2].axial = new_axial_locations[240 + 2*90: 240 + 2*90 +90]
        self.thin[3].axial = new_axial_locations[240 + 3*90: 240 + 3*90 +90]
        self.thin[4].axial = new_axial_locations[240 + 4*90: 240 + 4*90 +90]
        self.thin[5].axial = new_axial_locations[240 + 5*90: 240 + 5*90 +90]
        self.thin[6].axial = new_axial_locations[240 + 6*90: 240 + 6*90 +90]
        self.thin[7].axial = new_axial_locations[240 + 7*90: 240 + 7*90 +90]
        
        return

    def Newton(self):
        '''
        Newton's method nonlinear solver finds x0 such that F(x0)=0
        
        iteratively updates the guess x by:
            
            x_new_guess = x_old_guess - f(x_old_guess) / f'(x_old_guess)
        
        Jacobian is the matrix form of f' containing the entries df_i/dx_j
        
        see: https://en.wikipedia.org/wiki/Newton%27s_method#Systems_of_equations
        '''
        
        # current force
        F = np.concatenate((np.concatenate([i.axial_force() for i in self.thick]), 
                                     np.concatenate([i.axial_force() for i in self.thin])))
        
        num = 0
        # iterate as long as abs(F) > .06, arbitrary number
        while np.max(np.abs(F)) > .06:
            
            # initial guess is current configuration
            guess = np.concatenate((np.concatenate([i.axial for i in self.thick]), np.concatenate([i.axial for i in self.thin])))
            
            # get Jacobian matrix J for current sarcomere configuration
            # elements i,j are the derivative of the force on node i with respect to axial location of node j
            # see self.Jacobian_matrix and hs_stiffness_matrix
            J = self.Jacobian_Matrix()
            
            # solve J*delta_guess = F with sparse matrix solver spsolve, 
            # solve(a,b) is faster taking inverse
            new_guess = guess - spsolve(J, F)
            
            if np.any(np.isnan(new_guess)):
                pdb.set_trace()
            
            # update new axial spacings
            self.Update_axial_locations(new_guess)
                     
            # find new F - rerun if max(abs(F)) < cut-off value of .06 pN  
            F = np.concatenate((np.concatenate([i.axial_force() for i in self.thick]), 
                                         np.concatenate([i.axial_force() for i in self.thin])))
            
            F_ = self.Force_on_each_node()
            
            num = num + 1
            if num > 100:
                # pdb.set_trace()
                # this should never happen, if it does something is wrong
                print('failed to converge after ' + str(num) + ' steps, max f = ' + str(np.max(np.abs(F))))
                # revert to old method
                # pdb.set_trace()
                self.settle()
                break
            
        return
            
    @property
    def current_timestep(self):
        """Return the current timestep"""
        return self._current_timestep

    @current_timestep.setter
    def current_timestep(self, new_timestep):
        """Set the current timestep"""
        # Update boundary conditions
        self.update_hiding_line()
        td = self.time_dependence
        i = new_timestep
        if td is not None:
            if 'lattice_spacing' in td:
                self.lattice_spacing = td['lattice_spacing'][i]
            if 'z_line' in td:
                self.z_line = td['z_line'][i]
            if 'pCa' in td:
                self.pCa = td['pCa'][i]
                self.ca = 10 ** (-td['pCa'][i]) # store concentration at half sarcomere level, to prevent recalculating at every site at every timestep
        self._current_timestep = i
        return

    @property
    def actin_permissiveness(self):
        """How active & open to binding, 0 to 1, are binding sites?"""
        return [thin.permissiveness for thin in self.thin]

    @property
    def z_line(self):
        """Axial location of the z-line, length of the half sarcomere"""
        return self._z_line

    @z_line.setter
    def z_line(self, new_z_line):
        """Set a new z-line, updating the lattice spacing at the same time"""
        self._z_line = new_z_line
        self.update_ls_from_poisson_ratio()
        self.update_volume()

    @property
    def lattice_spacing(self):
        """Return the current lattice spacing"""
        return self._lattice_spacing

    @lattice_spacing.setter
    def lattice_spacing(self, new_lattice_spacing):
        """Assign a new lattice spacing"""
        self._lattice_spacing = new_lattice_spacing

    @property
    def _tn_count(self):
        return self.tn_total - self.tnca_count

    @property
    def tnca_count(self):
        bound = 0
        for thin in self.thin:
            for tm in thin.tm:
                for tm_site in tm.sites:
                    if tm_site.state != 0:
                        bound += 1
        return bound

    @property
    def _tm_open(self):
        uncovered = 0
        for thin in self.thin:
            for tm in thin.tm:
                for tm_site in tm.sites:
                    if tm_site.state == 3:
                        uncovered += 1
        return uncovered

    @property
    def tn_total(self):
        total_tn = 0
        for thin in self.thin:
            for tm in thin.tm:
                total_tn += len(tm.sites)
        return total_tn

    def update_concentrations(self):
        self.c_tn = self._concentration(self._tn_count)
        self.c_tnca = self._concentration(self.tnca_count)

    @property
    def c_ca(self):
        return 10.0 ** (-self.pCa)

    @property
    def concentrations(self):
        return {"free_tm": self.c_tn,
                "free_ca": self.c_ca,
                "bound_tm": self.c_tnca}

    @property
    def volume(self):
        """return the current fluid volume of the half-sarcomere AMA-11JAN2020"""
        return self._volume

    # @volume.setter
    def update_volume(self):
        """re-calculate the fluid volume of the half sarcomere - ASSUMING CONSTANT LATTICE SPACING
        returns volume in L(AMA-3FEB2020)
        (Used to be and no longer is nm3 AMA-11JAN2020)"""
        ls = self._lattice_spacing
        length = self._z_line

        # calculate area of 4 hexagons, with edge length 9/2 + 16/2 + ls
        edge = 9 / 2 + 16 / 2 + ls
        area = 4 * 3 / 2 * np.sqrt(3) * edge * edge

        thin_volume = np.pi * 22659.75  # radius 4.5 * radius 4.5 * length 1119
        thick_volume = np.pi * 58624  # radius 8 * radius 8 * length 916
        filament_volume = 10 * thin_volume + 4 * thick_volume
        whole_volume = area * length
        fluid_volume = whole_volume - filament_volume
        nm3_p_L = 1e-24
        fluid_volume *= nm3_p_L
        self._volume = fluid_volume

    def _concentration(self, count):
        return count / self.volume

    @staticmethod
    def ls_to_d10(face_dist):
        """Convert face-to-face lattice spacing to d10 spacing.

        Governing equations:
            ls = ftf, the face to face distance
            filcenter_dist = face_dist + .5 * dia_actin + .5 * dia_myosin
            d10 = 1.5 * filcenter_dist
        Values:
            dia_actin: 9nm [1]_
            dia_myosin: 16nm [2]_
            example d10: 37nm for cardiac muscle at 2.2um [3]_
        References:
            .. [1] Egelman 1985, The structure of F-actin.
                   J Muscle Res Cell Motil, Pg 130, values from 9 to 10 nm
            .. [2] Woodhead et al. 2005, Atomic model of a myosin filament in
                   the relaxed state. Nature, Pg 1195, in tarantula filament
            .. [3] Millman 1998, The filament lattice of striated muscle.
                   Physiol Rev,  Pg 375
        Note: Arguably this should be moved to a support class as it really
        isn't something the half-sarcomere knows about or does. I'm leaving it
        here as a convenience for now.

        Parameters:
            face_dist: face to face lattice spacing in nm
        Returns:
            d10: d10 spacing in nm
        """
        filcenter_dist = face_dist + 0.5 * 9 + 0.5 * 16
        d10 = 1.5 * filcenter_dist
        return d10

    @staticmethod
    def d10_to_ls(d10):
        """Convert d10 spacing to face-to-face lattice spacing

        Governing equations: See ls_to_d10
        Values: See ls_to_d10

        Parameters:
            d10: d10 spacing in nm
        Returns:
            face_dist: face to face lattice spacing in nm
        """
        filcenter_dist = d10 * 2 / 3
        face_dist = filcenter_dist - 0.5 * 9 - 0.5 * 16
        return face_dist

    def axial_force(self):
        """Sum of each thick filament's axial force on the M-line """
        return sum([thick.effective_axial_force() for thick in self.thick])

    def titin_axial_force(self):
        """Sum of each thick filament's axial force on the M-line """
        return sum([titin.axial_force() for titin in self.titin])

    def radial_tension(self):
        """The sum of the thick filaments' radial tensions"""
        return sum([t.radial_tension() for t in self.thick])

    def radial_force(self):
        """The sum of all of the thick filaments' radial forces, as a (y,z) vector"""
        # pdb.set_trace()
        # count, value in enumerate(values)
        radial_forces = []
        for thick in self.thick:
            for cr in thick.crowns:
                for ind, xb in enumerate(cr.crossbridges):
                    pass
                
                    if xb.state in {'loose', 'tight_1', 'tight_2'}:
                        
                        F_mag = xb.radial_force()
                        orient = cr.orientations[ind]
                        
                        # force_mag = crossbridge.radial_force()
                        radial_forces.append(np.multiply(F_mag, orient))
                        
                        # print(xb.radial_force(), cr.orientations[ind] )
                        
                        
        if not radial_forces:
            return [0., 0.]
        else:
            return np.sum(radial_forces, 0)
        
        
        # return np.sum([t.radial_force_of_filament() for t in self.thick], 0)

    def _single_settle(self, factor=0.95):
        """Settle down now, just a little bit"""
        thick = [thick.settle(factor) for thick in self.thick]
        thin = [thin.settle(factor) for thin in self.thin]
        return np.max((np.max(np.abs(thick)), np.max(np.abs(thin))))

    def settle(self):
        """Jiggle those locations around until the residual forces are low

        We choose the convergence limit so that 95% of thermal forcing events
        result in a deformation that produces more axial force than the
        convergence value, 0.12pN.
        """
        converge_limit = 0.12  # see doc string
        converge = self._single_settle()
        while converge > converge_limit:
            converge = self._single_settle()

    def _get_residual(self):
        """Get the residual force at every point in the half-sarcomere"""
        thick_f = np.hstack([t.axial_force() for t in self.thick])
        thin_f = np.hstack([t.axial_force() for t in self.thin])
        mash = np.hstack([thick_f, thin_f])
        return mash

    def get_xb_frac_in_states(self):
        """Calculate the fraction of cross-bridges in each state
        
        see https://doi.org/10.1085/jgp.202012604 page 5
        
        SRX = parked state PS
        DRX = M.D.Pi
        loose = A.M.D.Pi
        tight_1 = A.M.D
        tight_2 = A.M
        free_2 = M.T
        
        {"DRX": 1, "loose": 2, "tight_1": 3, "tight_2": 4, "free_2": 5, "SRX": 6}
        
        """
        nested = [t.get_states() for t in self.thick]
        xb_states = [xb for fil in nested for face in fil for xb in face]
        num_in_state = [xb_states.count(state) for state in range(1,7)]
        frac_in_state = [n / float(len(xb_states)) for n in num_in_state]
        return frac_in_state

    def get_tm_frac_in_states(self):
        """Calculate the fraction of tm_sites in each state"""
        nested = [t.get_states() for t in self.thin]
        tm_states = [xb for fil in nested for face in fil for xb in face]
        num_in_state = [tm_states.count(state) for state in range(4)]
        frac_in_state = [n / float(len(tm_states)) for n in num_in_state]
        return frac_in_state

    def tm_rates(self):
        """Average rates of the contained TmSites (for monitoring)"""
        
        # rates = None
        # for thin in self.thin:
        #     if rates is None:
        #         rates = thin.get_tm_rates()
        #     else:
        #         for key, value in thin.get_tm_rates().items():
        #             rates[key] += value
        # for key in rates:
        #     rates[key] /= len(self.thin)

        # Qs = [bs.tm_site.Q_matrix for thin in self.thin for bs in thin.binding_sites]
        
        # average_tm_site_rates = np.mean(Qs, axis=0)
        
        average_tm_site_rates = self.tm_rates_ #np.mean(Qs, axis=0)
        
        rates = {
            'tm_rate_12': average_tm_site_rates[0,1], 
            'tm_rate_21': average_tm_site_rates[1,0], 
            'tm_rate_23': average_tm_site_rates[1,2], 
            'tm_rate_32': average_tm_site_rates[2,1], 
            'tm_rate_34': average_tm_site_rates[2,3], 
            'tm_rate_43': average_tm_site_rates[3,2], 
            'tm_rate_41': average_tm_site_rates[3,0], 
            'tm_rate_14': average_tm_site_rates[0,3]
            }
        
        return rates

    def update_ls_from_poisson_ratio(self):
        """Update the lattice spacing consistent with the poisson ratio,
        initial lattice spacing, current z-line, and initial z-line

        Governing equations
        ===================
        Poisson ratio := ν
            ν = dε_r/dε_z = Δr/r_0 / Δz/z_0
        From Mathematica derivation
        γ := center to center distance between filaments
            γ(ν, γ_0, z_0, Δz) = γ_0 (z_0/(z_0+Δz))^ν
        And since we want the face-to-face distance, aka ls, we convert with:
            γ = ls + 0.5 (dia_actin + dia_myosin)
        and
            γ_0 = ls_0 + 0.5 (dia_actin + dia_myosin)
        and the simplifying
            β = 0.5 (dia_actin + dia_myosin)
        to get
            ls = (ls_0 + β) (z_0/(z_0 + Δz))^ν - β
        which is what we implement below.
        Note: this is a novel derivation and so there is no current
            citation to be invoked.

        Values: See ls_to_d10

        Parameters:
            self: half-sarcomere, automatically passed
        Returns:
            None
        """
        beta = 0.5 * (9 + 16)
        ls_0 = self._initial_lattice_spacing
        z_0 = self._initial_z_line
        nu = self.poisson_ratio
        dz = self.z_line - z_0
        ls = (ls_0 + beta) * (z_0 / (z_0 + dz)) ** nu - beta
        self.lattice_spacing = ls
        return

    def update_hiding_line(self):
        """Update the line determining which actin sites are unavailable"""
        farthest_actin = min([min(thin.axial) for thin in self.thin])
        self.hiding_line = -farthest_actin

    def resolve_address(self, address):
        """Give back a link to the object specified in the address
        Addresses are formatted as the object type (string) followed by a list
        of the indices that the object occupies in each level of organization.
        Valid string values are:
            thin_fil
            thin_face
            bs
            thick_fil
            crown
            thick_face
            xb
        and an example valid address would be ('bs', 1, 14) for the binding
        site at index 14 on the thin filament at index 1.
        """
        if address[0] == 'thin_fil':
            return self.thin[address[1]]
        elif address[0] in ['thin_face', 'bs', 'tm', 'tm_site']:
            return self.thin[address[1]].resolve_address(address)
        elif address[0] == 'thick_fil':
            return self.thick[address[1]]
        elif address[0] in ['crown', 'thick_face', 'xb']:
            return self.thick[address[1]].resolve_address(address)
        import warnings
        warnings.warn("Unresolvable address: %s" % str(address))


sarc = hs()
