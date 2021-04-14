#!/usr/bin/env python
# encoding: utf-8
"""
ti.py - A titin filament, with variable compliance

MORE ABOUT HOW TITIN WORKS

Created by Joe Powers and Dave Williams on 2017-02-17
"""

import numpy as np
import warnings
import numpy.random as random


class Titin:
    """This is all about the titin filament"""

    # kwargs that can be used to edit titin phenotype
    # titin can also accept phenotype profiles
    VALID_PARAMS = {'ti_a': "pN", 'ti_b': "nm-1"}

    def __init__(self, parent_lattice, index, thick_face, thin_face, **ti_params):
        """Initialize the titin filament.

        Parameters:
            parent_lattice: calling half-sarcomere instance
            index: which titin filament this is (0-23)
            thick_face: List of thick filament faces' numerical orientation (0-5)
            thin_face: List of thin filament faces' numerical orientation (0-2)
            ti_params:  key-worded dictionary of parameters handled at the end of initialization,
            to override default constants, recorded in constant dictionary
        Returns:
            None
        """
        # noinspection PyArgumentList
        random.seed()  # Ensure proper seeding

        # Name of the titin molecule
        self.index = index
        self.address = ('titin', index)
        # A connection to the parent lattice
        self.parent_lattice = parent_lattice
        # Location of the titin filament relative thick filament
        self.thick_face = thick_face
        # Link titin to the thick filament face
        self.thick_face.link_titin(self)
        # location of the titin filament relative to the thin filament
        self.thin_face = thin_face
        # Link titin to that face of the thin filament
        self.thin_face.link_titin(self)
        # ## And now we declare titin properties that will be used to
        # ## calculate forces
        self.rest = 120  # nm, no citation TODO cite
        # Create the constants that determine force
        self.a = 240        # pN
        self.b = 0.0045     # nm-1

        """Handle ti_params"""
        # ## Handle ti_isomer calculations
        if 'ti_iso' in ti_params.keys():  # !!! This means we don't actually have settings to pass yet !!!
            profiles = ti_params['ti_iso']
            cum_sum = 0
            rolled_val = random.random()  # get the rolled value
            i = 0
            while cum_sum < rolled_val:
                probability = float(profiles[i]['iso_p'])
                cum_sum += probability
                i += 1
            ti_params = ti_params['ti_iso'][i - 1].copy()  # actually select the params and proceed as normal
            ti_params.pop('iso_p')

        """Handle key-worded ti_params - overriding set values"""
        self.constants = {}

        # Titin constant a
        if "ti_a" in ti_params.keys():
            self.a = ti_params.pop("ti_a")
        self.constants['ti_a'] = self.a

        # Titin constant b
        if "ti_b" in ti_params.keys():
            self.b = ti_params.pop("ti_b")
        self.constants['ti_b'] = self.b

        # Print kwargs not digested
        for key in ti_params.keys():
            print("Unknown ti_param:", key)

    def to_dict(self):
        """Create a JSON compatible representation of titin

        Usage example: json.dumps(titin.to_dict(), indent=1)

        Current output includes:
            a: force constant
            b: force constant
            address: largest to most local, indices for finding this
            rest: rest length
        """
        td = self.__dict__.copy()
        td.pop('index')
        td.pop('parent_lattice')
        td['thick_face'] = td['thick_face'].address
        td['thin_face'] = td['thin_face'].address
        return td

    def from_dict(self, td):
        """ Load values from a thin face dict. Values read in correspond to
        the current output documented in to_dict.
        """
        # Check for index mismatch
        read, current = tuple(td['address']), self.address
        assert read == current, "index mismatch at %s/%s" % (read, current)
        # Local keys
        self.a = td['a']
        self.b = td['b']
        self.rest = td['rest']
        self.thick_face = self.parent_lattice.resolve_address(
            td['thick_face'])
        self.thin_face = self.parent_lattice.resolve_address(
            td['thin_face'])

    def angle(self):
        """Calculate the angle the titin makes relative to thick filament"""
        act_loc = self.thin_face.parent_thin.parent_lattice.z_line
        myo_loc = self.thick_face.get_axial_location(-1)
        ls = self.parent_lattice.lattice_spacing
        angle = np.arctan2(ls, act_loc - myo_loc)
        return angle

    def length(self):
        """Calculate the length of the titin filament"""
        act_loc = self.thin_face.parent_thin.parent_lattice.z_line
        myo_loc = self.thick_face.get_axial_location(-1)
        ls = self.parent_lattice.lattice_spacing
        length = np.sqrt((act_loc - myo_loc) * (act_loc - myo_loc) + ls * ls)
        return length

    def stiffness(self):
        """Need instantaneous stiffness at the current length to normalize
        force for settling at each timestep. We get this as the derivative of
        force with respect to x. D[a*exp(b*x), x] = a*b*exp(b*x)
        """
        return self.force() * self.b

    def force(self):
        """Calculate the total force the titin filament exerts"""
        length = self.length()
        if length < self.rest:
            return 0  # titin buckles
        else:
            # # ## Exponential Model
            x = length - self.rest
            return self.a * np.exp(self.b * x)
        # else:
        # Cubic model
        # x = length - self.rest
        # a = 3.25e-5
        # b = -5e-3
        # c = 9e-1
        # d = -7e-5
        # return a*(x**3) + b*(x**2) + c*x + d #Linke et al. PNAS 1998
        # else:
        # # ## Sawtooth Model
        # extension = length - self.rest
        # a = 1.2e-6
        # b = 3
        # c = 0.5e-6#*random.rand()
        # d = 34
        # e = 3
        # return a*extension**b + c*np.fmod(extension,d)**e
        # else:
        # # ## Hookean Model
        # return self.k * (self.length() - self.rest)

    def axial_force(self):
        """Return the total force the titin filament exerts on the
        thick filament's final node, (negate for the thin filament side)
        """
        return self.force() * np.cos(self.angle())  # ## CHECK_JDP ## # TODO Check

    def radial_force(self):
        """Return the force in the radial direction (positive is compressive)
        TODO: The 'positive is compressive' part needs to be double checked
        """
        warnings.warn("Check radial force direction in titin")
        return self.force() * np.sin(self.angle())
