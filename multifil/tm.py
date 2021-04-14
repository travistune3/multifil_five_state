#!/usr/bin/env python
# encoding: utf-8
"""
tm.py - A tropomyosin filament
Create and maintain a tropomyosin filament and the subgroups that comprise it.
Created by Dave Williams on 2017-12-03.
"""
from _operator import indexOf

import numpy as np
random = np.random


class TmSite:
    """A single tm site, located over a single actin binding site. 
    
    Individual tm sites keep track of how the tropomyosin chain is regulating
    the binding site: states, links to nearest neighbors, etc
    Kinetics
    --------
    The equilibrium state of a transition, K, is given by the ratio of forward
    to reverse transition rates, K = k_forward/k_reverse. Taking our kinetics
    from Tanner 2007 we define the three equilibrium states and their attendant
    forward transition rates. The equilibrium rates, K1, K2, and K3,
    correspond to the transitions thus:
    
        Tm+Tn+Ca <-K1-> Tm+Tn.Ca <-K2-> Tm.Tn.Ca <-K3-> Tm+Tn+Ca
    
    The forward transition rates are labeled as k_12, k_23, k_31. Reverse
    transitions are calculated from the above values. K1 and K2 are explicitly
    defined. In Tanner2007 K3 is stated to be obtained from the balancing 
    equation for equilibrium conditions, K1*K2*K3 = 1, but is instead defined
    directly. This implies non-equilibrium conditions, fair enough. 
    Only K1 is dependent on [Ca], although it would make sense for K3 to be so
    as well as Ca2+ detachment is surely dependent on [Ca]. 
   
    No rates include a temperature dependence. 
    """

    # kwargs that can be used to edit tm_site phenotype
    # tm_site can also accept phenotype profiles
    VALID_PARAMS = {'tm_coop': "AU",
                    'tm_K1': "mol(Ca)-1", 'tm_k_12': "mol(Ca)-1*ms-1", 'tm_K2': "none", 'tm_k_23': "ms-1",
                    'tm_K3': "none", 'tm_k_34': "ms-1", 'tm_K4': "mol(Ca)", 'tm_k_41': "ms-1"}

    def __init__(self, parent_tm, binding_site, index, **tm_params):
        """ A single tropomyosin site, paired to a binding site
        Parameters
        ----------
        parent_tm: tropomyosin class instance
            the tropomyosin chain on which this site is located
        binding_site: actin binding site class instance
            the binding site this tm site regulates
        index: int
            where this tm site is along the tm chain
        """
        # noinspection PyArgumentList
        random.seed()

        # ## Who are you and where do you come from?
        self.parent_tm = parent_tm
        self.index = index
        self.address = ("tm_site", parent_tm.parent_thin.index,
                        parent_tm.index, index)
        # What are you regulating?
        self.binding_site = binding_site
        self.binding_site.tm_site = self
        self.state = 0
        # ## Kinetics from Tanner 2007, 2012, and thesis
        # Equilibrium constants
        # todo cite
        K1 = 1.47e5  # per mole Ca
        K2 = 1.3e2  # unit-less
        K3 = 9.1e-1  # unit-less
        K4 = None  # moles Ca
        # Forward Rate constants - !!! r14 is overridden to 0 ms-1 in _r14()
        # todo cite
        k_12 = 1.7e8  # per mole Ca per sec
        k_23 = 1.4e4  # per sec
        k_34 = 2e3  # per sec
        k_41 = 110  # per sec
        s_per_ms = 1e-3  # seconds per millisecond
        k_12 *= s_per_ms  # convert rates from
        k_23 *= s_per_ms  # occurrences per second
        k_34 *= s_per_ms  # to
        k_41 *= s_per_ms  # occurrences per ms
        coop = 1  # cooperative multiplier - default 1 (no cooperativity)
        self._K1, self._K2, self._K3, self._K4 = K1, K2, K3, K4
        self._k_12, self._k_23, self._k_34, self._k_41 = k_12, k_23, k_34, k_41
        self._coop = coop
        # todo TBD self._concentrations = None  # [c_1, c_2, c_3]

        """Handle tm_params"""
        # ## Handle tm_isomer calculations
        if 'tm_iso' in tm_params.keys():  # !!! This means we don't actually have settings to pass yet !!!
            profiles = tm_params['tm_iso']
            cum_sum = 0
            rolled_val = random.random()  # get the rolled value
            i = 0
            while cum_sum < rolled_val:
                probability = float(profiles[i]['iso_p'])
                cum_sum += probability
                i += 1
            tm_params = tm_params['tm_iso'][i - 1].copy()  # actually select the params and proceed as normal
            tm_params.pop('iso_p')

        self.constants = {}

        self._process_params(tm_params)

        for param in tm_params.keys():
            print("Unknown tm_param:", param)

    def __str__(self):
        """Representation of the tmsite for printing"""
        out = "TmSite #%02i State:%i Loc:%04.0f" % (
            self.index, self.state, self.axial_location)
        return out

    def to_dict(self):
        """Create a JSON compatible representation of the tropomyosin site
        Usage example:json.dumps(tmsite.to_dict(), indent=1)
        Current output includes:
            address: largest to most local, indices for finding this
            binding_site: address of binding site
            state: kinetic state of site
            binding_influence: what the state tells you
            span: how far an activation spreads
            _k_12 - _k_41: transition rates
            _K1 - _K4: kinetic balances for reverse rates
        Returns
        -------
        tms_d: dict
            tropomyosin site dictionary
        """
        tms_d = self.__dict__.copy()
        tms_d.pop('parent_tm')
        tms_d.pop('index')
        tms_d['binding_site'] = tms_d['binding_site'].address
        return tms_d

    def from_dict(self, tms_d):
        """ Load values from a tropomyosin site dict. 
        
        Values read correspond to output documented in :to_dict:.
        """
        # Check for index mismatch
        read, current = tuple(tms_d['address']), self.address
        assert read == current, "index mismatch at %s/%s" % (read, current)
        # Local/remote keys
        self.binding_site = self.parent_tm.parent_thin.parent_lattice. \
            resolve_address(tms_d['binding_site'])
        self.binding_site.tm_site = self
        # Local keys
        self.state = tms_d['_state']
        self._k_12 = tms_d['_k_12']
        self._k_23 = tms_d['_k_23']
        self._k_34 = tms_d['_k_34']
        self._k_41 = tms_d['_k_41']
        self._K1 = tms_d['_K1']
        self._K2 = tms_d['_K2']
        self._K3 = tms_d['_K3']
        self._K4 = tms_d['_K4']
        self._coop = tms_d['_coop']
        return

    @property
    def timestep_len(self):
        """Timestep size is stored at the half-sarcomere level"""
        return self.parent_tm.parent_thin.parent_lattice.timestep_len

    @property
    def pCa(self):
        """pCa stored at the half-sarcomere level"""
        pCa = self.parent_tm.parent_thin.parent_lattice.pCa
        assert pCa > 0, "pCa must be given in positive units by convention"
        return pCa

    @property
    def ca(self):
        """The calcium concentration stored at the half-sarcomere level"""
        Ca = 10.0 ** (-self.pCa)
        return Ca

    @property
    def axial_location(self):
        """Axial location of the bs we are piggybacking on"""
        return self.binding_site.axial_location

    @property
    def span(self):
        """What is the span of cooperative activation for this tm site?
        
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
        """
        base = self.parent_tm.span_base
        steep = self.parent_tm.span_steep
        f50 = self.parent_tm.span_force50
        f = self.binding_site.tension
        span = 0.5 * base * (1 - np.tanh(steep * (f50 + f)))
        return span

    @property
    def subject_to_cooperativity(self):
        """True if another TMS, within span, is in state 3"""
        # Set up
        site_loc = self.axial_location
        parent_locs = self.parent_tm.axial_locations
        span = self.span
        # Find within span and check state
        near_inds = np.nonzero(np.abs(parent_locs - site_loc) < span)[0]
        states = [self.parent_tm.sites[i].state for i in near_inds]
        return any([state == 2 for state in states])

    @property
    def state(self):
        """Get the current state of the tm site
        
        Here states are given numerically for convenience.
        State designations have changed since multifil version 1.3;
            From    B-C-M
                (B)blocked, (C)closed,              (M)myosin
            To      U-B-C-O
                (U)unbound, (B)bound,   (C)closed,  (O)open

        State 0 - "unbound" - No Calcium bound to TnC
        State 1 - "bound"   - Calcium, TnC  and TnI haven't interacted
        State 2 - "closed"  - Calcium, TnC+TnI, actin site covered
        State 3 - "open"    - Calcium, TnC+TnI, actin site available
            State 3-M0 -> myosin unbound
            State 3-M1 -> myosin weakly bound
            State 3-M2 -> myosin strongly bound
        """
        return self._state

    @state.setter
    def state(self, new_state):
        """Set the state and thus the binding influence"""
        self._state = new_state
        self.binding_influence = {0: 0, 1: 0, 2: 0, 3: 1}[new_state]

    @property
    def rates(self):
        return {'tm_rate_12': self._r12(),
                'tm_rate_21': self._r21(),
                'tm_rate_23': self._r23(),
                'tm_rate_32': self._r32(),
                'tm_rate_34': self._r34(),
                'tm_rate_43': self._r43(),
                'tm_rate_41': self._r41(),
                'tm_rate_14': self._r14()}

    def _r12(self):
        """Rate of Ca and TnC association, conditional on [Ca]"""
        # TODO concentration logic - reversible reaction rate tn concentration in HS
        # forward = self._k_12 * self.ca * self._concentrations['free_tm']
        # coop = self._coop if self.subject_to_cooperativity else 1
        # forward *= coop

        coop = self._coop if self.subject_to_cooperativity else 1
        forward = self._k_12 * self.ca * coop

        return forward

    def _r21(self):
        """Rate of Ca detachment from TnC, conditional on [Ca]"""
        # k_21 = self._k_12 / self._K1
        # reverse = k_21 * self._concentrations['bound_tm']

        k_21 = self._k_12 / self._K1
        reverse = k_21

        return reverse

    def _r23(self):
        """Rate of TnI TnC interaction"""
        # forward = self._k_23
        # concentration of tm = TnI?
        # forward *= (self._concentrations['bound_tm'] + self._concentrations['free_tm'])

        forward = self._k_23
        coop = self._coop if self.subject_to_cooperativity else 1
        forward *= coop

        return forward

    def _r32(self):
        """Rate of TnI TnC interaction reversal"""
        forward = self._k_23
        reverse = forward / self._K2

        return reverse

    def _r34(self):
        """Rate of tropomyosin movement - Uncovering"""
        forward = self._k_34
        coop = self._coop if self.subject_to_cooperativity else 1
        forward *= coop

        return forward

    def _r43(self):
        """Rate of tropomyosin movement - Covering"""
        reverse = self._k_34 / self._K3

        return reverse

    def _r41(self):
        """Rate of Actin site covering due to Ca disassociation induced TnI TnC disassociation"""
        # forward = self._k_41 * self.ca * self._concentrations['bound_tm']

        forward = self._k_41

        return forward

    def _r14(self):
        """Rate of simultaneous Ca binding, TnI TnC association and tropomyosin movement.
        Should be quite low.
        """
        # k_14 = self._k_41 / self._K4
        # reverse = k_14  * self.ca * self.ca * self._concentrations['free_tm']
        if self._K4 is not None:
            forward = self._k_41
            reverse = forward / self._K4
            reverse *= self.ca

        return 0  # TODO figure this out

    def _prob(self, rate):
        """ Convert a rate to a probability, based on the current timestep
        length and the assumption that the rate is for a Poisson process.
        We are asking, what is the probability that at least one Poisson
        distributed value would occur during the timestep.
        Parameters
        ----------
            rate: float
                a per millisecond rate to convert to probability
        Returns
        -------
            probability: float
                the probability the event occurs during a timestep
                of length determined by self.timestep
        """
        return 1 - np.exp(-rate * self.timestep_len)

    @staticmethod
    def _forward_backward(forward_p, backward_p, rand):
        """Transition forward or backward based on random variable, return
        "forward", "backward", or "none"
        """
        if rand < forward_p:
            return "forward"
        elif rand > (1 - backward_p):
            return "backward"
        return "none"

    def transition(self):
        """Transition from one state to the next, or back, or don't """
        # TODO deal with this later
        # self._concentrations = self.parent_tm.parent_thin.parent_lattice.concentrations
        rand = np.random.random()

        f, b = 0, 0     # avoid any chance of an UnboundLocalError

        # Select which rate calculations are relevant
        if self.state == 0:
            f, b = self._prob(self._r12()), self._prob(self._r14())
        elif self.state == 1:
            f, b = self._prob(self._r23()), self._prob(self._r21())
        elif self.state == 2:
            f, b = self._prob(self._r34()), self._prob(self._r32())
        elif self.state == 3:
            if self.binding_site.state == 1:
                return  # can't transition if bound
            f, b = self._prob(self._r41()), self._prob(self._r43())

        # Calculate probabilities and change state accordingly
        trans_word = self._forward_backward(f, b, rand)
        self.state = {"forward": (self.state + 1) % 4,
                      "backward": (self.state + 3) % 4,
                      "none": self.state}[trans_word]

        assert self.state in (0, 1, 2, 3), "Tropomyosin state has invalid value"

        return trans_word

    def _process_params(self, tm_params):
        # cooperative multiplier
        key = 'tm_coop'
        if key in tm_params.keys():
            self._coop = tm_params.pop(key)
        self.constants[key] = self._coop

        # ## First step ---------------------------
        # K1
        key = 'tm_K1'
        if key in tm_params.keys():
            self._K1 = tm_params.pop(key)
        self.constants[key] = self._K1

        # forward rate - k_12
        key = 'tm_k_12'
        if key in tm_params.keys():
            self._k_12 = tm_params.pop(key)
        self.constants[key] = self._k_12

        # ## Second step ---------------------------
        # K2
        key = 'tm_K2'
        if key in tm_params.keys():
            self._K2 = tm_params.pop(key)
        self.constants[key] = self._K2

        # forward rate - k_23
        key = 'tm_k_23'
        if key in tm_params.keys():
            self._k_23 = tm_params.pop(key)
        self.constants[key] = self._k_23

        # ## Third step ---------------------------
        key = 'tm_K3'
        if key in tm_params.keys():
            self._K3 = tm_params.pop(key)
        self.constants[key] = self._K3

        # forward rate - k_34
        key = 'tm_k_34'
        if key in tm_params.keys():
            self._k_34 = tm_params.pop(key)
        self.constants[key] = self._k_34

        # ## Fourth step (3 -> 0) -----------------
        key = 'tm_K4'
        if key in tm_params.keys():
            self._K4 = tm_params.pop(key)
        self.constants[key] = self._K4

        # forward rate - k_41
        key = 'tm_k_41'
        if key in tm_params.keys():
            self._k_41 = tm_params.pop(key)
        self.constants[key] = self._k_41


class Tropomyosin:
    """Regulate the binding permissiveness of actin strands. 
    
    Tropomyosin stands in for both the Tm and the Tn. 
    
    Structure
    ---------
    The arrangement of the tropomyosin on the thin filament is represented as
    having an ability to be calcium regulated at each binding site with the
    option to spread that calcium regulation to adjacent binding sites. I am
    only grudgingly accepting of this as a representation of the TmTn
    interaction structure, but it is a reasonable first pass. 
    """

    def __init__(self, parent_thin, binding_sites, index, **tm_params):
        """A strand of tropomyosin chains
        
        Save the binding sites along a set of tropomyosin strands, 
        in preparation for altering availability of binding sites. 
        
        Parameters
        ----------
        parent_thin: thin filament instance
            thin filament on which the tropomyosin lives
        binding_sites: list of list of binding site instances
            binding sites on this tm string
        index: int
            which tm chain this is on the thin filament
        """
        # ## Who are you and where do you come from?
        self.parent_thin = parent_thin
        self.index = index
        self.address = ("tropomyosin", parent_thin.index, index)

        # ## What is your population?
        self.sites = [TmSite(self, bs, ind, **tm_params) for ind, bs in
                      enumerate(binding_sites)]
        # ## How does activation spread?
        # Material properties belong to tm chain, but actual span is 
        # calculated at the site level (where tension is experienced)
        self.span_base = 62  # 11 g-actin influence span (Tanner 2012)
        self.span_steep = 1  # how steep the influence curve is
        self.span_force50 = -8  # force at which span is decreased by half
        self.span = None

        self.constants = {}
        for tm_site in self.sites:
            constants = tm_site.constants
            tm_index = str(self.parent_thin.index) + "_" + str(self.index) + '_' + str(tm_site.index)
            self.constants[tm_index] = constants

    def to_dict(self):
        """Create a JSON compatible representation of the tropomyosin chain
        Usage example:json.dumps(tm.to_dict(), indent=1)
        Current output includes:
            address: largest to most local, indices for finding this
            sites: tm sites
        """
        tmd = self.__dict__.copy()
        tmd.pop('parent_thin')
        tmd.pop('index')
        tmd['sites'] = [site.to_dict() for site in tmd['sites']]
        return tmd

    def from_dict(self, tmd):
        """ Load values from a tropomyosin dict. Values read correspond to 
        the current output documented in to_dict.
        """
        # Check for index mismatch
        read, current = tuple(tmd['address']), self.address
        assert read == current, "index mismatch at %s/%s" % (read, current)
        # Sub-structures
        for data, site in zip(tmd['sites'], self.sites):
            site.from_dict(data)
        self.span = tmd['span']

    def resolve_address(self, address):
        """Give back a link to the object specified in the address
        We should only see addresses starting with 'tm_site'
        """
        if address[0] == 'tm_site':
            return self.sites[address[3]]
        import warnings
        warnings.warn("Unresolvable address: %s" % str(address))

    @property
    def axial_locations(self):
        """Axial location of each Tm site"""
        return np.array([site.axial_location for site in self.sites])

    @property
    def states(self):
        """States of the contained TmSites (for monitoring)"""
        return [site.state for site in self.sites]

    @property
    def rates(self):
        """Average rates of the contained TmSites (for monitoring)"""
        rates = None
        for site in self.sites:
            if rates is None:
                rates = site.rates
            else:
                for key, value in site.rates.items():
                    rates[key] += value
        for key in rates:
            rates[key] /= len(self.sites)

        return rates

    def transition(self):
        """Chunk through all binding sites, transition if need be"""

        """Activation spread method - 1 Anthony's Activation Spreading"""
        # all_sites = self.sites.copy()
        # remaining_sites, transitions = self._roll_for_activation(all_sites)
        # for site in remaining_sites:
        #     transitions.append(site.transition())
        """Activation spread method - 2 Original Cooperation"""
        transitions = [site.transition() for site in self.sites]
        # Spread activation
        # self._spread_activation()   # TODO amend state numberings

        return transitions  # TODO figure out how to incorporate 'mechanical' transitions from _spread_activation()

    def _roll_for_activation(self, all_sites):
        """"Spread activation along the filament"""
        transitions = []
        # If nothing is in state 2, we're done
        if not max([site.state for site in all_sites]) < 2:
            # Find all my axial locations
            locs = self.axial_locations
            # ## Chunk through each site
            # spread activation to the left, spread activation to the right
            for site in all_sites:
                if site.state == 2:
                    all_sites.pop(indexOf(all_sites, site))
                    loc = site.axial_location
                    span = site.span
                    near_inds = np.nonzero(np.abs(locs - loc) < span)[0]
                    near = [self.sites[index] for index in near_inds]
                    index = 0
                    if site in near:
                        index = indexOf(near, site)
                    for n_site in near[index:]:
                        if n_site in all_sites:
                            all_sites.pop(indexOf(all_sites, n_site))
                            transitions.append(n_site.transition())
                    for n_site in reversed(near[:index]):
                        if n_site in all_sites:
                            all_sites.pop(indexOf(all_sites, n_site))
                            transitions.append(n_site.transition())
        return all_sites, transitions

    def _spread_activation(self):
        """"Spread activation along the filament"""
        # If nothing is in state 2, we're done
        if max([site.state for site in self.sites]) < 2:
            return
        # Find all my axial locations
        locs = self.axial_locations
        # Chunk through each site
        for site in self.sites:
            if site.state == 2:
                loc = site.axial_location
                span = site.span
                near_inds = np.nonzero(np.abs(locs - loc) < span)[0]
                near = [self.sites[index] for index in near_inds]
                for n_site in near:
                    if n_site.state != 2:
                        n_site.state = 1
        return


if __name__ == '__main__':
    print("tm.py is really meant to be called as a supporting module")
