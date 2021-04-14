#!/usr/bin/env python
# encoding: utf-8
"""
af.py - An actin filament

Create and maintain a thin filament and the subgroups that comprise it.

Created by Dave Williams on 2010-01-04.
"""

import numpy as np

from . import tm


class BindingSite:
    """A singular globular actin site"""
    def __init__(self, parent_thin_fil, index, orientation):
        """Create a binding site on the thin filament

        Parameters:
            parent_thin_fil: the calling thin filament instance
            index: the axial index on the parent thin filament
        Properties:
            address: largest to most local, indices for finding this
            orientation: select between six orientations (0-5)
        """
        # Remember passed attributes
        self.parent_thin = parent_thin_fil
        self.index = index
        self.address = ('bs', self.parent_thin.index, self.index)
        # Use the passed orientation index to choose the correct
        # orientation vector according to schema in ThinFilament docstring
        orientation_vectors = ((0.866, -0.5), (0, -1), (-0.866, -0.5),
                               (-0.866, 0.5), (0, 1), (0.866, 0.5))
        self.orientation = orientation_vectors[orientation]
        # Start off unlinked to a tropomyosin site
        self.tm_site = None
        # Create attributes to store things not yet present
        self.bound_to = None    # None if unbound, Crossbridge object otherwise

    def __str__(self):
        """Return the current situation of the binding site"""
        # noinspection PyListCreation
        result = ['Binding Site #' + str(self.index) + ' Info']
        result.append(14 * '=')
        result.append('State: ' + str(self.state))
        if self.state != 0:
            result.append('Forces: ' + str(self.axial_force()) + '/' + str(self.radial_force()))
        return '\n'.join(result)

    def to_dict(self):
        """Create a JSON compatible representation of the binding site

        Usage example:json.dumps(bs.to_dict(), indent=1)

        Current output includes:
            address: largest to most local, indices for finding this
            bound_to: T/F if the binding site is bound
            orientation: the y/z orientation of the binding site relative to
                the center of the thin filament
            tropomyosin: the tropomyosin the bs is governed by
        """
        bsd = self.__dict__.copy()
        bsd.pop('index')
        bsd.pop('parent_thin')
        if bsd['bound_to'] is not None:
            bsd['bound_to'] = bsd['bound_to'].address
        bsd['tm_site'] = bsd['tm_site'].address
        return bsd

    def from_dict(self, bsd):
        """ Load values from a binding site dict. Values read in correspond to
        the current output documented in to_dict.
        """
        # Check for index mismatch
        read, current = tuple(bsd['address']), self.address
        assert read == current, "index mismatch at %s/%s" % (read, current)
        # Local keys
        self.orientation = bsd['orientation']
        if bsd['bound_to'] is not None:
            self.bound_to = self.parent_thin.parent_lattice.\
                    resolve_address(bsd['bound_to'])
        else:
            self.bound_to = bsd['bound_to']
        self.tm_site = self.parent_thin.parent_lattice.resolve_address(
            bsd['tm_site'])

    def axial_force(self, axial_location=None):
        """Return the axial force of the bound cross-bridge, if any

        Parameters:
            axial_location: location of the current node (optional)
        Returns:
            f_x: the axial force generated by the cross-bridge
        """
        if self.bound_to is None:
            return 0.0
        # Axial force on actin is equal but opposite
        return -self.bound_to.axial_force(tip_axial_loc=axial_location)

    def radial_force(self):
        """Radial force vector of the bound cross-bridge, if any

        Returns:
            (f_y, f_z): the radial force vector of this binding site
        """
        if self.bound_to is None:
            return np.array([0.0, 0.0])
        force_mag = -self.bound_to.radial_force()  # Equal but opposite
        return np.multiply(force_mag, self.orientation)

    def bind_to(self, crossbridge):
        """Link this binding site to a given, cross-bridge object
        return a reference to ourselves"""
        if self.bound_to is None:  # are we available?
            self.bound_to = crossbridge  # record the xb's phone number
            return self  # give the xb our phone number
        else:  # we are already taken!!!
            # import sys
            # import multiprocessing as mp
            # thread = mp.current_process().name
            # msg = "\n\n" + str(thread) + " Captain's log: ts=" + str(self.parent_thin.parent_lattice.current_timestep)
            # msg += "\nTHIS ACTIN SITE: " + str(self.address)
            # msg += "\nALREADY BOUND TO:  " + str(self.bound_to.address)
            # msg += "\nTRYING TO BIND TO: " + str(crossbridge.address)
            # msg += "\n---Denying access---"
            # sys.stdout.write(msg)
            # sys.stdout.flush()
            return None  # don't give this xb our phone number

    def unbind(self):
        """Kill off any link to a crossbridge"""
        assert (self.bound_to is not None)  # Else why try to unbind?
        self.bound_to = None  # remove crossbridge's contact
        return None  # return our self - to remove us from our ex-crossbridge

    @property
    def tension(self):
        """How much load the thin filament bears at this binding site"""
        return self.parent_thin.tension_at_site(self.index)

    @property
    def permissiveness(self):
        """What is your availability to bind, based on tropomyosin status?"""
        return self.tm_site.binding_influence

    @property
    def state(self):
        """Return the current numerical state, 0/unbound or 1/bound"""
        return self.bound_to is not None

    @property
    def lattice_spacing(self):
        """Get lattice spacing from the parent filament"""
        return self.parent_thin.lattice_spacing

    @property
    def axial_location(self):
        """Return the current axial location of the binding site"""
        return self.parent_thin.axial[self.index]


class ThinFace:
    """Represent one face of an actin filament
    Deals with orientation in the typical fashion for thin filaments
        ================
        ||     m4     ||  ^
        || m3      m5 ||  |   ^
        ||     af     ||  Z  /
        || m2      m0 ||    X
        ||     m1     ||      Y-->
        ================
    """

    def __init__(self, parent_thin_fil, orientation, index, binding_sites):
        """Create the thin filament face

        Parameters:
            parent_thin_fil: the thin filament on which this face sits
            orientation: which myosin face is opposite this face (0-5)
            index: location on the thin filament this face occupies (0-2)
        Properties:
            address: largest to most local, indices for finding this
            binding_sites: links to the actin binding sites on this face
        """
        self.parent_thin = parent_thin_fil
        self.index = index
        self.address = ('thin_face', self.parent_thin.index, self.index)
        self.orientation = orientation
        self.binding_sites = binding_sites
        self.thick_face = None  # ThickFace instance this face interacts with
        self.titin_fil = None

    def to_dict(self):
        """Create a JSON compatible representation of the thin face

        Usage example: json.dumps(thin_face.to_dict(), indent=1)

        Current output includes:
            address: largest to most local, indices for finding this
            orientation: out of 0-5 directions, which this projects in
            binding_sites: address information for each binding site
        """
        thinface_d = self.__dict__.copy()
        thinface_d.pop('index')
        thinface_d.pop('parent_thin')
        thinface_d['thick_face'] = thinface_d['thick_face'].address
        thinface_d['binding_sites'] = [bs.address for bs in thinface_d['binding_sites']]
        thinface_d['titin_fil'] = thinface_d['titin_fil'].address
        return thinface_d

    def link_titin(self, titin_fil):
        """Add a titin filament to this face"""
        self.titin_fil = titin_fil

    def from_dict(self, thinface_d):
        """ Load values from a thin face dict. Values read in correspond to
        the current output documented in to_dict.
        """
        # Check for index mismatch
        read, current = tuple(thinface_d['address']), self.address
        assert read == current, "index mismatch at %s/%s" % (read, current)
        # Local keys
        self.orientation = thinface_d['orientation']
        self.thick_face = self.parent_thin.parent_lattice.resolve_address(
            thinface_d['thick_face'])
        self.titin_fil = self.parent_thin.parent_lattice.resolve_address(
            thinface_d['titin_fil'])
        # Sub-structure keys
        self.binding_sites = [self.parent_thin.resolve_address(bsa) for bsa in thinface_d['binding_sites']]

    def nearest(self, axial_location):
        """Where is the nearest binding site?
        There a fair number of cases that must be dealt with here. When
        the system becomes too short (and some nearest queries are being
        directed to a thin face that doesn't really have anything near
        that location) the face will just return the nearest location and
        let the kinetics deal with the fact that binding is about as likely
        as stepping into the same river twice.
        Parameters:
            axial_location: the axial coordinates to seek a match for
        Return:
            binding_site: the nearest binding site on this face
        """
        # Next three lines of code enforce a jittery hiding, sometimes the
        # binding site just beyond the hiding line can be accessed
        hiding_line = self.parent_thin.hiding_line
        axial_location = max(hiding_line, axial_location)
        face_locs = [site.axial_location for site in self.binding_sites]
        next_index = np.searchsorted(face_locs, axial_location)
        prev_index = next_index - 1
        # If not using a very short SL, where the end face loc is closest,
        # then find distances to two closest locations
        if next_index != len(face_locs):
            dists = np.abs((face_locs[prev_index] - axial_location,
                            face_locs[next_index] - axial_location))
        else:
            return self.binding_sites[prev_index] # If at end, return end
        # If prior site was closer, give it, else give next
        if dists[0] < dists[1]:
            return self.binding_sites[prev_index]
        else:
            return self.binding_sites[next_index]

    def radial_force(self):
        """What is the radial force this face experiences?

        A side note: This was where the attempt to write the model out in
        a functional manner broke down. I got this far with nothing ever
        asking another instance for any information and everything being
        passed by method parameters. This was a really nice idea and
        worked well until this point where I had to start performing
        overly complex mental calisthenics to understand how things were
        going to be passed around. This lead to the current system where
        each instance has an internal state that it is responsible for
        keeping. This might make debugging harder in the long run, but it
        made the model writable in the meanwhile. Some teeth gnashing is
        included below for reference.

        Teeth gnashing:
            The source of conflict here seems to be a competition between
            the desire to write this in a functional manner and have all
            information passed down to the function as is needed and the
            desire to be able to call any function of any module at any
            time and have it return something sensible. This makes
            testing some bits easier but means that it can become harder
            to track what is going on with the states of the various
            functions. I am unsure as to how this should be resolved at
            this time. I want the final design to be as uncluttered and
            easy to troubleshoot as is possible. Perhaps something where
            the storage of information is kept separate from the ways that
            the modules are acting upon it? The advantage of this is that
            passing information around becomes infinitely easier, the
            drawback is that I am not sure that this isn't just a step
            removed from declaring every variable to be global and making
            the whole thing a fair bit more brittle.

        Returns:
            radial_force: the radial force myosin heads on this face exert
        """
        # First, a sanity check
        if self.thick_face is None:
            raise AttributeError("Thick filament not assigned yet.")
        # Now find the forces on each cross-bridge
        radial_forces = [site.radial_force() for site in self.binding_sites]
        return np.sum(radial_forces, 1)

    def set_thick_face(self, myosin_face):
        """Link to the relevant myosin filament."""
        assert(self.orientation == myosin_face.index)
        self.thick_face = myosin_face
        return

    def get_axial_location(self, binding_site_id):
        """Get the axial location of the targeted binding site"""
        return self.binding_sites[binding_site_id].axial_location

    @property
    def lattice_spacing(self):
        """Return lattice spacing to the face's opposite number"""
        return self.parent_thin.lattice_spacing


class ThinFilament:
    """Each thin filament is made up of two actin strands.  The overall
    filament length at rest is 1119 nm [Tanner2007].  Each strand
    hosts 45 actin binding sites (or nodes) giving the whole filament
    90 actin nodes, plus one at the Z-line for accounting.

    These nodes are located every 24.8 nm on each actin strand and are
    rotated by 120 degrees relative to the prior node [Tanner2007].
    This organization does not specify the relative offsets of the two
    filament's nodes.

    ## Naive repeating geometry of the thin filament
    The binding nodes of the two actin filaments must be offset by a
    multiple of the angle (120 degrees)x(distance offset/(24.8 nm)), but
    not by 360 degrees, or one of the actin filaments would have no binding
    sites facing a neighboring thick filament.  We assume that the actin
    nodes on the two strands are offset by half of the distance between
    adjacent nodes (12.4 nm) and 180 degrees.  This means that if one actin
    filament has a binding site facing one myosin filament, the second actin
    filament will have a binding site facing a second myosin filament
    12.4 nm down the thin filament.  The second myosin filament will be
    240 degrees clockwise of the first myosin filament around the
    thin filament.

    ## Binding site numbering
    As in the thick filament, the nodes/binding sites on the thin filament
    are numbered from low at the left to high on the right. Thus the 90th
    node is adjacent to the Z-line.

    ## Tropomyosin tracks
    While the filament is modeled as a one-start helix, as described in
    [Squire1981], chains of tropomyosin polymers follow the more gently
    curving two-start helix representation as seen in [Gunning2015].
    Assignment to tm chain a or b occurs with respect to the odd-even
    value of a monomer's index in the initial list of monomers.

    [Tanner2007]:http://dx.doi.org/10.1371/journal.pcbi.0030115
    [Squire1981]:http://dx.doi.org/10.1007/978-1-4613-3183-4
    [Gunning2015]:http://dx.doi.org/10.1242/jcs.172502
    """

    VALID_PARAMS = {"af_k": "pN/nm"}

    def __init__(self, parent_lattice, index, face_orientations, start=0, **af_params):
        """Initialize the thin filament

        Parameters:
            parent_lattice: the calling half-sarcomere instance
            index: which thin filament this is (0-7)
            face_orientations: list of faces' numerical orientation (0-5)
            start: which of the 26 actin monomers in an actin
                repeating unit this filament begins with (defaults
                to the first)
        Returns:
            None
        ## Thin face arrangement
        The thin filaments faces correspond to the following diagram:
        ================
        ||     m4     ||  ^
        || m3      m5 ||  |   ^
        ||     af     ||  Z  /
        || m2      m0 ||    X
        ||     m1     ||      Y-->
        ================
        These orientations correspond to the orientations of the facing
        thick filaments. Each thin filament will link to either faces
        0, 2, and 4 or faces 1, 3, and 5.
        This will result in a set of unit vectors pointing from the
        thin filament to the thick faces that are either
        ((0, 1), (0.866, -0.5), (-0.866, -0.5))
        for the case on the left or, for the case on the right,
        ((-0.886, 0.5), (0.866, 0.5), (0, -1))
        The vectors govern both what radial force linking cross-bridges
        generate and which actin monomers are considered to be facing
        the adjacent filaments.
        """
        # Remember who created you
        self.parent_lattice = parent_lattice
        # Remember who you are
        self.index = index
        self.address = ('thin_fil', self.index)
        tm_params = {}
        if 'tm_params' in af_params.keys():
            tm_params = af_params.pop("tm_params")
        elif 'tm_iso' in af_params.keys():
            tm_params = {'tm_iso': af_params.pop('tm_iso')}
        # TODO The creation of the monomer positions and angles should be refactored into a static function of similar.
        # Figure out axial positions, see Howard pg 125
        mono_per_poly = 26  # actin monomers in an actin polymer unit
        poly_per_fil = 15   # actin polymers in a thin filament
        polymer_base_length = 72.0  # nm per polymer unit length
        polymer_base_turns = 12.0   # revolutions per polymer
        rev = 2*np.pi   # one revolution
        pitch = polymer_base_turns * rev / mono_per_poly
        rise = polymer_base_length / mono_per_poly
        # Monomer positions start near the m-line
        monomer_positions = [(
            self.z_line - mono_per_poly*poly_per_fil*rise) + m*rise
            for m in range(mono_per_poly*poly_per_fil)]
        monomer_angles = [(((m+start+1) % mono_per_poly) * pitch) % rev
                          for m in range(mono_per_poly * poly_per_fil)]
        # Convert face orientations to angles, then to angles from 0 to 2pi
        orientation_vectors = ((0.866, -0.5), (0, -1.0), (-0.866, -0.5),
                               (-0.866, 0.5), (0, 1.0), (0.866, 0.5))
        face_vectors = [orientation_vectors[o] for o in face_orientations]
        face_angles = [np.arctan2(v[1], v[0]) for v in face_vectors]
        face_angles = [v + rev if (v < 0) else v for v in face_angles]
        # Find which monomers are opposite each face
        wiggle = rev/24     # count faces within 15 degrees of opposite
        mono_in_each_face = [np.nonzero(np.abs(np.subtract(monomer_angles,
                                                           face_angles[i])) < wiggle)[0]
                             for i in range(len(face_angles))]
        # This is [(index_to_face_1, ...), (index_to_face_2, ...), ...]
        # Translate monomer position to binding site position
        axial_by_face = [[monomer_positions[mono_ind] for mono_ind in face]
                         for face in mono_in_each_face]
        axial_flat = np.sort(np.hstack(axial_by_face))
        # Tie the nodes on each face into the flat axial locations
        node_index_by_face = np.array([
            [np.nonzero(axial_flat == l_ax_flat)[0][0] for l_ax_flat in face
             ] for face in axial_by_face])
        # V AMA 3-4-2020 V
        # TODO figure out if this argument to np.tile is correct. Trust in CDW
        # noinspection PyTypeChecker
        face_index_by_node = np.tile(None, len(axial_flat))
        for face_ind in range(len(node_index_by_face)):
            for node_ind in node_index_by_face[face_ind]:
                face_index_by_node[node_ind] = face_ind
        # Create binding sites and thin faces
        self.binding_sites = []
        for index in range(len(axial_flat)):
            orientation = face_orientations[face_index_by_node[index]]
            self.binding_sites.append(BindingSite(self, index, orientation))
        self.thin_faces = []
        orientation = None
        face_binding_sites = None
        for face_index in range(len(node_index_by_face)):
            face_binding_sites = ([self.binding_sites[i] for i in node_index_by_face[face_index]])
            orientation = face_orientations[face_index]
            self.thin_faces.append(
                ThinFace(self, orientation, face_index, face_binding_sites))
        del(orientation, face_binding_sites)
        # Remember the axial locations, both current and rest
        self.axial = axial_flat
        self.rests = np.diff(np.hstack([self.axial, self.z_line]))
        # Create links to two tropomyosin filament tracks
        bs_by_two_start = [[], []]
        for bs, ax in zip(self.binding_sites, axial_flat):
            mono_index = monomer_positions.index(ax)
            bs_by_two_start[mono_index % 2].append(bs)
        self.tm = [tm.Tropomyosin(self, bs_chain, ind, **tm_params) for
                   ind, bs_chain in enumerate(bs_by_two_start)]
        # Other thin filament properties to remember
        self.number_of_nodes = len(self.binding_sites)
        self.thick_faces = None     # Set after creation of thick filaments
        self.k = 1743

        """Handle af_params"""
        self.tm_constants = {}
        for tropomyosin in self.tm:
            constants = tropomyosin.constants
            for key in constants.keys():
                if key in self.tm_constants.keys():
                    self.tm_constants[key].update(constants[key])
                else:
                    self.tm_constants[key] = constants[key]

        self.af_constants = {}     # A dictionary containing constants changed by the user

        # set spring constant
        if 'af_k' in af_params.keys():
            self.k = af_params.pop('af_k')
        self.af_constants['af_k'] = self.k

        for param in af_params:
            print("Unknown af_param:", param)

    def to_dict(self):
        """Create a JSON compatible representation of the thin filament

        Example usage: json.dumps(thin.to_dict(), indent=1)

        Current output includes:
            address: largest to most local, indices for finding this
            axial: axial locations of binding sites
            rests: rest spacings between axial locations
            thin_faces: each of the thin faces
            binding_sites: each of the binding sites
            k: stiffness of the thin filament
            number_of_nodes: number of binding sites
        """
        thin_d = self.__dict__.copy()
        thin_d.pop('index')
        thin_d.pop('parent_lattice')    # TODO: Spend a P on an id for the lattice
        thin_d['thick_faces'] = [tf.address for tf in thin_d['thick_faces']]
        thin_d['thin_faces'] = [tf.to_dict() for tf in thin_d['thin_faces']]
        thin_d['axial'] = list(thin_d['axial'])
        thin_d['rests'] = list(thin_d['rests'])
        thin_d['binding_sites'] = [bs.to_dict() for bs in
                                   thin_d['binding_sites']]
        thin_d['tm'] = [tropomyosin.to_dict() for tropomyosin in thin_d['tm']]
        return thin_d

    def from_dict(self, td):
        """ Load values from a thin filament dict. Values read in correspond
        to the current output documented in to_dict.
        """
        # Check for index mismatch
        read, current = tuple(td['address']), self.address
        assert read == current, "index mismatch at %s/%s" % (read, current)
        # Local keys
        self.axial = np.array(td['axial'])
        self.rests = np.array(td['rests'])
        self.k = td['k']
        self.number_of_nodes = td['number_of_nodes']
        # Sub-structure and remote keys
        self.thick_faces = tuple([self.parent_lattice.resolve_address(tfa)
                                  for tfa in td['thick_faces']])
        for data, bs in zip(td['binding_sites'], self.binding_sites):
            bs.from_dict(data)
        for data, face in zip(td['thin_faces'], self.thin_faces):
            face.from_dict(data)
        for data, chain in zip(td['tm'], self.tm):
            chain.from_dict(data)

    def resolve_address(self, address):
        """Give back a link to the object specified in the address
        We should only see addresses starting with 'thin_face', 'bs',
        'tropomyosin', or 'tm_site'
        """
        if address[0] == 'thin_face':
            return self.thin_faces[address[2]]
        elif address[0] == 'bs':
            return self.binding_sites[address[2]]
        elif address[0] == 'tropomyosin':
            return self.tm[address[2]]
        elif address[0] == 'tm_site':
            return self.tm[address[2]].resolve_address(address)
        import warnings
        warnings.warn("Unresolvable address: %s" % str(address))

    def set_thick_faces(self, thick_faces):
        """Set the adjacent thick faces and associated values

        Parameters:
            thick_faces: links to three surrounding thick faces, in the
                order (0, 2, 4) or (1, 3, 5)

        ## Myosin filament arrangement
        ==================================  ^
        ||      m4      ||  m3      m5  ||  |   ^
        ||              or      af      ||  Z  /
        ||      af      ||              ||    X
        ||  m2      m0  ||      m1      ||      Y-->
        ==================================
        """
        self.thick_faces = thick_faces
        for a_face, m_face in zip(self.thin_faces, self.thick_faces):
            a_face.set_thick_face(m_face)

    def effective_axial_force(self):
        """The axial force experienced at the Z-line from the thin filament

        This only accounts for the force at the Z-line due to the actin
        node adjacent to it, i.e. this is the force that the Z-line
        experiences, not the tension existing elsewhere along the thin
        filament.
        Return:
            force: the axial force at the Z-line
        """
        return (self.rests[-1] - (self.z_line - self.axial[-1])) * self.k

    def axial_force_of_each_node(self, axial_locations=None):
        """Return a list of the XB derived axial force at each node

        Parameters:
            axial_locations: location of each node (optional)
        Returns:
            axial_forces: a list of the XB axial force at each node
        """
        if axial_locations is None:
            axial_forces = [site.axial_force() for site in self.binding_sites]
        else:
            axial_forces = [site.axial_force(loc) for
                            site, loc in zip(self.binding_sites, axial_locations)]
        return axial_forces

    def axial_force(self, axial_locations=None):
        """Return a list of axial forces at each binding site node location

        This returns the force at each node location (including the z-disk
        connection point), this is the sum of the force that results from
        displacement of the nodes from their rest separation and the axial
        force created by any bound cross-bridges

        Parameters:
            axial_locations: location of each node (optional)
        Return:
            force: sum of force from the cross-bridges and node displacement
        """
        # Calculate the force exerted by the thin filament's backbone
        thin = self._axial_thin_filament_forces(axial_locations)
        # Calculate the force exerted by any existing cross-bridges
        binding_sites = self.axial_force_of_each_node(axial_locations)
        # Return the combination of the two
        return np.add(thin, binding_sites)

    def transition(self):
        """Give self, (well, TMs really) a chance to transition states"""
        return [tropomyosin.transition() for tropomyosin in self.tm]

    def settle(self, factor):
        """Reduce the total axial force on the system by moving the sites"""
        # Total axial force on each point
        forces = self.axial_force()
        # Individual displacements needed to balance force
        isolated = factor*forces/self.k
        isolated[0] *= 2    # First node has spring on only one side
        # Cumulative displacements, working back from z-disk
        cumulative = np.flipud(np.cumsum(np.flipud(isolated)))
        # New axial locations
        self.axial += cumulative
        return forces

    def radial_force_of_each_node(self):
        """Radial force produced by XBs at each binding site node

        Parameters:
            self
        Returns
            radial_forces: a list of (f_y, f_z) force vectors
        """
        radial_forces = [nd.radial_force() for nd in self.binding_sites]
        return radial_forces

    def radial_force_of_filament(self):
        """The sum of the radial force experienced by this filament

        Parameters:
            self
        Returns:
            radial_force: a single (f_y, f_z) vector
        """
        radial_force_list = self.radial_force_of_each_node()
        radial_force = np.sum(radial_force_list, 0)
        return radial_force

    def displacement_per_node(self):
        """Displacement from rest lengths of segments between nodes"""
        dists = np.diff(np.hstack([self.axial, self.z_line]))
        return dists - self.rests

    def displacement(self):
        """A metric of how much the thin filament locations are offset"""
        return np.sum(np.abs(self.displacement_per_node()))

    def _axial_thin_filament_forces(self, axial_locations=None):
        """The force of the filament binding sites, sans cross-bridges

        Parameters:
            axial_locations: location of each node (optional)
        Returns:
            net_force_on_each_binding_site: per-site force
        """
        # Use the thin filament's stored axial locations if none are passed
        if axial_locations is None:
            axial_locations = np.hstack([self.axial, self.z_line])
        else:
            axial_locations = np.hstack([axial_locations, self.z_line])
        # Find the distance from binding site to binding site
        dists = np.diff(axial_locations)
        # Find the compressive or expansive force on each spring
        spring_force = (dists - self.rests) * self.k
        # The first node's not connected, so that side has no force...
        spring_force = np.hstack([0, spring_force])
        # Convert this to the force on each node
        net_force_on_each_binding_site = np.diff(spring_force)
        return net_force_on_each_binding_site

    def tension_at_site(self, index):
        """The net tension born by a given binding site

        How much tension is a given binding site subject to? This is
        useful for tension-dependent binding site kinetics and was
        originally included to study force depression after shortening.

        Parameters
        ----------
        index: int
            The binding site of interest

        Returns
        -------
        tension: float
        """
        # Passed index is subject only to forces between it and the
        # m-line side of the thin filament
        subject_to_forces = self._axial_thin_filament_forces()[index:]
        tension = np.sum(subject_to_forces)
        return tension

    def update_axial_locations(self, flat_axial_locs):
        """Update the axial locations to the passed ones

        Parameters:
            flat_axial_locs: the new locations for all axial nodes
        Returns:
            None
        """
        # You aren't allowed to change the number of nodes
        assert(len(flat_axial_locs) == len(self.axial))
        self.axial = flat_axial_locs

    @property
    def z_line(self):
        return self.parent_lattice.z_line

    @property
    def hiding_line(self):
        """Return the distance below which actin binding sites are hidden"""
        return self.parent_lattice.hiding_line

    @property
    def permissiveness(self):
        """Return the permissiveness of each binding site"""
        return [site.permissiveness for site in self.binding_sites]

    def get_binding_site(self, index):
        """Return a link to the binding site site at index"""
        return self.binding_sites[index]

    @property
    def bound_sites(self):
        """Give a list of binding sites that are bound to an XB"""
        return filter(lambda bs: bs.bound_to is not None, self.binding_sites)

    def get_axial_location(self, index):
        """Return the axial location of the node at index"""
        return self.axial[index]

    @property
    def lattice_spacing(self):
        """Return the lattice spacing of the half-sarcomere"""
        return self.parent_lattice.lattice_spacing

    def get_states(self):
        """Return the numeric states (0,1,2) of each face's cross-bridges"""
        return [tropomyosin.states for tropomyosin in self.tm]

    def get_tm_rates(self):
        """Return the average transition rates for the sites on each tropomyosin"""
        rates = None
        for tropomyosin in self.tm:
            if rates is None:
                rates = tropomyosin.rates
            else:
                for key, value in tropomyosin.rates.items():
                    rates[key] += value
        for key in rates:
            rates[key] /= len(self.tm)

        return rates


if __name__ == '__main__':
    print("af.py is really meant to be called as a supporting module")