#!/usr/bin/env python
# encoding: utf-8
"""
run.py - control a run

run.manage controls a run, locally on laptop or locally on an aws node, in
either case without needing to know where it is running. It takes in a meta file
produced by setup_run.emit and uses it to configure a sarcomere

run.sarc_file manages recording of complete sarcomere logs to a local file

run.data_file manages recording abbreviated data logs to a local file

run.s3 maintains a persistent s3 connection through all of this

Created by Dave Williams on 2016-07-02
"""

import sys
import os
import shutil
import subprocess
import time
import multiprocessing as mp
import boto
import numpy as np

from multifil import hs
from multifil.utilities import use_aws, json


# ## Manage a local run
class manage:
    """Run, now with extra object flavor"""

    def __init__(self, metafile, unattended=True, use_sarc=True, live_update=None):
        """Create a managed instance of the sarc, optionally running it

        Parameters
        ----------
        metafile: string
            The location of the metafile describing the run to be worked
            through. Can be local or on S3. Assumed to be on S3 if the local
            file does not exist.
        unattended: boolean
            Whether to complete the run without further intervention or treat
            as an interactive session.
        """
        if use_aws:
            self.s3 = s3()
        else:
            self.s3 = None
        self.uuid = os.path.basename(metafile).split('.')[0]
        self.working_dir = self._make_working_dir(self.uuid)
        self.metafile = self._parse_metafile_location(metafile)
        self.meta = self.unpack_meta(self.metafile)
        self.sarc = self.unpack_meta_to_sarc(self.meta)
        self.use_sarc = use_sarc
        self.live_update = live_update
        self.sarcfile = None
        self.datafile = None
        self.zip_filename = None
        self.working_filename = None
        if unattended:
            try:
                self.run_and_save()
            except Exception as e:
                mp.current_process().terminate()
                print(e)

    @staticmethod
    def _make_working_dir(name):
        """Create a temporary working directory and return the name"""
        wd_name = '/tmp/_multifil_tmp/' + name + "/"
        os.makedirs(wd_name, exist_ok=True)
        return wd_name

    def _parse_metafile_location(self, metafile):
        """Parse the passed location, downloading the metafile if necessary"""
        if not os.path.exists(metafile):
            raise FileNotFoundError("meta file not found")
            # return self.s3.pull_from_s3(metafile, self.working_dir)
        else:
            mfn = '/' + os.path.basename(metafile)
            return shutil.copyfile(metafile, self.working_dir + mfn)

    @staticmethod
    def unpack_meta(meta_filename):
        """Unpack the local meta file to a dictionary"""
        with open(meta_filename, 'r') as metafile:
            meta = json.load(metafile)
        return meta

    @staticmethod
    def unpack_meta_to_sarc(meta):
        """Unpack the local meta file and instantiate a sarc as defined
        in the meta file
        """
        # Prep single values for instantiation of hs
        none_if_list = lambda s: None if type(meta[s]) is list else meta[s]
        lattice_spacing = none_if_list('lattice_spacing')
        z_line = none_if_list('z_line')
        pCa = none_if_list('pCa')

        # handle no kwargs case
        hs_params = {} if meta['hs_params'] is None else meta['hs_params']

        # Time dependent values
        time_dep_dict = {}
        for prop in ['z_line', 'pCa']:
            if type(meta[prop]) is list:
                time_dep_dict[prop] = meta[prop]

        # Instantiate sarcomere
        sarc = hs.hs(
            lattice_spacing=lattice_spacing,
            z_line=z_line,
            poisson=meta['poisson_ratio'],
            pCa=pCa,
            timestep_len=meta['timestep_length'],
            time_dependence=time_dep_dict,
            **hs_params
        )
        return sarc

    def _copy_file_to_final_location(self, temp_full_fn, final_loc=None):
        """Copy file from the temporary location to the final resting places

        Parameters
        ----------
        temp_full_fn: string
            the full location of the temporary file to be copied
        final_loc: string
            an optional string for an extra local directory to copy the file to
        """
        temp_loc = temp_full_fn
        file_name = '/' + temp_loc.split('/')[-1]
        # Upload to S3
        if self.meta['path_s3'] is not None:
            # TODO see line 50
            raise BrokenPipeError("AWS services are not currently setup for multifil.")
            # s3_loc = self.meta['path_s3']
            # self.s3.push_to_s3(temp_loc, s3_loc)
        # Store in final local path specified in the meta settings
        if self.meta['path_local'] is not None:
            local_loc = os.path.abspath(os.path.expanduser(
                self.meta['path_local'])) + file_name
            try:
                shutil.copyfile(temp_loc, local_loc)
            except shutil.SameFileError:
                pass
        # Save to passed local location - second directory
        if final_loc is not None:
            local_loc = os.path.abspath(os.path.expanduser(final_loc)) \
                        + file_name
            shutil.copyfile(temp_loc, local_loc)

    def run_and_save(self):
        """Complete a run according to the loaded meta configuration and save
        results to meta-specified s3 and local locations"""
        exitcode = None
        result = None

        try:
            # Initialize data and sarc
            if self.use_sarc:
                self.sarcfile = sarc_file(self.sarc, self.meta, self.working_dir)
            self.datafile = data_file(self.sarc, self.meta, self.working_dir)
            # Run away
            # noinspection PyArgumentList
            np.random.seed()
            tic = time.time()

            for timestep in range(self.meta['timestep_number']):
                self.sarc.timestep(timestep)
                self.datafile.append()
                if self.live_update is not None and timestep % self.live_update == 0:
                    self.datafile.finalize()
                if self.use_sarc:
                    self.sarcfile.append()
                # Update on how it is going
                self._run_status(timestep, tic, 100)

            # Finalize and save files to final locations
            self._log_it("model finished, uploading")
            exitcode = 0
        except KeyboardInterrupt:
            exitcode = 130
        except Exception as e:
            exitcode = 1
            import traceback
            print("/n")
            print(e)
            traceback.print_exc()
        finally:    # <- executed normally but also in the event of failure
            # In the event of general failure or user interrupt,
            # we need to finalize what we have.
            # READ: orphaned files in /tmp/ are disallowed now.
            if self.datafile is not None:
                result = self.datafile.data_dict.copy()
                data_final_name = self.datafile.finalize()
                self._copy_file_to_final_location(data_final_name)
                self.datafile.delete()  # clean up temp files

            if self.use_sarc and self.sarcfile is not None:
                sarc_final_name = self.sarcfile.finalize()
                self._copy_file_to_final_location(sarc_final_name)
                self.sarcfile.delete()  # clean up temp files

            self._copy_file_to_final_location(self.metafile)
            os.remove(self.metafile)

            os.rmdir(self.working_dir)
            try:
                os.rmdir('/tmp/_multifil_tmp/')
            except OSError:
                pass
            self._log_it("uploading finished, done with this run")
            return result, exitcode

    def _run_status(self, timestep, start, every):
        """Report the run status"""
        if timestep % every == 0 or timestep == 0:
            total_steps = self.meta['timestep_number']
            sec_passed = time.time() - start
            sec_left = int(sec_passed / (timestep + 1) * (total_steps - timestep - 1))
            proc_name = mp.current_process().name
            self.sarc.print_bar(i=timestep, time_steps=total_steps, toc=sec_left, proc_name=proc_name)

    @staticmethod
    def _log_it(message):
        """Print message to sys.stdout"""
        sys.stdout.write("run.py " + mp.current_process().name +
                         " # ## " + message + "\n")
        sys.stdout.flush()


# ## File management
class sarc_file:
    def __init__(self, sarc, meta, working_dir):
        """Handles recording a sarcomere dict to disk at each timestep"""
        self.sarc = sarc
        self.meta = meta
        self.working_directory = working_dir
        sarc_name = '/' + meta['name'] + '.sarc.json'
        self.working_filename = self.working_directory + sarc_name

        try:
            self.working_file = open(self.working_filename, 'a')
            self.next_write = '[\n'
            self.append(True)
        except Exception as e:
            print(e)

        self.zip_filename = None
        self.print_zip = False

    def append(self, first=False):
        """Add the current timestep sarcomere to the sarc file"""
        if not first:
            self.next_write += ',\n'
        self.next_write += json.dumps(self.sarc.to_dict(), sort_keys=True)
        self.working_file.write(self.next_write)
        self.next_write = ''

    def finalize(self):
        """Close the current sarcomere file for proper JSON formatting"""
        self.working_file.write('\n]')
        self.working_file.close()
        time.sleep(1)
        self.zip_filename = self.meta['name'] + '.sarc.tar.gz'
        cp = subprocess.run(['tar', 'czf', self.zip_filename,
                             '-C', self.working_directory,
                             self.working_filename])
        if self.print_zip:
            print(cp)
        os.remove(self.working_filename)
        return self.zip_filename

    def delete(self):
        """Delete the sarc zip file from disk"""
        try:
            # print("removing zip filename\n\t", self.zip_filename, "\n")
            os.remove(self.zip_filename)
        except FileNotFoundError:
            print("Error removing temp file, check C:\/tmp for", self.zip_filename)


class data_file:
    def __init__(self, sarc, meta, working_dir):
        """Generate the dictionary for use with the below data callback"""
        self.sarc = sarc
        self.meta = meta
        self.working_directory = working_dir
        self.working_filename = None
        self.data_dict = {
            'name': self.meta['name'],
            'constants': self.sarc.constants,
            'timestep_length': self.sarc.timestep_len,
            'timestep': [],
            'z_line': [],
            'lattice_spacing': [],
            'axial_force': [],
            'radial_force_y': [],
            'radial_force_z': [],
            'radial_tension': [],
            'xb_fraction_free': [],
            'xb_fraction_loose': [],
            'xb_fraction_tight': [],
            'xb_trans_12': [],
            'xb_trans_23': [],
            'xb_trans_31': [],
            'xb_trans_21': [],
            'xb_trans_32': [],
            'xb_trans_13': [],
            'xb_trans_static': [],
            'actin_permissiveness': [],
            'thick_displace_mean': [],
            'thick_displace_max': [],
            'thick_displace_min': [],
            'thick_displace_std': [],
            'thin_displace_mean': [],
            'thin_displace_max': [],
            'thin_displace_min': [],
            'thin_displace_std': [],
            "coop": [],
            "tm_fraction_bound": [],
            "ca": [],
            "tm_unbound": [],
            "tm_bound": [],
            "tm_closed": [],
            "tm_open": [],
            'tm_rate_12': [],
            'tm_rate_21': [],
            'tm_rate_23': [],
            'tm_rate_32': [],
            'tm_rate_34': [],
            'tm_rate_43': [],
            'tm_rate_41': [],
            'tm_rate_14': [],
            'titin_axial_force': [],
        }

    def append(self):
        """Digest out the non-vector values we want to record for each
        timestep and append them to the data_dict. This is called at each
        timestep to build a dict for inclusion in a pandas DataFrame.
        """
        # ## Lambda helpers
        ad = lambda n, v: self.data_dict[n].append(v)
        # ## Calculated components
        radial_force = self.sarc.radial_force()
        xb_fracs = self.sarc.get_xb_frac_in_states()
        xb_trans = sum(sum(self.sarc.last_transitions, []), [])
        tm_fracs = self.sarc.get_tm_frac_in_states()
        tm_rates = self.sarc.tm_rates()
        act_perm = np.mean(self.sarc.actin_permissiveness)
        thick_d = np.hstack([t.displacement_per_crown()
                             for t in self.sarc.thick])
        thin_d = np.hstack([t.displacement_per_node()
                            for t in self.sarc.thin])
        # ## Dictionary work
        ad('timestep', self.sarc.current_timestep)
        ad('z_line', self.sarc.z_line)
        ad('lattice_spacing', self.sarc.lattice_spacing)
        ad('axial_force', self.sarc.axial_force())
        ad('radial_force_y', radial_force[0])
        ad('radial_force_z', radial_force[1])
        ad('radial_tension', self.sarc.radial_tension())
        ad('xb_fraction_free', xb_fracs[0])
        ad('xb_fraction_loose', xb_fracs[1])
        ad('xb_fraction_tight', xb_fracs[2])
        ad('xb_trans_12', xb_trans.count('12'))
        ad('xb_trans_23', xb_trans.count('23'))
        ad('xb_trans_31', xb_trans.count('31'))
        ad('xb_trans_21', xb_trans.count('21'))
        ad('xb_trans_32', xb_trans.count('32'))
        ad('xb_trans_13', xb_trans.count('13'))
        ad('xb_trans_static', xb_trans.count(None))
        ad('actin_permissiveness', act_perm)
        ad('thick_displace_mean', np.mean(thick_d))
        ad('thick_displace_max', np.max(thick_d))
        ad('thick_displace_min', np.min(thick_d))
        ad('thick_displace_std', np.std(thick_d))
        ad('thin_displace_mean', np.mean(thin_d))
        ad('thin_displace_max', np.max(thin_d))
        ad('thin_displace_min', np.min(thin_d))
        ad('thin_displace_std', np.std(thin_d))
        ad("tm_fraction_bound", self.sarc.tnca_count / self.sarc.tn_total)
        ad("ca", self.sarc.c_ca)
        ad("tm_unbound", tm_fracs[0])
        ad("tm_bound", tm_fracs[1])
        ad("tm_closed", tm_fracs[2])
        ad("tm_open", tm_fracs[3])
        ad('tm_rate_12', tm_rates['tm_rate_12'])
        ad('tm_rate_21', tm_rates['tm_rate_21'])
        ad('tm_rate_23', tm_rates['tm_rate_23'])
        ad('tm_rate_32', tm_rates['tm_rate_32'])
        ad('tm_rate_34', tm_rates['tm_rate_34'])
        ad('tm_rate_43', tm_rates['tm_rate_43'])
        ad('tm_rate_41', tm_rates['tm_rate_41'])
        ad('tm_rate_14', tm_rates['tm_rate_14'])
        ad('titin_axial_force', self.sarc.titin_axial_force())

    def finalize(self):
        """Write the data dict to the temporary file location"""
        data_name = '/' + self.meta['name'] + '.data.json'
        self.working_filename = self.working_directory + data_name
        with open(self.working_filename, 'w') as datafile:
            json.dump(self.data_dict, datafile, sort_keys=True)
        return self.working_filename

    def delete(self):
        """Delete the data file from disk"""
        try:
            # print("removing working filename\n\t", self.working_filename, "\n")
            os.remove(self.working_filename)
        except FileNotFoundError:
            print("File not created yet")


class s3:
    def __init__(self):
        """Provide an interface to to S3 that hides some error handling"""
        self._refresh_s3_connection()

    def _refresh_s3_connection(self):
        """Reconnect to s3, the connection gets dropped sometimes"""
        self.s3 = boto.connect_s3()

    def _get_bucket(self, bucket_name):
        """Return link to a bucket name"""
        try:
            bucket = self.s3.get_bucket(bucket_name)
        except (boto.exception.BotoClientError,
                boto.exception.BotoClientError) as e:
            print(e)
            print("Trying to reconnect to s3")
            self._refresh_s3_connection()
            bucket = self.s3.get_bucket(bucket_name)
        return bucket

    def pull_from_s3(self, name, local='./'):
        # noinspection PyUnresolvedReferences
        """Given a key on S3, download it to a local file

        Parameters
        ----------
        name: string
            bucket/key-name to download. can be prefixed by 's3://', '/', or
            by nothing
        local: string
            local directory to download key into, defaults to current directory

        Returns
        -------
        None

        Examples
        --------
        >>>pull_from_s3('s3://just_a_test_bucket/test')
        >>>os.remove('test')
        >>>pull_from_s3('just_a_test_bucket/test')
        >>>os.remove('test')
        """
        # Parse name
        bucket_name = [n for n in name.split('/') if len(n) > 3][0]  # rm s3:// & /
        key_name = name[len(bucket_name) + name.index(bucket_name):]
        file_name = key_name.split('/')[-1]
        # Connect to bucket
        bucket = self._get_bucket(bucket_name)
        # Connect to key/file
        key = bucket.get_key(key_name)
        # Prep local dirs to receive key
        local = os.path.abspath(os.path.expanduser(local))
        os.makedirs(local, exist_ok=True)
        # Download key
        downloaded_name = local + '/' + file_name
        key.get_contents_to_filename(downloaded_name)
        if key.size != os.stat(downloaded_name).st_size:
            print("Size mismatch, downloading again for %s: " % downloaded_name)
            key.get_contents_to_filename(downloaded_name)
        return downloaded_name

    def push_to_s3(self, local, remote):
        """Given a local file, push it to a location on s3

        Parameters
        ----------
        local: string
            path to local file to be uploaded to s3
        remote: string
            bucket and (optional) folder to upload local file to, the resulting
            key will be remote+/+local_filename. Can be in formats:
                s3://bucket/optional_folder/key
                /bucket/optional_folder/key
                bucket/optional_folder/key

        Returns
        -------
        None
        """
        # Parse names
        file_name = local.split('/')[-1]
        bucket_name = [n for n in remote.split('/') if len(n) > 3][0]
        key_name = remote[len(bucket_name) + remote.index(bucket_name):]
        if len(key_name) == 0 or key_name[-1] != '/':
            key_name += '/'
        # Parse bucket and folders
        bucket = self._get_bucket(bucket_name)
        key = bucket.new_key(key_name + file_name)
        key.set_contents_from_filename(local)
        if key.size != os.stat(local).st_size:
            print("Size mismatch, uploading again for %s: " % local)
            key.set_contents_from_filename(local)
        return


# ## Manage multiple local runs
class manage_async:
    """This class allows the user to run concurrent local simulations using a multi-core cpu"""
    #   Modern computers typically have at least 2 physical cores in their cpu
    #       able to perform processes truly asynchronously, given that they don't need to exchange
    #       information to complete these processes. If they require info exchange, that is also possible
    #       to facilitate, but may end up requiring some wait time to ensure the processes complete correctly.
    #   In addition to physical cores, modern cpus have a hardware-based trick called hyper-threading
    #       that allows them to switch between two things quickly enough that they get very close
    #       to executing two processes at once.
    #   This means that a 4 core cpu with hyper-threading can have 8 things going on at the same time.

    def __init__(self, meta_files, unattended=True, use_sarc=True, force=False, live_update=None):
        """Create a managed batch of instances of sarc objects, optionally running them

        Parameters
        ----------
        meta_files: list of strings
            A list of the locations of the metafile describing the run to be worked
            through. Can be local or on S3. Assumed to be on S3 if the local
            file does not exist.
        unattended: boolean
            Whether to complete the run without further intervention or treat
            as an interactive session.
        """
        max_threads = max(2, int(0.75 * mp.cpu_count()))
        if len(meta_files) >= mp.cpu_count():
            print("More instances than threads,", end=" ")
            if not force:
                print("these files will not be run:")
                for meta_file in meta_files[max_threads:]:
                    print(meta_file)
                meta_files = meta_files[0:max_threads]
            else:
                print("forcing anyway."
                      "\nBehavior will be dependent on python's built in class multiprocessing.")
        elif len(meta_files) > max_threads:
            print("More that 75% of available threads will be used for simulations, proceeding anyway.")

        live_update_list = []
        if isinstance(live_update, list):
            live_update_list = live_update
        elif isinstance(live_update, (int, float)):
            live_update_list.append(live_update)
        while len(live_update_list) < len(meta_files):
            live_update_list.append(None)
        print(live_update_list)

        self.managers = []
        for meta_file, live_update in zip(meta_files, live_update_list):
            self.managers.append(manage(meta_file, unattended, use_sarc, live_update))

        self.processes = []
        for manager in self.managers:
            self.processes.append(mp.Process(target=manager.run_and_save, args=()))

            if unattended:
                try:
                    self.run_and_save()
                except Exception as e:
                    mp.current_process().terminate()
                    print(e)

    def run_and_save(self):
        start = time.time()
        for process in self.processes:
            process.start()
        for process in self.processes:
            process.join()

        end = time.time()
        print(end - start, "seconds")

        return None, 0
