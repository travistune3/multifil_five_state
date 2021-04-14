#!/usr/bin/env python
# encoding: utf-8
# noinspection PyUnresolvedReferences
"""
metas.py - create the meta file that will configure a run

This was originally included in the run.py file, but has grown complicated
enough to warrant the creation of a fully separate management system

metas.emit produces a meta file that describes what we want a run to do: the
values of the z_line, lattice spacing, and actin permissiveness through the run
and where it will be stored after completion.

Example
--------
>>> freq, phase = 10, .8
>>> time_trace = metas.time_trace(.1, 10)
>>> zline = metas.zline_workloop(1250, 25, freq, time_trace)
>>> activation = metas.actin_permissiveness_workloop(freq, phase, 10, 3, 3,
...  time_trace)
>>> metas.emit('./', None, time_trace, z_line=zline,
...  actin_permissiveness=activation, write=False, phase=phase, freq=freq)
{'actin_permissiveness': None,
...  'actin_permissiveness_func': None,
...  'comment': None,
...  'lattice_spacing': None,
...  'lattice_spacing_func': None,
...  'name': ...,
...  'path_local': './',
...  'path_s3': None,
...  'timestep_length': 0.1,
...  'timestep_number': 100,
...  'z_line': None,
...  'z_line_func': None}


Created by Dave Williams on 2017-03-08
"""

import os
import uuid
import numpy as np
import pdb
from multifil.utilities import json


# ## Define traces to be used in runs
def time_trace(timestep_length, run_length_in_ms):
    """Create a time series in ms. This is easily doable through other methods
    but this documents it a bit.

    timestep_length: float
        Length of a timestep in ms
    run_length_in_ms: int
        Number of timesteps run is simulated for
    """
    return np.arange(0, run_length_in_ms, timestep_length)


def zline_workloop(mean, amp, freq, time):
    """A sinusoidal oscillatory length trace.

    Parameters:
        mean: resting z-line value, will start here
        amp: peak-to-peak amplitude
        freq: frequency of oscillation
        time: time trace in ms to provide length trace for
    """
    period = 1000 / freq
    zline = mean + 0.5 * amp * np.cos(2 * np.pi * time / period)
    return zline


def zline_forcevelocity(L0, hold_time, L0_per_sec, time):
    """Takes initial length, time to hold there in ms, & shortening in L0/sec"""
    # Things we need to know for the shortening
    number_of_timesteps = len(time)  # for ease of reading
    timestep_length = np.diff(time)[0]
    hold_steps = int(hold_time / timestep_length)
    shorten_steps = number_of_timesteps - hold_steps
    nm_per_step = timestep_length * 1 / 1000 * L0_per_sec * L0
    # Construct the length signal
    zline = [L0 for _ in range(hold_steps)]
    for i in range(shorten_steps):
        zline.append(zline[-1] - nm_per_step)
    return zline


def actin_permissiveness_workloop(freq, phase, stim_duration,
                                  influx_time, half_life, time):
    """Requires cycle frequency, phase relative to longest length
    point, duration of on time, time from 10 to 90% influx level, and
    the half-life of the Ca2+ out-pumping.
    """
    # Convert frequency to period in ms
    period = 1000 / freq
    # Things we need to know for the shape of a single cycle
    decay_rate = np.log(1 / 2) / half_life
    growth_rate = 0.5 * influx_time
    max_signal = 1.0
    # Things we need to know for the cyclical nature of the signal
    number_of_timesteps = len(time)  # for ease of reading
    timestep_length = np.diff(time)[0]
    cycle_step_number = int(period / timestep_length)
    cycle_time_trace = np.arange(0, period, timestep_length)
    try:
        steps_before_stim = np.argwhere(
            cycle_time_trace >= (period * (phase % 1)))[0][0]
    except IndexError:
        assert 0 == len(np.argwhere(
            cycle_time_trace >= (period * (phase % 1))))
        steps_before_stim = 0  # b/c phase was 0.999 or similar
    stim_step_number = int(stim_duration / timestep_length)
    no_stim_step_number = cycle_step_number - stim_step_number
    # Things we need to know for smoothing
    sd = 1  # standard deviation of smoothing window in ms
    sw = 3  # smoothing window in ms
    base_normal = np.exp(-np.arange(-sw, sw, timestep_length) ** 2 / (2 * sd ** 2))
    normal = base_normal / sum(base_normal)
    # Step through, generating signal
    out = [0.1]
    for i in range(steps_before_stim):
        out.append(out[-1])
    while len(out) < (4 * cycle_step_number + number_of_timesteps):
        for i in range(stim_step_number):
            growth = timestep_length * out[-1] * growth_rate * \
                     (1 - out[-1] / max_signal)
            out.append(out[-1] + growth)
        for i in range(no_stim_step_number):
            decay = timestep_length * out[-1] * decay_rate
            out.append(out[-1] + decay)
    # Smooth signal
    out = np.convolve(normal, out)
    return out[2 * cycle_step_number:2 * cycle_step_number + number_of_timesteps]


# assumes millisecond time input
def calcium_transient(t, amp, tp, w, asy):
    t /= 1000
    return amp * np.exp(-((t ** asy - tp) / w) ** 2)


def twitch_pCa_trace(amp, tp, w, asy, time, stt, dur):
    ca = []
    front = []
    for t in time:
        if t < dur:
            ca.append(calcium_transient(t, amp, tp, w, asy))
        if t < stt:
            front.append(-1)
    # pdb.set_trace()
    baseline = -np.log10(ca[-1])
    pCa = [baseline for _ in front]
    for c_ca in ca:
        pCa.append(-np.log10(c_ca))
    while len(pCa) < len(time):
        pCa.append(baseline)
    return pCa


# ## Configure a run via a saved meta file
def emit(path_local, path_s3, time, poisson=0.0, ls=None, z_line=None,
         pCa=None, comment=None, write=True, hs_params=None, **kwargs):
    """Produce a structured JSON file that will be consumed to create a run

    Import emit into an interactive workspace and populate a directory with
    run configurations to be executed by a cluster.

    Parameters
    ----------
    path_local: string
        The local (absolute or relative) directory to which we save both
        emitted files and run output.
    path_s3: string
        The s3 bucket (and optional folder) to save run output to and to which
        the emitted files should be uploaded.
    time: iterable
        Time trace for run, in ms
    poisson: float
        poisson ratio of lattice. 0.5 const vol; 0 default const lattice;
        negative for auxetic
    ls: float, optional
        Specifies the initial starting lattice spacing which will act as a
        zero or mean for the spacing. If not given, the default lattice
        spacing from hs.hs will be used.
    z_line: float or iterable, optional
        If not given, default distance specified in hs.hs is used. If given as
        float, the z-line distance for the run. If given as an iterable, used as
        trace for run, timestep by timestep.
    pCa: float or iterable, optional
        Same as for z-line.
    comment: string, optional
        Space for comment on the purpose or other characteristics of the run

    write: bool, optional
        True (default) writes file to path_local/name.meta.json. Other values
        don't. In both cases the dictionary describing the run is returned.
    hs_params: a dict of settings that were passed to the sarcomere to override settings
    **kwargs:
        Further keyword args will be included in the output dictionary. These
        are used to sort the resulting runs by their properties of interest.
        For example, where we are varying phase of activation across a series
        of runs we would include the argument, e.g. 'phase=0.2', in order to
        sort over phase when looking at results.

    Returns
    -------
    run_d: dict
        Copy of run dictionary saved to disk as json.

    Examples
    --------
    # >>> emit('./', None, .1, 100, write=False)
    {'actin_permissiveness': None,
    ...  'actin_permissiveness_func': None,
    ...  'comment': None,
    ...  'lattice_spacing': None,
    ...  'lattice_spacing_func': None,
    ...  'name': ...,
    ...  'path_local': './',
    ...  'path_s3': None,
    ...  'timestep_length': 0.1,
    ...  'timestep_number': 100,
    ...  'z_line': None,
    ...  'z_line_func': None}
    """
    # Ensure that the output_dir exists
    os.makedirs(path_local, exist_ok=True)

    run_d = {}
    name = str(uuid.uuid1())
    # ## Build dictionary
    run_d['name'] = name
    run_d['comment'] = comment
    run_d['path_local'] = path_local
    run_d['path_s3'] = path_s3
    run_d['poisson_ratio'] = poisson
    run_d['lattice_spacing'] = ls
    run_d['z_line'] = z_line
    run_d['pCa'] = pCa
    run_d['hs_params'] = hs_params
    run_d['timestep_length'] = np.diff(time)[0]
    run_d['timestep_number'] = len(time)
    # ## Include kwargs
    for k in kwargs:
        run_d[k] = kwargs[k]
    # # ## Ensure vanilla JSON compatibility - vanilla json is not able to read/write numpy arrays
    for key, value in run_d.items():
        if isinstance(value, np.ndarray):
            run_d[key] = list(value)
    # ## Write out the run description
    if write is True:
        try:
            output_filename = os.path.join(path_local, name + '.meta.json')
            with open(output_filename, 'w') as metafile:
                json.dump(run_d, metafile, indent=4)
        except TypeError as e:
            for key, value in run_d.items():
                print(key, type(value))
            print()
            print("Probably, one of the above types is not json compatible,"
                  "\nsee bottom of following error message for more info")
            print()
            raise e
    return run_d
