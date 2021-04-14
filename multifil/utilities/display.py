def display_axial_force_end(half_sarcomere):
    """ Show an end view with axial forces of face pairs

    Parameters:
        half_sarcomere: The half-sarcomere object to be displayed
    Returns:
        None
    """
    # Note: The display requires the form:
    #  [[M0_A0, M0_A1, ..., M0_A5], ..., [M3_A0, ..., M3_A5]]
    forces = [[face.axial_force() for face in thick.thick_faces]
              for thick in half_sarcomere.thick]
    # Display the forces
    display_ends(forces, "Axial force of face pairs", True)


def display_state_end(half_sarcomere, states=(1, 2)):
    """ Show an end view of the current state of the cross-bridges

    Parameters:
        half_sarcomere: The half-sarcomere object to be displayed
        states: List of states to count in the display, defaults
                to [1,2] showing the number of bound cross-bridges
    Returns:
        None
    """
    # Compensate if the passed states aren't iterable
    try:
        iter(states)
    except TypeError:
        states = [states]
    # Retrieve and process cross-bridge states
    # Note: The display requires the form:
    #  [[M0_A0, M0_A1, ..., M0_A5], ..., [M3_A0, ..., M3_A5]]
    state_count = []
    for thick in half_sarcomere.thick:
        state_count.append([])  # Append list for this thick filament
        for face in thick.thick_faces:
            crossbridges = face.get_xb()
            # Retrieve states
            xb_states = [xb.numeric_state for xb in crossbridges]
            # Count states that match our passed states of interest
            count = sum([state in states for state in xb_states])
            state_count[-1].append(count)
    # Display the cross-bridge states
    display_ends(state_count, ("Cross-bridge count in state(s) "
                               + str(states)), False)


def display_state_side(half_sarcomere, states=(1, 2)):
    """ Show a side view of the current state of the cross-bridges

    Parameters:
        half_sarcomere: The half-sarcomere object to be displayed
        states: List of states to count in the display, defaults
                to [1,2] showing the number of bound cross-bridges
    Returns:
        None
    """
    # Compensate if the passed states aren't iterable
    try:
        iter(states)
    except TypeError:
        states = [states]
    # Retrieve and process cross-bridge states
    # Note: The display requires the form:
    # [[A0_0,... A0_N], [M0A0_0,... M0A0_N], ...
    #  [M0A1_0,... M0A1_N], [A1_0,... A1_N]]
    azo = lambda x: 0 if (x is None) else 1  # Actin limited to zero, one
    odd_even = 0
    vals = []
    for thick in half_sarcomere.thick:
        vals.append([])
        for face in thick.thick_faces:
            m_s = [xb.numeric_state for xb in face.get_xb()]
            m_s = [m in states for m in m_s]
            while len(m_s) < 40:
                m_s.append(False)
            a_s = [azo(bs.bound_to) for bs in face.thin_face.binding_sites]
            if odd_even == 0:
                vals[-1].append([])
                vals[-1][-1].append(a_s)
                vals[-1][-1].append(m_s)
                odd_even = 1
            elif odd_even == 1:
                vals[-1][-1].append(m_s)
                vals[-1][-1].append(a_s)
                odd_even = 0
    # Display the cross-bridge states
    title = ("Cross-bridges in state(s) " + str(states))
    for fil in vals:
        for pair in fil:
            half_sarcomere.display_side(pair, title=title)


def display_ends(graph_values, title=None, display_as_float=None):
    """ Show the state of some interaction between the filaments

    Parameters:
        graph_values: Array of values to display in the format:
            [[M0_A0, M0_A1, ..., M0_A5], ..., [M3_A0, ..., M3_A5]]
        title: Name of what is being shown (optional)
        display_as_float: Display values as floats? Tries to determine
            which type of value was passed, but can be manually set to
            True or False (optional)
    Returns:
        None

    The display is of the format:
     +-----------------------------------------------------+
     |           [AA]              [AA]                    |
     |                                                     |
     |  [AA]     0200     [AA]     0300     [AA]           |
     |                                                     |
     |      0200      0010    0100      0050               |
     |           (MM)              (MM)                    |
     |      0100      0010    0100      0010               |
     |                                                     |
     |  [AA]     0100     [AA]     0100     [AA]           |
     |                                                     |
     |           [AA]     0400     [AA]     0100     [AA]  |
     |                                                     |
     |               0200      0020    0200      0020      |
     |                    (MM)              (MM)           |
     |               0200      0010    0300      0020      |
     |                                                     |
     |           [AA]     0600     [AA]     0300     [AA]  |
     |                                                     |
     |                    [AA]              [AA]           |
     +-----------------------------------------------------+
    """
    # Functions for converting numbers to easily displayed formats
    left_float = lambda x: "%-4.1f" % x
    right_float = lambda x: "%4.1f" % x
    left_int = lambda x: "%-4i" % x
    right_int = lambda x: "%4i" % x
    if display_as_float:
        n_l = left_float  # left  number = n_l
        n_r = right_float  # right number = n_r
    elif type(graph_values[0][0]) == int or not display_as_float:
        n_l = left_int
        n_r = right_int
    else:
        n_l = left_float
        n_r = right_float
    # Print the title, or not
    if title is not None:
        print("  +" + title.center(53, "-") + "+")
    else:
        print("  +" + 53 * "-" + "+")
    # Print the rest
    v = graph_values  # Shorthand
    print(
        "  |           [AA]              [AA]                    |\n" +
        "  |                                                     |\n" +
        "  |  [AA]     %s     [AA]     %s     [AA]           |\n"
        % (n_l(v[0][1]), n_l(v[1][1])) +
        "  |                                                     |\n" +
        "  |      %s      %s    %s      %s               |\n"
        % (n_l(v[0][0]), n_r(v[0][2]), n_l(v[1][0]), n_r(v[1][2])) +
        "  |           (MM)              (MM)                    |\n" +
        "  |      %s      %s    %s      %s               |\n"
        % (n_l(v[0][5]), n_r(v[0][3]), n_l(v[1][5]), n_r(v[1][3])) +
        "  |                                                     |\n" +
        "  |  [AA]     %s     [AA]     %s     [AA]           |\n"
        % (n_l(v[0][4]), n_l(v[1][4])) +
        "  |                                                     |\n" +
        "  |           [AA]     %s     [AA]     %s     [AA]  |\n"
        % (n_l(v[2][1]), n_l(v[3][1])) +
        "  |                                                     |\n" +
        "  |               %s      %s    %s      %s      |\n"
        % (n_l(v[2][0]), n_r(v[2][2]), n_l(v[3][0]), n_r(v[3][2])) +
        "  |                    (MM)              (MM)           |\n" +
        "  |               %s      %s    %s      %s      |\n"
        % (n_l(v[2][5]), n_r(v[2][3]), n_l(v[3][5]), n_r(v[3][3])) +
        "  |                                                     |\n" +
        "  |           [AA]     %s     [AA]     %s     [AA]  |\n"
        % (n_l(v[2][4]), n_l(v[3][4])) +
        "  |                                                     |\n" +
        "  |                    [AA]              [AA]           |\n" +
        "  +-----------------------------------------------------+")
    return


def display_side(graph_values, ends=(0, 0, 0), title=None,
                 labels=("A ", "M ", "A "), display_zeros=True):
    """Show the states of the filaments, as seen from their sides

    The input is essentially a list of dictionaries, each of which
    contains the values necessary to produce one of the panels this
    outputs. Each of those dictionaries contains the title (if any)
    for that panel, the side titles, the end values, and the numeric
    interaction values. Currently, the interaction values are limited
    to integers.

    Parameters:
        graph_values: Values to display in the format
            [[A0_0, A0_1, ..., A0_N],
             [M0A0_0, M0A0_1, ..., M0A0_N],
             [M0A1_0, M0A1_1, ..., M0A1_N],
             [A1_0, A1_1, ..., A1_N]]
        ends: None or values for ends in the format
            [A0_end, M0_end, A1_end]
        title: None or a title string
        labels: None or filament labels in the format
            ['A0', 'M0', 'A1']
        display_zeros: Defaults to True
    Returns:
        None

    The printed output is of the format:
     +-----------------------------------------------------------+----+
     | Z-disk                                                    |    |
     | ||----*--*--*--*--*--*--*--*--*--*--*--*--*--*--*         | A0 |
     | 000   00 00 00 00 00 00 00 00 00 00 00 00 00 00 00        |    |
     |                                                           |    |
     |      00 00 00 00 00 00 00 00 00 00 00 00 00 00 00    000  |    |
     |      #==#==#==#==#==#==#==#==#==#==#==#==#==#==#======||  | M0 |
     |      00 00 00 00 00 00 00 00 00 00 00 00 00 00 00  M-line |    |
     |                                                           |    |
     | 000   00 00 00 00 00 00 00 00 00 00 00 00 00 00 00        |    |
     | ||----*--*--*--*--*--*--*--*--*--*--*--*--*--*--*         | A1 |
     | Z-disk                                                    |    |
     +-----------------------------------------------------------+----+
     |                                                           |    |
     | ||---*--*--*--*--*--*--*--*--*--*--*--*--*--*--*          | A2 |
     |                                                           |    |
     |                                                           |    |
     |         #==#==#==#==#==#==#==#==#==#==#==#==#==#==#===||  | M1 |
     |                                                           |    |
     |                                                           |    |
     | ||---*--*--*--*--*--*--*--*--*--*--*--*--*--*--*          | A3 |
     |                                                           |    |
     +-----------------------------------------------------------+----+
    ... and so on.
    """
    # Functions for converting numbers to easily displayed formats
    filter_zeros = lambda x: x if (display_zeros or (x != 0)) else None
    l_l = lambda x: "%-2i" % filter_zeros(int(x))
    l_bl = lambda x: "%-3i" % filter_zeros(int(x))
    # l_r = lambda x: "%2i" % filter_zeros(int(x))
    l_br = lambda x: "%3i" % filter_zeros(int(x))
    # Print the title, if any
    if title is not None:
        print("  +" + title.center(134, "-") + "+----+")
    else:
        print("  +" + 134 * "-" + "+----+")
    # Print the rest
    vals = [[l_bl(ends[0])] + list(map(l_l, graph_values[0])),
            list(map(l_l, graph_values[1])) + [l_br(ends[1])],
            list(map(l_l, graph_values[2])),
            [l_bl(ends[2])] + list(map(l_l, graph_values[3]))]  # Shorthand
    print(
        "  | Z-disk                                                                                                                               |    |\n" +
        "  | ||----*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*                                       | %s |\n"
        % labels[0] +
        "  | %s   %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s                                      |    |\n"
        % tuple(vals[0]) +
        "  |                                                                                                                                      |    |\n" +
        "  |      %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s    %s  |    |\n"
        % tuple(vals[1]) +
        "  |      #==#==#==#==#==#==#==#==#==#==#==#==#==#==#==#==#==#==#==#==#==#==#==#==#==#==#==#==#==#==#==#==#==#==#==#==#==#==#==#======||  | %s |\n"
        % labels[1] +
        "  |      %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s  M-line |    |\n"
        % tuple(vals[2]) +
        "  |                                                                                                                                      |    |\n" +
        "  | %s   %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s                                      |    |\n"
        % tuple(vals[3]) +
        "  | ||----*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*                                       | %s |\n"
        % labels[2] +
        "  | Z-disk                                                                                                                               |    |\n" +
        "  +--------------------------------------------------------------------------------------------------------------------------------------+----+\n"
    )
    # +-----------------------------------------------------------+----+
    # | Z-disk                                                    |    |
    # | ||----*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--...    | A0 |
    # | 000   00 00 00 00 00 00 00 00 00 00 00 00 00 00 00        |    |
    # |                                                           |    |
    # |      00 00 00 00 00 00 00 00 00 00 00 00 00 00 00         |    |
    # |      #==#==#==#==#==#==#==#==#==#==#==#==#==#==#==...     | M0 |
    # |      00 00 00 00 00 00 00 00 00 00 00 00 00 00 00         |    |
    # |                                                           |    |
    # | 000   00 00 00 00 00 00 00 00 00 00 00 00 00 00 00        |    |
    # | ||----*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--...    | A1 |
    # | Z-disk                                                    |    |
    # +-----------------------------------------------------------+----+
    # |                                                           |    |
    # |  ...--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*         | A0 |
    # |       00 00 00 00 00 00 00 00 00 00 00 00 00 00 00        |    |
    # |                                                           |    |
    # |      00 00 00 00 00 00 00 00 00 00 00 00 00 00 00         |    |
    # |  ...=#==#==#==#==#==#==#==#==#==#==#==#==#==#==#==...     | M0 |
    # |      00 00 00 00 00 00 00 00 00 00 00 00 00 00 00         |    |
    # |                                                           |    |
    # |       00 00 00 00 00 00 00 00 00 00 00 00 00 00 00        |    |
    # |  ...--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*         | A1 |
    # |                                                           |    |
    # +-----------------------------------------------------------+----+
    # |                                                           |    |
    # |                                                           | A0 |
    # |                                                           |    |
    # |                                                           |    |
    # |      00 00 00 00 00 00 00 00 00 00  000                   |    |
    # |  ...=#==#==#==#==#==#==#==#==#==#======||                 | M0 |
    # |      00 00 00 00 00 00 00 00 00 00  M-line                |    |
    # |                                                           |    |
    # |                                                           |    |
    # |                                                           | A1 |
    # |                                                           |    |
    # +-----------------------------------------------------------+----+
    #
    #
    return
