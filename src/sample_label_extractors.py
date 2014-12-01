"""
Some sample label extraction functions to 
pass into parse_fa_tissue

Dist:
    forebrain => 309
    midbrain => 253
    hindbrain => 236
    neural tube => 170
    limb => 162
    other => 596
"""

def lftd_binary(description):
    """ converts descriptions to multi-label
    tissue label.
    Indices correspond to:
        brain => 0
        limb => 1
        heart => 2
        neural => 3
    """
    for raw in description.split("|")[4:]:
        line = raw.strip()
        # if "brain" in line:
        #     return 1
        if "limb" in line:
            return 1
        if "heart" in line:
            return -1
        # if "neural" in line:
        #     return 1
    return None


def lftd(description):
    """ converts descriptions to multi-label
    tissue label.
    Indices correspond to:
        brain => 0
        limb => 1
        heart => 2
        neural => 3
    """
    label = [0, 0]
    for raw in description.split("|")[4:]:
        line = raw.strip()
        if "brain" in line:
            label[0] = 1
        # if "limb" in line:
        #     label[1] = 1
        # if "heart" in line:
        #     label[2] = 1
        # if "neural" in line:
        #     label[3] = 1
    return label


def lfbd(description):
    """ converts descriptions to multi-label
    brain label.
    Indices in label correspond to:
        1 => "forebrain"
        2 => "midbrain"
        3 => "hindbrain"
    """
    label = [0, 0, 0]
    for raw in description.split("|")[4:]:
        line = raw.strip()
        if "midbrain" in line:
            label[0] = 1
        if "forebrain" in line:
            label[1] = 1
        if "hindbrain" in line:
            label[2] = 1
    return label
