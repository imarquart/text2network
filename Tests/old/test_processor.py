import numpy as np


def test_norm( processor):

    # Regular normalization
    x=np.array([1,2,3])
    expected_x=np.array([1/6,2/6,3/6])
    normed_x=processor.norm(x,min_zero=False)
    assert (normed_x==expected_x).all(), "Regular normalization. x: {} - expected: {}".format(normed_x,expected_x)
    assert np.sum(normed_x)== 1, "Regular normalization. sum(x): {} - expected: {}".format(np.sum(normed_x),1)

    # Zero normalization
    x=np.array([0,1,2,3,0])
    expected_x=np.array([0,1/6,2/6,3/6,0])
    normed_x=processor.norm(x,min_zero=False)
    assert (normed_x==expected_x).all(), "Regular normalization. x: {} - expected: {}".format(normed_x,expected_x)
    assert np.sum(normed_x)== 1, "Regular normalization. sum(x): {} - expected: {}".format(np.sum(normed_x),1)

    # Fractional normalization
    x=np.array([0,1/2,1/4,0])
    expected_x=np.array([0,2/3,1/3,0])
    normed_x=processor.norm(x,min_zero=False)
    assert (normed_x==expected_x).all(), "Regular normalization. x: {} - expected: {}".format(normed_x,expected_x)
    assert np.sum(normed_x)== 1, "Regular normalization. sum(x): {} - expected: {}".format(np.sum(normed_x),1)

    # List normalization
    x=[1,2,3]
    expected_x=np.array([1/6,2/6,3/6])
    normed_x=processor.norm(x,min_zero=False)
    assert (normed_x==expected_x).all(), "Regular normalization. x: {} - expected: {}".format(normed_x,expected_x)
    assert np.sum(normed_x)== 1, "Regular normalization. sum(x): {} - expected: {}".format(np.sum(normed_x),1)


    # Delete Minimum normalization
    x=np.array([0,1/2,1/4,1/4])
    expected_x=np.array([0,1,0,0])
    normed_x=processor.norm(x,min_zero=True)
    assert (normed_x==expected_x).all(), "Regular normalization. x: {} - expected: {}".format(normed_x,expected_x)
    assert np.sum(normed_x)== 1, "Regular normalization. sum(x): {} - expected: {}".format(np.sum(normed_x),1)

def test_calculate_cutoffs( processor):

    # Vector of ties of length 5, with degree 4 since one is zero
    x = np.array([0, 0.5, 0.25, 0.15, 0.1])

    sortx = np.sort(x)[::-1]
    max_degree=1000

    # No cutoffs
    percent=100
    expected_degree=len(x[x>0])
    expected_cutoff=0.1
    cutoff_degree, cut_prob = processor.calculate_cutoffs(x, method="percent", percent=percent, max_degree=max_degree, min_cut=0)
    assert cutoff_degree == expected_degree, "No cutoffs. cutoff_degree: {} - expected: {}".format(cutoff_degree, expected_degree)
    assert cut_prob <= expected_cutoff, "No cutoffs. cut_prob: {} - expected: {}".format(cut_prob, expected_cutoff)
    implied_mass=np.sum(sortx[0:cutoff_degree])
    assert implied_mass >= percent/100, "No cutoffs. implied_mass: {} - expected: {}".format(implied_mass*100, percent)

    # 50% cutoff
    percent=50
    expected_degree=1
    expected_cutoff=0.5
    cutoff_degree, cut_prob = processor.calculate_cutoffs(x, method="percent", percent=percent, max_degree=max_degree, min_cut=0)
    assert cutoff_degree == expected_degree, "No cutoffs. cutoff_degree: {} - expected: {}".format(cutoff_degree, expected_degree)
    assert cut_prob <= expected_cutoff, "No cutoffs. cut_prob: {} - expected: {}".format(cut_prob, expected_cutoff)
    implied_mass=np.sum(sortx[0:cutoff_degree])
    assert implied_mass >= percent/100, "No cutoffs. implied_mass: {} - expected: {}".format(implied_mass*100, percent)

    # 75% cutoff
    percent=75
    expected_degree=2
    expected_cutoff=0.25
    cutoff_degree, cut_prob = processor.calculate_cutoffs(x, method="percent", percent=percent, max_degree=max_degree, min_cut=0)
    assert cutoff_degree == expected_degree, "No cutoffs. cutoff_degree: {} - expected: {}".format(cutoff_degree, expected_degree)
    assert cut_prob <= expected_cutoff, "No cutoffs. cut_prob: {} - expected: {}".format(cut_prob, expected_cutoff)
    implied_mass=np.sum(sortx[0:cutoff_degree])
    assert implied_mass >= percent/100, "No cutoffs. implied_mass: {} - expected: {}".format(implied_mass*100, percent)


    # Mean cutoff
    xmean=np.mean(x)
    expected_degree=2
    expected_cutoff=0.2
    cutoff_degree, cut_prob = processor.calculate_cutoffs(x, method="mean", percent=percent,
                                                               max_degree=max_degree, min_cut=0)
    assert cutoff_degree == expected_degree, "No cutoffs. cutoff_degree: {} - expected: {}".format(cutoff_degree, expected_degree)
    assert cut_prob <= expected_cutoff, "No cutoffs. cut_prob: {} - expected: {}".format(cut_prob, expected_cutoff)
    implied_mass=np.sum(sortx[0:cutoff_degree])
    assert implied_mass >= percent/100, "No cutoffs. implied_mass: {} - expected: {}".format(implied_mass*100, percent)

def test_get_weighted_edgelist(processor):

    # Vector of ties of length 5, with degree 4 since one is zero
    x = np.array([0, 0.5, 0.25, 0.15, 0.1])
    max_degree=5

    # 100 Percent
    percent=100
    cutoff_degree, cut_prob = processor.calculate_cutoffs(x, method="percent", percent=percent, max_degree=max_degree, min_cut=0)
    print("Degree: {}, Probability: {}".format(cutoff_degree,cut_prob))
    ties=processor.get_weighted_edgelist(token=100, x=x, time=1995, cutoff_number=cutoff_degree, cutoff_probability=cut_prob, seq_id=100, pos=1, p1="p1",
                          p2="p2", p3="p3", p4="p4", max_degree=max_degree)
    assert len(ties) == cutoff_degree, "{}% cutoffs. Returned ties: {} - expected: {}".format(percent, len(ties),cutoff_degree)


    # 75 Percent
    percent=75
    cutoff_degree, cut_prob = processor.calculate_cutoffs(x, method="percent", percent=percent, max_degree=max_degree, min_cut=0)
    print("Degree: {}, Probability: {}".format(cutoff_degree,cut_prob))
    ties=processor.get_weighted_edgelist(token=100, x=x, time=1995, cutoff_number=cutoff_degree, cutoff_probability=cut_prob, seq_id=100, pos=1, p1="p1",
                          p2="p2", p3="p3", p4="p4", max_degree=max_degree)
    assert len(ties) == cutoff_degree, "{}% cutoffs. Returned ties: {} - expected: {}".format(percent, len(ties),cutoff_degree)



    # 25 Percent
    percent=25
    cutoff_degree, cut_prob = processor.calculate_cutoffs(x, method="percent", percent=percent, max_degree=max_degree, min_cut=0)
    print("Degree: {}, Probability: {}".format(cutoff_degree,cut_prob))
    ties=processor.get_weighted_edgelist(token=100, x=x, time=1995, cutoff_number=cutoff_degree, cutoff_probability=cut_prob, seq_id=100, pos=1, p1="p1",
                          p2="p2", p3="p3", p4="p4", max_degree=max_degree)
    assert len(ties) == cutoff_degree, "{}% cutoffs. Returned ties: {} - expected: {}".format(percent, len(ties),cutoff_degree)

