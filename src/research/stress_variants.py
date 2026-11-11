STRESS_VARIANTS = {
    'baseline': {'obs_noise_model': 'uniform_bounded'},
    'random_missing': {'missing_rate': 0.35, 'obs_noise_model': 'uniform_bounded'},
    'contiguous_dropout': {'contiguous_dropout': True, 'missing_rate': 0.1, 'obs_noise_model': 'uniform_bounded'},
    'missing_boundary': {'boundary_shift': 0.0, 'missing_rate': 0.3, 'obs_noise_model': 'uniform_bounded'},
    'shifted_demand': {'boundary_shift': 0.25, 'obs_noise_model': 'uniform_bounded'},
    'wrong_interval': {'wrong_interval': True, 'obs_noise_model': 'uniform_bounded'},
    'truth_outside_interval': {'truth_outside_interval': True, 'obs_noise_model': 'uniform_bounded'},
    'od_prior_shift': {'od_prior_shift': 0.3, 'obs_noise_model': 'uniform_bounded'},
    'gaussian_misspecification': {
        'obs_noise_model': 'gaussian_misspecification',
        'missing_rate': 0.15,
    },
    'inconsistent_od_counts': {
        'inconsistent_od_counts': True,
        'obs_noise_model': 'uniform_bounded',
    },
    'intentional_boundary_inconsistency': {
        'truth_outside_interval': True,
        'wrong_interval': True,
        'obs_noise_model': 'uniform_bounded',
    },
}
