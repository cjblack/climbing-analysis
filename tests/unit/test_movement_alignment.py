import numpy as np
import pandas as pd
from neurokinematics.multi_modal.alignment import align_movements_to_ephys

# class TestSortingExtractor:
#     def __init__(self, spike_trains):
#         self.spike_trains = spike_trains
#         self.unit_ids = list(spike_trains.keys())
#     def get_unit_spike_train_in_seconds(self, unit_id):
#         return np.asarray(self.spike_trains[unit_id])
    
def test_movement_alignment(generate_alignment_inputs):
    """Test for aligning movement events to ephys recording

    Args:
        generate_alignment_inputs (_type_): _description_
    """
    results = align_movements_to_ephys(generate_alignment_inputs, fs=30000., fps=200.)
    events_ts = results['event_times_ts']
    events_samples = results['event_times_samples']
    assert len(results) == 8
    assert events_ts.dtype == 'float64'
    assert events_samples.dtype == 'int64'
    assert events_ts.unique().size == events_ts.size
    assert events_samples.unique().size == events_samples.size
