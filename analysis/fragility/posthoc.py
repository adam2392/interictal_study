from eztrack import (
    read_result_eztrack
)
from eztrack.io.base import _add_desc_to_bids_fname, DERIVATIVETYPES
from mne_bids.path import _parse_ext


def read_perturbation_result(deriv_path, source_basename, description):
    if description not in [DERIVATIVETYPES.ROWPERTURB_MATRIX.value, DERIVATIVETYPES.PERTURB_MATRIX.value]:
        raise RuntimeError(f'Perturbation matrix derivative description is only '
                           f'one of accepted values.')

    source_basename, ext = _parse_ext(source_basename)
    source_basename = source_basename + '.json'
    deriv_basename = _add_desc_to_bids_fname(
        source_basename, description=DERIVATIVETYPES.ROWPERTURB_MATRIX.value, verbose=False
    )
    deriv_fpath = deriv_path / deriv_basename

    result = read_result_eztrack(deriv_fname=deriv_fpath,
                                 description=description,
                                 normalize=False)

    return result
