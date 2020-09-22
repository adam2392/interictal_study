
EZTRACK Interictal Study
========================

Master repo with eztrack's UI, data manager, and analysis backend.

.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :target: https://github.com/ambv/black
   :alt: Code style: black

Data Organization
-----------------

Data should be organized in the BIDS-iEEG format:

https://github.com/bids-standard/bids-specification/blob/master/src/04-modality-specific-files/04-intracranial-electroencephalography.md

Additional data components:

.. code-block::

   1. electrode_layout.xlsx 
       A layout of electrodes by contact number denoting white matter (WM), outside brain (OUT), csf (CSF), ventricle (ventricle), or other bad contacts.

   2. clinical_metadata.xlsx     
       A database column layout of subject identifiers and their metadata.


Installation Guide
==================

Setup environment from pipenv

.. code-block::

   pipenv install --dev

   # use pipenv to install private repo
   pipenv install -e git+git@github.com:adam2392/eztrack

   # or
   pipenv install -e /Users/adam2392/Documents/eztrack

   # if dev versions are needed
   pipenv install https://api.github.com/repos/mne-tools/mne-python/zipball/master
   pipenv install https://api.github.com/repos/mne-tools/mne-bids/zipball/master