"""Portable entry point. Optional preinstalled library root for managed hosts."""
import os, sys
from pathlib import Path
approved=os.environ.get('Q1_APPROVED_SITE_PACKAGES')
if approved:
    if not Path(approved).is_dir(): raise RuntimeError('Configured preinstalled site-packages does not exist')
    sys.path.insert(0,approved)
os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).resolve().parents[1]/'.runtime'/'matplotlib'))
from pipeline import main
if __name__=='__main__':main()
