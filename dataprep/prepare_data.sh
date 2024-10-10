#!/bin/sh

pip install zstandard
pip install datasets

script_dir=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd ${script_dir}/..
python dataprep/prepare_data.py
