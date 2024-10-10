#!/bin/bash

rcv1_dir=$(realpath $1)
dataprep_dir=$(cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
repo_dir=$(realpath $dataprep_dir/..)

tmp_dir=$(mktemp -d)
pushd ${tmp_dir}/

seeded_rng()
{
  seed="$1"
  openssl enc -aes-256-ctr -pass pass:"$seed" -nosalt </dev/zero 2>/dev/null
}

# split xml files into train valid test
find ${rcv1_dir} -name "*.xml" \
  | sort \
  | sort --random-source=<(seeded_rng 0) -R \
  > shuf.list
sed -n 1,5000p shuf.list > test.list
sed -n 5001,10000p shuf.list > valid.list
sed -n 10001,'$'p shuf.list > train.list

# extract text data from xml for each split
for split in test valid train ; do
  cat ${split}.list \
    | xargs -n 1000 python ${dataprep_dir}/rcv1.py \
    | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//' \
    | sed -e '/^[[:space:]]*$/d' \
    | sed -e 's/\t/ /g' \
  > ${split}.txt
done

popd

# copy to data directory and cleanup temp space.
dir=${repo_dir}/data/rcv1
mkdir -p ${dir}
cp ${tmp_dir}/{train,valid,test}.* ${dir}/
rm -r ${tmp_dir}
