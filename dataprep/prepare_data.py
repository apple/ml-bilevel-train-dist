import os

from datasets import load_dataset


def load_all():
    load_dataset("allenai/c4", "en")
    load_dataset("monology/pile-uncopyrighted")
    load_dataset("monology/pile-test-val")


def write_as_text(limit, max_len, output, ds):
    need_parent_directory(output)
    with open(output, "w") as fout:
        for i, example in enumerate(ds):
            x = example["text"]
            x = (
                x.replace("\n", " ")
                .replace("\r", " ")
                .replace("\t", " ")
                .replace("\x00", "")
            )
            x = x[:max_len]
            if x and not x.isspace():  # skip empty documents.
                fout.write(x)
                fout.write("\n")
            if limit > 0 and i + 1 == limit:
                break


def need_parent_directory(filename):
    # Get the parent directory from the filename
    parent_directory = os.path.dirname(filename)
    os.makedirs(parent_directory, exist_ok=True)


def prepare_c4_split(limit, max_len, split, output):
    ds = load_dataset("allenai/c4", "en", split=split)
    write_as_text(limit, max_len, output, ds)


def prepare_pile_split(limit, max_len, subset, split, output):
    if split == "train":
        ds = load_dataset("monology/pile-uncopyrighted", split=split)
    else:
        ds = load_dataset("monology/pile-test-val", split=split)
    ds = ds.filter(lambda example: example["meta"]["pile_set_name"] == subset)
    write_as_text(limit, max_len, output, ds)


def prepare_gutenberg():
    # gutenberg is different since the validation set contains only few books.
    # we make a new split from the training set.
    ds = load_dataset("monology/pile-uncopyrighted", split="train")
    ds = ds.filter(lambda x: x["meta"]["pile_set_name"] == "Gutenberg (PG-19)")

    is_valid = lambda x: x[0] % 10 == 0
    ds_valid = map(lambda x: x[1], filter(is_valid, enumerate(ds)))
    write_as_text(
        limit=-1,
        max_len=256,
        output="data/pile_100k_gutenberg_resplit/valid.txt",
        ds=ds_valid,
    )

    is_train = lambda x: not is_valid(x)
    ds_train = map(lambda x: x[1], filter(is_train, enumerate(ds)))
    write_as_text(
        limit=-1,
        max_len=256,
        output="data/pile_100k_gutenberg_resplit/train.txt",
        ds=ds_train,
    )


def prepare_c4():
    prepare_c4_split(
        limit=35631728,
        max_len=256,
        split="train",
        output="data/c4/train.txt",
    )
    prepare_c4_split(
        limit=364608,
        max_len=256,
        split="validation",
        output="data/c4/validation.txt",
    )


def prepare_pile():
    subsets = [
        ("arxiv", "ArXiv"),
        ("europarl", "EuroParl"),
        ("freelaw", "FreeLaw"),
        ("pubmed_abstracts", "PubMed Abstracts"),
        ("stackexchange", "StackExchange"),
        ("wikipedia_en", "Wikipedia (en)"),
    ]
    for dir, subset in subsets:
        prepare_pile_split(
            limit=100_000,
            max_len=256,
            subset=subset,
            split="train",
            output=f"data/pile_100k_{dir}/train.txt",
        )
        prepare_pile_split(
            limit=-1,
            max_len=256,
            subset=subset,
            split="validation",
            output=f"data/pile_100k_{dir}/valid.txt",
        )


if __name__ == "__main__":
    prepare_c4()
    prepare_pile()
    prepare_gutenberg()
