import argparse
from os.path import isfile, exists

from typing import List

import re

import torch as th
import gensim

from tqdm import tqdm


def parse_wiki_dump(wiki_xml_file: str, limit=2000000) -> List[str]:
    xml_f = open(wiki_xml_file, "r")

    sentences = []
    for i, l in tqdm(enumerate(xml_f)):
        if i > limit:
            break
        sentences.append(l)

    res = [re.sub(r"</?[a-z]+>", "", s) for s in tqdm(sentences)]

    return res


def gen_embedding(ft_model_file: str, sentence: str, hidden_length: int, n_channel: int) -> th.Tensor:
    words = re.findall(r"[\w]+", sentence)
    words = words[:-(len(words) % hidden_length)]
    res = th.zeros(1, len(words), n_channel)

    ft = gensim.models.fasttext.FastText.load(ft_model_file)

    for i, w in enumerate(words):
        vec = th.tensor(ft.wv[w], dtype=th.float)
        res[0, i, :] = vec

    return res.permute(0, 2, 1)


def main() -> None:
    parser = argparse.ArgumentParser("Train word embedding")

    parser.add_argument("--out-fasttext", type=str, required=True, dest="out_fasttext")
    parser.add_argument("--input-file", type=str, required=True, dest="input_file")
    parser.add_argument("--size", type=int, required=True)

    args = parser.parse_args()

    out_ft_file = args.out_fasttext
    input_file = args.input_file
    size = args.size

    assert exists(input_file), f"Input text file \"{input_file}\" doesn't exist."
    assert isfile(input_file), f"\"{input_file}\" is not a file."

    sentences = parse_wiki_dump(input_file)

    print(len(sentences))

    ft = gensim.models.fasttext.FastText(sentences=sentences, size=size, workers=8, window=5, word_ngrams=5, cbow_mean=0)

    ft.save(out_ft_file)


if __name__ == '__main__':
    main()
