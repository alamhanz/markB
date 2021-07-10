# https://gist.github.com/BrikerMan/7bd4e4bd0a00ac9076986148afc06507
from gensim.models import KeyedVectors
import pandas as pd

# Load gensim word2vec
PATH_EMBED = '../../../models/'
PATH_MODEL = '../../artifact/'
PATH_DATA = '../../../data/interim/'
w2v_path = PATH_EMBED+'decaf_v7.wv'
# w2v_path = PATH_EMBED+'decaf_v10.wv'

# w2v = KeyedVectors.load_word2vec_format(w2v_path)
w2v = KeyedVectors.load(w2v_path)

print('done')

import io

# Vector file, `\t` seperated the vectors and `\n` seperate the words
"""
0.1\t0.2\t0.5\t0.9
0.2\t0.1\t5.0\t0.2
0.4\t0.1\t7.0\t0.8
"""
out_v = io.open(PATH_MODEL+'vecs_15k_v7.tsv', 'w', encoding='utf-8')

# Meta data file, `\n` seperated word
"""
token1
token2
token3
"""
out_m = io.open(PATH_MODEL+'meta_15k_v7.tsv', 'w', encoding='utf-8')


# certain_words = int(len(w2v.index2word))

# Write meta file and vector file
# for index in range(certain_words):
#     word = w2v.index2word[index]
#     vec = w2v.vectors[index]
#     out_m.write(word + "\n")
#     out_v.write('\t'.join([str(x) for x in vec]) + "\n")

## Certain Words
df_words = pd.read_csv(PATH_DATA+'top20k.csv')
df_words = df_words[:15000]
certain_words = df_words['item_dish_std3'].unique()
print(len(certain_words))

for word in certain_words :
  try :
    vec = w2v[word]
    out_m.write(word + "\n")
    out_v.write('\t'.join([str(x) for x in vec]) + "\n")
  except:
    pass
  
out_v.close()
out_m.close()

# Then we can visuale using the `http://projector.tensorflow.org/` to visualize those two files.

# 1. Open the Embedding Projector.
# 2. Click on "Load data".
# 3. Upload the two files we created above: vecs.tsv and meta.tsv.