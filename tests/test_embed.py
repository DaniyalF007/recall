from sentence_transformers import util
from sentence_transformers import SentenceTransformer

# load the embedding model (downloads once, then cached)
model = SentenceTransformer("all-MiniLM-L6-v2")

# a few example sentences to test
sentences = [
    "The cat sat on the mat.",
    "A kitten rested on the rug.",
    "Python is a programming language."
]

# turn each sentence into numbers
embeddings = model.encode(sentences)

print("Shape of embeddings:", embeddings.shape)
print("First embedding (first 10 numbers):")
print(embeddings[0][:10])

# compare how similar each pair of sentences is
sim_1_2 = util.cos_sim(embeddings[0], embeddings[1])
sim_1_3 = util.cos_sim(embeddings[0], embeddings[2])

print("Cat/mat vs kitten/rug similarity:", sim_1_2.item())
print("Cat/mat vs Python similarity:", sim_1_3.item())
