from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np


def chunk_text(text, tokenizer, max_length=510):
    """Split text into chunks that fit within token limit"""
    tokens = tokenizer.encode(text, add_special_tokens=False)

    chunks = []
    for i in range(0, len(tokens), max_length):
        chunk = tokens[i:i + max_length]
        chunks.append(tokenizer.decode(chunk))

    return chunks


def encode_long_text(text, tokenizer, model):
    """Encode long text by chunking and averaging embeddings"""
    chunks = chunk_text(text, tokenizer)

    embeddings = []
    for chunk in chunks:
        inputs = tokenizer(chunk, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
            chunk_embedding = outputs.last_hidden_state[:, 0, :].numpy()
            embeddings.append(chunk_embedding)

    # Average all chunk embeddings
    final_embedding = np.mean(embeddings, axis=0)

    return final_embedding


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
    model = AutoModel.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
                                      use_safetensors=True)

    omim_description = """
    Amyotrophic lateral sclerosis-18 (ALS18) is caused by heterozygous mutation in the PFN1 gene (176610) on chromosome 17p13.
    Among 22 patients with ALS that resulted from mutations in PFN1, all displayed limb onset. Given that bulbar onset represents approximately 25% of ALS cases, Wu et al. (2012) proposed that their observation suggests a common clinical phenotype among ALS patients with PFN1 mutations. The age of onset for familial ALS18 patients was 44.8 +/- 7.4 years.
    Wu et al. (2012) performed exome capture followed by deep sequencing on 2 large ALS families of Caucasian (family 1) and Sephardic Jewish (family 2) origin. Both displayed an autosomal dominant inheritance mode and were negative for known ALS-causing mutations. For each family, 2 affected members with maximum genetic distance were selected for exome sequencing. More than 150X coverage was achieved. Using a variety of filters, Wu et al. (2012) were able to reduce the number of candidate mutations to 2 within family 1 and 3 within family 2.
    Wu et al. (2012) identified 4 different missense mutations in the PFN1 gene in 7 families segregating autosomal dominant ALS (176610.0001-176610.0004). Sequencing of the PFN1 coding region in 816 sporadic ALS samples identified 2 samples containing the E117G mutation (176610.0004). In each of the mutations, the altered amino acid was evolutionarily conserved down to the level of zebrafish.
    """

    # Check if truncation is needed
    tokens = tokenizer.encode(omim_description)
    print(f"Total tokens: {len(tokens)}")

    if len(tokens) > 512:
        print("⚠️ Text exceeds 512 tokens - using chunking strategy")
        embedding = encode_long_text(omim_description, tokenizer, model)
    else:
        print("✓ Text fits within limit")
        inputs = tokenizer(omim_description, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
            embedding = outputs.last_hidden_state[:, 0, :].numpy()

    print("Embedding: ")
    print(embedding)
    print(len(embedding[0]))
