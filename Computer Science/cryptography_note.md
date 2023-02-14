# Cryptography Note

**cryptosystem**

A cryptosystem is a five-tuple $\cal(P, C, K, E, D)$, where the following conditions are satisfied:

1. $\cal P$ is a finite set of possible plaintexts;

1. $\cal C$ is a finite set of possible ciphertexts;

1. $\cal K$, the keyspace, is a finite set of possible keys;

1. For each $K \in \cal K$, there is an entryption rule $e_K \in \cal E$ and a corresponding decryption rule $d_K \in \cal D$. Each $e_K: \cal P \rightarrow \cal C$ and $d_K: \cal C \rightarrow \cal P$ are functions such that $d_K (e_K (x)) = x$ for every plaintext element $x \in \cal P$.



