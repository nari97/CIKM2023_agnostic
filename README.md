# CIKM2023_agnostic
This github repo contains the code and some results for the paper titled "A Model-Agnostic Method to Interpret Link Prediction Evaluation of Knowledge Graph Embedding" to be published in CIKM2023

Paper link - https://dl.acm.org/doi/pdf/10.1145/3583780.3614763

Results and datasets link - https://drive.google.com/drive/folders/1UcCRLww0_dUAIzUEI_AnL8S8yfViEsZl?usp=sharing

Datasets description:
There are 11 datatsets in the dataset folder: BioKG, FB13, FB15K, FB15K-237, Hetionet, NELL-995, Royals(Test dataset), WN11, WN18, WN18RR and YAGO3-10

Each folder contains:
1. relation2id.txt - Maps the relations to IDs
2. entity2id.txt - Maps the entities to IDs
3. train2id.txt - Contains the triples in the training splits seperated by "\t" or " "
4. valid2id.txt - Contains the triples in the validation splits seperated by "\t" or " "
5. test2id.txt - Contains the triples in the test splits seperated by "\t" or " "

Results description:
1. Materializations - This folder contains the materializations for each dataset and each model
2. MinedRules - This folder contains the rules mined from the materializations for each dataset and model
3. BestRules - This folder contains the best rules obtained for each predicate from the MinedRules for each dataset and model
4. Tables - This folder contains the aggregated results for each dataset

Code description:

1. Python folder contains the code to extract materializations and combining rules and creating results
2. RuleMiner folder contains a gradle project to extract PCA and HC instantiations which is used in the python scripts when combining rules. This was done so that all the instantiations were created using Neo4j embedded in JAVA.
