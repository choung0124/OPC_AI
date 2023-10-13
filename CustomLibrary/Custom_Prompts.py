Final_Answer_Template_Alpaca = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
Give a detailed answer the question, using information from the context below. Infer mechanisms or pathways from the relational information in the context to add more depth to your answer.
Your Previous Answers:
{input}

### Input:
Quesiton: {question}

### Response: """

OPC_Entity_type_Template = """### User:
Give the entity types of all the bimedical entities provided in the input, choose the entity types that accurately represent each entity provided in the input. Your response should be a list of tuples, use double quotes instead of single quotes. Choose one for each entity from the list below:
Entity Types:
Amino_acid_sequence, Analytical_sample, Biological_process, Biological_sample, Cellular_component, Chromosome, Clinical_variable, Clinically_relevant_variant, Complex, Disease, Drug, Experiment, Experimental_factor, Food, Functional_region, GWAS_study, Gene, Known_variant, Metabolite, Modification, Modified_protein, Molecular_function, Pathway, Peptide, Phenotype, Project, Protein, Protein_structure, Publication, Subject, Timepoint, Tissue, Transcript, Units, User
Your response should strictly follow this format:
Entities: [("Entity1_name", "Entity1_ype"), ("Entity2_name", "Entity2_type")]
Input: {input}

### Assistant:
"""


OPC_Entity_Extraction_Template = """### User:
Do not answer the question. Extract all entities from the question below. Do not put square brackets around each extracted entity. Do not extract verbs or adjectives, only nouns.
Question: {input}
Response template examples:
Your response should look like this if there is only one entity in the question:
Entities: ["entity"]
Your response should look like this if there are two entities in the question:
Entities: ["entity1", "entity2"]
Your response should look like this if there are three entities in the question:
Entities: ["entity1", "entity2", "entity3"]
Your response should look like this if there are four entities in the question:
Entities: ["entity1", "entity2", "entity3", "entity4"]
Your response should strictly follow this format if there are no entities in the question:
Entities: []
PLEASE DO NOT DO SOMETHING LIKE THIS:
Entities: [["entity1"], ["entity2"]]

### Assistant:
"""


Graph_Answer_Gen_Template_alpaca = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction: As if you were a healthcare professional/biomedical researcher, give a detailed answer the question, using information from the context below. Infer mechanisms or pathways from the relational information in the context to add more depth to your answer.
Question: {question}
Context: 
{input}

### Response:
"""

