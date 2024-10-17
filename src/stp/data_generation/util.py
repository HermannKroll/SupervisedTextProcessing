def clean_entity_type(ent_type):
    # remove prefix of ChemProt entities
    if ent_type in {"GENE-N", "GENE-Y"}:
        ent_type = "GENE"
    # generalize specific entity types in CDR
    elif ent_type in {"group", "drug_n", "brand"}:
        ent_type = "DRUG"
    return ent_type.upper()
