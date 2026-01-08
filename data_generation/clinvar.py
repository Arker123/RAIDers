import lxml.etree as ET
import csv

XML_FILE = "ClinVarVariationRelease.xml"
OUT_FILE = "clinvar_extracted.csv"

def get_text(elem, path):
    found = elem.find(path)
    return found.text if found is not None else None

def get_attr(elem, path, attr):
    found = elem.find(path)
    return found.get(attr) if found is not None else None

with open(OUT_FILE, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "variation_id",
        "rs_id",
        "gene",
        "variant_type",
        "consequence",
        "chromosome",
        "position",
        "clinical_sig",
        "disease_name",
        "species",
        "mim_gene",
        "mim_disease"
    ])

    context = ET.iterparse(
        XML_FILE,
        events=("end",),
        tag="VariationArchive"
    )

    for _, va in context:
        variation_id = va.get("VariationID")

        species = get_text(va, "Species")

        gene = get_attr(
            va,
            ".//GeneList/Gene",
            "Symbol"
        )

        variant_type = get_text(
            va,
            ".//SimpleAllele/VariantType"
        )

        consequence = get_attr(
            va,
            ".//MolecularConsequence",
            "Type"
        )

        chrom = None
        pos = None
        for loc in va.findall(".//SequenceLocation"):
            if loc.get("Assembly") == "GRCh38":
                chrom = loc.get("Chr")
                pos = loc.get("start")
                break

        rs_id = None
        for x in va.findall(".//XRef"):
            if x.get("DB") == "dbSNP":
                rs_id = x.get("ID")
                break

        clinical_sig = get_text(
            va,
            ".//Classifications/GermlineClassification/Description"
        )

        # --- Disease name (Preferred) ---
        disease_name = None
        for ev in va.findall(".//Trait[@Type='Disease']//ElementValue"):
            if ev.get("Type") == "Preferred":
                disease_name = ev.text
                break

        mim_gene = get_text(
            va,
            ".//Gene/OMIM"
        )

        mim_disease = None
        for x in va.findall(".//XRef"):
            if x.get("DB") == "OMIM" and x.get("Type") == "MIM":
                mim_disease = x.get("ID")
                break

        writer.writerow([
            variation_id,
            rs_id,
            gene,
            variant_type,
            consequence,
            chrom,
            pos,
            clinical_sig,
            disease_name,
            species,
            mim_gene,
            mim_disease
        ])

        va.clear()
        while va.getprevious() is not None:
            del va.getparent()[0]
