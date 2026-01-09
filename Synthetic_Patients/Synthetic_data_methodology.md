### Technical Specification: Synthetic Patient Generation & Interaction Modeling

This document provides a detailed technical breakdown of the **RAIDers** (RAre disease AI and Radar) synthetic data pipeline. It describes the mathematical and biological logic used to transform static variant annotations into a dynamic, 15,000-patient cohort partitioned across five ancestral nodes.

---

#### 1. The Challenge: Rare Variant Data Sparsity

In real-world populations, pathogenic ALS variants are exceptionally rare, often with an Allele Frequency (AF) of  or lower.

* **The Sparsity Problem**: Directly applying empirical gnomAD frequencies to a 15,000-person cohort would result in zero carriers for the vast majority of pathogenic sites, rendering machine learning models ineffective.
* **The Solution (Signal Amplification)**: The pipeline utilizes a **Probabilistic Estimation Engine** that "amplifies" these frequencies into a learning-accessible range (0.01% – 0.2%). This ensures that every pathogenic variant is statistically visible to the AI while strictly preserving the **relative ancestral ratios** provided by gnomAD.

---

### 2. The Interaction Model: From Annotations to Phenotypes

The core of the **RAIDers** framework is the **Contextual Interaction Model**, which moves beyond deterministic gene-to-phenotype mapping. By treating the ancestral background as a modifier of a pathogenic "anchor," we simulate the biological reality of **variable penetrance** and **expressivity**.

#### 2.1 Therotical Framework

#### Step A: The Mutation Anchor ($I$)

Baseline severity is derived from ClinVar clinical significance:

$$I = \begin{cases} 0.8 & \text{Pathogenic} \,\ 0.5 & \text{Likely Pathogenic} \end{cases}$$

Step B: The Ancestral Modifier ($M$)

The ancestral genomic context is quantified by the Relative Allelic Ratio ($R$):


**$$R = \frac{AF_{\text{Population}}}{AF_{\text{Global}}}$$**

The modifier ($M$) is then assigned based on biological sensitivity or tolerance:

$$M = \\begin{cases} 0.8 & \text{if } R > 1.5 \quad \text{(Tolerant/Protective)}, \\ 1.2 & \text{if } R < 0.5 \quad \text{(Sensitive/Aggravating)}, \\ 1.0 & \text{if } 0.5 \le R \le 1.5 \quad \text{(Neutral)} \end{cases}$$

Step C: Final Interaction Score ($S$)

The interaction between the anchor and the modifier, adjusted by stochastic noise ($\epsilon$):

$$S = (I \times M) + \epsilon$$
​

**Step B: The Ancestral Modifier ()**
The modifier is calculated using the **Relative Allelic Ratio ()**, which compares population-specific rarity to the global baseline:

The modifier  is then assigned via a threshold-based mapping:

**Tolerant/Protective Background**: If R > 1.5, then M = 0.8 (High prevalence suggests evolved tolerance).

**Sensitive/Aggravating Background**: If R < 0.5, then M = 1.2 (Extreme rarity suggests high sensitivity).

***Neutral Background:** If 0.5 ≤ R ≤ 1.5, then M = 1.0.

**Step C: Final Interaction Score ()**
The interaction between the anchor and the modifier, adjusted by stochastic noise (), determines the final phenotype:

Where  is a random variable sampled from a normal distribution , representing environmental or unobserved factors.

---


#### 2.2 Worked Calculation Example: rs80356732 (TARDBP)

Case 1: Patient in EUR Superpopulation (Protective Context)

Anchor ($I$): $0.8$

Ratio ($R$):

$$R = \frac{0.00010}{0.00005} = 2.0$$

Modifier ($M$): Since $R > 1.5$, $M = 0.8$.

Interaction Score ($S$): (Assuming $\epsilon = 0.01$)

$$S = (0.8 \times 0.8) + 0.01 = 0.65$$

Phenotype Mapping: $0.60 < S \le 0.85 \rightarrow$ Slow Progression.

Case 2: Patient in AFR Superpopulation (Aggravating Context)

Anchor ($I$): $0.8$

Ratio ($R$):

$$R = \frac{0.00001}{0.00005} = 0.2$$

Modifier ($M$): Since $R < 0.5$, $M = 1.2$.

Interaction Score ($S$): (Assuming $\epsilon = 0.01$)

$$S = (0.8 \times 1.2) + 0.01 = 0.97$$

Phenotype Mapping: $S > 0.85 \rightarrow$ Fast Progression.

---

#### 2.3 Phenotype Categorization Thresholds

The continuous interaction score () is discretized into discrete clinical labels to provide the "Ground Truth" for federated subtyping:

| Final Score (S) | Clinical Label|
| --- | --- |
|S > 0.85 | Fast Progression|
| 0.60 < S ≤ 0.85 | Slow Progression |
| S ≤ 0.60 | Asymptomatic / Low Penetrance |

By implementing this logic, we ensure that **Federated Learning** algorithms (XGBoost/K-Means) must resolve the complex relationship between the **Genotype** and the **Hashed Ancestral Neighborhood** to successfully discover molecular subtypes.

---

#### 3. Data Transformation Example

The following table demonstrates how a single variant from `clinvar.cleaned.csv` results in diverse clinical outcomes based on the ancestral background.

| rsID | Gene | Clinical Sig | Base Impact () | Ancestry | AF Ratio () | Modifier () | Final Score | **Resulting Phenotype** |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **rs80356732** | *TARDBP* | Pathogenic | 0.8 | **EUR** | 2.1 (High) | 0.8 | 0.64 | **Slow Progression** |
| **rs80356732** | *TARDBP* | Pathogenic | 0.8 | **AFR** | 0.2 (Low) | 1.2 | 0.96 | **Fast Progression** |

---

#### 4. Core Implementation: Interaction Logic

The following Python snippet implements the label-assignment logic used to populate the federated nodes:

```python
def assign_contextual_phenotype(variant_row, pop_id):
    """
    Assigns clinical phenotypes based on the interaction between 
    mutation impact and ancestral modifiers.
    """
    # 1. Impact Anchor
    base_impact = 0.8 if "Pathogenic" in variant_row['clinical_sig'] else 0.5
    
    # 2. Ancestral Modifier (gnomAD Ratio Logic)
    ratio = variant_row[f'gnomAD_AF_{pop_id}'] / variant_row['gnomAD_AF']
    modifier = 0.8 if ratio > 1.5 else (1.2 if ratio < 0.5 else 1.0)
    
    # 3. Final Interaction with Stochastic Noise (5-10%)
    noise = np.random.normal(0, 0.05)
    score = (base_impact * modifier) + noise
    
    return "Fast Progression" if score > 0.85 else "Slow Progression"

```

---

#### 5. Stochastic Genotype Generation (HWE Simulation)

To convert estimated frequencies into a patient-level genotype matrix (0/1/2), the pipeline employs a **Hardy-Weinberg Equilibrium (HWE)** simulation. This ensures that the synthetic cohort follows the statistical distribution of real-world populations.

```python
def simulate_genotypes_hwe(af, n_samples):
    """
    Simulates individual genotypes based on Allele Frequency (p)
    using the p^2 + 2pq + q^2 distribution.
    """
    p = af
    q = 1.0 - p
    
    # Genotype Probabilities: [Homozygous Ref, Heterozygous, Homozygous Alt]
    probs = [q**2, 2*p*q, p**2]
    
    # Generate Genotypes (0, 1, or 2) for N patients
    genotypes = np.random.choice([0, 1, 2], size=n_samples, p=probs)
    return genotypes

```

This stochastic generation ensures that **Federated Subtyping** (via K-Means or XGBoost) must identify the underlying biological signal across diverse genomic backgrounds, proving the framework's scalability for real-world biobank integration.
