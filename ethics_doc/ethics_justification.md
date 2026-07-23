# Ethics Justification Statement
## Data Strategy for the XAI Audit Toolkit for UK SME Recruitment

**Project Title:** Developing an Explainable AI Audit Toolkit for Regulatory Compliance in UK SME Recruitment

**Prepared by:** [Your Name]

**Institution:** [Your University]

**Supervisor:** [Supervisor Name]

**Date:** March 2026

**Version:** 1.0

---

## 1. Overview

This document sets out the ethical rationale for the data strategy adopted in this research project. It explains why synthetic data — calibrated against verified real UK demographic and labour market statistics — is the appropriate and ethically sound approach, and why attempting to use real individual-level recruitment records would be both legally problematic and practically unnecessary for the project's aims.

This statement is intended to accompany any ethics committee submission and to demonstrate compliance with the institution's research ethics policy, the UK GDPR, the Data Protection Act 2018, and the British Psychological Society's Code of Ethics and Conduct (where applicable).

---

## 2. Research Aims and Data Requirements

The primary aim of this project is to build and evaluate an AI audit toolkit that can detect demographic bias, generate explainability outputs, and produce regulatory compliance reports for UK SME recruitment processes.

To achieve this, the project requires:

1. **Representative demographic distributions** — proportions of gender, ethnicity, age, disability, and region in the UK working-age population.
2. **Plausible individual candidate records** — synthetic profiles with CV features (experience, education, keyword match scores) that a real screening model might process.
3. **Ground-truth labels** — binary hire/reject decisions, both unbiased and intentionally biased, to enable toolkit testing and validation.

Critically, the project does **not** require:

- Real identities of actual candidates.
- Actual private hiring decisions from real employers.
- Any personally identifiable information.

This distinction is fundamental to the data strategy described below.

---

## 3. Why Real Individual-Level Recruitment Data Was Not Used

### 3.1 Legal Barriers

Individual-level recruitment data is classified as **personal data** under the UK General Data Protection Regulation (UK GDPR) and the Data Protection Act 2018. Specifically:

- Candidate names, contact details, CV content, and hiring decisions constitute personal data under **Article 4(1) UK GDPR**.
- Where the data includes protected characteristics (gender, ethnicity, disability, age), it constitutes **special category data** under **Article 9 UK GDPR**, attracting heightened protections.
- Processing such data for research purposes requires a **lawful basis** (Article 6 UK GDPR) and, for special category data, an **additional condition** (Article 9(2) UK GDPR), such as explicit consent or a substantial public interest basis.

Obtaining meaningful written consent from a sufficient number of real candidates across multiple UK SMEs — ensuring they understand their data will be used in AI fairness research — is not feasible within the scope and timeframe of this project. Alternative lawful bases (e.g., legitimate interests, public task) are unlikely to apply to a student research project without institutional data processing agreements.

Additionally, UK employers are not legally obligated to share recruitment data and, for commercial sensitivity and data protection reasons, would be unlikely to do so voluntarily.

### 3.2 Practical and Methodological Barriers

Even if legal access were theoretically possible:

- Real recruitment datasets are almost never labelled with both the automated model decision and the true hiring outcome, making ground-truth fairness evaluation impossible.
- Real datasets from a single SME would not be representative of the wider UK population, introducing uncontrolled sampling bias into the research findings.
- Real datasets would not include a "known-biased" variant, which is necessary to validate that the toolkit correctly detects bias.
- Protecting anonymity whilst retaining analytical utility is extremely difficult; re-identification risks in small datasets are significant.

### 3.3 Disproportionate Risk with No Scientific Benefit

Collecting real personal data when synthetic alternatives are available and scientifically equivalent introduces risk — to participants, to the institution, and to the researcher — with no compensating methodological benefit. This contravenes the principle of **data minimisation** under Article 5(1)(c) UK GDPR and the ethical principle of **non-maleficence**.

---

## 4. The Synthetic Data Approach

### 4.1 What "ONS-Calibrated Synthetic Data" Means

The synthetic dataset used in this project is generated programmatically using verified, publicly available UK demographic statistics as its calibration source. No real individual's data is used at any point. The generation process works as follows:

1. **Marginal distributions** for gender, ethnicity, age, region, disability, and education level are drawn directly from the sources listed in Section 4.2.
2. **Individual candidate records** are then sampled from these distributions using a probabilistic generation algorithm. Each record is an artificially constructed profile that does not correspond to any real person.
3. **Employment-related features** (years of experience, keyword match scores, employment gap duration) are modelled using distributions informed by labour market research but are not copied from real data.
4. **Hiring labels** are assigned via a scoring function, with an additional "bias injection" variant that intentionally introduces demographic penalties — solely to allow the toolkit's bias detection capabilities to be tested against a known-biased scenario.

### 4.2 Data Sources Used for Calibration

The following publicly available, aggregate-level data sources inform the distributions used in synthetic data generation. No individual-level records from these sources are used.

| Source | Data Used | URL |
|--------|-----------|-----|
| ONS Labour Force Survey (LFS) 2023 | Age and gender distributions in UK working-age population | ons.gov.uk/employmentandlabourmarket |
| ONS Annual Population Survey (APS) 2023 | Ethnicity breakdowns in UK employment | ons.gov.uk |
| NOMIS Labour Market Statistics 2023 | Regional employment distributions | nomisweb.co.uk |
| ONS Disability and Employment (2023) | Disability prevalence in working-age population | ons.gov.uk |
| ONS Qualifications and the Labour Market (2023) | Education level distributions | ons.gov.uk |
| EHRC Workforce Equality Data (2023) | Differential employment rates by protected characteristic | equalityhumanrights.com |
| Sutton Trust Graduate Outcomes Report (2023) | University type and employment outcomes | suttontrust.com |

All sources are freely and publicly available. Accessing them does not require ethics approval, as they contain no personal data.

### 4.3 Scientific Validity

The use of ONS-calibrated synthetic data is a **well-established and academically accepted** methodology in AI fairness research. Key precedents and supporting literature include:

- **Beutel et al. (2019)** — "Putting fairness principles into practice: Challenges, metrics, and improvements." *AIES 2019.* Uses synthetic data calibrated to real population statistics for fairness evaluation.
- **Mehrabi et al. (2021)** — "A Survey on Bias and Fairness in Machine Learning." *ACM Computing Surveys.* Notes that synthetic datasets grounded in real demographic distributions are appropriate for bias detection research.
- **Bellamy et al. (2019)** — "AI Fairness 360: An Extensible Toolkit for Detecting and Mitigating Algorithmic Bias." *IBM Journal of Research and Development.* The AIF360 toolkit itself includes synthetic datasets for exactly this purpose.
- **ICO (2022)** — "Explaining decisions made with AI." Endorses the use of realistic but non-personal data in AI transparency research and auditing.

Calibrating synthetic data to real demographic distributions ensures that:
- The dataset reflects the actual composition of the UK labour market.
- Fairness metrics computed on it are meaningful and comparable to real-world scenarios.
- The toolkit's outputs are externally valid and applicable to real UK SME contexts.

---

## 5. Ethical Principles Applied

### 5.1 Respect for Autonomy

No real individuals are involved in data collection. No consent is required or sought. The autonomy of real job candidates is fully protected because their data is not used at any point.

### 5.2 Non-Maleficence

The synthetic approach eliminates all risk of harm to real individuals from data breaches, re-identification, or misuse. The biased dataset variant is clearly labelled as intentionally constructed for toolkit testing and is not used to make any real hiring decisions.

### 5.3 Beneficence

The research aims to improve fairness in UK recruitment by providing SMEs with accessible tools to detect and explain algorithmic bias. This is a direct social benefit, particularly for candidates from protected groups who face discrimination risks in automated screening processes.

### 5.4 Justice

By grounding the synthetic data in ONS statistics — which capture the full diversity of the UK working-age population including underrepresented ethnic minority groups, disabled workers, and older workers — the research ensures that marginalised groups are accurately represented in the analysis, rather than being further excluded by convenience sampling.

### 5.5 Data Minimisation and Privacy by Design

In line with Article 5(1)(c) UK GDPR and the principle of **Privacy by Design** (Article 25 UK GDPR), only the minimum data necessary for the research is generated and used. No names, email addresses, postcode-level data, or other identifiers are retained in the research dataset beyond what is necessary for plausibility testing. The "name" field generated is entirely fictional.

---

## 6. Data Governance and Security

| Aspect | Approach |
|--------|----------|
| Storage | All data stored on encrypted university-managed storage |
| Access | Restricted to researcher and supervisory team only |
| Retention | Datasets retained for 5 years post-publication in line with university policy; deleted thereafter |
| Sharing | Synthetic datasets may be published as open data (no personal data risk) |
| Anonymisation | Not required — data is synthetic and contains no personal information |
| DPO Notification | Not required — no personal data processed |

---

## 7. Scope of Ethics Review Required

Based on the above, this project:

- **Does not** involve human participants in any data collection activity.
- **Does not** process personal data.
- **Does not** involve deception, covert observation, or any intervention.
- **Does not** involve vulnerable populations.

Under most UK university ethics frameworks, this would typically qualify for **low-risk / expedited review** rather than full committee review. The researcher should:

1. Complete the institution's standard ethics screening checklist.
2. Submit this statement as supporting documentation.
3. Confirm that no personal data is collected or used.

If the institution's ethics board requires full review regardless, this document provides the necessary substantive justification.

---

## 8. Summary Statement

This research uses synthetic data generated from publicly available UK demographic statistics (ONS, APS, NOMIS) as a privacy-preserving, ethically sound, and scientifically valid substitute for real individual-level recruitment data. This approach is consistent with UK GDPR data minimisation principles, established AI fairness research methodology, and the ethical principles of non-maleficence, beneficence, and justice.

The use of real personal recruitment data would introduce significant legal risk, provide no additional scientific benefit, and would be disproportionate to the research aims. Accordingly, the synthetic data approach represents best practice for this type of research.

---

## 9. Declaration

I confirm that:

- [ ] The research described in this document does not involve the collection or use of personal data.
- [ ] All data used is either synthetically generated or drawn from publicly available aggregate statistics.
- [ ] The research has been designed to minimise risk of harm and to comply with UK GDPR and the Data Protection Act 2018.
- [ ] This ethics statement has been reviewed and approved by my supervisor.

**Researcher Signature:** _______________________ **Date:** _____________

**Supervisor Signature:** _______________________ **Date:** _____________

---

*This document should be retained as part of the research ethics record for the duration of the project and for five years thereafter.*

---

**References**

Bellamy, R. K. E., et al. (2019). AI Fairness 360: An Extensible Toolkit for Detecting and Mitigating Algorithmic Bias. *IBM Journal of Research and Development, 63*(4/5), 4:1–4:15.

Beutel, A., et al. (2019). Putting Fairness Principles into Practice. *Proceedings of the 2019 AAAI/ACM Conference on AI, Ethics, and Society.*

ICO. (2022). *Explaining Decisions Made with AI*. Information Commissioner's Office. ico.org.uk

Mehrabi, N., et al. (2021). A Survey on Bias and Fairness in Machine Learning. *ACM Computing Surveys, 54*(6), 1–35.

ONS. (2023). *Labour Force Survey User Guide*. Office for National Statistics.

UK GDPR. (2018). *General Data Protection Regulation as retained in UK law by the European Union (Withdrawal) Act 2018.*
