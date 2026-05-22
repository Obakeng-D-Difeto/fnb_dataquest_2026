# AI Usage Log

## Research Phase

- **Phase:** Research
- **Tool & Prompt:** Claude. "Synthesise the existing Research_Findings.docx with the eight academic papers into one research section. Ensure inline citations and address all requirements in student_task.pdf."
- **Why we used AI:** To merge our existing notes with several peer-reviewed papers into a coherent, cited section faster than reading every paper end-to-end.
- **What we kept:** Section structure (GLMs → Interpretability → WoE/IV → Metrics → Regulatory). The metrics section retained the core explanations from our prior notes. Citations and the WoE/IV theoretical framing (Sudjianto & Burakov, 2025; Sharma et al., 2023) were AI-sourced.
- **What we modified:** Tightened the regulatory section, removed inline tool attributions from the existing draft (those moved here). Mapped regulatory concerns to actual loan-book columns ourselves.
- **What we rejected:** A suggestion to lengthen the GLM section with deeper mathematical derivations so it would be accessible for a non-technical audience.

### I have a research task to tackle, below are the expectations:

• Brief paragraphs, equations, and plots that show the difference between
Generalised Linear Models and Non-Linear Models specifically for classifi-
cation.
• The balance between interpretability and complexity.
• A brief explanation of credit-modelling concepts such as Weights of Evi-
dence Encoding and Information Value. Explain why these are useful in a
credit space.
• A brief description on key metrics such as Accuracy, AUC, Gini, Precision,
Recall and F1. Focusing on how these relate to credit modelling.
• Despite the fact that the data are simulated, research which feature(s) a
regulator may disapprove of and why.
provide search terms I can enter into google search that will yield the best results in learning about these concepts for the paragraphs

- **Why we used AI:** Search terms produced by AI would yield results I could trust and it guided us to the best research papers for the research section that met the requirements of the task paper

---

- **Tool & Prompt:** Claude. Port person A’s logic into our structure with the modelling page, business page, and univariate WoE section

- **Why we used AI:** This was a two man job. Person A was responsible for backend logic and building the credit risk model, person b was responsible for the research section, presentation, script and building the front end shell. Due to the remote nature of our work, to make extra sure the front end and backend meshed well, claude ai was prompted to combine our work into one.

## EDA Phase

- **Tool & Prompt:** Claude. Prompt: What data visualisation techniques are most useful in credit risk modelling

- **Why we used AI:** Given Claude had all the necessary context from my research paper and research task , this prompt was especially helpful because it informed me what a business audience would actual care to see visually before making business decisions.

- **Tool & Prompt:** Claude. What are some notable findings in you can highlight from this dataset?

- **Why we used AI:** This is what was revealed the high percentage of missing data in the months since last delinquency column and mixed data formats. Ai was especially helpful in highlighting these as we weren’t required to go over the csv file manually.

## Modelling Phase

- **Tool & Prompt:** Claude. Lets go through the phases for building the backend logic
  Q: How do you want to pace this? A: Build it but stop for review after each step
  Q: How aggressive should the improved model be? (more features + tighter binning = higher AUC but more justification needed) A: Yes — push close to the LightGBM ceiling
  Decision 2:
  Q: interest_rate — drop or keep? A: Keep it — it's available at decision time, it's a legitimate feature
  Q: loan_purpose dirty data — clean it? A: Yes — normalise the strings before WoE binning (will likely make it a useful feature)
  Q: Include age (medium IV but regulatorily borderline)? A: Yes — go aggressive, include borderline features
  Age was included as I hoped it would be meaningful. And it was as it ranked 5th in our information value for by feature plot in the EDA.
  Q: How do we frame the baseline? A: A — Accept 0.76 as baseline, push improved to ~0.80+
  The other options were "B: Build a deliberately weaker baseline" which was not chosen because it's aritificially nature, and "C:Have two Baselines (artificial and non-artificial)", which was not chosen because it required further explanation in the document which we did not find ideal for our current understanding of credit modeling . Option A was chosen because it was a honest representation and allowed us to fully focus on logistic regression modelling.

---

## App Development Phase

### 2026-05-20 — Streamlit app shell scaffolding

- **Phase:** App
- **Tool & Prompt:** Claude. "Build the Streamlit app shell with the agreed file structure: views, content, core, components, utils."
- **Why we used AI:** Scaffolding 20+ Python files with consistent conventions, caching, and stubs is mechanical work where AI is faster than typing manually.
- **What we kept:** Directory structure, navigation via `st.navigation`, cached data and content loaders, reusable chart and KPI components.

- **Tool & Prompt:** Claude. Build the following data visualisation techniques: Score distribution histogram, Gains / Lift curve, Default rate by decile, Model calibration plot, Expected loss table in R, Missingness visual. Ensure there are interactive. The viewer must be able to change x and y axis categories where necessary.

- **Why we used AI:** AI is especially efficient at looking at the findings discovered and plotting them in tables and graphs that even we can interpret as non data science students.

---

## Presentation Phase

- **Tool & Prompt:** Claude. Create presentation slides. Ensure the followig are included. The key findings from the data quality page,
  The final logistic regression formula in the final model. A comparison of the given baseline, our improved baseline and ceiling baseline using LightGBM. The business recomendations.

- **Why we used AI:** AI's effeciency in extracting all the data from our various webpages and formatting them into 4 slides in a compact manner was especially useful here and saved a lot of time and headache.
