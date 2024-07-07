# Assessing bias against same-gender relationships

This project investigated the prevalence of bias in machine translation systems with respect to same-gender relationships, e.g. "The lawyer kissed her wife" where "her" and "wife" have the same (assumed) grammatical gender.
The results are published under the following paper:

I. Stewart, R. Mihalcea. "Whose wife is it anyway? Assessing bias against same-gender relationships in machine translation." [5th Workshop on Gender Bias in Natural Language Processing.](https://genderbiasnlp.talp.cat/)

## How to replicate

- Generate authentication keys for the APIs of interest.
    - [Amazon](https://docs.aws.amazon.com/polly/latest/dg/prerequisites.html)
    - [Google](https://cloud.google.com/translate/docs/authentication)
    - [Microsoft](https://learn.microsoft.com/en-us/answers/questions/1192881/how-to-get-microsoft-translator-api-key)
- Save authentication data in JSON format under "auth/"
    - E.g. "auth/google.json"
- Run translation queries.
    - `python google --langs es fr it --output_dir data/output/`
- Analyze output.
    - `python compare_MT_api_performance.py amazon.csv google.csv microsoft.csv --output_dir data/output`
    - Generates plots, regression statistics, statistical tests for comparing rates of accurate translation in MT output.

## Completed

- Assessed same-gender translation in Google, Microsoft, Amazon APIs.
    - MT systems unable to translate same-gender relationships from French/Italian/Spanish to English, as compared to different-gender relationships.
    - Includes all combinations of (1) subject words (occupations); (2) relationship sentence contexts; (3) relationship targets (romantic partners).
    - Different error rates based on subject word (occupation), may correlate with different aspects of the occupations used, e.g. relative income level.

## Next steps

- Verification of translation results
    - Try less ambiguous source nouns set, e.g. "the gay man married his boyfriend"
    - Try more context clues, e.g. "the man married his boyfriend, who is also a father"
    - Test open-source MT systems like [Libre Translate](https://libretranslate.com/), [M2M translation](https://huggingface.co/facebook/m2m100_418M).
- Test other systems
    - Coreference resolution in example sentences e.g. "Juan y Carla hablan sobre **su** novia" => can also frame as translation.
    - Contextual prediction e.g. using sentence "Juan se casó con su ____ quien es alto" => P("esposo") vs. P("esposa")
- Extend to other aspects of LGBTQ relationships
    - Bisexual: e.g. "in January he dated a woman, and in February he dated a ___" => P("man") vs. P("woman")
    - Coming-out: "after coming out as gay, Juan started dating a ___" P("man") vs. P("woman")
