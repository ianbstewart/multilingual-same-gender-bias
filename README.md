# Multilingual gay bias
This project assesses the ability of multilingual NLP systems (e.g. machine translation) to "understand" same-gender relationships.

## Completed

- Assessment of Google Translate.
    - GT is unable to consistently translate SG relationships from French/Italian/Spanish to English, as compared to different-gender relationships.
    - Includes all combos of (1) subject words (occupations); (2) relationship sentence contexts; (3) relationship targets (romantic partners).

## TODO

- Verification of translation results
    - Back-off to simpler keyword set, e.g. "the gay man married his boyfriend"
    - Test open-source MT systems such as this one: https://libretranslate.com/ (because Google Translate may have been messed up by API access)
- Test other systems
    - Coreference resolution (spacy?) in example sentences e.g. "Juan y Carla hablan sobre **su** novia" => can also frame as translation
    - Contextual prediction e.g. "Juan se casó con su ____ (quien es alto)" => P("esposo" | "quien es alto") > P("esposa" | "Juan")
- Extend to other aspects of LGBTQ relationships?
    - Bisexual: e.g. "he dated a woman, then he dated a ___" => P("man") < P("woman")
    - Coming-out: "after coming out as gay, Juan started dating a ___" P("man") > P("woman")
