# Identification of causal discourse relations in French text using machine-translated training resources

This project is the result of the thesis in partial fullfilment of the MSc. Data Science and Society at Tilburg University, The Netherlands (Spring 2022).

Causality is a central scientific concept which answers the question: "What was the cause of event A?". It is particularly useful in policy making as governments want to know the real impact of their policies or programs, their mechanisms and unintended consequences. Such information can be extracted from text (citizen consultations, social media, etc.) but is very unstructured and requires expertise and heavy hand labelling.

Causality in texts can be expressed a discourse relations, which are logical links between 2 free-standing, meaningful spans of texts. A free-standing span of text, also known as an argument, generally contains one idea, event or action. Two arguments can be linked by a relation of opposition, entailment, causality, etc. This thesis aimed at automatically identify causal discourse relations in texts in an effort to structure such information.

Research on this topic is already extensive in English, but has been poor in other languages like French. Yet the French government has led many citizen consultation campaigns on topics such as crime, marijuana liberalization etc... yielding 10 000s answers and was only able to run very superficial analyses. The lack of French NLU models able to extract and structure the logical relations present in texts impedes the full exploitation of such valuable texts resources.

The scarcity of research in French is mainly due to the lack of French hand-labelled training data. Indeed, hand-labelling discourse relation requires linguistics expertise and can be very labor intensive. Alternatives have been studied with multi-lingual model and zero-shot training. This thesis explores a simple and low-cost alternative to the lack of French training data: machine-translated training data from English to French. Results showed that machine-translated data can indeed be used to train a French causal discourse relation classifier but that bigger and more advanced models like BERT should be preferred.

The repo contains the full thesis report (thesis-text.pdf) as well as the 2 datasets used in this thesis: a English-to-French machine-translated version of the PDTB2 and the French discourse relation database Explicadis. Finally, two Jupyter notebooks show the models' details and how their respective training.
