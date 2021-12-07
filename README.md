# NLP With Deep Learning (W266)

Submission by *Carolina Arriaga, Ayman, Abhi Sharma*

Winter 2021 | UC Berkeley

## ShapSum: A Framework to Predict Human Judgement Multi-Dimensional Quality Scores for Text Summarization

Text summarization is the task of producing a shorter version of a document.  Model performance  has  been  compared  amongst  each other based mainly on their ROUGE score. The metric has been widely criticized because it only assesses content selection and does not account for other quality metrics such as fluency, grammaticality, coherence, consistency and relevance (Ruder). (Lin, 2004) Combined score metrics like BLEND or DPMFcomb incorporate lexical, syntactic and semantic based metrics and achieve high correlation with human judgement  (Yu et  al., 2015) in  the MT and text generation context.  However,  none of  these  combined  metrics  have  been  tested in  summaries,  and  particularly,  have  moved away from human scores based on Pyramid and Responsiveness scores. Our findings show that multiple metrics used in the summarization field are predictive of multidimensional quality evaluations from experts.  We produced four saturated models using decision trees and the corresponding surrogate Shapley explanation models to measure metric contribution against four dimensions of evaluation (fluency,  rele-vance, consistency, coherence). We hope that our work can be used as a standard evaluation framework  to  compare  summary  quality  between new summarization models.

If you are looking for the auxiliary analysis done by the team regarding varying length summary output vs summary length, along with additional exploration - that can be found [here](https://github.com/abhisha1991/w266_final_project/tree/main/analysis).

#### Project outputs
1. [Link](https://drive.google.com/drive/folders/1_EzQMxyx_lvsHvgrJs7FoYFALpsZM3Xe?usp=sharing) to Google Drive folder.
2. [Link](https://github.com/abhisha1991/w266_final_project/blob/main/report/ShapSum__A_framework_to_predict_human_judgement_multi_dimensional_qualities_for_text_summarization_.pdf) to paper.
3. [Link](https://docs.google.com/presentation/d/1QM0jkJZ2foetrGy1y6AL8szoAcRdYTnJ8TeHR9dOSR4/edit?usp=sharing) to presentation.

