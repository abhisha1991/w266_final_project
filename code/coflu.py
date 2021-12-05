# coding=utf-8
# Copyright 2020 The HuggingFace Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" ROUGE metric from Google Research github repo. """

# The dependencies in https://github.com/google-research/google-research/blob/master/rouge/requirements.txt
import absl  # Here to have a nice missing dependency error message early on
import nltk  # Here to have a nice missing dependency error message early on
import numpy  # Here to have a nice missing dependency error message early on
import six  # Here to have a nice missing dependency error message early on
from rouge_score import rouge_scorer, scoring

import datasets
import statistics


_CITATION = """\
@inproceedings{TBD,
    title = "TBD",
    author = "Arriaga, Moawad, Abhisha",
    booktitle = NONE,
    month = Dec,
    year = "2021",
    address = "San Francisco, California",
    publisher = "TBD",
    url = "TBD",
    pages = "X--Y",
}
"""

_DESCRIPTION = """\
COFLU, or Content-Fluence Evaluation, is a metric that combines ROUGE at a sentence level and summary level
based on given weights to each component to measure the content captured in a summary with a Fluency evaluation
based on an unsupervised score that relies on a LSTM LM approach. No references are required to compute the
fluency score of a summary. 
The metric seeks to capture the content and fluency in a summary (0,1), where 1 is the highest score.

Note that ROUGE is case insensitive, meaning that upper case letters are treated the same way as lower case letters.
Fluency evaluation is not.

This metric is a wrapper around Google Research reimplementation of ROUGE:
https://github.com/google-research/google-research/tree/master/rouge

Also, includes the QRNN used in Salesforce aws-lstm-lm to model the Fluency:
https://github.com/salesforce/awd-lstm-lm.git
"""

_KWARGS_DESCRIPTION = """
Calculates content and fluency scores for a list of hypotheses and references
Args:
    predictions: list of predictions to score. Each predictions
        should be a string with tokens separated by spaces.
    references: list of reference for each prediction. Each
        reference should be a string with tokens separated by spaces.
    weights: list of weights [w1,w2] for the sentence-level ROUGE and summary-level ROUGE score.
    They must add to 1.
    
Returns:
    mean_rouge_socre: rouge.recall(1, 2, L),
    mean_rouge_summary: rouge.recall(LSum),
    mean_weighted_rouge: w1*mean_rouge_score + w2*mean_rouge_summary,
    
Examples:

    >>> coflu = load_metric('datasets/metrics/coflu')
    >>> predictions = ["hello there", "general kenobi"]
    >>> references = ["hello there", "general kenobi"]
    >>> weights = [0.2, 0.8]
    >>> results = rouge.compute(predictions=predictions, references=references, weights=weights)
    >>> print(results)
    >>> (0.16209485347416383, 0.18423423423423424, 0.17316454385419905
"""


@datasets.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class Rouge(datasets.Metric):
    def _info(self):
        return datasets.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=datasets.Features(
                {
                    "predictions": datasets.Value("string", id="sequence"),
                    "references": datasets.Value("string", id="sequence"),
                }
            ),
            codebase_urls=["https://github.com/google-research/google-research/tree/master/rouge"],
            reference_urls=[
                "https://en.wikipedia.org/wiki/ROUGE_(metric)",
                "https://github.com/google-research/google-research/tree/master/rouge",
            ],
        )


    def get_rouge_scores(self, predictions, references, rouge_types=None, use_agregator=True, use_stemmer=False):
      pass
        

    def _compute(self, predictions, references, weights, rouge_types=None, use_agregator=True, use_stemmer=False):
        if rouge_types is None:
            rouge_types = ["rouge1", "rouge2", "rougeL", "rougeLsum"]

        scorer = rouge_scorer.RougeScorer(rouge_types=rouge_types, use_stemmer=use_stemmer)
        if use_agregator:
            aggregator = scoring.BootstrapAggregator()
        else:
            scores = []

        for ref, pred in zip(references, predictions):
            score = scorer.score(ref, pred)
            if use_agregator:
                aggregator.add_scores(score)
            else:
                scores.append(score)

        if use_agregator:
            result = aggregator.aggregate()
        else:
            result = {}
            for key in scores[0]:
                result[key] = list(score[key] for score in scores)

        # Get sentence all Rouge Scores
        # rouge_scores = get_rouge_scores(predictions, references)
        
        # Get sentence level Rouge Score (1,2,L)
        mean_rouge_sentences = statistics.mean([result['rouge1'].mid.recall, 
                  result['rouge2'].mid.recall, 
                  result['rougeL'].mid.recall])
        
        # Get sentence level Rouge Score (LSum)
        mean_rouge_summary = result['rougeLsum'].mid.recall

        # Print results
        print(f"Mean rouge (1,2,L) = {mean_rouge_sentences}")
        print(f"Mean rouge (LSum) = {mean_rouge_summary}")

        # Use weights to promote more Sentence Level or Summary level Rouge
        output = weights[0]*mean_rouge_sentences + weights[1]*mean_rouge_summary
        print(f"Mean w1*R_bar(1,2,L) + w2*R(LSum) = {output}")
        
        return mean_rouge_sentences, mean_rouge_summary, output
