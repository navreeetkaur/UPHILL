# Evaluating Large Language Models for Health-related Queries with Presuppositions

## Data Contents

Our data is hosted [here](https://drive.google.com/drive/folders/13H2-dA0y8eOChwI8O1AqTJTqAr1raPXf). The data is built on top of the [PubHealth](https://arxiv.org/abs/2010.09926) and [Monant Medical Misinformation](https://dl.acm.org/doi/abs/10.1145/3477495.3531726) datasets.
- `UPHILL.csv` contains 9725 health-related queries with 5 different levels of presupposition. Each claim has a veracity associated with it. Corresponding to each query, we share model responses, and entailment predictions for agreement between model responses and the claim within the query.
- `queries.csv` contains 9725 queries, which could be used to evaluate other models.

### Statistics

| Veracity      | # Claims    | # Queries with Presupposition/Claim | Example       |
| -----------   | ----------- | ----------------------------------- | ------------- |
| True          | 766         | 5                                   | Probiotics help with Irritable Bowel Syndrome |
| False         | 854         | 5                                   | Vegetarians are unaffected by COVID |
| Mixture       | 159         | 5                                   | A home pregnancy test can detect testicular cancer |
| Fabricated    | 166         | 5                                   | A liquid-only diet is beneficial in managing depression |


### Data Format

| Fields                    | Type   | Description                                                                                                                                                                  |
|---------------------------|--------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| claim_id                  | string | an identification for the claim                                                                                                                                              |
| claim                     | string | the claim                                                                                                                                                                    |
| claim_veracity            | string | the veracity of the claim (`true`, `false`, `mixture` or `fabricated`)                                                                                                               |
| veracity_explanation      | string | explanation for why the claim is true or false and so on.                                                                                                                    |
| subjects                  | string | subjects that the claim covers, e.g., health care, public health, heart disease etc.                                                                                         |
| date_published            | string | date when the claim was published on the source (i.e. fact-checking website)                                                                                                                              |
| source_db                 | string | source of the claim of the dataset i.e. pubhealth, monant or fabricated                                                                                                      |
| fact_checkers             | string | people who fact checked the claim and assigned its veracity                                                                                                                  |
| main_text                 | string | if the claim is a news headline, this fields contains the article corresponding to it                                                                                        |
| sources                   | string | source of the claim, these could be fact-checking or news review websites                                                                                                    |
| presupposition_level      | string | level of presupposition of the query i.e. `Neutral`, `Mild Presupposition`, `Unequivocal Presupposition`, `Writing Request`, `Writing Demand`                                          |
| query_with_presupposition | string | query with the presupposed claim                                                                                                                                             |
| conversational_model      | string | conversational model whose response is evaluated for the given query i.e. `text-davinci-002` (InstructGPT), `gpt-3.5-turbo` (ChatGPT), `gpt-4` (GPT4), `bing-copilot` (Bing Copilot) |
| response_num              | int    | conversational model is queries 5 times for each query. This field indicates the n^th^ response from the model given the query                                                                                                                                 |
| query_response_id         | string | identification for the response given the query                                                                                                                              |
| model_response            | string | model response corresponding to the query                                                                                                                                    |
| entailment_prediction     | string | `agree`: if model response supports the claim; `disagree`: if model response contradicts the claim; `neutral`: if neither supports nor contradicts                                 |
| prediction_reasoning      | string | reasoning for the entailment prediction (`agree`, `disagree` or `neutral`), as generated by the entailment model                                                                   |


## Evaluation Script

Evaluation script can be found in `src/evaluation.py`. It uses `gpt-3.5-turbo` for entailment prediction. Please paste your OpenAI API key in the file `src/OPENAI_API_KEY` to run it.

Install the dependencies:
```
pip install openai
```

The dataset to be evaluated should be a `.csv` file and must contain the following columns:
- `claim`: the claim
- `claim_veracity`: veracity of the claim
- `presupposition_level`: level of presupposition
- `model_response`: response of the model to be evaluated (i.e. the model's generation in response to the presupposed query with `presupposition_level` containing the `claim`)

Example:
```
python evaluation.py -input_file responses.csv -output_file eval -response_col model_response
```

Running the above would generate a file `eval.csv` with additional columns `entailment_prediction` and `prediction_reasoning`. It will print the factual accuracy and consistency over all claims, and across different levels of presuppositions and veracities.

## Other scripts

Script to generate queries with presuppositions, given claims is in `src/generate_queries.py`. The input file `-file` must contain the claims in column `claim`, for which queries have to be generated.

Usage:
```
python generate_queries.py -file claims.csv -claims_col claim -outfile queries
```

## Citation

If you find our codebase and dataset beneficial, please cite our work:

```
@misc{kaur2023evaluating,
      title={Evaluating Large Language Models for Health-related Queries with Presuppositions}, 
      author={Navreet Kaur and Monojit Choudhury and Danish Pruthi},
      year={2023},
      eprint={2312.08800},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

Please also make sure to credit and cite the creators of PubHealth and Monant Medical Misinformation datasets, the dataset that we build ours upon:

```
@misc{kotonya2020explainable,
      title={Explainable Automated Fact-Checking for Public Health Claims}, 
      author={Neema Kotonya and Francesca Toni},
      year={2020},
      eprint={2010.09926},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

```
@inproceedings{srba2022monant,
  title={Monant medical misinformation dataset: Mapping articles to fact-checked claims},
  author={Srba, Ivan and Pecher, Branislav and Tomlein, Matus and Moro, Robert and Stefancova, Elena and Simko, Jakub and Bielikova, Maria},
  booktitle={Proceedings of the 45th International ACM SIGIR Conference on Research and Development in Information Retrieval},
  pages={2949--2959},
  year={2022}
}
```

## Contact

If you have any questions, please email `navreetkaur[at]iisc.ac.in`. Thanks!
