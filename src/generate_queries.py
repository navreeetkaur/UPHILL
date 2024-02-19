import argparse
import pandas as pd
from tqdm import tqdm
from random import randrange
import utils
import config
import os

def main():
    parser = argparse.ArgumentParser(
        description='Enter inputs to run this file')
    parser.add_argument('-file', '--filepath',
                        help='Path of the input data containing claims', required=True)
    parser.add_argument('-claims_col', '--claims_col',
                        help='Column in the CSV file that contains claims', required=True)
    parser.add_argument('-outfile', '--outfile',
                        help='Name for output file', required=True)
    args = parser.parse_args()

    datasets_path = os.path.dirname(args.filepath)
    claim_col = args.claims_col
    df = pd.read_csv(args.filepath, header=0)

    if claim_col not in list(df.columns):
        raise ValueError(f"{claim_col} column is not present in the input file.")
    
    df_queries = pd.DataFrame(columns=list(df.columns) +
                          ['presupposition_level', 'query_with_presupposition'])
        
    for index, row in tqdm(df.iterrows()):
        claim = row[claim_col]
        for level in config.query_templates.keys():
            num_prompts = len(config.query_templates[level])
            template = config.query_templates[level][randrange(num_prompts)]
            query_generator_prompt = config.query_generator_prompt_template.format(claim=claim, template=template)
            query_with_presupposition = utils.get_gpt_response(
                model_name=config.query_generator_model, prompt=query_generator_prompt, num_responses=1, get_chat_completion=True)
            df_queries.loc[len(df_queries)] = list(df.loc[index]) + [level, query_with_presupposition]
            df_queries.to_csv(
                f"{datasets_path}{args.outfile}.csv", index=False)

if __name__ == "__main__":
    main()    