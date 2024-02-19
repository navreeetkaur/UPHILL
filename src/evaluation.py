import os
import argparse
from tqdm import tqdm
import utils
import pandas as pd
import asyncio
import config

async def predict(datasets_path, df, response_col, prompt_dict, config, output_filename):
    pred_response_col = [f'prediction_reasoning']
    pred_col = [f'entailment_prediction']

    for index, row in tqdm(df.iterrows()):
        messages = utils.get_messages(
            claim=row['claim'], model_response=row[response_col], prompt_dict=prompt_dict)
        entailment_response = await utils.get_entailment_response(config, messages)
        df.loc[index, pred_response_col] = entailment_response

        try:
            df.loc[index, pred_col] = utils.get_prediction_from_entailment_response(entailment_response)
        except:
            with open("logging_predict_gpt.txt", "a") as f:
                f.write(
                    f"file: {datasets_path}/{output_filename}\tindex: {index}\tresponse_col: {response_col}")
        df.to_csv(
            datasets_path + f'{output_filename}.csv', index=False)
    return df


def print_evaluation_results(df):
    df['presupposition_level'] = pd.Categorical(df['presupposition_level'], config.query_templates.keys())
    print('- ' * 30 + "\n" + '- ' * 10 + "Evaluation Results " + '- ' * 10 + "\n" + '- ' * 30 + "\n")
    print(f'Overall factual accuracy: {utils.get_accuracy(df)}\n')
    print(f'Overall consistency: {utils.get_consistency(df)}\n')

    print("\n" + '- ' * 5 + "Factual accuracy across veracities " + '- ' * 5)
    for veracity in list(df['claim_veracity'].unique()):
        if veracity not in ["true", "false", "mixture"]:
            continue
        veracity_df = df[df['claim_veracity']==veracity]
        veracity_acc = utils.get_accuracy(veracity_df)
        print(f'{veracity}: {veracity_acc}')
        
    print("\n" + '- ' * 5 +"Factual accuracy across presupposition levels " + '- ' * 5)
    for level in list(df['presupposition_level'].unique()):
        level_df = df[df['presupposition_level']==level]
        level_acc = utils.get_accuracy(level_df)
        print(f'{level}: {level_acc}')
        
    print("\n" + '- ' * 5 + "Agreement distribution across veracities for all claims " +'- ' * 5)
    print(df.groupby(['entailment_prediction', 'claim_veracity']).size().groupby('claim_veracity', group_keys=False).apply(utils.get_percentage).unstack(fill_value=0))

    print("\n" + '- ' * 5 + "Consistency across veracities " + '- ' * 5)
    for veracity in list(df['claim_veracity'].unique()):
        veracity_df = df[df['claim_veracity']==veracity]
        veracity_acc = utils.get_consistency(veracity_df)
        print(f'{veracity}: {veracity_acc}')

    print("\n" + '- ' * 5 + "Agreement distribution across presupposition levels for all claims " +'- ' * 5)
    print(df.groupby(['presupposition_level', 'entailment_prediction']).size().groupby('presupposition_level', group_keys=False).apply(utils.get_percentage).unstack(fill_value=0))

    for veracity in list(df['claim_veracity'].unique()):
        veracity_df = df[df['claim_veracity']==veracity]
        print("\n" + '- ' * 5 + f"Agreement distribution across presupposition levels for {veracity} claims " +'- ' * 5)
        print(veracity_df.groupby(['presupposition_level', 'entailment_prediction']).size().groupby('presupposition_level', group_keys=False).apply(utils.get_percentage).unstack(fill_value=0))


def main():
    parser = argparse.ArgumentParser(
        description='Enter inputs to run this file')
    parser.add_argument('-input_file', '--input_file',
                        help='Path of the input data; data should be in .csv format', required=True)
    parser.add_argument('-output_file', '--output_file',
                        help='Name for output file', required=True)
    parser.add_argument('-response_col', '--response_col',
                        help='Column containing model responses in the CSV file', required=True)
    args = parser.parse_args()

    datasets_path = os.path.dirname(args.input_file)
    df_filename = os.path.basename(args.input_file)
    output_filename = args.output_file
    response_col = args.response_col

    print(f"Reading the input dataset {df_filename} . . . ")
    df = pd.read_csv(datasets_path + df_filename, header=0)
    if response_col not in list(df.columns):
        raise ValueError(f"{response_col} column is not present in the input file.")

    print(f"Predicting agreement between model responses and claims for {response_col}")
    asyncio.run(predict(datasets_path=datasets_path, df=df, response_col=response_col,
                prompt_dict=config.entailment_prompt, config=config.entailment_model_config, output_filename=output_filename))
    
    eval_df = pd.read_csv(datasets_path + f'{output_filename}.csv', header=0)
    print_evaluation_results(eval_df)


if __name__ == "__main__":
    main()