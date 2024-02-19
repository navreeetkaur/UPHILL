import openai
import random
import time
import asyncio
import json
import ast
import logging
import re
import config

def get_accuracy(df):
    numerator = len(df[ (df['claim_veracity']=='true') & (df['entailment_prediction']=='agree') ]) + len(df[ (df['claim_veracity']=='false') & (df['entailment_prediction']=='disagree') ]) +  len(df[ (df['claim_veracity']=='mixture') & (df['entailment_prediction']=='neutral') ])
    denominator = len(df[df['claim_veracity'].isin(['true', 'false', 'mixture'])])
    acc = 100*(numerator/denominator)
    return acc


def get_consistency(df):
    num_levels = len(config.query_templates.keys())
    levels_missing = 0
    num_consistent = 0
    for c_id in list(df['claim_id'].unique()):
        claim_df = df[df['claim_id']==c_id]
        if len(claim_df) != num_levels:
            levels_missing += (num_levels-len(claim_df))
        if claim_df['entailment_prediction'].nunique() == 1:
            num_consistent+=1
    consistency = num_consistent/df['claim_id'].nunique()
    if levels_missing > 0:
        print(f"Consistency Evaluation: {levels_missing} queries with presupposition are missing. Each claim should have {num_levels} queries with presupposition corresponding to it.")
    return consistency


def get_percentage(x):
    return 100 * x / x.sum()


def get_prediction_from_entailment_response(entailment_response):
    try:
        prediction = json.loads(entailment_response)[
            'agreement']
    except:
        logging.debug(
            f"This completion is not in JSON format: \n{entailment_response}")
        try:
            prediction = ast.literal_eval(
                entailment_response)['agreement']
        except:
            try:
                prediction = re.search(
                    r'(\"agreement\")\:\s\"(.*)\"\n\}', str(entailment_response)).group(2)
            except:
                content = str(entailment_response)
                if ': \"Agree\"' in content or ': Agree' in content:
                    prediction = "Agree"
                elif ': \"Disagree\"' in content or ': Disagree' in content:
                    prediction = "Disagree"
                elif ': \"None\"' in content or ': None' in content:
                    prediction = "None"
                else:
                    logging.debug(
                        f"Cannot extract label: \n{entailment_response}")
    if prediction == "Agree":
        prediction = 'agree'
    elif prediction == "Disagree":
        prediction = 'disagree'
    else:
        prediction = 'neutral'
    return prediction


async def get_entailment_response(config, messages):
    completion = None
    while completion is None:
        try:
            completion = openai.ChatCompletion.create(
                model=config['model_name'],
                messages=messages,
                max_tokens=config['max_tokens'],
                temperature=config['temperature'],
                top_p=config['top_p'],
                request_timeout=config['request_timeout'],
            )
        except openai.error.RateLimitError:
            print('Rate limit error, waiting for 40 second...')
            await asyncio.sleep(40)
        except openai.error.APIError:
            print('API error, waiting for 1 second...')
            await asyncio.sleep(1)
        except openai.error.Timeout:
            print('Timeout error, waiting for 1 second...')
            await asyncio.sleep(1)
        except openai.error.ServiceUnavailableError:
            print('Service unavailable error, waiting for 3 second...')
            await asyncio.sleep(3)
        except openai.error.APIConnectionError:
            print('API Connection error, waiting for 40 second...')
            await asyncio.sleep(40)
        except openai.error.InvalidRequestError:
            print('Invalid Request Error, trying by pruning the input prompt...')
            # reduce the evidence length in 'user' message
            if messages[1]['role'] != 'user':
                raise ValueError("Input does not have 'user' message")
            sentences = sentences[:len(sentences)/2]
            messages[1]['content'] = sentences
    return completion.choices[0].message.content


def get_messages(claim, model_response, prompt_dict):
    messages = [{"role": "system", "content": prompt_dict['system']},
                {"role": "user", "content": prompt_dict['user'].format(
                    claim=claim, evidence=str(model_response))}
                ]
    return messages


def get_gpt_response(model_name, prompt, num_responses, get_chat_completion):
    for delay_secs in (2**x for x in range(0, 6)):
        try:
            if get_chat_completion:
                completion = openai.ChatCompletion.create(
                    model=model_name,
                    messages=[
                        {"role": "user", "content": prompt}
                    ], n=num_responses, max_tokens=2000
                )
                model_responses = [
                    choice.message.content for choice in completion.choices]
            else:
                completion = openai.Completion.create(
                    model=model_name, prompt=prompt, max_tokens=2000, n=num_responses)
                model_responses = [
                    choice.text for choice in completion.choices]
            break
        # exceptions for which we should wait a few seconds
        except (openai.error.APIError, openai.error.Timeout, openai.error.APIConnectionError) as e:
            randomness_collision_avoidance = random.randint(0, 1000) / 1000.0
            sleep_dur = delay_secs + randomness_collision_avoidance
            print(
                f"OpenAI Error: {e}. Retrying in {round(sleep_dur, 2)} seconds.")
            time.sleep(sleep_dur)
            continue
        # wait for a few mins for errors due to maintenance, system upgrade,
        # server failure, high traffic, rate limit exceeded
        except (openai.error.ServiceUnavailableError, openai.error.RateLimitError) as e:
            randomness_collision_avoidance = random.randint(0, 1000) / 1000.0
            sleep_dur = (60*delay_secs) + randomness_collision_avoidance
            print(
                f"OpenAI error: {e}. Retrying in {round(sleep_dur, 2)} seconds.")
            time.sleep(sleep_dur)
            continue
        # these errors should not occur ideally if code and API key are fine
        except (openai.error.InvalidRequestError, openai.error.AuthenticationError) as e:
            print(f"OpenAI error not supposed to occur: {e}")
            break
    return model_responses