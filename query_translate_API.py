"""
Query translation API, using generic classes to
represent APIs.
"""
import os, requests, uuid, json
from argparse import ArgumentParser
from data_helpers import load_clean_relationship_sent_data
from tqdm import tqdm
import boto3
from google.cloud.translate_v3 import TranslationServiceClient
tqdm.pandas()
class TranslationAPI:
    def __init__(self, auth_params):
        self.auth_params = auth_params

    def query(self, source_text, source_lang, target_lang):
        pass

class MicrosoftTranslationAPI(TranslationAPI):

    def __init__(self, auth_params):
        super().__init__(auth_params)
        self.subscription_key = auth_params['subscription_key']
        self.region = auth_params['region']
        self.endpoint = auth_params['endpoint']
        self.path = '/translate?api-version=3.0'
        os.environ['TRANSLATOR_TEXT_SUBSCRIPTION_KEY'] = self.subscription_key
        os.environ['TRANSLATOR_TEXT_REGION'] = self.region
        os.environ['TRANSLATOR_TEXT_ENDPOINT'] = self.endpoint

    def query(self, source_text, source_lang, target_lang):
        query_params = f'&from={source_lang}&to={target_lang}'
        query_url = self.endpoint + self.path + query_params
        headers = {
            'Ocp-Apim-Subscription-Key': self.subscription_key,
            'Ocp-Apim-Subscription-Region': self.region,
            'Content-type': 'application/json',
            'X-ClientTraceId': str(uuid.uuid4())
        }
        body = [{
            'text' : source_text,
        }]
        request = requests.post(query_url, headers=headers, json=body)
        response = request.json()
        target_text = response[0]['translations'][0]['text']
        return target_text

class AmazonTranslationAPI(TranslationAPI):

    def __init__(self, auth_params):
        super().__init__(auth_params)
        os.environ['AWS_ACCESS_KEY_ID'] = auth_params['access_key']
        os.environ['AWS_SECRET_ACCESS_KEY'] = auth_params['secret_access_key']
        self.client = boto3.client(
            service_name='translate',
            region_name=auth_params['region'],
            use_ssl=True
        )

    def query(self, source_text, source_lang, target_lang):
        result = self.client.translate_text(
            Text=source_text,
            SourceLanguageCode=source_lang,
            TargetLanguageCode=target_lang
        )
        output = result.get('TranslatedText')
        return output

class GoogleTranslationAPI(TranslationAPI):

    def __init__(self, auth_params):
        super().__init__(auth_params)
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = auth_params['file']
        self.client = TranslationServiceClient()

    def query(self, source_text, source_lang, target_lang):
        location = 'global'
        parent = f"projects/{self.auth_params['project_id']}/locations/{location}"
        response = self.client.translate_text(
            request={
                'parent' : parent,
                'contents' : [source_text],
                'mime_type' : 'text/plain',
                'source_language_code': source_lang,
                'target_language_code' : target_lang,
            }
        )
        output = response.translations[0].translated_text
        return output

translators = {
    'microsoft' : MicrosoftTranslationAPI,
    'amazon' : AmazonTranslationAPI,
    'google' : GoogleTranslationAPI,
}
def main():
    parser = ArgumentParser()
    parser.add_argument(
        'translator_type',
        choices=['microsoft', 'amazon', 'google'],
        help='Translator API service to query'
    )
    parser.add_argument(
        '--langs',
        nargs='+',
        default=['es', 'fr', 'it'],
        help='Source languages to test'
    )
    parser.add_argument(
        '--output_dir',
        default='data/output/'
    )
    args = parser.parse_args()

    ## load data
    auth_file = os.path.join('auth', f'{args.translator_type}.json')
    auth_data = json.load(open(auth_file))
    ## Google: add file path to params
    if(os.path.basename(auth_file) == 'google.json'):
        auth_data['file'] = auth_file

    ## load data
    relationship_sent_data = load_clean_relationship_sent_data(langs=args.langs)

    ## get translator
    translator = translators[args.translator_type](auth_data)
    target_lang = 'en'
    target_lang_sents = relationship_sent_data.progress_apply(
        lambda x: translator.query(x.loc['sent'], x.loc['lang'], target_lang),
        axis=1,
    )

    ## save data
    output_data = relationship_sent_data.assign(**{
        'target_sent' : target_lang_sents,
    })
    output_data_file = os.path.join(
        args.output_dir, f'{args.translator_type}.csv'
    )
    output_data.to_csv(output_data_file, index=False)

if __name__ == '__main__':
    main()
