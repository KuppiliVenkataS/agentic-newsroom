
from google.cloud import bigquery
from google.oauth2 import service_account
creds = service_account.Credentials.from_service_account_file('/Users/santhilatakv/secrets/agentic-newsroom-132bc7cb81ee.json')
client = bigquery.Client(project='agentic-newsroom', credentials=creds)
query = '''
SELECT SourceCommonName, DocumentIdentifier, V2Themes
FROM `gdelt-bq.gdeltv2.gkg_partitioned`
WHERE DATE(_PARTITIONTIME) = '2026-05-21'
AND (LOWER(V2Themes) LIKE '%oil%' OR LOWER(V2Themes) LIKE '%crude%')
LIMIT 5
'''
result = list(client.query(query).result())
print('Rows returned:', len(result))
for row in result:
    print(row['SourceCommonName'], '|', row['DocumentIdentifier'][:60])