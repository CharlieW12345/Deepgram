import requests
import json
import CountVectorizer
from text2digits import text2digits
#manual keyword boosting cosmosDB: https://api.deepgram.com/v1/listen?model=general&numerals=true&language=en-GB&keywords=cosmos db:6&keywords=key:6&keywords=azure:6 : lowest WER: 0.122 # what important is actually the imporvement that can be made
#manual keyword boosting music combination of both techniques : https://api.deepgram.com/v1/listen?model=general&language=en-GB&keywords=hoody:10&keywords=rage:6&replace=cage:luke%20cage" : lowest : 0.58
"&tier=enhanced"
#Method to loop through array of countvectorization and produce string that can appended to the end of the post request 
url = "https://api.deepgram.com/v1/listen?model=general&numerals=true&language=en-GB&teir=nova" #+ CountVectorizer.unigram(10) # specify the n-gram and the amount n-gram ranked from most to least
headers = {
  'Content-Type': 'audio/mp3',
  'Authorization': 'Token 1e0d4fb8962729fad7b364b45cdfb12594dae73f'
}

with open('Transcripts/transcript1/cosmosdb.mp3','rb') as f:
    response = requests.request("POST", url, headers=headers, data=f)

transcript = json.loads(response.text)
print(transcript["results"]["channels"][0]["alternatives"][0]["transcript"])

t2d = text2digits.Text2Digits()
result = t2d.convert(transcript["results"]["channels"][0]["alternatives"][0]["transcript"])
with open("Enhanced.txt","w") as f:
          f.write(result)