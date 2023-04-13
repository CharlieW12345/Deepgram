
from deepgram import Deepgram
import asyncio, json
from collections import Counter
import re

# The API key we created in step 3
DEEPGRAM_API_KEY = '1e0d4fb8962729fad7b364b45cdfb12594dae73f'
# Replace with your file path and audio mimetype
PATH_TO_FILE = 'transcripts/transcript1/cosmosDB.mp3'
#PATH_TO_FILE = 'music.mp3'
MIMETYPE = 'audio/mp3'


async def standard(): 
    # Initializes the Deepgram SDK
    dg_client = Deepgram(DEEPGRAM_API_KEY)
    
    with open(PATH_TO_FILE, 'rb') as audio:
        source = {'buffer': audio, 'mimetype': MIMETYPE}
        options = { "punctuate": True, "diarize" : False, "paragraphs" : True, "model": "general", "language": "en-US" }  
        response = await dg_client.transcription.prerecorded(source, options)
        transcript = response["results"]["channels"][0]["alternatives"][0]["paragraphs"]["transcript"] 
        with open("Standard.txt","w") as f:
          f.write(transcript)
        print(json.dumps(transcript, indent=4))


async def enhanced(): 
    # Initializes the Deepgram SDK
    dg_client = Deepgram(DEEPGRAM_API_KEY) 
    
    with open(PATH_TO_FILE, 'rb') as audio:
        source = {'buffer': audio, 'mimetype': MIMETYPE}
        options = {"punctuate": True, "diarize" : False, "paragraphs" : True, "model": "general", "language": "en-US", "tier": "enhanced" }
        response = await dg_client.transcription.prerecorded(source, options)
        transcript = response["results"]["channels"][0]["alternatives"][0]["paragraphs"]["transcript"] 
        with open("Enhanced.txt","w") as f:
          f.write(transcript)
        print(json.dumps(transcript, indent=4))


async def keywords(): 
    # Initializes the Deepgram SDK
    dg_client = Deepgram(DEEPGRAM_API_KEY) 
    
    with open(PATH_TO_FILE, 'rb') as audio:
        source = {'buffer': audio, 'mimetype': MIMETYPE}
        options = { "keywords" : ["partition key:6","request unit:6","request units:6","logical partition:6"],"punctuate": True, "diarize" : False, "paragraphs" : True, "model": "general", "language": "en-US", "tier": "enhanced" }
        # ["database", "server", "sql","authorization","api", "subset","latency"]
        response = await dg_client.transcription.prerecorded(source, options)
        transcript = response["results"]["channels"][0]["alternatives"][0]["paragraphs"]["transcript"] 
        with open("Keywords.txt","w") as f:
          f.write(transcript)
        print(json.dumps(transcript, indent=4))


#Method for calculating Word Error Rate (WER)
def wer(ref, hyp ,debug=True):
    r = ref.split()
    h = hyp.split()
    #costs will holds the costs, like in the Levenshtein distance algorithm
    costs = [[0 for inner in range(len(h)+1)] for outer in range(len(r)+1)]
    # backtrace will hold the operations we've done.
    # so we could later backtrace, like the WER algorithm requires us to.
    backtrace = [[0 for inner in range(len(h)+1)] for outer in range(len(r)+1)]
 
    OP_OK = 0
    OP_SUB = 1
    OP_INS = 2
    OP_DEL = 3
    DEL_PENALTY = 1
    INS_PENALTY = 1
    SUB_PENALTY = 1
    
    # First column represents the case where we achieve zero
    # hypothesis words by deleting all reference words.
    for i in range(1, len(r)+1):
        costs[i][0] = DEL_PENALTY*i
        backtrace[i][0] = OP_DEL
    
    # First row represents the case where we achieve the hypothesis
    # by inserting all hypothesis words into a zero-length reference.
    for j in range(1, len(h) + 1):
        costs[0][j] = INS_PENALTY * j
        backtrace[0][j] = OP_INS
    
    # computation
    for i in range(1, len(r)+1):
        for j in range(1, len(h)+1):
            if r[i-1] == h[j-1]:
                costs[i][j] = costs[i-1][j-1]
                backtrace[i][j] = OP_OK
            else:
                substitutionCost = costs[i-1][j-1] + SUB_PENALTY # penalty is always 1
                insertionCost    = costs[i][j-1] + INS_PENALTY   # penalty is always 1
                deletionCost     = costs[i-1][j] + DEL_PENALTY   # penalty is always 1
                 
                costs[i][j] = min(substitutionCost, insertionCost, deletionCost)
                if costs[i][j] == substitutionCost:
                    backtrace[i][j] = OP_SUB
                elif costs[i][j] == insertionCost:
                    backtrace[i][j] = OP_INS
                else:
                    backtrace[i][j] = OP_DEL
                 
    # back trace though the best route:
    i = len(r)
    j = len(h)
    numSub = 0
    numDel = 0
    numIns = 0
    numCor = 0
    if debug:
        print("OP\tREF\tHYP")
        lines = []
        subs1 = []
        subs2 = []
    while i > 0 or j > 0:
        if backtrace[i][j] == OP_OK:
            numCor += 1
            i-=1
            j-=1
            if debug:
                lines.append("OK\t" + r[i]+"\t"+h[j])
        elif backtrace[i][j] == OP_SUB:
            numSub +=1
            i-=1
            j-=1
            if debug:
                lines.append("SUB\t" + r[i]+"\t"+h[j])
                subs1.append(r[i])
                subs2.append(h[j])
        elif backtrace[i][j] == OP_INS:
            numIns += 1
            j-=1
            if debug:
                lines.append("INS\t" + "****" + "\t" + h[j])
        elif backtrace[i][j] == OP_DEL:
            numDel += 1
            i-=1
            if debug:
                lines.append("DEL\t" + r[i]+"\t"+"****")
    if debug:
        lines = reversed(lines)
        for line in lines: # this views idividual lines
            print(line)
        print("#cor " + str(numCor))
        print("#sub " + str(numSub))
        print("#del " + str(numDel))
        print("#ins " + str(numIns))
        print("substituted: ")
        print(Counter(subs1))
        print("subs: " )
        print(Counter(subs2)) #Just to work out what the common subs are 
    # return (numSub + numDel + numIns) / (float) (len(r))
    wer_result = round( (numSub + numDel + numIns) / (float) (len(r)), 3)
    print('WER : ' + str(wer_result) + ', numCor: ' + str(numCor) + ', numSub: ' + str(numSub) +  ', numIns: ' + str(numIns) + ', numDel: ' + str(numDel) +  ", numCount: " +  str(len(r)))

def compareWER(ref,hyp):
    with open(ref,'r') as file:
      ref = file.read()
      ref = ref.lower()
      ref = re.sub(r'[^\w\s]',"",ref) #removes punctuation
      #ref = ref.replace("\n", "")
    with open(hyp,'r') as file:
      hyp = file.read()
      hyp = hyp.lower()
      hyp = re.sub(r'[^\w\s]',"",hyp) #removes punctuation 
      #hyp = hyp.replace("\n","")
    wer(ref,hyp)



    
#Choose which Deepgram version to run here:      
#asyncio.get_event_loop().run_until_complete(enhanced())

#Run Word Error Rate check here:
compareWER("transcripts/transcript1/cosmosDB.txt","Enhanced.txt")
#compareWER("lyrics.txt","Enhanced.txt")







# For music.mp3
#Using Deepgram
#Standard WER : 0.651, numCor: 299, numSub: 208, numIns: 10, numDel: 322, numCount: 829
#Enhanced WER : 0.525, numCor: 415, numSub: 352, numIns: 21, numDel: 62, numCount: 829
#Key word boosting WER : 0.525, numCor: 415, numSub: 352, numIns: 21, numDel: 62, numCount: 829
#Lower the WER the better
#Using Falcon AI

#For comosDB.mp3
#Standard 
#Enhanced WER : 0.15, numCor: 5659, numSub: 439, numIns: 436, numDel: 45, numCount: 6143
#Key word boosting (unigram)
# Bigram 
# Trigram  
#Using Falcon AI
#Perhaps using different ASR provider, could talk about why the accuracy of this is higher and why it may differ to the lierature review
