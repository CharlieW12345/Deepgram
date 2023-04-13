from text2digits import text2digits
t2d = text2digits.Text2Digits()

with open("Transcripts/transcript1/CosmosDB.txt",'r') as f:
    file = f.read()
    # Convert numeric words to numbers
    # Using join() + split()
    res = t2d.convert(file)
    print(res)
    with open("Transcripts/transcript1/CosmosDB.txt","w") as f:
        f.write(res)



